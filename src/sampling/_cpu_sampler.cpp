// High-performance C++17 CPU neighbor sampler, CSR transposer, bucketer, and row gatherer.
// Uses a lightweight persistent C++17 std::thread pool (NO OpenMP / libgomp dependency) and a
// pure extern "C" ABI so one compiled .so works across Python and PyTorch versions via ctypes.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

static inline int64_t current_pid() {
#if defined(__unix__) || defined(__APPLE__)
    return static_cast<int64_t>(::getpid());
#else
    return 0;
#endif
}

static inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t rand_range(uint64_t& state, uint64_t bound) {
    uint64_t r = splitmix64(state);
    state += 0x9e3779b97f4a7c15ULL;
    return static_cast<uint64_t>((static_cast<unsigned __int128>(r) * static_cast<unsigned __int128>(bound)) >> 64);
}

static inline uint64_t next_pow2(uint64_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

static inline int ctz64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(v);
#else
    int c = 0;
    while ((v & 1ULL) == 0) {
        v >>= 1;
        ++c;
    }
    return c;
#endif
}

static inline size_t hash_u64(uint64_t k, size_t mask) {
    return static_cast<size_t>((k * 11400714819323198485ULL) >> 32) & mask;
}

static constexpr int64_t kDenseThreshold = 4'000'000;
static constexpr int kNumPartitions = 32;
static constexpr int kMaxThreads = 64;

// Hybrid lock-free spin-then-sleep sense-reversing barrier.
// Fast path (all threads arrive within ~8 us): zero mutexes, zero syscalls.
// Slow path (a thread is preempted by the OS on a shared host): waiting threads sleep on a
// condition variable instead of spinning in sched_yield(), preventing preemption convoys.
struct SpinBarrier {
    explicit SpinBarrier(int n) : total_(n), count_(n), phase_(0), waiters_(0) {}

    void wait(uint64_t& local_phase) {
        const uint64_t target = ++local_phase;
        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            count_.store(total_, std::memory_order_relaxed);
            phase_.store(target, std::memory_order_seq_cst);
            if (waiters_.load(std::memory_order_seq_cst) > 0) {
                std::lock_guard<std::mutex> lk(mu_);
                cv_.notify_all();
            }
            return;
        }
        // ~100 us spin covers normal per-thread work skew without futex syscalls, then sleeps
        // cleanly if a sibling thread was OS-preempted on a loaded host.
        for (int spin = 0; spin < 3000; ++spin) {
            if (phase_.load(std::memory_order_acquire) >= target) return;
            cpu_relax();
        }
        std::unique_lock<std::mutex> lk(mu_);
        waiters_.fetch_add(1, std::memory_order_seq_cst);
        cv_.wait(lk, [this, target]() {
            return phase_.load(std::memory_order_seq_cst) >= target;
        });
        waiters_.fetch_sub(1, std::memory_order_relaxed);
    }

    int total_;
    std::atomic<int> count_;
    std::atomic<uint64_t> phase_;
    std::atomic<int> waiters_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// Lightweight per-state C++17 thread pool (no OpenMP, no global threads, fork-safe because each
// CpuSamplerState creates threads lazily only when num_threads > 1 is actually called on it).
// Every worker acknowledges each generation before run() returns, so consecutive calls with
// different thread counts are 100% race-free.
class ThreadPool {
public:
    ThreadPool() : owner_pid_(current_pid()) {}
    ~ThreadPool() { shutdown(); }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    bool is_forked_child() const { return current_pid() != owner_pid_; }

    void ensure_workers(int total_threads) {
        const int need = std::max(0, std::min(total_threads, kMaxThreads) - 1);
        if (need <= static_cast<int>(workers_.size())) return;
        uint64_t init_gen = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            init_gen = job_gen_.load(std::memory_order_relaxed);
        }
        const int old_n = static_cast<int>(workers_.size());
        workers_.reserve(static_cast<size_t>(need));
        for (int w = old_n; w < need; ++w) {
            const int tid = w + 1;
            workers_.emplace_back([this, tid, init_gen]() { worker_loop(tid, init_gen); });
        }
    }

    template <typename Fn>
    void run(int total_threads, Fn&& fn) {
        const int nt = std::clamp(total_threads, 1, kMaxThreads);
        if (nt == 1) {
            fn(0, 1);
            return;
        }
        ensure_workers(nt);
        const int num_workers = static_cast<int>(workers_.size());
        auto trampoline = [](void* ctx, int tid, int nthreads) {
            (*static_cast<Fn*>(ctx))(tid, nthreads);
        };
        // Dekker seq_cst publish/check guarantees zero lost wakeups without taking mu_ when spinning.
        fn_ = trampoline;
        fn_ctx_ = &fn;
        active_nt_ = nt;
        remaining_.store(num_workers, std::memory_order_relaxed);
        job_gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_workers_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_work_.notify_all();
        }

        fn(0, nt);

        for (int spin = 0; spin < 3000; ++spin) {
            if (remaining_.load(std::memory_order_acquire) == 0) return;
            cpu_relax();
        }
        std::unique_lock<std::mutex> lk(mu_);
        caller_sleeping_.store(true, std::memory_order_seq_cst);
        cv_done_.wait(lk, [this]() { return remaining_.load(std::memory_order_seq_cst) == 0; });
        caller_sleeping_.store(false, std::memory_order_relaxed);
    }

    void shutdown() {
        if (workers_.empty()) return;
        if (is_forked_child()) {
            // Threads from the parent do not exist after fork(); never join or lock parent mutexes.
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true, std::memory_order_relaxed);
            job_gen_.fetch_add(1, std::memory_order_release);
        }
        cv_work_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        stop_.store(false, std::memory_order_relaxed);
    }

private:
    void worker_loop(int tid, uint64_t seen_gen) {
        while (true) {
            uint64_t g = job_gen_.load(std::memory_order_acquire);
            if (g == seen_gen) {
                // Spin ~100 us for back-to-back layers, then sleep to avoid burning cores when idle.
                for (int spin = 0; spin < 3000; ++spin) {
                    g = job_gen_.load(std::memory_order_acquire);
                    if (g != seen_gen) break;
                    cpu_relax();
                }
            }
            if (g == seen_gen) {
                std::unique_lock<std::mutex> lk(mu_);
                sleeping_workers_.fetch_add(1, std::memory_order_seq_cst);
                cv_work_.wait(lk, [this, seen_gen]() {
                    return stop_.load(std::memory_order_acquire) ||
                           job_gen_.load(std::memory_order_seq_cst) != seen_gen;
                });
                sleeping_workers_.fetch_sub(1, std::memory_order_relaxed);
                g = job_gen_.load(std::memory_order_acquire);
            }
            if (stop_.load(std::memory_order_acquire)) return;
            seen_gen = g;

            const int nt = active_nt_;
            auto fn = fn_;
            void* ctx = fn_ctx_;
            if (tid < nt) {
                fn(ctx, tid, nt);
            }
            if (remaining_.fetch_sub(1, std::memory_order_seq_cst) == 1) {
                if (caller_sleeping_.load(std::memory_order_seq_cst)) {
                    std::lock_guard<std::mutex> lk(mu_);
                    cv_done_.notify_one();
                }
            }
        }
    }

    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::atomic<uint64_t> job_gen_{0};
    std::atomic<int> remaining_{0};
    std::atomic<int> sleeping_workers_{0};
    std::atomic<bool> caller_sleeping_{false};
    std::atomic<bool> stop_{false};
    void (*fn_)(void*, int, int) = nullptr;
    void* fn_ctx_ = nullptr;
    int active_nt_ = 1;
    int64_t owner_pid_ = 0;
    std::vector<std::thread> workers_;
};

struct HashSlot {
    int64_t key;
    int32_t local_id;
    uint32_t gen;
};

struct CpuSamplerState {
    int64_t num_nodes = 0;
    bool use_dense = false;
    uint32_t cur_gen = 0;

    std::unique_ptr<ThreadPool> pool_ptr;
    std::vector<int32_t> map32;  // initialized to -1; restored to -1 for active nodes after each layer
    std::vector<HashSlot> htab;
    size_t hmask = 0;

    // Reusable scratch buffers (avoid per-batch malloc / zero-page faults).
    std::vector<int32_t> raw_i32;
    std::vector<int32_t> part_i32;
    std::vector<int32_t> part_v_i32;
    std::vector<int64_t> raw_i64;
    std::vector<int64_t> part_u_i64;
    std::vector<int64_t> part_v_i64;
    std::vector<int64_t> src_nodes;
    std::vector<std::vector<int32_t>> p_new;
    std::vector<int32_t> bwd_counts_i32;
    std::vector<int32_t> bwd_cursor_i32;
    std::vector<int64_t> bwd_counts_i64;
    std::vector<int64_t> bwd_cursor_i64;
    std::vector<int32_t> fwd_light_tmp;
    std::vector<int32_t> fwd_heavy_tmp;
    std::vector<int32_t> bwd_light_tmp;
    std::vector<int32_t> bwd_heavy_tmp;

    explicit CpuSamplerState(int64_t n) : num_nodes(n), use_dense(n > 0 && n <= kDenseThreshold), p_new(kNumPartitions) {
        if (use_dense) {
            map32.assign(static_cast<size_t>(n), -1);
        }
    }

    ~CpuSamplerState() {
        if (pool_ptr && pool_ptr->is_forked_child()) {
            // In a forked child, abandon the dead parent pool without running ~ThreadPool()/pthread_join.
            pool_ptr.release();
        }
    }

    ThreadPool& get_pool() {
        if (pool_ptr && pool_ptr->is_forked_child()) {
            pool_ptr.release();
        }
        if (!pool_ptr) {
            pool_ptr = std::make_unique<ThreadPool>();
        }
        return *pool_ptr;
    }

    void shutdown_pool() {
        if (!pool_ptr) return;
        if (pool_ptr->is_forked_child()) {
            pool_ptr.release();
        } else {
            pool_ptr->shutdown();
            pool_ptr.reset();
        }
    }

    uint32_t next_gen() {
        cur_gen++;
        if (cur_gen == 0) {
            cur_gen = 1;
            if (!htab.empty()) {
                std::memset(htab.data(), 0, htab.size() * sizeof(HashSlot));
            }
        }
        return cur_gen;
    }

    void ensure_htab(size_t min_items) {
        size_t want = static_cast<size_t>(next_pow2(std::max<uint64_t>(64, min_items * 2 + 16)));
        if (htab.size() < want) {
            htab.assign(want, HashSlot{0, -1, 0});
            hmask = want - 1;
            cur_gen = 1;
        }
    }
};

static inline int choose_threads(int requested, int64_t work_items, int64_t items_per_thread) {
    if (requested <= 1 || work_items < items_per_thread * 2) return 1;
    const int64_t useful = work_items / items_per_thread;
    return static_cast<int>(std::min<int64_t>(std::min(requested, kMaxThreads), std::max<int64_t>(1, useful)));
}

// In-stack Floyd sampling without replacement for f <= 64.
static inline void floyd_picks_small(int64_t deg, int64_t f, int64_t* __restrict__ picks, uint64_t rng) {
    for (int64_t it = 0; it < f; ++it) {
        const int64_t j = deg - f + it;
        int64_t t = static_cast<int64_t>(rand_range(rng, static_cast<uint64_t>(j + 1)));
        for (int64_t p = 0; p < it; ++p) {
            if (picks[p] == t) {
                t = j;
                break;
            }
        }
        picks[it] = t;
    }
}

template <typename InIdxT, typename OutRawT>
static inline void sample_row_large_fanout(
    const InIdxT* __restrict__ indices,
    int64_t start,
    int64_t deg,
    int64_t fanout,
    OutRawT* __restrict__ out,
    uint64_t rng
) {
    const size_t cap = static_cast<size_t>(next_pow2(static_cast<uint64_t>(fanout) * 2 + 16));
    const size_t mask = cap - 1;
    std::vector<int64_t> seen(cap, -1);
    for (int64_t it = 0; it < fanout; ++it) {
        const int64_t j = deg - fanout + it;
        int64_t t = static_cast<int64_t>(rand_range(rng, static_cast<uint64_t>(j + 1)));
        size_t pos = hash_u64(static_cast<uint64_t>(t), mask);
        bool dup = false;
        while (seen[pos] != -1) {
            if (seen[pos] == t) {
                dup = true;
                break;
            }
            pos = (pos + 1) & mask;
        }
        if (dup) {
            t = j;
            pos = hash_u64(static_cast<uint64_t>(t), mask);
            while (seen[pos] != -1) pos = (pos + 1) & mask;
        }
        seen[pos] = t;
        out[it] = static_cast<OutRawT>(indices[start + t]);
    }
}

// Zero-atomic parallel sampling + 32-way 4KB-page-aligned partitioned dedup for N <= kDenseThreshold.
// Runs inside ONE ThreadPool::run() call using a lock-free SpinBarrier between phases.
template <typename InIdxT, typename OutIdxT>
static int64_t sample_relabel_dense_partitioned(
    CpuSamplerState& st,
    const int64_t* __restrict__ indptr,
    const InIdxT* __restrict__ indices,
    const int64_t* __restrict__ dst,
    int64_t n,
    int64_t fanout,
    const int64_t* __restrict__ out_indptr,
    OutIdxT* __restrict__ out_col,
    uint64_t seed,
    int num_threads
) {
    const int64_t total_edges = out_indptr[n];
    if (total_edges == 0) {
        st.src_nodes.resize(static_cast<size_t>(n));
        if (n > 0) std::memcpy(st.src_nodes.data(), dst, static_cast<size_t>(n) * sizeof(int64_t));
        return n;
    }

    int32_t* __restrict__ map32 = st.map32.data();
    // Scale threads with layer size so small layers (~5K-50K edges) don't over-parallelize.
    const int threads = choose_threads(num_threads, total_edges, 32768);

    // Single-threaded fused fast path.
    if (threads == 1) {
        st.src_nodes.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const int64_t v = dst[i];
            map32[v] = static_cast<int32_t>(i);
            st.src_nodes[static_cast<size_t>(i)] = v;
        }
        int64_t num_src = n;
        int64_t picks[64];
        int32_t u_buf[64];

        auto emit_one = [&](int64_t u) -> OutIdxT {
            int32_t lid = map32[u];
            if (lid < 0) {
                lid = static_cast<int32_t>(num_src++);
                map32[u] = lid;
                st.src_nodes.push_back(u);
            }
            return static_cast<OutIdxT>(lid);
        };

        for (int64_t i = 0; i < n; ++i) {
            if (i + 8 < n) __builtin_prefetch(indptr + dst[i + 8], 0, 0);
            const int64_t v = dst[i];
            const int64_t start = indptr[v];
            const int64_t deg = indptr[v + 1] - start;
            const int64_t out_pos = out_indptr[i];
            if (deg == 0) continue;

            if (fanout < 0 || deg <= fanout) {
                for (int64_t k = 0; k < deg; ++k) {
                    if (k + 12 < deg) __builtin_prefetch(map32 + static_cast<int64_t>(indices[start + k + 12]), 1, 0);
                    out_col[out_pos + k] = emit_one(static_cast<int64_t>(indices[start + k]));
                }
            } else if (fanout <= 64) {
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                floyd_picks_small(deg, fanout, picks, rng);
                for (int64_t k = 0; k < fanout; ++k) __builtin_prefetch(indices + start + picks[k], 0, 0);
                for (int64_t k = 0; k < fanout; ++k) {
                    const int32_t u = static_cast<int32_t>(indices[start + picks[k]]);
                    u_buf[k] = u;
                    __builtin_prefetch(map32 + u, 1, 0);
                }
                for (int64_t k = 0; k < fanout; ++k) {
                    out_col[out_pos + k] = emit_one(static_cast<int64_t>(u_buf[k]));
                }
            } else {
                if (st.raw_i32.size() < static_cast<size_t>(fanout)) st.raw_i32.resize(static_cast<size_t>(fanout));
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                sample_row_large_fanout(indices, start, deg, fanout, st.raw_i32.data(), rng);
                for (int64_t k = 0; k < fanout; ++k) {
                    out_col[out_pos + k] = emit_one(static_cast<int64_t>(st.raw_i32[static_cast<size_t>(k)]));
                }
            }
        }
        for (int64_t i = 0; i < num_src; ++i) {
            map32[st.src_nodes[static_cast<size_t>(i)]] = -1;
        }
        return num_src;
    }

    if (st.raw_i32.size() < static_cast<size_t>(total_edges)) {
        st.raw_i32.resize(static_cast<size_t>(total_edges));
    }
    if (st.part_i32.size() < static_cast<size_t>(total_edges)) {
        st.part_i32.resize(static_cast<size_t>(total_edges));
    }
    int32_t* __restrict__ raw = st.raw_i32.data();
    int32_t* __restrict__ part_buf = st.part_i32.data();

    int64_t hist[kMaxThreads][kNumPartitions];
    int64_t part_start[kNumPartitions + 1];
    int64_t offs[kNumPartitions + 1];
    int64_t num_src = n;
    const int64_t row_chunk = (n + threads - 1) / threads;
    std::atomic<int> next_dedup{0};
    std::atomic<int> next_assign{0};
    std::atomic<int64_t> next_relabel{0};
    std::atomic<int64_t> next_reset{n};  // [n, num_src) are guaranteed unique; [0, n) reset on tid == 0
    SpinBarrier barrier(threads);

    st.get_pool().run(threads, [&](int tid, int nt) {
        uint64_t sense = 0;
        const int64_t r_lo = std::min(n, static_cast<int64_t>(tid) * row_chunk);
        const int64_t r_hi = std::min(n, r_lo + row_chunk);

        // Phase 1: tid == 0 marks dst[0..n-1] sequentially (race-free even if seeds contain duplicates;
        // map32 is not read until Phase 3), while all threads sample neighbors + count partition histograms.
        if (tid == 0) {
            for (int64_t i = 0; i < n; ++i) {
                map32[dst[i]] = static_cast<int32_t>(i);
            }
        }
        int64_t lhist[kNumPartitions] = {0};
        int64_t picks[64];
        for (int64_t i = r_lo; i < r_hi; ++i) {
            if (i + 8 < r_hi) __builtin_prefetch(indptr + dst[i + 8], 0, 0);
            const int64_t v = dst[i];
            const int64_t s = indptr[v];
            const int64_t deg = indptr[v + 1] - s;
            const int64_t op = out_indptr[i];
            if (deg == 0) continue;

            if (fanout < 0 || deg <= fanout) {
                for (int64_t k = 0; k < deg; ++k) {
                    const int32_t u = static_cast<int32_t>(indices[s + k]);
                    raw[op + k] = u;
                    lhist[(u >> 10) & (kNumPartitions - 1)]++;
                }
            } else if (fanout <= 64) {
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                floyd_picks_small(deg, fanout, picks, rng);
                for (int64_t k = 0; k < fanout; ++k) __builtin_prefetch(indices + s + picks[k], 0, 0);
                for (int64_t k = 0; k < fanout; ++k) {
                    const int32_t u = static_cast<int32_t>(indices[s + picks[k]]);
                    raw[op + k] = u;
                    lhist[(u >> 10) & (kNumPartitions - 1)]++;
                }
            } else {
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                sample_row_large_fanout(indices, s, deg, fanout, raw + op, rng);
                for (int64_t k = 0; k < fanout; ++k) {
                    lhist[(raw[op + k] >> 10) & (kNumPartitions - 1)]++;
                }
            }
        }
        for (int p = 0; p < kNumPartitions; ++p) hist[tid][p] = lhist[p];

        barrier.wait(sense);

        // Compute per-thread scatter offsets locally from hist[] (saves a full barrier!).
        int64_t cur[kNumPartitions];
        {
            int64_t tot = 0;
            for (int p = 0; p < kNumPartitions; ++p) {
                if (tid == 0) part_start[p] = tot;
                for (int t = 0; t < nt; ++t) {
                    if (t == tid) cur[p] = tot;
                    tot += hist[t][p];
                }
            }
            if (tid == 0) part_start[kNumPartitions] = tot;
        }

        // Phase 2: Scatter raw -> part_buf by partition.
        for (int64_t i = r_lo; i < r_hi; ++i) {
            const int64_t op = out_indptr[i];
            const int64_t end = out_indptr[i + 1];
            for (int64_t e = op; e < end; ++e) {
                const int32_t u = raw[e];
                part_buf[cur[(u >> 10) & (kNumPartitions - 1)]++] = u;
            }
        }
        barrier.wait(sense);

        // Phase 3: Zero-atomic partitioned dedup (disjoint 1024-node / 4KB-aligned chunks of map32).
        while (true) {
            const int p = next_dedup.fetch_add(1, std::memory_order_relaxed);
            if (p >= kNumPartitions) break;
            auto& my = st.p_new[static_cast<size_t>(p)];
            my.clear();
            const int64_t s = part_start[p];
            const int64_t e_end = part_start[p + 1];
            for (int64_t e = s; e < e_end; ++e) {
                if (e + 16 < e_end) __builtin_prefetch(map32 + part_buf[e + 16], 1, 0);
                const int32_t u = part_buf[e];
                if (map32[u] == -1) {
                    map32[u] = -2;
                    my.push_back(u);
                }
            }
        }
        barrier.wait(sense);

        if (tid == 0) {
            offs[0] = n;
            for (int p = 0; p < kNumPartitions; ++p) {
                offs[p + 1] = offs[p] + static_cast<int64_t>(st.p_new[static_cast<size_t>(p)].size());
            }
            num_src = offs[kNumPartitions];
            st.src_nodes.resize(static_cast<size_t>(num_src));
        }
        barrier.wait(sense);

        // Phase 4: Copy dst into src_nodes + assign final local IDs per partition.
        if (r_lo < r_hi) {
            std::memcpy(st.src_nodes.data() + r_lo, dst + r_lo, static_cast<size_t>(r_hi - r_lo) * sizeof(int64_t));
        }
        while (true) {
            const int p = next_assign.fetch_add(1, std::memory_order_relaxed);
            if (p >= kNumPartitions) break;
            const int64_t base = offs[p];
            const auto& my = st.p_new[static_cast<size_t>(p)];
            const size_t m = my.size();
            for (size_t k = 0; k < m; ++k) {
                const int32_t u = my[k];
                map32[u] = static_cast<int32_t>(base + static_cast<int64_t>(k));
                st.src_nodes[static_cast<size_t>(base) + k] = static_cast<int64_t>(u);
            }
        }
        barrier.wait(sense);

        // Phase 5: Dynamic work-stealing relabel out_col[e] = map32[raw[e]].
        while (true) {
            const int64_t e_lo = next_relabel.fetch_add(8192, std::memory_order_relaxed);
            if (e_lo >= total_edges) break;
            const int64_t e_hi = std::min(total_edges, e_lo + 8192);
            for (int64_t e = e_lo; e < e_hi; ++e) {
                if (e + 16 < e_hi) __builtin_prefetch(map32 + raw[e + 16], 0, 0);
                out_col[e] = static_cast<OutIdxT>(map32[raw[e]]);
            }
        }
        barrier.wait(sense);

        // Phase 6: Reset touched entries in map32 back to -1.
        // dst[0..n-1] may contain duplicate seeds, so tid == 0 resets [0, n) sequentially while all
        // threads dynamically reset the guaranteed-unique new sources [n, num_src) (disjoint from dst).
        if (tid == 0) {
            for (int64_t i = 0; i < n; ++i) {
                map32[dst[i]] = -1;
            }
        }
        while (true) {
            const int64_t s_lo = next_reset.fetch_add(8192, std::memory_order_relaxed);
            if (s_lo >= num_src) break;
            const int64_t s_hi = std::min(num_src, s_lo + 8192);
            for (int64_t i = s_lo; i < s_hi; ++i) {
                map32[st.src_nodes[static_cast<size_t>(i)]] = -1;
            }
        }
    });

    return num_src;
}

// Hash-table path for huge graphs (N > kDenseThreshold): O(n + E_batch) memory only.
template <typename InIdxT, typename OutIdxT>
static int64_t sample_relabel_hash(
    CpuSamplerState& st,
    const int64_t* __restrict__ indptr,
    const InIdxT* __restrict__ indices,
    const int64_t* __restrict__ dst,
    int64_t n,
    int64_t fanout,
    const int64_t* __restrict__ out_indptr,
    OutIdxT* __restrict__ out_col,
    uint64_t seed,
    int num_threads
) {
    const int64_t total_edges = out_indptr[n];
    if (total_edges == 0) {
        st.src_nodes.resize(static_cast<size_t>(n));
        if (n > 0) std::memcpy(st.src_nodes.data(), dst, static_cast<size_t>(n) * sizeof(int64_t));
        return n;
    }
    if (st.raw_i64.size() < static_cast<size_t>(total_edges)) {
        st.raw_i64.resize(static_cast<size_t>(total_edges));
    }
    int64_t* __restrict__ raw_src = st.raw_i64.data();

    const int threads = choose_threads(num_threads, total_edges, 16384);
    const int64_t row_chunk = (n + threads - 1) / threads;
    st.get_pool().run(threads, [&](int tid, int /*nt*/) {
        const int64_t lo = std::min(n, static_cast<int64_t>(tid) * row_chunk);
        const int64_t hi = std::min(n, lo + row_chunk);
        int64_t picks[64];
        for (int64_t i = lo; i < hi; ++i) {
            if (i + 8 < hi) __builtin_prefetch(indptr + dst[i + 8], 0, 0);
            const int64_t v = dst[i];
            const int64_t start = indptr[v];
            const int64_t deg = indptr[v + 1] - start;
            const int64_t out_pos = out_indptr[i];
            if (deg == 0) continue;
            if (fanout < 0 || deg <= fanout) {
                for (int64_t k = 0; k < deg; ++k) raw_src[out_pos + k] = static_cast<int64_t>(indices[start + k]);
            } else if (fanout <= 64) {
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                floyd_picks_small(deg, fanout, picks, rng);
                for (int64_t k = 0; k < fanout; ++k) __builtin_prefetch(indices + start + picks[k], 0, 0);
                for (int64_t k = 0; k < fanout; ++k) raw_src[out_pos + k] = static_cast<int64_t>(indices[start + picks[k]]);
            } else {
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                sample_row_large_fanout(indices, start, deg, fanout, raw_src + out_pos, rng);
            }
        }
    });

    st.ensure_htab(static_cast<size_t>(n + total_edges));
    const uint32_t gen = st.next_gen();
    HashSlot* __restrict__ tab = st.htab.data();
    const size_t mask = st.hmask;

    st.src_nodes.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = dst[i];
        st.src_nodes[static_cast<size_t>(i)] = v;
        size_t pos = hash_u64(static_cast<uint64_t>(v), mask);
        while (tab[pos].gen == gen && tab[pos].key != v) pos = (pos + 1) & mask;
        tab[pos].key = v;
        tab[pos].local_id = static_cast<int32_t>(i);
        tab[pos].gen = gen;
    }

    int64_t num_src = n;
    for (int64_t e = 0; e < total_edges; ++e) {
        if (e + 16 < total_edges) {
            __builtin_prefetch(tab + hash_u64(static_cast<uint64_t>(raw_src[e + 16]), mask), 1, 0);
        }
        const int64_t u = raw_src[e];
        size_t pos = hash_u64(static_cast<uint64_t>(u), mask);
        while (true) {
            HashSlot& s = tab[pos];
            if (s.gen != gen) {
                s.key = u;
                s.local_id = static_cast<int32_t>(num_src);
                s.gen = gen;
                st.src_nodes.push_back(u);
                out_col[e] = static_cast<OutIdxT>(num_src);
                num_src++;
                break;
            }
            if (s.key == u) {
                out_col[e] = static_cast<OutIdxT>(s.local_id);
                break;
            }
            pos = (pos + 1) & mask;
        }
    }
    return num_src;
}

// Zero-atomic, zero-branch, 32-way parallel O(num_src + E) counting-sort CSR transpose.
// Preserves exact ascending dst order inside every transposed source row with zero post-sorting!
template <typename IdxT>
static void transpose_csr_impl(
    CpuSamplerState& st,
    const IdxT* __restrict__ fwd_indptr,
    const IdxT* __restrict__ fwd_indices,
    int64_t num_dst,
    int64_t num_src,
    int64_t num_edges,
    IdxT* __restrict__ bwd_indptr,
    IdxT* __restrict__ bwd_indices,
    int num_threads
) {
    if (num_src == 0) {
        bwd_indptr[0] = 0;
        return;
    }
    if (num_edges == 0) {
        std::memset(bwd_indptr, 0, static_cast<size_t>(num_src + 1) * sizeof(IdxT));
        return;
    }

    const int threads = choose_threads(num_threads, num_edges, 32768);
    if (threads == 1 || num_src < 1024) {
        if constexpr (sizeof(IdxT) == 4) {
            st.bwd_counts_i32.assign(static_cast<size_t>(num_src), 0);
            int32_t* __restrict__ counts = st.bwd_counts_i32.data();
            for (int64_t e = 0; e < num_edges; ++e) counts[fwd_indices[e]]++;
            int32_t acc = 0;
            st.bwd_cursor_i32.resize(static_cast<size_t>(num_src));
            int32_t* __restrict__ cursor = st.bwd_cursor_i32.data();
            for (int64_t u = 0; u < num_src; ++u) {
                bwd_indptr[u] = acc;
                cursor[u] = acc;
                acc += counts[u];
            }
            bwd_indptr[num_src] = acc;
            for (int64_t v = 0; v < num_dst; ++v) {
                const int32_t s = fwd_indptr[v];
                const int32_t e_end = fwd_indptr[v + 1];
                for (int32_t e = s; e < e_end; ++e) {
                    bwd_indices[cursor[fwd_indices[e]]++] = static_cast<int32_t>(v);
                }
            }
        } else {
            st.bwd_counts_i64.assign(static_cast<size_t>(num_src), 0);
            int64_t* __restrict__ counts = st.bwd_counts_i64.data();
            for (int64_t e = 0; e < num_edges; ++e) counts[fwd_indices[e]]++;
            int64_t acc = 0;
            st.bwd_cursor_i64.resize(static_cast<size_t>(num_src));
            int64_t* __restrict__ cursor = st.bwd_cursor_i64.data();
            for (int64_t u = 0; u < num_src; ++u) {
                bwd_indptr[u] = acc;
                cursor[u] = acc;
                acc += counts[u];
            }
            bwd_indptr[num_src] = acc;
            for (int64_t v = 0; v < num_dst; ++v) {
                const int64_t s = fwd_indptr[v];
                const int64_t e_end = fwd_indptr[v + 1];
                for (int64_t e = s; e < e_end; ++e) {
                    bwd_indices[cursor[fwd_indices[e]]++] = v;
                }
            }
        }
        return;
    }

    // Power-of-two block size (>= 16 for 64-byte cacheline alignment) gives <= 32 contiguous partitions.
    const uint64_t block_size = std::max<uint64_t>(16ULL, next_pow2(static_cast<uint64_t>((num_src + kNumPartitions - 1) / kNumPartitions)));
    const int shift = ctz64(block_size);
    const int num_parts = static_cast<int>((static_cast<uint64_t>(num_src) + block_size - 1) / block_size);

    IdxT* part_u = nullptr;
    IdxT* part_v = nullptr;
    if constexpr (sizeof(IdxT) == 4) {
        if (st.part_i32.size() < static_cast<size_t>(num_edges)) st.part_i32.resize(static_cast<size_t>(num_edges));
        if (st.part_v_i32.size() < static_cast<size_t>(num_edges)) st.part_v_i32.resize(static_cast<size_t>(num_edges));
        part_u = st.part_i32.data();
        part_v = st.part_v_i32.data();
    } else {
        if (st.part_u_i64.size() < static_cast<size_t>(num_edges)) st.part_u_i64.resize(static_cast<size_t>(num_edges));
        if (st.part_v_i64.size() < static_cast<size_t>(num_edges)) st.part_v_i64.resize(static_cast<size_t>(num_edges));
        part_u = st.part_u_i64.data();
        part_v = st.part_v_i64.data();
    }

    st.bwd_counts_i64.resize(static_cast<size_t>(num_src));
    st.bwd_cursor_i64.resize(static_cast<size_t>(num_src));
    int64_t* __restrict__ counts = st.bwd_counts_i64.data();
    int64_t* __restrict__ cursor = st.bwd_cursor_i64.data();

    int64_t hist[kMaxThreads][kNumPartitions];
    int64_t part_offs[kMaxThreads][kNumPartitions];
    int64_t part_start[kNumPartitions + 1];
    const int64_t dst_chunk = (num_dst + threads - 1) / threads;
    std::atomic<int> next_part{0};
    SpinBarrier barrier(threads);

    st.get_pool().run(threads, [&](int tid, int nt) {
        uint64_t sense = 0;
        const int64_t v_lo = std::min(num_dst, static_cast<int64_t>(tid) * dst_chunk);
        const int64_t v_hi = std::min(num_dst, v_lo + dst_chunk);

        // Phase 1: Per-thread histogram over partitions p = u >> shift.
        int64_t lhist[kNumPartitions] = {0};
        if (v_lo < v_hi) {
            const int64_t e_begin = static_cast<int64_t>(fwd_indptr[v_lo]);
            const int64_t e_end = static_cast<int64_t>(fwd_indptr[v_hi]);
            for (int64_t e = e_begin; e < e_end; ++e) {
                lhist[static_cast<uint64_t>(fwd_indices[e]) >> shift]++;
            }
        }
        for (int p = 0; p < num_parts; ++p) hist[tid][p] = lhist[p];

        barrier.wait(sense);
        if (tid == 0) {
            int64_t tot = 0;
            for (int p = 0; p < num_parts; ++p) {
                part_start[p] = tot;
                for (int t = 0; t < nt; ++t) {
                    part_offs[t][p] = tot;
                    tot += hist[t][p];
                }
            }
            part_start[num_parts] = tot;
            bwd_indptr[num_src] = static_cast<IdxT>(tot);
        }
        barrier.wait(sense);

        // Phase 2: Scatter (u, v) pairs in ascending v order into disjoint partition slices.
        int64_t cur[kNumPartitions];
        for (int p = 0; p < num_parts; ++p) cur[p] = part_offs[tid][p];
        for (int64_t v = v_lo; v < v_hi; ++v) {
            const int64_t s = static_cast<int64_t>(fwd_indptr[v]);
            const int64_t e_end = static_cast<int64_t>(fwd_indptr[v + 1]);
            for (int64_t e = s; e < e_end; ++e) {
                const IdxT u = fwd_indices[e];
                const int64_t pos = cur[static_cast<uint64_t>(u) >> shift]++;
                part_u[pos] = u;
                part_v[pos] = static_cast<IdxT>(v);
            }
        }
        barrier.wait(sense);

        // Phase 3: Independent counting-sort transpose per contiguous source partition (zero atomics!).
        while (true) {
            const int p = next_part.fetch_add(1, std::memory_order_relaxed);
            if (p >= num_parts) break;
            const int64_t u_lo = static_cast<int64_t>(static_cast<uint64_t>(p) << shift);
            const int64_t u_hi = std::min(num_src, static_cast<int64_t>(static_cast<uint64_t>(p + 1) << shift));
            if (u_lo >= u_hi) continue;
            std::memset(counts + u_lo, 0, static_cast<size_t>(u_hi - u_lo) * sizeof(int64_t));
            const int64_t e_lo = part_start[p];
            const int64_t e_hi = part_start[p + 1];
            for (int64_t e = e_lo; e < e_hi; ++e) {
                counts[part_u[e]]++;
            }
            int64_t acc = e_lo;
            for (int64_t u = u_lo; u < u_hi; ++u) {
                bwd_indptr[u] = static_cast<IdxT>(acc);
                cursor[u] = acc;
                acc += counts[u];
            }
            for (int64_t e = e_lo; e < e_hi; ++e) {
                bwd_indices[cursor[part_u[e]]++] = part_v[e];
            }
        }
    });
}

template <typename IdxT>
static void partition_buckets_impl(
    CpuSamplerState& st,
    const IdxT* __restrict__ indptr,
    int64_t n,
    int64_t threshold,
    int deg_order,
    int num_threads,
    std::vector<int32_t>& light_tmp,
    std::vector<int32_t>& heavy_tmp
) {
    heavy_tmp.clear();
    if (n <= 0) {
        light_tmp.clear();
        return;
    }

    if (threshold < 0) {
        light_tmp.resize(static_cast<size_t>(n));
        int32_t* __restrict__ out = light_tmp.data();
        const int threads = choose_threads(num_threads, n, 32768);
        const int64_t chunk = (n + threads - 1) / threads;
        st.get_pool().run(threads, [&](int tid, int /*nt*/) {
            const int64_t lo = std::min(n, static_cast<int64_t>(tid) * chunk);
            const int64_t hi = std::min(n, lo + chunk);
            for (int64_t i = lo; i < hi; ++i) {
                out[i] = static_cast<int32_t>(i);
            }
        });
    } else {
        light_tmp.clear();
        light_tmp.reserve(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const int64_t d = static_cast<int64_t>(indptr[i + 1] - indptr[i]);
            if (d >= threshold) {
                heavy_tmp.push_back(static_cast<int32_t>(i));
            } else {
                light_tmp.push_back(static_cast<int32_t>(i));
            }
        }
    }

    if (deg_order) {
        auto cmp = [indptr](int32_t a, int32_t b) {
            const int64_t da = static_cast<int64_t>(indptr[a + 1] - indptr[a]);
            const int64_t db = static_cast<int64_t>(indptr[b + 1] - indptr[b]);
            return da > db;
        };
        if (light_tmp.size() > 1) std::stable_sort(light_tmp.begin(), light_tmp.end(), cmp);
        if (heavy_tmp.size() > 1) std::stable_sort(heavy_tmp.begin(), heavy_tmp.end(), cmp);
    }
}

}  // namespace

extern "C" {

void* cpu_sampler_state_create(int64_t num_nodes) {
    return new CpuSamplerState(num_nodes);
}

void cpu_sampler_state_destroy(void* ptr) {
    delete static_cast<CpuSamplerState*>(ptr);
}

void cpu_sampler_state_shutdown_pool(void* ptr) {
    if (ptr != nullptr) {
        static_cast<CpuSamplerState*>(ptr)->shutdown_pool();
    }
}

int64_t cpu_sample_layer_indptr(
    void* state_ptr,
    const int64_t* indptr,
    const int64_t* dst,
    int64_t n,
    int64_t fanout,
    void* out_indptr,
    int out_is_i32,
    int64_t* out_indptr_i64_buf,
    int num_threads
) {
    // Single-threaded indptr prefix sum takes <0.35 ms for n <= 131K (avoids thread wakeup + barriers).
    const int threads = (state_ptr != nullptr && n >= 131072) ? choose_threads(num_threads, n, 65536) : 1;
    if (threads > 1) {
        auto& st = *static_cast<CpuSamplerState*>(state_ptr);
        const int64_t chunk = (n + threads - 1) / threads;
        int64_t t_sums[kMaxThreads + 1] = {0};
        SpinBarrier barrier(threads);

        st.get_pool().run(threads, [&](int tid, int nt) {
            uint64_t sense = 0;
            const int64_t lo = std::min(n, static_cast<int64_t>(tid) * chunk);
            const int64_t hi = std::min(n, lo + chunk);
            int64_t s = 0;
            for (int64_t i = lo; i < hi; ++i) {
                if (i + 12 < hi) __builtin_prefetch(indptr + dst[i + 12], 0, 0);
                const int64_t v = dst[i];
                int64_t deg = indptr[v + 1] - indptr[v];
                if (fanout >= 0 && deg > fanout) deg = fanout;
                out_indptr_i64_buf[i] = deg;
                s += deg;
            }
            t_sums[tid + 1] = s;
            barrier.wait(sense);
            if (tid == 0) {
                for (int t = 1; t <= nt; ++t) t_sums[t] += t_sums[t - 1];
                out_indptr_i64_buf[n] = t_sums[nt];
                if (out_is_i32) {
                    static_cast<int32_t*>(out_indptr)[n] = static_cast<int32_t>(t_sums[nt]);
                } else {
                    static_cast<int64_t*>(out_indptr)[n] = t_sums[nt];
                }
            }
            barrier.wait(sense);
            int64_t acc = t_sums[tid];
            if (out_is_i32) {
                auto* ip = static_cast<int32_t*>(out_indptr);
                for (int64_t i = lo; i < hi; ++i) {
                    const int64_t deg = out_indptr_i64_buf[i];
                    out_indptr_i64_buf[i] = acc;
                    ip[i] = static_cast<int32_t>(acc);
                    acc += deg;
                }
            } else {
                auto* ip = static_cast<int64_t*>(out_indptr);
                for (int64_t i = lo; i < hi; ++i) {
                    const int64_t deg = out_indptr_i64_buf[i];
                    out_indptr_i64_buf[i] = acc;
                    ip[i] = acc;
                    acc += deg;
                }
            }
        });
        return out_indptr_i64_buf[n];
    }

    int64_t acc = 0;
    if (out_is_i32) {
        auto* ip = static_cast<int32_t*>(out_indptr);
        for (int64_t i = 0; i < n; ++i) {
            if (i + 12 < n) __builtin_prefetch(indptr + dst[i + 12], 0, 0);
            ip[i] = static_cast<int32_t>(acc);
            out_indptr_i64_buf[i] = acc;
            const int64_t v = dst[i];
            int64_t deg = indptr[v + 1] - indptr[v];
            if (fanout >= 0 && deg > fanout) deg = fanout;
            acc += deg;
        }
        ip[n] = static_cast<int32_t>(acc);
        out_indptr_i64_buf[n] = acc;
    } else {
        auto* ip = static_cast<int64_t*>(out_indptr);
        for (int64_t i = 0; i < n; ++i) {
            if (i + 12 < n) __builtin_prefetch(indptr + dst[i + 12], 0, 0);
            ip[i] = acc;
            out_indptr_i64_buf[i] = acc;
            const int64_t v = dst[i];
            int64_t deg = indptr[v + 1] - indptr[v];
            if (fanout >= 0 && deg > fanout) deg = fanout;
            acc += deg;
        }
        ip[n] = acc;
        out_indptr_i64_buf[n] = acc;
    }
    return acc;
}

int64_t cpu_sample_layer_edges(
    void* state_ptr,
    const int64_t* indptr,
    const void* indices,
    int indices_is_i32,
    const int64_t* dst,
    int64_t n,
    int64_t fanout,
    const int64_t* out_indptr_i64,
    void* out_col,
    int out_is_i32,
    uint64_t seed,
    int num_threads
) {
    auto& st = *static_cast<CpuSamplerState*>(state_ptr);
    if (st.use_dense) {
        if (indices_is_i32) {
            const auto* ix = static_cast<const int32_t*>(indices);
            return out_is_i32
                       ? sample_relabel_dense_partitioned(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int32_t*>(out_col), seed, num_threads)
                       : sample_relabel_dense_partitioned(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int64_t*>(out_col), seed, num_threads);
        } else {
            const auto* ix = static_cast<const int64_t*>(indices);
            return out_is_i32
                       ? sample_relabel_dense_partitioned(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int32_t*>(out_col), seed, num_threads)
                       : sample_relabel_dense_partitioned(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int64_t*>(out_col), seed, num_threads);
        }
    }
    if (indices_is_i32) {
        const auto* ix = static_cast<const int32_t*>(indices);
        return out_is_i32
                   ? sample_relabel_hash(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int32_t*>(out_col), seed, num_threads)
                   : sample_relabel_hash(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int64_t*>(out_col), seed, num_threads);
    } else {
        const auto* ix = static_cast<const int64_t*>(indices);
        return out_is_i32
                   ? sample_relabel_hash(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int32_t*>(out_col), seed, num_threads)
                   : sample_relabel_hash(st, indptr, ix, dst, n, fanout, out_indptr_i64, static_cast<int64_t*>(out_col), seed, num_threads);
    }
}

void cpu_copy_src_nodes(void* state_ptr, int64_t* dst_buf, int64_t num_src) {
    auto& st = *static_cast<CpuSamplerState*>(state_ptr);
    if (num_src > 0) {
        std::memcpy(dst_buf, st.src_nodes.data(), static_cast<size_t>(num_src) * sizeof(int64_t));
    }
}

void cpu_transpose_and_bucket_prep(
    void* state_ptr,
    const void* fwd_indptr,
    const void* fwd_indices,
    int64_t num_dst,
    int64_t num_src,
    int64_t num_edges,
    int idx_is_i32,
    void* bwd_indptr,
    void* bwd_indices,
    int64_t fwd_threshold,
    int64_t bwd_threshold,
    int fwd_deg_order,
    int bwd_deg_order,
    int num_threads,
    int64_t* out_bucket_sizes
) {
    auto& st = *static_cast<CpuSamplerState*>(state_ptr);
    if (idx_is_i32) {
        const auto* fip = static_cast<const int32_t*>(fwd_indptr);
        const auto* fix = static_cast<const int32_t*>(fwd_indices);
        auto* bip = static_cast<int32_t*>(bwd_indptr);
        auto* bix = static_cast<int32_t*>(bwd_indices);
        transpose_csr_impl(st, fip, fix, num_dst, num_src, num_edges, bip, bix, num_threads);
        partition_buckets_impl(st, fip, num_dst, fwd_threshold, fwd_deg_order, num_threads, st.fwd_light_tmp, st.fwd_heavy_tmp);
        partition_buckets_impl(st, bip, num_src, bwd_threshold, bwd_deg_order, num_threads, st.bwd_light_tmp, st.bwd_heavy_tmp);
    } else {
        const auto* fip = static_cast<const int64_t*>(fwd_indptr);
        const auto* fix = static_cast<const int64_t*>(fwd_indices);
        auto* bip = static_cast<int64_t*>(bwd_indptr);
        auto* bix = static_cast<int64_t*>(bwd_indices);
        transpose_csr_impl(st, fip, fix, num_dst, num_src, num_edges, bip, bix, num_threads);
        partition_buckets_impl(st, fip, num_dst, fwd_threshold, fwd_deg_order, num_threads, st.fwd_light_tmp, st.fwd_heavy_tmp);
        partition_buckets_impl(st, bip, num_src, bwd_threshold, bwd_deg_order, num_threads, st.bwd_light_tmp, st.bwd_heavy_tmp);
    }
    out_bucket_sizes[0] = static_cast<int64_t>(st.fwd_light_tmp.size());
    out_bucket_sizes[1] = static_cast<int64_t>(st.fwd_heavy_tmp.size());
    out_bucket_sizes[2] = static_cast<int64_t>(st.bwd_light_tmp.size());
    out_bucket_sizes[3] = static_cast<int64_t>(st.bwd_heavy_tmp.size());
}

void cpu_copy_buckets(
    void* state_ptr,
    void* fwd_light,
    void* fwd_heavy,
    void* bwd_light,
    void* bwd_heavy,
    int idx_is_i32
) {
    auto& st = *static_cast<CpuSamplerState*>(state_ptr);
    auto copy_vec = [idx_is_i32](const std::vector<int32_t>& src, void* dst) {
        if (src.empty()) return;
        if (idx_is_i32) {
            std::memcpy(dst, src.data(), src.size() * sizeof(int32_t));
        } else {
            auto* out = static_cast<int64_t*>(dst);
            for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<int64_t>(src[i]);
        }
    };
    copy_vec(st.fwd_light_tmp, fwd_light);
    copy_vec(st.fwd_heavy_tmp, fwd_heavy);
    copy_vec(st.bwd_light_tmp, bwd_light);
    copy_vec(st.bwd_heavy_tmp, bwd_heavy);
}

// Parallel row gather using the state's C++17 ThreadPool (zero OpenMP).
void cpu_gather_rows(
    void* state_ptr,
    const void* src_ptr,
    const int64_t* idx_ptr,
    void* dst_ptr,
    int64_t n,
    int64_t row_bytes,
    int num_threads
) {
    if (n <= 0 || row_bytes <= 0) return;
    const auto* src = static_cast<const uint8_t*>(src_ptr);
    auto* dst = static_cast<uint8_t*>(dst_ptr);
    const int threads = (state_ptr != nullptr) ? choose_threads(num_threads, n * row_bytes, 1024 * 1024) : 1;
    if (threads <= 1) {
        for (int64_t i = 0; i < n; ++i) {
            if (i + 12 < n) __builtin_prefetch(src + idx_ptr[i + 12] * row_bytes, 0, 0);
            std::memcpy(dst + i * row_bytes, src + idx_ptr[i] * row_bytes, static_cast<size_t>(row_bytes));
        }
        return;
    }
    auto& st = *static_cast<CpuSamplerState*>(state_ptr);
    const int threads_eff = std::min(threads, kMaxThreads);
    const int64_t chunk = (n + threads_eff - 1) / threads_eff;
    st.get_pool().run(threads_eff, [&](int tid, int /*nt*/) {
        const int64_t lo = std::min(n, static_cast<int64_t>(tid) * chunk);
        const int64_t hi = std::min(n, lo + chunk);
        for (int64_t i = lo; i < hi; ++i) {
            if (i + 12 < hi) __builtin_prefetch(src + idx_ptr[i + 12] * row_bytes, 0, 0);
            std::memcpy(dst + i * row_bytes, src + idx_ptr[i] * row_bytes, static_cast<size_t>(row_bytes));
        }
    });
}

}  // extern "C"
