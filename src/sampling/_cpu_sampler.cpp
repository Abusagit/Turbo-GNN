// High-performance C++17 + OpenMP CPU neighbor sampler, CSR transposer, bucketer, and row gatherer.
// Pure extern "C" ABI so one compiled .so works across Python / PyTorch versions via ctypes.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

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

static inline size_t hash_u64(uint64_t k, size_t mask) {
    return static_cast<size_t>((k * 11400714819323198485ULL) >> 32) & mask;
}

// Compact int32 direct-mapped table for graphs with N <= kDenseThreshold (4 bytes/node, e.g. 9.8 MB for 2.45M nodes):
// - Zero atomics: parallel dedup partitions nodes by 4KB-aligned blocks `(u >> 10) & (P - 1)` so each
//   partition has exclusive cacheline ownership of its slice of `map32`.
// - Reset only touches the `num_src` active nodes at the end of each layer (O(num_src), never O(N)).
// Graphs with N > kDenseThreshold (e.g. ogbn-papers100M) use a generation-stamped O(n + E_batch) hash table.
static constexpr int64_t kDenseThreshold = 4'000'000;
static constexpr int kNumPartitions = 32;
static constexpr int kMaxThreads = 64;

struct HashSlot {
    int64_t key;
    int32_t local_id;
    uint32_t gen;
};

struct CpuSamplerState {
    int64_t num_nodes = 0;
    bool use_dense = false;
    uint32_t cur_gen = 0;

    std::vector<int32_t> map32;  // initialized to -1; restored to -1 for active nodes after each layer
    std::vector<HashSlot> htab;
    size_t hmask = 0;

    // Reusable scratch buffers (zero per-batch malloc / zero-page faults).
    std::vector<int32_t> raw_i32;
    std::vector<int32_t> part_i32;
    std::vector<int64_t> raw_i64;
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

// Zero-atomic parallel sampling + 32-way page-aligned partitioned dedup for N <= kDenseThreshold.
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
    const int threads = (total_edges >= 16384 && num_threads > 1) ? std::min(num_threads, kMaxThreads) : 1;

    // Single-threaded fused fast path for small inner layers (zero barrier / scatter overhead).
    if (threads == 1) {
        st.src_nodes.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const int64_t v = dst[i];
            map32[v] = static_cast<int32_t>(i);
            st.src_nodes[static_cast<size_t>(i)] = v;
        }
        int64_t num_src = n;
        int64_t picks[64];

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
                    out_col[out_pos + k] = emit_one(static_cast<int64_t>(indices[start + k]));
                }
            } else if (fanout <= 64) {
                uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
                floyd_picks_small(deg, fanout, picks, rng);
                for (int64_t k = 0; k < fanout; ++k) __builtin_prefetch(indices + start + picks[k], 0, 0);
                for (int64_t k = 0; k < fanout; ++k) {
                    out_col[out_pos + k] = emit_one(static_cast<int64_t>(indices[start + picks[k]]));
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
        // Reset only the `num_src` touched entries back to -1.
        for (int64_t i = 0; i < num_src; ++i) {
            map32[st.src_nodes[static_cast<size_t>(i)]] = -1;
        }
        return num_src;
    }

    // Parallel zero-atomic path for large layers (single OpenMP region so `nt = omp_get_num_threads()`
    // is invariant across histogram, prefix-sum, scatter, dedup, relabel, and reset).
    if (st.raw_i32.size() < static_cast<size_t>(total_edges)) {
        st.raw_i32.resize(static_cast<size_t>(total_edges));
    }
    if (st.part_i32.size() < static_cast<size_t>(total_edges)) {
        st.part_i32.resize(static_cast<size_t>(total_edges));
    }
    int32_t* __restrict__ raw = st.raw_i32.data();
    int32_t* __restrict__ part_buf = st.part_i32.data();

    int64_t hist[kMaxThreads][kNumPartitions];
    int64_t part_offs[kMaxThreads][kNumPartitions];
    int64_t part_start[kNumPartitions + 1];
    int64_t offs[kNumPartitions + 1];
    int64_t num_src = n;

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        // Step 1: Mark dst[0..n-1] + parallel neighbor sampling + per-thread 32-way partition histogram.
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < n; ++i) {
            map32[dst[i]] = static_cast<int32_t>(i);
        }

        int64_t lhist[kNumPartitions] = {0};
        int64_t picks[64];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
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

        #pragma omp barrier
        #pragma omp single
        {
            int64_t tot = 0;
            for (int p = 0; p < kNumPartitions; ++p) {
                part_start[p] = tot;
                for (int t = 0; t < nt; ++t) {
                    part_offs[t][p] = tot;
                    tot += hist[t][p];
                }
            }
            part_start[kNumPartitions] = tot;
        }

        // Step 2: Scatter raw -> part_buf by partition, then zero-atomic partitioned dedup!
        int64_t cur[kNumPartitions];
        for (int p = 0; p < kNumPartitions; ++p) cur[p] = part_offs[tid][p];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const int64_t op = out_indptr[i];
            const int64_t end = out_indptr[i + 1];
            for (int64_t e = op; e < end; ++e) {
                const int32_t u = raw[e];
                part_buf[cur[(u >> 10) & (kNumPartitions - 1)]++] = u;
            }
        }

        // Each partition owns disjoint 1024-node (4KB-aligned) chunks of map32 -> zero atomics, zero false sharing.
        #pragma omp for schedule(dynamic, 1)
        for (int p = 0; p < kNumPartitions; ++p) {
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

        #pragma omp single
        {
            offs[0] = n;
            for (int p = 0; p < kNumPartitions; ++p) {
                offs[p + 1] = offs[p] + static_cast<int64_t>(st.p_new[static_cast<size_t>(p)].size());
            }
            num_src = offs[kNumPartitions];
            st.src_nodes.resize(static_cast<size_t>(num_src));
        }

        // Step 3: Assign final local IDs, copy src_nodes, relabel out_col, and reset map32 back to -1.
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < n; ++i) {
            st.src_nodes[static_cast<size_t>(i)] = dst[i];
        }
        #pragma omp for schedule(dynamic, 1)
        for (int p = 0; p < kNumPartitions; ++p) {
            const int64_t base = offs[p];
            const auto& my = st.p_new[static_cast<size_t>(p)];
            const size_t m = my.size();
            for (size_t k = 0; k < m; ++k) {
                const int32_t u = my[k];
                map32[u] = static_cast<int32_t>(base + static_cast<int64_t>(k));
                st.src_nodes[static_cast<size_t>(base) + k] = static_cast<int64_t>(u);
            }
        }

        #pragma omp for schedule(static)
        for (int64_t e = 0; e < total_edges; ++e) {
            if (e + 16 < total_edges) __builtin_prefetch(map32 + raw[e + 16], 0, 0);
            out_col[e] = static_cast<OutIdxT>(map32[raw[e]]);
        }

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < num_src; ++i) {
            map32[st.src_nodes[static_cast<size_t>(i)]] = -1;
        }
    }
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
    if (st.raw_i64.size() < static_cast<size_t>(total_edges)) {
        st.raw_i64.resize(static_cast<size_t>(total_edges));
    }
    int64_t* __restrict__ raw_src = st.raw_i64.data();

    const int threads = (total_edges >= 4096 && num_threads > 1) ? num_threads : 1;
    #pragma omp parallel for schedule(guided, 64) num_threads(threads) if (threads > 1)
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = dst[i];
        const int64_t start = indptr[v];
        const int64_t deg = indptr[v + 1] - start;
        const int64_t out_pos = out_indptr[i];
        if (deg == 0) continue;
        if (fanout < 0 || deg <= fanout) {
            for (int64_t k = 0; k < deg; ++k) raw_src[out_pos + k] = static_cast<int64_t>(indices[start + k]);
        } else if (fanout <= 64) {
            int64_t picks[64];
            uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
            floyd_picks_small(deg, fanout, picks, rng);
            for (int64_t k = 0; k < fanout; ++k) __builtin_prefetch(indices + start + picks[k], 0, 0);
            for (int64_t k = 0; k < fanout; ++k) raw_src[out_pos + k] = static_cast<int64_t>(indices[start + picks[k]]);
        } else {
            uint64_t rng = seed ^ (static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
            sample_row_large_fanout(indices, start, deg, fanout, raw_src + out_pos, rng);
        }
    }

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
    (void)num_threads;
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
}

template <typename IdxT>
static void partition_buckets_impl(
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
        const int threads = (n >= 65536 && num_threads > 1) ? std::min(num_threads, 8) : 1;
        #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
        for (int64_t i = 0; i < n; ++i) {
            out[i] = static_cast<int32_t>(i);
        }
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

int64_t cpu_sample_layer_indptr(
    const int64_t* indptr,
    const int64_t* dst,
    int64_t n,
    int64_t fanout,
    void* out_indptr,
    int out_is_i32,
    int64_t* out_indptr_i64_buf,
    int num_threads
) {
    if (n >= 8192 && num_threads > 1) {
        // Parallel prefix sum for large frontiers; use actual OpenMP team size `nt`.
        const int threads = std::min(num_threads, 24);
        int64_t t_sums[65] = {0};
        #pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            const int64_t chunk = (n + nt - 1) / nt;
            const int64_t lo = std::min(n, tid * chunk);
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
            #pragma omp barrier
            #pragma omp single
            {
                for (int t = 1; t <= nt; ++t) t_sums[t] += t_sums[t - 1];
                out_indptr_i64_buf[n] = t_sums[nt];
                if (out_is_i32) {
                    static_cast<int32_t*>(out_indptr)[n] = static_cast<int32_t>(t_sums[nt]);
                } else {
                    static_cast<int64_t*>(out_indptr)[n] = t_sums[nt];
                }
            }
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
        }
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
        partition_buckets_impl(fip, num_dst, fwd_threshold, fwd_deg_order, num_threads, st.fwd_light_tmp, st.fwd_heavy_tmp);
        partition_buckets_impl(bip, num_src, bwd_threshold, bwd_deg_order, num_threads, st.bwd_light_tmp, st.bwd_heavy_tmp);
    } else {
        const auto* fip = static_cast<const int64_t*>(fwd_indptr);
        const auto* fix = static_cast<const int64_t*>(fwd_indices);
        auto* bip = static_cast<int64_t*>(bwd_indptr);
        auto* bix = static_cast<int64_t*>(bwd_indices);
        transpose_csr_impl(st, fip, fix, num_dst, num_src, num_edges, bip, bix, num_threads);
        partition_buckets_impl(fip, num_dst, fwd_threshold, fwd_deg_order, num_threads, st.fwd_light_tmp, st.fwd_heavy_tmp);
        partition_buckets_impl(bip, num_src, bwd_threshold, bwd_deg_order, num_threads, st.bwd_light_tmp, st.bwd_heavy_tmp);
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

void cpu_gather_rows(
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
    const int threads = (n * row_bytes >= 65536 && num_threads > 1) ? num_threads : 1;
    #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
    for (int64_t i = 0; i < n; ++i) {
        if (i + 12 < n) {
            __builtin_prefetch(src + idx_ptr[i + 12] * row_bytes, 0, 0);
        }
        std::memcpy(dst + i * row_bytes, src + idx_ptr[i] * row_bytes, static_cast<size_t>(row_bytes));
    }
}

}  // extern "C"
