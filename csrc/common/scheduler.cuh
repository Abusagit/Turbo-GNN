#pragma once

#include <algorithm>
#include <string>

#include "common/misc.cuh"
#include "common/traits.cuh"

// Node -> thread-block scheduling policies.
//
// These kernels historically launched with `grid.x == number of output nodes`, so the block
// count was dictated by the graph rather than the GPU, and work could not be balanced across
// blocks. These policies decouple "which block am I" from "which node do I process", letting
// a kernel launch persistently (grid.x ~ C * SM_count) and loop over several nodes.
//
// Canonical kernel body:
//
//     using Sched = NodeScheduler<SK, index_t, /*SyncBlock=*/true>;
//     __shared__ typename Sched::SharedStorage sched_smem;
//     Sched sched(sched_params, sched_smem);
//
//     for (auto w = sched.first(); sched.valid(w); w = sched.next(w)) {
//         const index_t node_i = sched.node(w);      // was: node_indices[blockIdx.x]
//         ... existing per-node body, `return` -> `continue` ...
//     }
//
// ---------------------------------------------------------------------------------------
// Barrier discipline -- the part that is easy to get wrong
// ---------------------------------------------------------------------------------------
//
// The inter-iteration barrier lives in `next()`, NOT at the bottom of the kernel body.
// That is not a style choice. `continue` jumps to the loop's increment expression, so a
// barrier written at the bottom of the body is *skipped* whenever the body takes the
// isolated-node (degree 0) early-out -- which these kernels do. The block would then split
// into threads that executed the barrier and threads that did not. Emitting it from `next()`
// makes `continue` structurally safe.
//
// One barrier per iteration is enough for the body's shared memory. `next()`'s barrier sits
// between body_k and body_{k+1} for every thread, so it fences the pattern these kernels all
// share: warp 0 reads every other warp's accumulator plane at the bottom of an iteration,
// and some warp rewrites that plane at the top of the next one.
//
// Kernels whose body touches no shared memory pass `SyncBlock = false` and pay nothing.
//
// Every policy's loop condition is block-uniform -- it derives from `blockIdx`/`gridDim`
// (uniform by construction), from a kernel parameter, or from a shared slot read immediately
// after a barrier. That is what makes `__syncthreads()` inside the loop legal: all threads of
// a block take the same number of trips.
//
// There is no grid-wide synchronisation anywhere, and no block ever waits on another, so no
// policy can deadlock at any occupancy. This matters because the build has no `-rdc=true`,
// which rules out cooperative groups.
//
// Reference: FlashAttention's hopper/tile_scheduler.hpp, whose Single / StaticPersistent /
// DynamicPersistent split this mirrors, including the `atomicAdd(...) + gridDim.x` trick.

namespace turbo_gnn::sched {

template <typename T>
__host__ __device__ inline constexpr T ceil_div(T a, T b) {
    return (a + b - T{1}) / b;
}

enum class ScheduleKind : int {
    OneNodePerBlock = 0,  // one block per node: the historical behaviour, zero overhead
    GridStride      = 1,  // persistent, stride by gridDim.x, no stored assignment
    PrecomputedList = 2,  // persistent, host-assigned contiguous slice per block
    DynamicQueue    = 3,  // persistent, atomic work queue (default)
};

inline constexpr int kNumScheduleKinds = 4;

/// Ints reserved per `blockIdx.y`. Padded to a 128 B L2 sector so concurrent heads never
/// contend on the same sector when claiming ranks or work items.
inline constexpr int kCounterStride = 32;
inline constexpr int kRankSlot      = 0;
inline constexpr int kWorkSlot      = 1;

/// One POD for every policy, so the kernel signature does not change with the policy --
/// only the template argument does. Unused fields stay null and are dead code in the
/// specialisations that never read them.
///
/// `nodes` is the existing indirection array: a light/heavy bucket, or an LPT permutation
/// ordering nodes by descending degree. Null means identity (work item *is* the node id),
/// which is what the non-bucketed kernels want.
template <typename index_t>
struct SchedulerParams {
    index_t const *nodes     = nullptr;  ///< work item -> node id; null = identity
    int const *block_offsets = nullptr;  ///< PrecomputedList: [gridDim.x + 1], CSR-style
    int *counters            = nullptr;  ///< [heads * kCounterStride], zeroed by the host
    int count                = 0;        ///< number of work items
    int chunk                = 1;        ///< DynamicQueue: work items claimed per atomic
};

struct WorkTile {
    int idx;   ///< position in the ordered node list
    bool ok;   ///< false => the loop terminates
};

namespace detail {

/// True for exactly one thread per block, whatever the block shape.
__device__ __forceinline__ bool is_block_leader() {
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

/// Two slots so the broadcast can be double-buffered; see `DynamicQueue::next`.
struct SharedSlots {
    int slot[2];
};

/// Make the leader's value visible to the whole block, and fence the previous iteration's
/// shared-memory traffic on the way through.
__device__ __forceinline__ int publish(SharedSlots &st, int v, int buf) {
    if (is_block_leader()) {
        st.slot[buf] = v;
    }
    __syncthreads();
    return st.slot[buf];
}

/// Claim a dense logical block id that ascends in *start* order.
///
/// Kept for reference and for any future policy whose assignment depends on when a block
/// began rather than on which slice it owns. Nothing uses it today, and that is deliberate:
///
///  * `DynamicQueue` gets the property for free -- its cursor hands out chunks in demand
///    order, so the claim *is* the ascending id and a separate rank would be a second atomic.
///  * `GridStride` and `PrecomputedList` have a *static* assignment. Their logical id only has
///    to be a bijection onto `[0, gridDim.x)`; which block holds which slice is immaterial
///    because the slices are fixed by the host. `blockIdx.x` is already such a bijection, and
///    it is free.
///
/// That is worth roughly 7%: with the grid raised until each block held exactly one node --
/// a launch shaped identically to `one_per_block` -- `GridStride` still measured 0.93x, and
/// the rank claim was the only thing left to account for it. One atomic plus one barrier per
/// block, all blocks hitting a single counter, is not free at 131k blocks.
__device__ __forceinline__ int claim_rank(int *rank_counter, SharedSlots &st) {
    int r = 0;
    if (is_block_leader()) {
        r = atomicAdd(rank_counter, 1);
    }
    return publish(st, r, /*buf=*/0);
}

}  // namespace detail

template <ScheduleKind Kind, typename index_t, bool SyncBlock = true>
struct NodeScheduler;

/// Members every policy shares.
template <typename index_t>
struct SchedulerBase {
    using SharedStorage = detail::SharedSlots;
    using Params        = SchedulerParams<index_t>;

    index_t const *nodes_;
    int count_;

    __device__ __forceinline__ explicit SchedulerBase(Params const &p)
        : nodes_(p.nodes), count_(p.count) {}

    __device__ __forceinline__ bool valid(WorkTile w) const { return w.ok; }

    /// Node id through the indirection array -- the bucketed kernels.
    __device__ __forceinline__ index_t node(WorkTile w) const {
        return nodes_ ? nodes_[w.idx] : static_cast<index_t>(w.idx);
    }

    /// Position within the ordered list, for kernels indexing a compacted output buffer
    /// (the packed-heavy scratch is indexed by compacted position, not by node id).
    __device__ __forceinline__ int slot(WorkTile w) const { return w.idx; }
};

// ---------------------------------------------------------------------------------------
// 1. OneNodePerBlock -- the historical behaviour.
//
// No atomics, no barriers, no rank claim; `blockIdx.x` used directly. `ok` is traceable to
// a constant, so nvcc peels the single iteration and the loop disappears. This must stay
// the zero-overhead baseline that proves the conversion cost nothing.
// ---------------------------------------------------------------------------------------
template <typename index_t, bool SyncBlock>
struct NodeScheduler<ScheduleKind::OneNodePerBlock, index_t, SyncBlock> : SchedulerBase<index_t> {
    using Base          = SchedulerBase<index_t>;
    using SharedStorage = typename Base::SharedStorage;
    using Params        = typename Base::Params;

    static constexpr bool kPersistent = false;

    __device__ __forceinline__ NodeScheduler(Params const &p, SharedStorage &) : Base(p) {}

    __device__ __forceinline__ WorkTile first() const {
        const int i = static_cast<int>(blockIdx.x);
        return WorkTile{i, i < this->count_};
    }
    __device__ __forceinline__ WorkTile next(WorkTile) const { return WorkTile{0, false}; }
};

// ---------------------------------------------------------------------------------------
// 2. GridStride -- persistent, nothing stored.
//
// The assignment is recomputed from `gridDim.x` rather than held in memory, which is the
// point when the node list is large enough that a per-block assignment array is not worth
// its bytes.
// ---------------------------------------------------------------------------------------
template <typename index_t, bool SyncBlock>
struct NodeScheduler<ScheduleKind::GridStride, index_t, SyncBlock> : SchedulerBase<index_t> {
    using Base          = SchedulerBase<index_t>;
    using SharedStorage = typename Base::SharedStorage;
    using Params        = typename Base::Params;

    static constexpr bool kPersistent = true;

    __device__ __forceinline__ NodeScheduler(Params const &p, SharedStorage &) : Base(p) {}

    /// `blockIdx.x`, not an atomically-claimed rank: the stride assignment is the same set of
    /// nodes whichever block walks it, so paying an atomic to permute the ids buys nothing.
    __device__ __forceinline__ WorkTile first() const {
        const int i = static_cast<int>(blockIdx.x);
        return WorkTile{i, i < this->count_};
    }

    __device__ __forceinline__ WorkTile next(WorkTile w) {
        if constexpr (SyncBlock) {
            __syncthreads();  // fences body_k against body_{k+1}; see the header comment
        }
        const int nxt = w.idx + static_cast<int>(gridDim.x);
        return WorkTile{nxt, nxt < this->count_};
    }
};

// ---------------------------------------------------------------------------------------
// 3. PrecomputedList -- persistent, host-assigned contiguous slice.
//
// Block `rank` walks `[block_offsets[rank], block_offsets[rank + 1])`. Strictly more general
// than GridStride: the host can bin-pack nodes so each block gets roughly equal *total
// degree* rather than an equal count, which is what actually balances these kernels.
// Costs `gridDim.x + 1` ints.
// ---------------------------------------------------------------------------------------
template <typename index_t, bool SyncBlock>
struct NodeScheduler<ScheduleKind::PrecomputedList, index_t, SyncBlock> : SchedulerBase<index_t> {
    using Base          = SchedulerBase<index_t>;
    using SharedStorage = typename Base::SharedStorage;
    using Params        = typename Base::Params;

    static constexpr bool kPersistent = true;

    int const *block_offsets_;
    int end_ = 0;  ///< block-uniform: both bounds come from a uniform address

    __device__ __forceinline__ NodeScheduler(Params const &p, SharedStorage &)
        : Base(p), block_offsets_(p.block_offsets) {}

    /// `blockIdx.x` indexes the host's assignment directly -- no atomic, no barrier, and no
    /// shared slot, so `first()` is two uniform loads. Both addresses are block-uniform, so
    /// each is a single broadcast transaction.
    __device__ __forceinline__ WorkTile first() {
        const int bid = static_cast<int>(blockIdx.x);
        const int beg = block_offsets_[bid];
        end_          = block_offsets_[bid + 1];
        return WorkTile{beg, beg < end_};
    }

    __device__ __forceinline__ WorkTile next(WorkTile w) {
        if constexpr (SyncBlock) {
            __syncthreads();
        }
        const int nxt = w.idx + 1;
        return WorkTile{nxt, nxt < end_};
    }
};

// ---------------------------------------------------------------------------------------
// 4. DynamicQueue -- persistent atomic work queue. The default.
//
// A single monotone cursor. Every block draws its *first* chunk from the same counter as
// every later one, so items are handed out strictly in demand order: 0, 1, 2, ... This is
// the whole design, and the reason is locality, not fairness.
//
// **Why not FlashAttention's `atomicAdd(...) + gridDim.x`.** That form pre-assigns item
// `rank` to block `rank` statically and only goes to the counter for the tail, which saves
// one atomic per block. It is correct here too, and it was the first implementation -- but it
// is only *locality-preserving* when the grid is exactly one wave, which is the regime
// FlashAttention launches in. Ours is not: on ogbn-proteins the light bucket runs
// grid.x = 6912 against roughly 864 resident blocks, so a block that finished node 5 jumped
// straight to node 6912. The set of nodes in flight fragmented into scattered windows within
// a few iterations, and the CSR rows being streamed stopped being adjacent.
//
// That cost is large and it was measured. Against the one-block-per-node baseline on
// ogbn-proteins (p50 degree 402), the pre-assigning form reached only 0.57x on min_aggr and
// 0.77x on GT, and -- the tell -- it got *monotonically worse* as `chunk` grew (0.57 -> 0.34)
// and *better* as the grid grew, i.e. the closer the launch came to one-block-per-node the
// faster it ran. Both trends point at the access pattern rather than at overhead: raising the
// grid narrows the window each block jumps over, and raising the chunk widens it.
//
// With one monotone cursor the in-flight items are always a contiguous window whose width is
// the number of *resident* blocks -- exactly the pattern the hardware block scheduler
// produces for the historical launch, which is why that launch was so hard to beat. We keep
// that access pattern and add dynamic balancing and ~N fewer block launches on top.
//
// **Chunking.** A block claims `params.chunk` consecutive items per atomic. This trades
// window width for atomic traffic: on a sparse graph (ogbn-arxiv, p50 degree 1) the per-node
// work is so small that one atomic per node is visible, and chunk 4-8 is worth the wider
// window; on a dense graph the per-node work dwarfs the atomic and chunk 1 is right. It is a
// tunable for exactly that reason, and the default is set for the sparse case because that is
// where it changes anything.
//
// Pairing this with a `nodes` array sorted by descending degree gives longest-processing-
// time-first scheduling: the expensive nodes go out first and the cheap ones fill the tail,
// the standard makespan-minimising order. It also widens the effective window, so it is a
// tunable too, not a default.
//
// The broadcast slot is double-buffered, which keeps the cost at one barrier per iteration.
// With a single slot the leader's write for iteration k+1 could land while another thread was
// still reading iteration k's value, forcing a second barrier. With parity flipping, the
// write to `slot[b]` at iteration k+2 happens after the barrier at k+1, and every read of
// `slot[b]` happened before that same barrier -- write-after-all-reads, no race.
//
// No rank is claimed at all: the cursor *is* the identity, and it is ascending in start order
// by construction, which is what the persistent policies needed a rank for in the first place.
// ---------------------------------------------------------------------------------------
template <typename index_t, bool SyncBlock>
struct NodeScheduler<ScheduleKind::DynamicQueue, index_t, SyncBlock> : SchedulerBase<index_t> {
    using Base          = SchedulerBase<index_t>;
    using SharedStorage = typename Base::SharedStorage;
    using Params        = typename Base::Params;

    static constexpr bool kPersistent = true;

    SharedStorage *st_;
    int *work_counter_;
    int chunk_;
    int chunk_end_ = 0;  ///< one past the last item of the chunk this block currently holds
    int buf_       = 0;  ///< block-uniform double-buffer parity

    __device__ __forceinline__ NodeScheduler(Params const &p, SharedStorage &st)
        : Base(p), st_(&st),
          work_counter_(p.counters + blockIdx.y * kCounterStride + kWorkSlot),
          chunk_(p.chunk < 1 ? 1 : p.chunk) {}

    /// Draw a chunk from the shared cursor. Block-uniform on return: `publish` broadcasts the
    /// leader's value and carries the barrier that fences the previous iteration's shared
    /// memory, so this doubles as the inter-iteration fence.
    __device__ __forceinline__ WorkTile claim() {
        int claimed = 0;
        if (detail::is_block_leader()) {
            // Peek first. Every block ends by claiming once past the end, and at 131k blocks
            // that terminating atomic is a storm on a single counter for a guaranteed miss. A
            // plain load of the same (read-mostly, L2-resident) line is far cheaper and is
            // safe in both directions: the counter only increases, so a value at or past the
            // end proves every chunk has already been handed out, while a stale-low read just
            // falls through to the atomic and stays correct.
            const int seen = *reinterpret_cast<int volatile *>(work_counter_);
            claimed = (seen * chunk_ >= this->count_) ? this->count_
                                                      : atomicAdd(work_counter_, 1) * chunk_;
        }
        const int base = detail::publish(*st_, claimed, buf_);
        chunk_end_     = base + chunk_;
        return WorkTile{base, base < this->count_};
    }

    __device__ __forceinline__ WorkTile first() {
        buf_ = 0;
        return claim();
    }

    __device__ __forceinline__ WorkTile next(WorkTile w) {
        const int nxt = w.idx + 1;
        if (nxt < chunk_end_) {
            // Still inside the claimed chunk: no atomic, no broadcast. `chunk_end_` and
            // `w.idx` are both block-uniform, so the whole block takes this branch together
            // and the barrier count stays uniform.
            if constexpr (SyncBlock) {
                __syncthreads();  // inter-iteration fence for the body's shared memory
            }
            return WorkTile{nxt, nxt < this->count_};
        }
        buf_ ^= 1;  // block-uniform: every thread flips it
        return claim();
    }
};

/// Maps the `MakeIntVariant` dispatch constant back to a policy type, so the scheduler slots
/// into the existing `std::visit` chain as one more axis.
template <int Kind, typename index_t, bool SyncBlock = true>
using PolicyFor_t = NodeScheduler<static_cast<ScheduleKind>(Kind), index_t, SyncBlock>;

// ---------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------

/// Only `DynamicQueue` touches global state now. The other two derive everything from
/// `blockIdx.x` and the host's assignment, so allocating and zeroing a counter slab for them
/// was pure launch overhead -- an `at::zeros` plus a memset on the stream, per launch, on a
/// kernel that can itself run in 600 us.
inline bool needs_counters(ScheduleKind k) { return k == ScheduleKind::DynamicQueue; }

inline int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

/// Number of blocks along x.
///
/// `OneNodePerBlock` keeps the historical `grid.x == count`. The persistent policies target a
/// *total* of `blocks_per_sm * SM_count` blocks; when a second grid axis is in use (heads on
/// `gridDim.y`) the target is divided by it, because `(C * SM, H)` would queue up on y and
/// would not be persistent at all. Rounded up: truncating under-fills, e.g. 108 SMs with
/// C=1, H=8 gives 104 blocks truncated versus 112 rounded.
///
/// WARNING on sizing. `blocks_per_sm` is a *resident blocks per SM* target, not a wave count.
/// These kernels launch small blocks -- the light bucket uses 1-warp blocks -- so a value of
/// 1 leaves the GPU almost empty: with H=8 on 108 SMs it yields grid.x=14, i.e. 112 blocks of
/// 32 threads for the whole device. Pass something near what actually fits per SM (use
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the call site, where the instantiation
/// and dynamic shared memory are known), or tune it per bucket.
///
/// Never exceeds `count`: surplus blocks would claim a rank, find nothing and exit.
inline int persistent_grid_x(
    ScheduleKind kind, int count, int blocks_per_sm, int grid_y = 1, int chunk = 1
) {
    if (kind == ScheduleKind::OneNodePerBlock) {
        return count;
    }
    const int c      = std::max(1, blocks_per_sm);
    const int y      = std::max(1, grid_y);
    const int g      = (kind == ScheduleKind::DynamicQueue) ? std::max(1, chunk) : 1;
    const int target = std::max(1, ceil_div(sm_count() * c, y));
    // A DynamicQueue block covers `chunk` items per claim, so the grid needed to cover the
    // work shrinks accordingly; launching beyond that just leaves blocks with nothing to do.
    return std::min(target, std::max(1, ceil_div(count, g)));
}

/// Backing store for the per-`blockIdx.y` counters, one row per bucket launch so a light
/// launch cannot leave dirt for the heavy launch that follows.
///
/// `at::zeros` issues its memset on the current stream, so the zeroing is correctly ordered
/// against a kernel launched on that same stream -- which the launch site must be. Returns an
/// undefined tensor for `OneNodePerBlock`, which needs no counters at all.
inline at::Tensor make_counters(ScheduleKind kind, int heads, int num_launches, at::Device device) {
    if (!needs_counters(kind)) {
        return at::Tensor{};
    }
    const int64_t rows = std::max(1, num_launches);
    const int64_t cols = static_cast<int64_t>(std::max(1, heads)) * kCounterStride;
    return at::zeros({rows, cols}, at::TensorOptions().dtype(at::kInt).device(device));
}

/// Even contiguous split of `count` items across `num_blocks`, as CSR-style offsets.
///
/// The fallback assignment for `PrecomputedList` when the caller has not supplied a
/// degree-balanced one. Built on-device and stream-ordered, so it costs no host round trip.
inline at::Tensor default_block_offsets(int count, int num_blocks, at::Device device) {
    const int nb = std::max(1, num_blocks);
    auto edges   = at::linspace(0, count, nb + 1, at::TensorOptions().dtype(at::kFloat).device(device));
    return edges.round().to(at::kInt).contiguous();
}

/// Contiguous slices carrying roughly equal *total degree*, rather than equal node count.
///
/// This is the assignment `PrecomputedList` exists for. Cost per node is proportional to its
/// degree, so an equal-count split balances nothing on a skewed graph -- and combined with a
/// descending-degree (LPT) order it is actively pathological, because block 0 then receives
/// every hub in the graph. Splitting on the degree prefix sum instead gives each block the
/// same number of *edges*, while keeping each block's nodes contiguous, so the streaming
/// access pattern that makes the historical launch fast is preserved exactly.
///
/// Built with ATen ops on the current stream: a cumsum over the bucket and a searchsorted,
/// both small. It is still work per launch, which is why it is worth caching on the graph for
/// a repeated call; for a single measurement it is a few tens of microseconds.
inline at::Tensor degree_balanced_block_offsets(
    at::Tensor const &indptr, at::Tensor const &nodes, int count, int num_blocks
) {
    const int nb = std::max(1, num_blocks);
    if (count <= 0) {
        return at::zeros({nb + 1}, at::TensorOptions().dtype(at::kInt).device(indptr.device()));
    }
    auto ip  = indptr.to(at::kLong);
    auto deg = ip.slice(0, 1) - ip.slice(0, 0, -1);
    auto d   = nodes.defined() ? deg.index_select(0, nodes.to(at::kLong)) : deg.slice(0, 0, count);
    auto cum = d.cumsum(0);  // [count], non-decreasing

    const auto opts = at::TensorOptions().dtype(at::kLong).device(indptr.device());
    // Interior boundaries at 1/nb, 2/nb, ... of the total edge count. `searchsorted` on a
    // non-decreasing key with non-decreasing targets returns non-decreasing offsets, so the
    // slices are well formed even when a single node holds more than 1/nb of all edges (it
    // simply gets a slice to itself and its neighbours get empty ones).
    auto tgt  = at::arange(1, nb, opts) * cum.index({-1}).item<int64_t>() / nb;
    auto mid  = at::searchsorted(cum, tgt);
    auto offs = at::cat({at::zeros({1}, opts), mid, at::full({1}, static_cast<int64_t>(count), opts)});
    return offs.to(at::kInt).contiguous();
}

/// Fill in the pointers a policy reads. `counters` must come from `make_counters` with the
/// same `kind`; `launch_index` selects its row.
template <typename index_t>
inline SchedulerParams<index_t> make_params(
    ScheduleKind kind, index_t const *nodes, int count, at::Tensor const &counters, int heads = 1,
    int launch_index = 0, at::Tensor const &block_offsets = {}, int chunk = 1
) {
    SchedulerParams<index_t> p;
    p.nodes = nodes;
    p.count = count;
    p.chunk = chunk < 1 ? 1 : chunk;

    if (needs_counters(kind)) {
        TORCH_CHECK(counters.defined(), "scheduler: persistent policies need a zeroed counter slab");
        p.counters = counters.data_ptr<int>() +
                     static_cast<int64_t>(launch_index) * std::max(1, heads) * kCounterStride;
    }
    if (kind == ScheduleKind::PrecomputedList) {
        TORCH_CHECK(block_offsets.defined(), "scheduler: PrecomputedList needs block_offsets");
        TORCH_CHECK(
            block_offsets.is_cuda() && block_offsets.is_contiguous() && block_offsets.scalar_type() == at::kInt,
            "scheduler: block_offsets must be a contiguous int32 CUDA tensor"
        );
        p.block_offsets = block_offsets.data_ptr<int>();
    }
    return p;
}

inline ScheduleKind schedule_from_string(std::string const &name) {
    if (name == "one_per_block") return ScheduleKind::OneNodePerBlock;
    if (name == "grid_stride") return ScheduleKind::GridStride;
    if (name == "precomputed") return ScheduleKind::PrecomputedList;
    if (name == "dynamic") return ScheduleKind::DynamicQueue;
    TORCH_CHECK(false, "scheduler: unknown schedule '", name,
                "'; expected one_per_block | grid_stride | precomputed | dynamic");
}

inline ScheduleKind schedule_from_int(int v) {
    TORCH_CHECK(v >= 0 && v < kNumScheduleKinds,
                "scheduler: schedule must be 0..3 (0=one_per_block, 1=grid_stride, "
                "2=precomputed, 3=dynamic), got ", v);
    return static_cast<ScheduleKind>(v);
}

}  // namespace turbo_gnn::sched
