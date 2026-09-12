#pragma once

#include <cstdint>

#include "gsddmm/gsddmm.cuh"

// =============================================================================
// GSDDMM forward: host-side launch plumbing shared by the per-op translation
// units.
//
// The kernel is fully templated on (op, lhs member, rhs member, warps/block, D,
// dtype, index type, pipeline stages), so every combination is a separate
// instantiation and the size of the dispatch grid alone drives this kernel's
// compile time. nvcc scales superlinearly in instantiations per translation
// unit, so the grid is sharded by op: gsddmm_launch_<op>.cu instantiates only
// its own op's slice of the grid and the six files compile in parallel.
//
// gsddmm_binding.cu keeps the argument checking and output allocation (it
// instantiates no kernels at all) and routes to one of the entry points below.
// Adding an op means adding an enumerator, an entry point here, and one more
// gsddmm_launch_<op>.cu; extending the dtype / index / D / warps / stages axes
// happens once in gsddmm_dispatch.cuh and costs every shard.
// =============================================================================

namespace gsddmm {

// (lhs member, rhs member, op) triple. A structural type, so it can be used as
// a non-type template parameter and fold the three enums into one dispatch axis.
struct LRO {
    GSDDMM_MEMBER l, r;
    GSDDMM_OP op;

    bool operator==(LRO other) const { return (l == other.l) && (r == other.r) && (op == other.op); }
};

// Everything a per-op launcher needs, assembled once by gsddmm_binding.cu after
// validation. The tensors are borrowed for the duration of the call; O is the
// (already allocated) output.
struct GsddmmLaunchArgs {
    const torch::Tensor& L;
    const torch::Tensor& R;
    torch::Tensor& O;
    const torch::Tensor& row_ptr;
    const torch::Tensor& col_idx;
    const torch::Tensor& light_nodes;
    const torch::Tensor& heavy_nodes;
    const at::cuda::CUDAStream& stream;
    uint64_t N;
    uint64_t D;
    LRO key;
    uint16_t light_warps_per_block;
    uint16_t heavy_warps_per_block;
    uint8_t pipeline_stages;
    // Heavy-node chunking (nullptr / 0 = one block per heavy node): heavy_nodes
    // then lists the node of every heavy block and heavy_block_parts its chunk index.
    const torch::Tensor *heavy_block_parts;
    uint32_t heavy_edges_per_block;
    // Run the light bucket on a pool stream forked from / joined back to `stream`
    // while the heavy bucket runs on `stream` itself (see gsddmm_dispatch).
    bool overlap_buckets;
};

// Per-op entry points; one translation unit each (gsddmm_launch_<op>.cu). Each
// covers the member pairs that op is instantiated for and raises from
// MakeEnumVariant on any other pair. The declarations are X-macro'd over the
// op list so they cannot drift from the routing switches or the shards.
#define XX(ENUM, name) void gsddmm_forward_launch_##name(const GsddmmLaunchArgs& args);
#include "gsddmm/gsddmm_ops.inc"
#undef XX

struct GsddmmLaunchArgsEdge {
    const torch::Tensor& L;
    const torch::Tensor& R;
    torch::Tensor& O;
    ulonglong2 const *__restrict__ edge_nodes_idx;
    // Nullable [E] canonical edge id of every traversal slot; indexes the Edge
    // operand rows and the output row, so a source-grouped list can still emit
    // forward-CSR-numbered output. nullptr: traversal order is canonical.
    unsigned long long const *__restrict__ canonical_edge_idx;
    const at::cuda::CUDAStream& stream;
    uint64_t E;
    uint64_t D;
    LRO key;
    uint8_t pipeline_stages;  // cp.async prefetch depth per warp, 0 = direct loads
    uint8_t edges_per_warp;   // contiguous edges per warp, 1..kGsddmmEdgeMaxEdgesPerWarp
    uint8_t warps_per_block;  // independent warps packed per block, 1..kGsddmmEdgeMaxWarpsPerBlock
};

// Per-op entry points; one translation unit each (gsddmm_launch_<op>.cu). Each
// covers the member pairs that op is instantiated for and raises from
// MakeEnumVariant on any other pair. The declarations are X-macro'd over the
// op list so they cannot drift from the routing switches or the shards.
#define XX(ENUM, name) void gsddmm_forward_edge_launch_##name(const GsddmmLaunchArgsEdge& args);
#include "gsddmm/gsddmm_ops.inc"
#undef XX

// =============================================================================
// Backward
// =============================================================================
//
// A call runs one pass per node operand (see GSDDMM_REDUCE in gsddmm.cuh): the
// Dst pass walks the forward CSR, the Src pass the backward one. Both CSRs and
// both bucket pairs are therefore passed in, and the dispatch launches only the
// passes the member pair actually needs. dL / dR are pre-allocated by the
// binding, so a launcher never allocates.
struct GsddmmBackwardLaunchArgs {
    const torch::Tensor& L;
    const torch::Tensor& R;
    const torch::Tensor& dO;
    torch::Tensor& dL;
    torch::Tensor& dR;
    // Forward CSR (rows = destinations) and its node buckets: the Dst pass.
    const torch::Tensor& row_ptr;
    const torch::Tensor& col_idx;
    const torch::Tensor& light_nodes;
    const torch::Tensor& heavy_nodes;
    // Backward CSR (rows = sources) and its buckets: the Src pass. Its slots are
    // CSC positions, so canonical_edge_idx maps them onto dO's numbering (null
    // only when the graph aliases its two CSRs, i.e. the orders coincide).
    const torch::Tensor& row_ptr_T;
    const torch::Tensor& col_idx_T;
    const torch::Tensor& light_nodes_T;
    const torch::Tensor& heavy_nodes_T;
    unsigned long long const *__restrict__ canonical_edge_idx;
    const at::cuda::CUDAStream& stream;
    uint64_t N;
    uint64_t D;
    LRO key;
    uint16_t light_warps_per_block;
    uint16_t heavy_warps_per_block;
};

#define XX(ENUM, name) void gsddmm_backward_launch_##name(const GsddmmBackwardLaunchArgs& args);
#include "gsddmm/gsddmm_ops.inc"
#undef XX

// Edge-parallel backward. Each pass traverses the edge list grouped by the node
// it reduces, so that a warp's chunk shares its target row and the accumulation
// collapses to one atomic per tile: the Dst pass takes edge_nodes_idx_dst (which
// is already in canonical order), the Src pass edge_nodes_idx_src together with
// canonical_edge_idx.
//
// Node gradients land in fp32 scratch (dL_f32 / dR_f32, zeroed by the binding,
// which casts them back); an edge operand's gradient is a plain store straight
// into dL / dR.
struct GsddmmBackwardLaunchArgsEdge {
    const torch::Tensor& L;
    const torch::Tensor& R;
    const torch::Tensor& dO;
    torch::Tensor& dL;
    torch::Tensor& dR;
    torch::Tensor& dL_f32;
    torch::Tensor& dR_f32;
    ulonglong2 const *__restrict__ edge_nodes_idx_dst;
    ulonglong2 const *__restrict__ edge_nodes_idx_src;
    unsigned long long const *__restrict__ canonical_edge_idx;
    const at::cuda::CUDAStream& stream;
    uint64_t E;
    uint64_t D;
    LRO key;
    uint8_t edges_per_warp;
    uint8_t warps_per_block;
};

#define XX(ENUM, name) void gsddmm_backward_edge_launch_##name(const GsddmmBackwardLaunchArgsEdge& args);
#include "gsddmm/gsddmm_ops.inc"
#undef XX

};  // namespace gsddmm
