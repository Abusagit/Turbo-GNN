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

    bool operator==(LRO other) const {
        return (l == other.l) && (r == other.r) && (op == other.op);
    }
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
    at::cuda::CUDAStream stream;
    int64_t N;
    int64_t D;
    LRO key;
    uint32_t light_warps_per_block;
    uint32_t heavy_warps_per_block;
    uint32_t pipeline_stages;
};

// Per-op entry points; one translation unit each (gsddmm_launch_<op>.cu). Each
// covers the member pairs that op is instantiated for and raises from
// MakeEnumVariant on any other pair.
void gsddmm_forward_launch_add(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_sub(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_mul(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_div(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_dot(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_copy(const GsddmmLaunchArgs& args);

};  // namespace gsddmm
