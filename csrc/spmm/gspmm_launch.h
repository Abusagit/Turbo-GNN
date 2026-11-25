#pragma once

#include <torch/extension.h>

#include <cstdint>

#include "common/gspmm_ops.cuh"
#include "reduction/reduction_launch.cuh"

// =============================================================================
// g-SpMM: host-side plumbing between the binding (gspmm.cu), which validates and plans, and the
// per-op, per-dtype translation units (gspmm_shard_<op>_<dtype>.cu), which instantiate the kernels.
// Every kernel is templated on op x dtype x index width x reducer x row shape (and the broadcast and
// edge-map axes of the edge operand), so the grid is split across translation units that nvcc
// compiles in parallel; gspmm.cu instantiates none of it.
// =============================================================================

namespace turbo_gnn::gspmm {

// One forward: a reduction walk with a g-SpMM message (reduction/reduction_common.cuh).
struct ForwardArgs {
    at::Tensor X;         // node operand [N, D], unit stride over D, row stride < 2^32; undefined for copy_e
    at::Tensor E;         // edge operand, contiguous [E, D] or [E] (broadcast); undefined for copy_u
    at::Tensor edge_map;  // [E], the row of E each walked edge reads; undefined on a forward walk
    at::Tensor row_ptr;
    at::Tensor col_idx;
    at::Tensor light;
    at::Tensor heavy;
    reduction::ForwardLaunch const *plan;
    int64_t slice_size;
    ReductionOp op;
    bool bcast;
    size_t elem_size;
};

// The sum backward's edge gradient (gspmm_edge_grad.cuh).
struct EdgeGradArgs {
    at::Tensor G;   // grad_out [N, D], unit stride over D, row stride < 2^32
    at::Tensor X;   // node operand, as in ForwardArgs; read by mul and div only
    at::Tensor E;   // edge operand, contiguous; read by div only
    at::Tensor dE;  // its gradient, shaped as E, contiguous
    at::Tensor row_ptr;
    at::Tensor col_idx;
    at::Tensor light;
    at::Tensor heavy;
    reduction::Slices const *slices;
    int64_t slice_size;
    int64_t D;
    bool bcast;
    bool wide;
    int warps;
};

}  // namespace turbo_gnn::gspmm

// The shard entry points, one per op and dtype (copy_u has no edge gradient).
#define GSPMM_DECLARE_SHARD(name, suffix)                                                   \
    void gspmm_forward_##name##_##suffix(turbo_gnn::gspmm::ForwardArgs const& args);       \
    void gspmm_edge_grad_##name##_##suffix(turbo_gnn::gspmm::EdgeGradArgs const& args);
#define GSPMM_DECLARE_SHARDS(name) GSPMM_DECLARE_SHARD(name, f32) GSPMM_DECLARE_SHARD(name, f16) GSPMM_DECLARE_SHARD(name, bf16)
GSPMM_DECLARE_SHARDS(copy_u)
GSPMM_DECLARE_SHARDS(copy_e)
GSPMM_DECLARE_SHARDS(add)
GSPMM_DECLARE_SHARDS(sub)
GSPMM_DECLARE_SHARDS(mul)
GSPMM_DECLARE_SHARDS(div)
#undef GSPMM_DECLARE_SHARDS
#undef GSPMM_DECLARE_SHARD
