#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/gspmm_ops.cuh"

// =============================================================================
// The scatter that the min/max backward of g-SpMM takes on dense graphs (see csrc/spmm/gspmm.cu).
// It was reduction_aggr's backward before that moved to reduction_backward.cuh; g-SpMM grew it the
// *edge operand* axes:
//
//   BOp            which binary op combined node and edge data.  The default BinaryOp::COPY_U
//                  ignores `rhs` entirely, and `rhs` is then passed as nullptr.
//   RHS_BROADCAST  edge data is one scalar per edge (`rhs[eid]`) rather than one value per (edge,
//                  feature) (`rhs[eid * d + f]`).
//   ARG_IS_EDGE    arg_idx holds the winning *edge position* instead of the winning source node.
// =============================================================================

// Backward of an arg-tracking reducer: only the winning edge of each output
// element contributed, so one scatter over arg_idx yields every gradient.
//
// With the defaults this is plain reduction_aggr -- arg_idx holds the source
// node, the message was the node value, and grad_x is the only output.  Under
// ARG_IS_EDGE the index is a CSR edge position instead, `edge_idx` recovers the
// source node from it, and the edge half of the gradient lands in grad_rhs
// (shaped [E, d], or [E] when the edge operand broadcast over the features).
//
// grad_t is the accumulate dtype of the *node* gradient: cuda_t for
// reduction_aggr, float for g-SpMM, whose Python layer casts back -- several
// destinations can pick the same source, so that write accumulates, and fp16
// atomicAdd is arch-gated and lossy.
//
// grad_rhs_t is separate because the edge gradient usually does not accumulate
// at all: a full-width edge operand gives every (arg, f) pair a slot of its
// own -- distinct destinations own disjoint edge ranges, and distinct features
// land at distinct offsets -- so the write is exactly-once and can go straight
// out in the operand dtype.  Only a broadcast operand, which folds all d
// features of an edge into one slot, still needs an atomic (and hence float).
template <
    size_t WARPS_PER_BLOCK,
    FloatingNum cuda_t,
    typename index_t,
    BinaryOp BOp           = BinaryOp::COPY_U,
    bool RHS_BROADCAST     = false,
    bool ARG_IS_EDGE       = false,
    FloatingNum grad_t     = cuda_t,
    FloatingNum accum_t    = float,
    FloatingNum grad_rhs_t = grad_t
>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) reduction_aggr_backward_typed(
    const cuda_t *__restrict__ grad_out,
    const index_t *__restrict__ arg_idx,
    grad_t *__restrict__ grad_x,
    size_t num_nodes,
    size_t d,
    const index_t *__restrict__ edge_idx = nullptr,
    const cuda_t *__restrict__ lhs       = nullptr,
    const cuda_t *__restrict__ rhs       = nullptr,
    grad_rhs_t *__restrict__ grad_rhs    = nullptr
) {
    using Sentinel = IndexSentinel<index_t>;
    using BOps     = BinaryOps<BOp>;

    size_t block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    size_t tid = threadIdx.x;
    // blockDim.x, not WARPS_PER_BLOCK * kWarpSize: reduction_aggr launches with
    // exactly that many threads, but g-SpMM pins WARPS_PER_BLOCK to the CUDA
    // maximum (so it needs one instantiation, not one per warp count) and picks
    // its block size at runtime.  WARPS_PER_BLOCK still sets __launch_bounds__.
    const size_t block_dim   = blockDim.x;
    const size_t base_offset = block_idx * d;

    for (size_t f = tid; f < d; f += block_dim) {
        const index_t arg = arg_idx[base_offset + f];
        if (!Sentinel::is_valid(arg)) {
            continue;  // node with no in-edges: nothing reached it, nothing flows back
        }

        const index_t u    = ARG_IS_EDGE ? edge_idx[static_cast<size_t>(arg)] : arg;
        const accum_t g    = static_cast<accum_t>(grad_out[base_offset + f]);
        const size_t e_off = RHS_BROADCAST ? static_cast<size_t>(arg) : static_cast<size_t>(arg) * d + f;

        // Only mul and div differentiate to something that reads the operands;
        // for the rest the loads would be dead, and for copy_u/copy_e the
        // pointer they would read is null.
        accum_t u_val{};
        accum_t e_val{};
        if constexpr (BOps::GRAD_USES_OPERANDS) {
            u_val = static_cast<accum_t>(lhs[static_cast<size_t>(u) * d + f]);
            e_val = static_cast<accum_t>(rhs[e_off]);
        }

        if constexpr (BOps::USE_LHS) {
            atomicAdd(&grad_x[static_cast<size_t>(u) * d + f], static_cast<grad_t>(BOps::grad_lhs(u_val, e_val, g)));
        }
        if constexpr (BOps::USE_RHS) {
            const grad_rhs_t contribution = static_cast<grad_rhs_t>(BOps::grad_rhs(u_val, e_val, g));
            if constexpr (RHS_BROADCAST) {
                // One slot per edge, shared by all d features of it.
                atomicAdd(&grad_rhs[e_off], contribution);
            } else {
                // (arg, f) is unique across the whole output, so nothing else
                // ever touches this slot.  Slots of edges that won nothing keep
                // the zero the allocation gave them.
                grad_rhs[e_off] = contribution;
            }
        }
    }
}
