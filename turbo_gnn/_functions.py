"""torch.autograd.Function subclasses wrapping turbo_gnn._C CUDA kernels.

Each class bridges the Python API (:mod:`turbo_gnn.ops`) to the C++/CUDA
extension module (``turbo_gnn._C``), implementing custom forward/backward
passes with AMP support (``custom_fwd`` / ``custom_bwd``).
"""

from __future__ import annotations

import warnings
from math import ceil

import torch

import turbo_gnn._C as _C

WARP_SIZE = 32
FOUR_BYTES_CONSTANT = 4


def _next_power_of_two(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x


class ReductionAggrFunction(torch.autograd.Function):
    """Min/max/sum reduction aggregation over CSR neighbors.

    Forward: calls ``_C.reduction_aggr_forward_partitioned`` which splits nodes
    into light (atomic kernel) and heavy (tiled reduction kernel) buckets.
    For min/max it saves argmin/argmax indices for the backward pass.

    Backward depends on the reducer:

    - **min/max**: scatters ``grad_out`` to source nodes using the saved arg
      indices via ``_C.reduction_aggr_backward`` (only the "winning" source
      gets gradient).
    - **sum**: every source contributed, so the gradient is itself a sum
      aggregation -- the same forward kernel run over the *transposed* CSR.
      This needs the backward adjacency, which the ``bwd_*`` arguments carry;
      without them a ``reduce="sum"`` tensor cannot require grad.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        edge_ptr,
        edge_idx,
        X,
        light,
        heavy,
        max_degree,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel=False,
        features_per_block=32,
        tiles_y=8,
        reduce="min",
        pipeline_stages=0,
        bwd_edge_ptr=None,
        bwd_edge_idx=None,
        bwd_light=None,
        bwd_heavy=None,
        bwd_max_degree=-1,
    ):
        if torch.is_autocast_enabled():
            X = X.to(torch.get_autocast_gpu_dtype())

        num_of_threads_invoked = WARP_SIZE * warps_per_block
        num_features_per_thread = FOUR_BYTES_CONSTANT // X.dtype.itemsize

        num_threads_needed = ceil(X.shape[-1] / num_features_per_thread)

        if num_threads_needed < num_of_threads_invoked:
            warps_per_block_needed = ceil(num_threads_needed / WARP_SIZE)
            warnings.warn(
                f"Number of threads involved for ReductionAggr is {num_of_threads_invoked} "
                f"({warps_per_block} warps per thread block requested). "
                f"However, number of threads needed is {num_threads_needed} "
                f"({warps_per_block_needed} warps). Setting this value instead."
            )

            warps_per_block = warps_per_block_needed
            if warps_per_block not in {1, 2, 4, 8, 16, 32, 64}:
                warps_per_block = _next_power_of_two(warps_per_block)

        out, arg_idx = _C.reduction_aggr_forward_partitioned(
            edge_ptr,
            edge_idx,
            X,
            light,
            heavy,
            max_degree,
            warps_per_block,
            edges_per_block_heavy_nodes,
            use_2d_kernel,
            features_per_block,
            tiles_y,
            reduce,
            pipeline_stages,
        )
        ctx.save_for_backward(arg_idx, bwd_edge_ptr, bwd_edge_idx, bwd_light, bwd_heavy)
        ctx.num_src_nodes = X.size(0)
        ctx.warps_per_block = warps_per_block
        ctx.reduce = reduce
        ctx.bwd_max_degree = bwd_max_degree
        ctx.edges_per_block_heavy_nodes = edges_per_block_heavy_nodes
        ctx.features_per_block = features_per_block
        ctx.tiles_y = tiles_y
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        arg_idx, bwd_edge_ptr, bwd_edge_idx, bwd_light, bwd_heavy = ctx.saved_tensors
        num_src_nodes = ctx.num_src_nodes

        if ctx.reduce == "sum":
            if bwd_edge_ptr is None:
                raise RuntimeError(
                    "reduction_aggr(reduce='sum') backward needs the transposed CSR. "
                    "Pass the backward adjacency (bwd_edge_ptr/bwd_edge_idx/bwd_light/bwd_heavy/"
                    "bwd_max_degree) to ReductionAggrFunction.apply."
                )
            # grad_x[u] = sum over out-edges u->v of grad_out[v]: the same
            # aggregation, walked on the transpose.
            grad_x, _ = _C.reduction_aggr_forward_partitioned(
                bwd_edge_ptr,
                bwd_edge_idx,
                grad_out.contiguous(),
                bwd_light,
                bwd_heavy,
                ctx.bwd_max_degree,
                ctx.warps_per_block,
                ctx.edges_per_block_heavy_nodes,
                True,  # sum has no packed path; the 2-D kernel is the only option
                ctx.features_per_block,
                ctx.tiles_y,
                "sum",
                0,
            )
        else:
            grad_x = _C.reduction_aggr_backward(grad_out, arg_idx, num_src_nodes, ctx.warps_per_block)

        return (
            None,
            None,
            grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class gatv2_function(torch.autograd.Function):
    """GATv2 fused forward/backward pass.

    Forward (``_C.gatv2_forward``): for each edge (u -> v), computes
    ``e = attn^T * LeakyReLU(x_left[v] + x_right[u])``, applies numerically
    stable edge softmax (returns log-sum-exp for backward), and aggregates
    ``out[v] = sum alpha_{uv} * x_right[u]``.

    Backward (``_C.gatv2_backward``): computes gradients for x_left, x_right,
    and attention weights. The backward kernel walks the *transposed* CSR
    (backward adjacency) to scatter gradients to source nodes. The
    ``grad_A_reduce_row_chunk_size`` parameter controls shared-memory usage
    in the attention-gradient reduction kernel.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        indptr_forward,
        indices_forward,
        indptr_backward,
        indices_backward,
        x_left,
        x_right,
        attention_weights,
        negative_slope,
        grad_A_reduce_row_chunk_size,
        fwd_light_nodes,
        fwd_heavy_nodes,
        bwd_light_nodes,
        bwd_heavy_nodes,
        forward_light_warps,
        forward_heavy_warps,
        backward_light_warps,
        backward_heavy_warps,
        is_directed,
        pipeline_stages=0,
        backward_pipeline_stages=0,
    ):
        if torch.is_autocast_enabled():
            attention_weights = attention_weights.to(torch.get_autocast_gpu_dtype())

        output, logsumexp = _C.gatv2_forward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            attention_weights,
            negative_slope,
            fwd_light_nodes,
            fwd_heavy_nodes,
            forward_light_warps,
            forward_heavy_warps,
            pipeline_stages,
        )
        ctx.negative_slope = negative_slope
        ctx.grad_A_reduce_row_chunk_size = grad_A_reduce_row_chunk_size
        ctx.backward_light_warps = backward_light_warps
        ctx.backward_heavy_warps = backward_heavy_warps
        ctx.backward_pipeline_stages = backward_pipeline_stages
        ctx.is_directed = is_directed
        ctx.heads = x_left.shape[1]
        ctx.head_dim = x_left.shape[2]

        ctx.save_for_backward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
            fwd_light_nodes,
            fwd_heavy_nodes,
            bwd_light_nodes,
            bwd_heavy_nodes,
        )

        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
            fwd_light_nodes,
            fwd_heavy_nodes,
            bwd_light_nodes,
            bwd_heavy_nodes,
        ) = ctx.saved_tensors

        num_heads = ctx.heads
        head_dim = ctx.head_dim

        grad_output = grad_output.view(-1, num_heads, head_dim)

        grad_x_left, grad_x_right, grad_attention = _C.gatv2_backward(
            grad_output,
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
            ctx.negative_slope,
            ctx.grad_A_reduce_row_chunk_size,
            fwd_light_nodes,
            fwd_heavy_nodes,
            bwd_light_nodes,
            bwd_heavy_nodes,
            ctx.backward_light_warps,
            ctx.backward_heavy_warps,
            ctx.is_directed,
            ctx.backward_pipeline_stages,
        )

        # 4 CSR tensors + 3 gradients + 13 non-Variable args = 20 total
        return (None, None, None, None, grad_x_left, grad_x_right, grad_attention) + (None,) * 13


class _FusedGraphAttention(torch.autograd.Function):
    """Fused multi-head graph transformer attention (forward + backward).

    Forward (``_C.gt_forward_csr_mh``): computes per-edge dot-product attention
    scores ``Q[src] . K[dst] * scale``, edge softmax via log-sum-exp, and
    weighted value aggregation -- all in a single kernel over the forward CSR.

    Backward (``_C.gt_backward_csr_mh``): computes dQ, dK, dV using the
    transposed CSR (backward adjacency) and the saved logsumexp + output
    tensors for the softmax Jacobian.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        edge_ptr,
        edge_idx,
        edge_ptr_T,
        edge_idx_T,
        Q,
        K,
        V,
        scale,
        fwd_light_nodes,
        fwd_heavy_nodes,
        bwd_light_nodes,
        bwd_heavy_nodes,
        forward_light_warps,
        forward_heavy_warps,
        backward_light_warps,
        backward_heavy_warps,
        is_directed,
        pipeline_stages=0,
        backward_pipeline_stages=0,
    ):
        scale = scale or 1 / (Q.shape[-1] ** 0.5)
        out, logsumexp = _C.gt_forward_csr_mh(
            edge_ptr,
            edge_idx,
            Q,
            K,
            V,
            scale,
            fwd_light_nodes,
            fwd_heavy_nodes,
            forward_light_warps,
            forward_heavy_warps,
            pipeline_stages,
        )

        ctx.scale = scale
        ctx.is_directed = is_directed
        ctx.num_heads = Q.shape[1]
        ctx.head_dim = Q.shape[2]
        ctx.backward_light_warps = backward_light_warps
        ctx.backward_heavy_warps = backward_heavy_warps
        ctx.backward_pipeline_stages = backward_pipeline_stages
        ctx.save_for_backward(
            edge_ptr, edge_idx, edge_ptr_T, edge_idx_T, Q, K, V, out, logsumexp, bwd_light_nodes, bwd_heavy_nodes
        )

        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (
            edge_ptr,
            edge_idx,
            edge_ptr_T,
            edge_idx_T,
            Q,
            K,
            V,
            out,
            logsumexp,
            bwd_light_nodes,
            bwd_heavy_nodes,
        ) = ctx.saved_tensors
        scale = ctx.scale
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        grad_output = grad_output.reshape(-1, num_heads, head_dim).contiguous()

        dQ, dK, dV = _C.gt_backward_csr_mh(
            edge_ptr,
            edge_idx,
            edge_ptr_T,
            edge_idx_T,
            Q,
            K,
            V,
            out,
            grad_output,
            logsumexp,
            scale,
            bwd_light_nodes,
            bwd_heavy_nodes,
            ctx.backward_light_warps,
            ctx.backward_heavy_warps,
            ctx.is_directed,
            ctx.backward_pipeline_stages,
        )

        return (None,) * 4 + (dQ, dK, dV) + (None,) * 12


class _CudaSpMMConvFn(torch.autograd.Function):
    """cuSPARSE SpMM with AdjacencyForwardBackwardWithNodeBuckets graph format."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim):
        ctx.save_for_backward(forward_indptr, forward_indices)
        ctx.norm_type = norm_type
        ctx.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        ctx.block_dim = block_dim

        return csr_SPMM_normalized(
            indptr=forward_indptr,
            indices=forward_indices,
            features=x,
            edge_weights=None,
            norm=norm_type,
            algorithm=cu_sparse_algorithm_id,
            do_transpose_a=False,
            block_dim=block_dim,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        forward_indptr, forward_indices = ctx.saved_tensors
        grad_x = csr_SPMM_normalized(
            indptr=forward_indptr,
            indices=forward_indices,
            features=grad_outputs[0],
            edge_weights=None,
            norm=ctx.norm_type,
            algorithm=ctx.cu_sparse_algorithm_id,
            do_transpose_a=True,
            block_dim=ctx.block_dim,
        )
        return grad_x, None, None, None, None, None


def csr_SPMM_normalized(
    indptr,
    indices,
    features,
    edge_weights=None,
    norm="none",
    algorithm=-1,
    use_cache=True,
    do_transpose_a=False,
    block_dim=256,
):
    """Normalized SpMM: ``out = norm(A) @ features`` via cuSPARSE.

    Wraps ``_C.csr_SPMM_normalized`` which computes degree-based normalization
    weights on the fly and calls ``cusparseSpMM``.

    Args:
        indptr: CSR row pointers, shape ``[N+1]``.
        indices: CSR column indices, shape ``[E]``.
        features: Node feature matrix, shape ``[N, F]``.
        edge_weights: Optional per-edge weights, shape ``[E]``. None = all ones.
        norm: Normalization mode -- ``"none"`` (sum), ``"right"`` (mean),
            ``"left"`` (random-walk), ``"both"`` (symmetric GCN).
        algorithm: cuSPARSE algorithm id (-1 = auto select).
        use_cache: Cache the cuSPARSE descriptor across calls.
        do_transpose_a: If True, multiply by A^T instead of A (used in backward).
        block_dim: CUDA block size for the normalization pre-pass kernel.

    Returns:
        Result tensor, shape ``[N, F]``.
    """
    if edge_weights is None:
        edge_weights_gpu = torch.empty(0, device=features.device, dtype=torch.float32)
    else:
        edge_weights_gpu = edge_weights.to(device=features.device, dtype=torch.float32)

    out = _C.csr_SPMM_normalized(
        indptr, indices, features.contiguous(), edge_weights_gpu, norm, algorithm, use_cache, do_transpose_a, block_dim
    )

    return out


class GSpMMFunction(torch.autograd.Function):
    """Generalized SpMM: ``out[v] = reduce_{(u,e) in in(v)} op(lhs[u], rhs[e])``.

    Covers the ``dgl.ops.gspmm`` operator family -- ``op`` in
    {copy_u, copy_e, add, sub, mul, div} times ``reduce`` in {sum, min, max}.

    Forward calls ``_C.gspmm_forward``, which splits nodes into light/heavy
    buckets and, for min/max, records the winning CSR edge position per output
    element.

    Backward has two shapes:

    - **min/max**: only the winning edge of each output element contributed, so
      one scatter over the saved indices (``_C.gspmm_backward_arg``) yields both
      gradients at once.
    - **sum**: every edge contributed, and the two gradients decouple. The node
      gradient is this same g-SpMM re-run on the *transposed* CSR; the edge
      gradient is ``_C.gspmm_backward_edge`` over the forward CSR. For ``mul``
      and ``div`` the transposed run still needs the edge values, which live in
      forward-CSR order, so ``graph.backward_edge_map`` goes to the kernel and
      the read is redirected there.
    """

    # Operations whose node gradient depends on the edge value; their sum
    # backward needs edge data remapped into backward-CSR order.
    _EDGE_WEIGHTED_OPS = ("mul", "div")

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        lhs,
        rhs,
        op,
        reduce,
        edge_ptr,
        edge_idx,
        light,
        heavy,
        bwd_edge_ptr,
        bwd_edge_idx,
        bwd_light,
        bwd_heavy,
        bwd_edge_map,
        warps_per_block=8,
        features_per_block=32,
        tiles_y=8,
        pipeline_stages=0,
    ):
        if torch.is_autocast_enabled():
            target = torch.get_autocast_gpu_dtype()
            if lhs is not None and lhs.is_floating_point():
                lhs = lhs.to(target)
            if rhs is not None and rhs.is_floating_point():
                rhs = rhs.to(target)

        # Absent operands cross the pybind boundary as empty tensors rather than
        # None, matching how the cuSPARSE path passes absent edge weights.
        ref = lhs if lhs is not None else rhs
        if ref is None:
            raise ValueError(f"gspmm(op='{op}') needs at least one of lhs/rhs, got neither")
        empty = torch.empty(0, device=ref.device, dtype=ref.dtype)
        lhs_t = empty if lhs is None else lhs.contiguous()
        rhs_t = empty if rhs is None else rhs.contiguous()

        out, arg_eid = _C.gspmm_forward(
            edge_ptr,
            edge_idx,
            lhs_t,
            rhs_t,
            light,
            heavy,
            op,
            reduce,
            warps_per_block,
            features_per_block,
            tiles_y,
            pipeline_stages,
        )

        ctx.save_for_backward(
            lhs_t, rhs_t, arg_eid, edge_ptr, edge_idx, bwd_edge_ptr, bwd_edge_idx, bwd_light, bwd_heavy, bwd_edge_map
        )
        ctx.op = op
        ctx.reduce = reduce
        ctx.warps_per_block = warps_per_block
        ctx.features_per_block = features_per_block
        ctx.tiles_y = tiles_y
        ctx.pipeline_stages = pipeline_stages
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        (
            lhs_t,
            rhs_t,
            arg_eid,
            edge_ptr,
            edge_idx,
            bwd_edge_ptr,
            bwd_edge_idx,
            bwd_light,
            bwd_heavy,
            bwd_edge_map,
        ) = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        op = ctx.op
        lhs_needs_grad, rhs_needs_grad = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_lhs = None
        grad_rhs = None

        if ctx.reduce in ("min", "max"):
            g_lhs, g_rhs = _C.gspmm_backward_arg(grad_out, arg_eid, edge_idx, lhs_t, rhs_t, op, ctx.warps_per_block)
            # Only the node gradient is staged in fp32 -- its scatter is
            # atomicAdd-based, since several destinations can share a winning
            # source.  The edge gradient already comes back in the operand
            # dtype (see gspmm_backward_arg).
            if lhs_needs_grad:
                grad_lhs = g_lhs.to(lhs_t.dtype)
            if rhs_needs_grad:
                grad_rhs = g_rhs
        else:
            if lhs_needs_grad:
                if bwd_edge_ptr is None:
                    raise RuntimeError(
                        f"gspmm(op='{op}', reduce='sum') backward needs the transposed CSR. "
                        "Pass the backward adjacency through GSpMMFunction.apply."
                    )
                # d(sum)/d(lhs[u]) sums the per-edge factor over u's out-edges,
                # which is the same g-SpMM walked on the transpose.
                if op in GSpMMFunction._EDGE_WEIGHTED_OPS:
                    if bwd_edge_map is None:
                        raise RuntimeError(
                            f"gspmm(op='{op}', reduce='sum') backward needs graph.backward_edge_map "
                            "to read edge data while walking the transposed CSR."
                        )
                    # Handed to the kernel rather than used to permute rhs
                    # here: materializing rhs[bwd_edge_map] cost a full [E, d]
                    # gather and a second [E, d] allocation on every backward.
                    rhs_bwd = rhs_t
                    edge_map = bwd_edge_map
                    bwd_op = op
                else:
                    # add/sub/copy_u contribute a factor of 1 per edge, so the
                    # edge operand drops out of the node gradient entirely.
                    rhs_bwd = torch.empty(0, device=grad_out.device, dtype=grad_out.dtype)
                    # Absent, spelled as an empty tensor of the index dtype:
                    # the pybind signature takes a Tensor, not an optional.
                    edge_map = bwd_edge_ptr[:0]
                    bwd_op = "copy_u"

                grad_lhs, _ = _C.gspmm_forward(
                    bwd_edge_ptr,
                    bwd_edge_idx,
                    grad_out,
                    rhs_bwd,
                    bwd_light,
                    bwd_heavy,
                    bwd_op,
                    "sum",
                    ctx.warps_per_block,
                    ctx.features_per_block,
                    ctx.tiles_y,
                    ctx.pipeline_stages,
                    edge_map,
                )

            if rhs_needs_grad:
                # Already in the operand dtype: this kernel writes every slot
                # exactly once, so it has no float staging buffer to cast back.
                grad_rhs = _C.gspmm_backward_edge(edge_ptr, edge_idx, grad_out, lhs_t, rhs_t, op, ctx.warps_per_block)

        return (
            grad_lhs,
            grad_rhs,
            None,  # op
            None,  # reduce
            None,  # edge_ptr
            None,  # edge_idx
            None,  # light
            None,  # heavy
            None,  # bwd_edge_ptr
            None,  # bwd_edge_idx
            None,  # bwd_light
            None,  # bwd_heavy
            None,  # bwd_edge_map
            None,  # warps_per_block
            None,  # features_per_block
            None,  # tiles_y
            None,  # pipeline_stages
        )
