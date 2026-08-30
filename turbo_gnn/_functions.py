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

#: Node -> thread-block scheduling policies, mirroring ``csrc/common/scheduler.cuh``.
#:
#: ``one_per_block`` is the historical behaviour (grid.x == node count). The other three are
#: persistent: the grid is sized to ``blocks_per_sm * SM_count`` and each block loops over
#: several nodes -- ``grid_stride`` strides by gridDim.x, ``precomputed`` walks a host-assigned
#: contiguous slice balanced by edge count, ``dynamic`` claims from an atomic work queue.
SCHEDULES = {"one_per_block": 0, "grid_stride": 1, "precomputed": 2, "dynamic": 3}

#: The default is ``one_per_block``, i.e. the historical launch. This reverses the original
#: design decision ("persistent dynamic by default") and it is worth saying why, because the
#: reason is not that the persistent path is broken -- it is bit-exact and often faster.
#:
#: Measured over 108 (graph, conv, head dim, direction) cells -- every graph in
#: ``configs/datasets/main/`` plus ogbn-proteins, head dims 128 and 256, forward and backward,
#: on idle GPUs -- no persistent policy is a safe blanket default:
#:
#:     policy                geomean   worst    best   >=1.0
#:     grid_stride/bps1024      0.98    0.58    1.35   41/108
#:     precomputed/bps1024      0.90    0.37    1.10   16/108
#:     dynamic/bps256/c4        0.83    0.46    1.16   19/108
#:     dynamic/bps256/c1        0.77    0.22    1.08   23/108
#:
#: Picking the best policy per cell (an oracle no runtime can have) gives 1.02x. The reason
#: the baseline is so hard to beat is that the hardware block scheduler is already a dynamic
#: work queue *and* a locality-optimal one, at zero cost -- see ``SCHEDULER_PERF.md``.
#:
#: So the policies stay available and tunable, and the default costs nobody anything. Where
#: they do pay, they pay well: min_aggr backward on small sparse graphs reaches 1.24-1.35x
#: (tolokers-2, avazu-ctr, city-roads-M), min_aggr forward on cache-bound ogbn-proteins reaches
#: 1.10x with ``precomputed``, and gat_v2 at head dim 256 reaches 1.08x with ``dynamic``.
DEFAULT_SCHEDULE = "one_per_block"

#: Resident blocks per SM targeted by the persistent policies; ignored by ``one_per_block``.
#: Low values are catastrophic (12x slower at 1, ~2x at 8) because the grid under-fills the
#: GPU. It flattens out above ~128 and 1024 was the best single value in the sweep, which is
#: also why sizing the grid from ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` was tried
#: and dropped: above the knee the exact grid size stops mattering.
DEFAULT_BLOCKS_PER_SM = 1024

#: Consecutive work items ``dynamic`` claims per atomic. Read by that policy only.
#:
#: This started as a fix for what looked like the bottleneck -- one global atomic per node --
#: and it is kept because it does help sparse graphs, where per-node work is small enough for
#: the atomic to show. It is *not* a free win: on a cache-bound graph a larger chunk widens the
#: window of nodes in flight and costs L2 reuse, monotonically (0.57x at chunk 1 down to 0.34x
#: at chunk 32 on ogbn-proteins). Hence a tunable rather than a constant.
DEFAULT_SCHED_CHUNK = 4

#: How the light and heavy node buckets are launched relative to each other.
#:
#: These convolutions split nodes by degree quantile and run a kernel per bucket. The two touch
#: disjoint output rows and have no data dependence, but historically went out back to back on
#: one stream, so the heavy launch could not begin until the light one had drained.
#: ``concurrent`` puts them on separate streams, heavy issued first.
#:
#: Forward and backward are controlled separately, because they want different answers.
#: Measured over 192 cells (16 graphs x 3 convs x head dims 128/256 x both passes):
#:
#:     head dim 128, forward     1.140      head dim 128, backward    0.921
#:     head dim 256, forward     1.113      head dim 256, backward    0.988
#:
#: Concurrent on forward and sequential on backward is worth 1.061x overall against 1.037x for
#: turning it on everywhere, and raises cells at or above baseline from 147/192 to 177/192.
#: Making the two independent lets the autotuner find that split per graph rather than being
#: forced into one answer for both.
#:
#: A third mode, "heavy_first" -- reordering the two launches on a single stream -- was
#: implemented, measured at 0.964 geomean, and removed. Reordering cannot help when the second
#: launch still waits for the first to drain; all of the gain is in the overlap.
BUCKET_LAUNCHES = {"sequential": 0, "concurrent": 1}
DEFAULT_BUCKET_LAUNCH = "sequential"


def resolve_bucket_launch(bucket_launch) -> int:
    """Accept either a name from :data:`BUCKET_LAUNCHES` or the raw int the kernel takes."""
    if isinstance(bucket_launch, str):
        try:
            return BUCKET_LAUNCHES[bucket_launch]
        except KeyError:
            raise ValueError(
                f"unknown bucket_launch {bucket_launch!r}; expected one of {', '.join(BUCKET_LAUNCHES)}"
            ) from None
    if bucket_launch not in BUCKET_LAUNCHES.values():
        raise ValueError(f"bucket_launch must be one of {sorted(BUCKET_LAUNCHES.values())}, got {bucket_launch!r}")
    return int(bucket_launch)


def resolve_schedule(schedule) -> int:
    """Accept either the policy name or its raw int, and validate."""
    if isinstance(schedule, int):
        if schedule not in SCHEDULES.values():
            raise ValueError(
                f"schedule must be one of {sorted(SCHEDULES.values())} or {sorted(SCHEDULES)}, got {schedule}"
            )
        return schedule
    try:
        return SCHEDULES[schedule]
    except KeyError:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {sorted(SCHEDULES)}") from None


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
    """Min/max reduction aggregation over CSR neighbors.

    Forward: calls ``_C.reduction_aggr_forward_partitioned`` which splits nodes
    into light (atomic kernel) and heavy (tiled reduction kernel) buckets.
    Saves argmin/argmax indices for the backward pass.

    Backward: scatters ``grad_out`` to source nodes using the saved arg indices
    via ``_C.reduction_aggr_backward`` (only the "winning" source gets gradient).
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
        schedule=DEFAULT_SCHEDULE,
        blocks_per_sm=DEFAULT_BLOCKS_PER_SM,
        sched_chunk=DEFAULT_SCHED_CHUNK,
        forward_bucket_launch=DEFAULT_BUCKET_LAUNCH,
        backward_bucket_launch=DEFAULT_BUCKET_LAUNCH,
        forward_heavy_edge_slice=0,
        fwd_chunk_node=None,
        fwd_chunk_start=None,
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
            resolve_schedule(schedule),
            blocks_per_sm,
            sched_chunk,
            resolve_bucket_launch(forward_bucket_launch),
            fwd_chunk_node if fwd_chunk_node is not None else _empty_i32(X.device),
            fwd_chunk_start if fwd_chunk_start is not None else _empty_i32(X.device),
            forward_heavy_edge_slice,
        )
        ctx.save_for_backward(arg_idx)
        ctx.num_src_nodes = X.size(0)
        ctx.warps_per_block = warps_per_block
        ctx.schedule = resolve_schedule(schedule)
        ctx.blocks_per_sm = blocks_per_sm
        ctx.sched_chunk = sched_chunk
        # Backward gets its own value: concurrency helps the forward buckets and hurts the
        # backward ones, so forcing one answer on both leaves most of the gain behind.
        ctx.bucket_launch = resolve_bucket_launch(backward_bucket_launch)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        (arg_idx,) = ctx.saved_tensors
        num_src_nodes = ctx.num_src_nodes
        grad_x = _C.reduction_aggr_backward(
            grad_out,
            arg_idx,
            num_src_nodes,
            ctx.warps_per_block,
            ctx.schedule,
            ctx.blocks_per_sm,
            ctx.sched_chunk,
            ctx.bucket_launch,
        )
        # 14 trailing forward args, plus the 3 carrying the heavy-node edge-slice table.
        return (None, None, grad_x) + (None,) * 17


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
        schedule=DEFAULT_SCHEDULE,
        blocks_per_sm=DEFAULT_BLOCKS_PER_SM,
        sched_chunk=DEFAULT_SCHED_CHUNK,
        forward_bucket_launch=DEFAULT_BUCKET_LAUNCH,
        backward_bucket_launch=DEFAULT_BUCKET_LAUNCH,
        forward_heavy_edge_slice=0,
        fwd_chunk_node=None,
        fwd_chunk_start=None,
        fwd_node_chunk_offset=None,
        backward_heavy_edge_slice=0,
        bwd_chunk_node=None,
        bwd_chunk_start=None,
        bwd_node_chunk_offset=None,
    ):
        if torch.is_autocast_enabled():
            attention_weights = attention_weights.to(torch.get_autocast_gpu_dtype())
        schedule_id = resolve_schedule(schedule)

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
            schedule_id,
            blocks_per_sm,
            sched_chunk,
            resolve_bucket_launch(forward_bucket_launch),
            fwd_chunk_node if fwd_chunk_node is not None else _empty_i32(x_left.device),
            fwd_chunk_start if fwd_chunk_start is not None else _empty_i32(x_left.device),
            fwd_node_chunk_offset if fwd_node_chunk_offset is not None else _empty_i32(x_left.device),
            forward_heavy_edge_slice,
        )
        ctx.schedule = schedule_id
        ctx.blocks_per_sm = blocks_per_sm
        ctx.sched_chunk = sched_chunk
        # Backward gets its own value: concurrency helps the forward buckets and hurts the
        # backward ones, so forcing one answer on both leaves most of the gain behind.
        ctx.bucket_launch = resolve_bucket_launch(backward_bucket_launch)
        # The undirected backward slices the *forward* CSR, since that is the adjacency it walks.
        empty = _empty_i32(x_left.device)
        ctx.backward_heavy_edge_slice = backward_heavy_edge_slice
        ctx.bwd_chunk_node = bwd_chunk_node if bwd_chunk_node is not None else empty
        ctx.bwd_chunk_start = bwd_chunk_start if bwd_chunk_start is not None else empty
        ctx.bwd_node_chunk_offset = bwd_node_chunk_offset if bwd_node_chunk_offset is not None else empty
        ctx.negative_slope = negative_slope
        ctx.grad_A_reduce_row_chunk_size = grad_A_reduce_row_chunk_size
        ctx.backward_light_warps = backward_light_warps
        ctx.backward_heavy_warps = backward_heavy_warps
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
            ctx.schedule,
            ctx.blocks_per_sm,
            ctx.sched_chunk,
            ctx.bucket_launch,
            ctx.bwd_chunk_node,
            ctx.bwd_chunk_start,
            ctx.bwd_node_chunk_offset,
            ctx.backward_heavy_edge_slice,
        )

        # 4 CSR tensors + 3 gradients + 13 non-Variable args = 20 total
        # 16 trailing forward args, plus the 4 carrying the heavy-node edge-slice table.
        return (None, None, None, None, grad_x_left, grad_x_right, grad_attention) + (None,) * 24


_EMPTY_I32: dict[torch.device, torch.Tensor] = {}


def _empty_i32(device: torch.device) -> torch.Tensor:
    """Placeholder for an unused slice table.

    The C++ side takes the table by value, and `None` does not convert to a `torch::Tensor`, so
    the node-per-block path still has to hand over something. Cached per device because it would
    otherwise be a fresh allocation on every call.
    """
    t = _EMPTY_I32.get(device)
    if t is None:
        t = torch.empty(0, dtype=torch.int32, device=device)
        _EMPTY_I32[device] = t
    return t


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
        schedule=DEFAULT_SCHEDULE,
        blocks_per_sm=DEFAULT_BLOCKS_PER_SM,
        sched_chunk=DEFAULT_SCHED_CHUNK,
        forward_bucket_launch=DEFAULT_BUCKET_LAUNCH,
        backward_bucket_launch=DEFAULT_BUCKET_LAUNCH,
        forward_heavy_edge_slice=0,
        fwd_chunk_node=None,
        fwd_chunk_start=None,
        fwd_node_chunk_offset=None,
        backward_heavy_edge_slice=0,
        bwd_chunk_node=None,
        bwd_chunk_start=None,
        bwd_node_chunk_offset=None,
    ):
        scale = scale or 1 / (Q.shape[-1] ** 0.5)
        schedule_id = resolve_schedule(schedule)
        empty = _empty_i32(Q.device)
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
            schedule_id,
            blocks_per_sm,
            sched_chunk,
            resolve_bucket_launch(forward_bucket_launch),
            fwd_chunk_node if fwd_chunk_node is not None else empty,
            fwd_chunk_start if fwd_chunk_start is not None else empty,
            fwd_node_chunk_offset if fwd_node_chunk_offset is not None else empty,
            forward_heavy_edge_slice,
        )

        ctx.schedule = schedule_id
        ctx.blocks_per_sm = blocks_per_sm
        ctx.sched_chunk = sched_chunk
        # Backward gets its own value: concurrency helps the forward buckets and hurts the
        # backward ones, so forcing one answer on both leaves most of the gain behind.
        ctx.bucket_launch = resolve_bucket_launch(backward_bucket_launch)
        ctx.scale = scale
        ctx.is_directed = is_directed
        ctx.num_heads = Q.shape[1]
        ctx.head_dim = Q.shape[2]
        ctx.backward_light_warps = backward_light_warps
        ctx.backward_heavy_warps = backward_heavy_warps
        # The backward bucket has its own table: it slices the transpose CSR, not the forward one.
        ctx.backward_heavy_edge_slice = backward_heavy_edge_slice
        ctx.bwd_chunk_node = bwd_chunk_node if bwd_chunk_node is not None else empty
        ctx.bwd_chunk_start = bwd_chunk_start if bwd_chunk_start is not None else empty
        ctx.bwd_node_chunk_offset = bwd_node_chunk_offset if bwd_node_chunk_offset is not None else empty
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
        grad_output = grad_output.view(-1, num_heads, head_dim)

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
            ctx.schedule,
            ctx.blocks_per_sm,
            ctx.sched_chunk,
            ctx.bucket_launch,
            ctx.bwd_chunk_node,
            ctx.bwd_chunk_start,
            ctx.bwd_node_chunk_offset,
            ctx.backward_heavy_edge_slice,
        )

        # 15 trailing forward args, plus the 4 forward-table and 4 backward-table arguments.
        return (None,) * 4 + (dQ, dK, dV) + (None,) * 23


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
