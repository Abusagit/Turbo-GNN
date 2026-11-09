"""torch.autograd.Function subclasses wrapping turbo_gnn._C CUDA kernels.

Each class bridges the Python API (:mod:`turbo_gnn.ops`) to the C++/CUDA
extension module (``turbo_gnn._C``), implementing custom forward/backward
passes with AMP support (``custom_fwd`` / ``custom_bwd``).
"""

from __future__ import annotations

import torch

from turbo_gnn._extension import _C

#: Node -> thread-block scheduling policy, mirroring ``csrc/common/scheduler.cuh``.
#:
#: Only ``one_per_block`` (grid.x == node count) is compiled into the attention kernels. The
#: hardware block scheduler already hands blocks out as a dynamic, locality-preserving queue, and
#: the persistent policies that scheduler.cuh also implements were never a safe default; keeping
#: them as a template axis quadrupled the attention kernels' build time. Load balance comes from
#: degree bucketing, degree ordering, concurrent bucket launch and edge slicing instead.
SCHEDULES = {"one_per_block": 0}
DEFAULT_SCHEDULE = "one_per_block"

#: Resident blocks per SM targeted by the persistent policies; ignored by ``one_per_block``.
#: Low values are catastrophic (12x slower at 1, ~2x at 8) because the grid under-fills the
#: GPU. It flattens out above ~128 and 1024 was the best single value in the sweep, which is
#: also why sizing the grid from ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` was tried
#: and dropped: above the knee the exact grid size stops mattering.
# Target resident blocks per SM for the persistent schedules. 32 is the sm_80 hardware limit,
# so it is the largest value for which the name is truthful: persistent_grid_x computes
# SM_count * this as a *total* grid size and never checks it against the limit, so anything
# above 32 silently stops being persistent (at 1024 the grid is 32 waves deep -- effectively
# one-block-per-node with extra steps). Below 32 the GPU is simply under-filled: the light
# bucket uses 1-warp blocks, so 8 blocks/SM is 8 of 64 warps.
# Historical note: C++ defaulted to 8 and Python to 1024; since every call arrives from
# Python, 1024 is what actually ran, and neither value was measured.
DEFAULT_BLOCKS_PER_SM = 32

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


#: Rows that are not whole 16-byte vectors -- odd head dims, or e.g. 100 in fp16 -- are read by the
#: attention kernels element by element, and those longer than 64 elements also fall to the
#: two-pass path. The attention functions copy such inputs into zero-padded rows of whole vectors
#: when the call is large enough to repay the copies, whose cost is mostly fixed: about 100 us of
#: launches and host time per call. Measured on 14 real graphs (padded / element-wise time, 464
#: cells; gathered bytes = edges x heads x padded head dim x element size):
#:
#:   rows over 64 elements, gathering >= 16 MiB     2.97x geomean, 1 cell of 176 slower (0.90x)
#:   rows of 16 B to 64 elements, >= 256 MiB        1.41x geomean, 7 of 136 slower (>= 0.76x);
#:                                                  below 256 MiB padding them lost (0.76x geomean)
#:   rows under 16 bytes                            0.91x on the backward even at 1-10 GiB: one
#:                                                  sector a row either way, so never padded
#:
#: Thresholds in gathered bytes, for rows longer than _PAD_SHORT_ROW_ELEMENTS and for the others.
_PAD_MIN_GATHER_BYTES = 1 << 24
_PAD_MIN_GATHER_BYTES_SHORT_ROWS = 1 << 28
_PAD_SHORT_ROW_ELEMENTS = 64


def _vector_rows(t: torch.Tensor) -> bool:
    """Whether the kernels read ``t``'s rows (its last dim) in 16-byte vectors.

    Mirrors ``rows::rows_16b_aligned`` in ``csrc/common/row_shape.cuh``: every row starts on a
    16-byte boundary and spans a whole number of 16-byte vectors.

    Args:
        t: A tensor with unit stride over its last dim.

    Returns:
        bool: True when the rows qualify.
    """
    elem = t.element_size()
    if (t.shape[-1] * elem) % 16 or t.data_ptr() % 16:
        return False
    return all((stride * elem) % 16 == 0 for size, stride in zip(t.shape[:-1], t.stride()[:-1]) if size > 1)


def _pad_rows(t: torch.Tensor, width: int) -> torch.Tensor:
    """A fresh, contiguous copy of ``t`` with its last dim widened to ``width`` by zeros.

    One op either way (``F.pad`` or ``clone``): host time dominates the cost of the copies on
    small graphs, and three ops per tensor measured 20 us of it against 13.

    Args:
        t: Tensor ``[..., D]``.
        width: New last-dim size, at least ``D``.

    Returns:
        torch.Tensor: ``[..., width]``; the first ``D`` columns are ``t``, the rest zero.
    """
    d = t.shape[-1]
    if width == d:
        return t.clone(memory_format=torch.contiguous_format)
    return torch.nn.functional.pad(t, (0, width - d))


def _pad_attention_rows(num_edges: int, heads: int, *tensors: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], int]:
    """Widen the attention inputs' rows to whole 16-byte vectors when that pays (see
    ``_PAD_MIN_GATHER_BYTES``).

    The zero columns change nothing the kernels compute: they add 0 to every dot product and
    LeakyReLU(0) * 0 to every GATv2 score, give zero output columns, and receive gradients that are
    cut away. Only tensors that need it are copied.

    Args:
        num_edges: Edges of the forward CSR.
        heads: Attention heads.
        *tensors: The inputs, each ``[..., D]`` with the same ``D``.

    Returns:
        tuple: The tensors to hand to the kernels, and the head dim they have -- ``D`` unless
        they were padded.
    """
    d = tensors[0].shape[-1]
    elem = tensors[0].element_size()
    # A non-unit stride over D is the kernels' to reject, whatever the size.
    if d * elem < 16 or all(_vector_rows(t) for t in tensors) or any(t.stride(-1) != 1 for t in tensors if d > 1):
        return tensors, d
    per_vector = 16 // elem
    width = -(-d // per_vector) * per_vector
    threshold = _PAD_MIN_GATHER_BYTES if d > _PAD_SHORT_ROW_ELEMENTS else _PAD_MIN_GATHER_BYTES_SHORT_ROWS
    if num_edges * heads * width * elem < threshold:
        return tensors, d
    return tuple(t if width == d and _vector_rows(t) else _pad_rows(t, width) for t in tensors), width


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


def _slice_kwargs(table) -> dict:
    """The slice-table keyword arguments of ``_C.reduction_aggr_forward_partitioned``.

    Args:
        table: An ``EdgeSliceTable`` of the heavy bucket, or None to let the extension build one.

    Returns:
        The table's tensors by argument name; empty when ``table`` is None.
    """
    if table is None:
        return {}
    return {
        "chunk_node": table.chunk_node,
        "chunk_start": table.chunk_start,
        "node_chunk_offset": table.node_chunk_offset,
    }


def _walk_kwargs(walk) -> dict:
    """The slice table and light-degree bound of a ``WalkPartition`` as ``_C.gspmm_forward``
    keyword arguments; empty when ``walk`` is None."""
    if walk is None:
        return {}
    return {**_slice_kwargs(walk.slices), "light_degree_bound": walk.light_degree}


def reduction_graph_args(graph, slice_size: int, reduce: str) -> tuple:
    """The rows of the reduction walk of ``reduction_aggr`` and ``gspmm``, cached on the graph.

    Args:
        graph: An ``AdjacencyForwardBackwardWithNodeBuckets``.
        slice_size: Edges per slice (``edges_per_block_heavy_nodes``): rows of more edges are cut
            into slices.
        reduce: The reducer; only ``"sum"`` walks the backward adjacency.

    Returns:
        ``(fwd, bwd)``: the forward and backward adjacencies' ``WalkPartition`` s -- the rows walked
        whole, the rows cut into slices and their ``EdgeSliceTable``, and a bound on the whole rows'
        degrees, which the kernel weighs against the launch's total work to pick its build. ``bwd``
        is None unless ``reduce`` is ``"sum"``.
    """
    fwd = graph.walk_partition("forward", slice_size)
    if reduce != "sum":
        return fwd, None
    return fwd, graph.walk_partition("backward", slice_size)


class ReductionAggrFunction(torch.autograd.Function):
    """Min/max/sum reduction aggregation over CSR neighbors, for any feature dim.

    Forward: calls ``_C.reduction_aggr_forward_partitioned``. ``light`` rows take one lane group or
    warp each; ``heavy`` rows are cut into slices of ``edges_per_block_heavy_nodes`` edges, one
    worker each, whose partial results are merged in edge order. ``fwd_slices`` is that slice table;
    ``reduction_graph_args`` gives the partition (every row of more edges than a slice is cut, see
    ``graph.walk_partition``) cached per graph. Without a table the extension builds one on every
    call, at the cost of a host sync. For min/max the forward also saves the winning source of every
    output element.

    Backward depends on the reducer:

    - **min/max**: scatters ``grad_out`` to source nodes using the saved arg
      indices via ``_C.reduction_aggr_backward`` (only the "winning" source
      gets gradient).
    - **sum**: every source contributed, so the gradient is itself a sum
      aggregation -- the same forward kernel run over the *transposed* CSR.
      This needs the backward adjacency, which the ``bwd_*`` arguments carry
      (``bwd_slices`` being its slice table); without them a ``reduce="sum"``
      tensor cannot require grad.

    ``max_degree`` (and ``bwd_max_degree``) bound the degree of the light rows -- see
    ``reduction_graph_args`` -- and may be any larger value, or -1 if unknown. ``use_2d_kernel``,
    ``features_per_block``, ``tiles_y`` and ``pipeline_stages`` are accepted for compatibility and
    have no effect.
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
        fwd_slices=None,
        bwd_slices=None,
    ):
        if torch.is_autocast_enabled():
            X = X.to(torch.get_autocast_gpu_dtype())

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
            **_slice_kwargs(fwd_slices),
        )
        ctx.save_for_backward(arg_idx, bwd_edge_ptr, bwd_edge_idx, bwd_light, bwd_heavy)
        ctx.bwd_slices = bwd_slices
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
                grad_out,
                bwd_light,
                bwd_heavy,
                ctx.bwd_max_degree,
                ctx.warps_per_block,
                ctx.edges_per_block_heavy_nodes,
                False,
                ctx.features_per_block,
                ctx.tiles_y,
                "sum",
                0,
                **_slice_kwargs(ctx.bwd_slices),
            )
        else:
            grad_x = _C.reduction_aggr_backward(grad_out, arg_idx, num_src_nodes, ctx.warps_per_block)

        return (None, None, grad_x) + (None,) * 17


class gatv2_function(torch.autograd.Function):
    """GATv2 fused forward/backward pass.

    Forward (``_C.gatv2_forward``): for each edge (u -> v), computes
    ``e = attn^T * LeakyReLU(x_left[v] + x_right[u])``, applies numerically
    stable edge softmax (returns log-sum-exp for backward), and aggregates
    ``out[v] = sum alpha_{uv} * x_right[u]``.

    Backward (``_C.gatv2_backward``): computes gradients for x_left, x_right,
    and attention weights. On a directed graph the x_right gradient is gathered
    over the *transposed* CSR (backward adjacency); an undirected graph needs
    only the forward one. The softmax term uses ``<grad_output, output>`` per
    row, so the output is saved for the backward. ``grad_A_reduce_row_chunk_size``
    is accepted for compatibility and unused: the attention-weight gradient is
    reduced inside the backward kernels.
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
        pipeline_stages=0,
        heavy_pipeline_stages=0,
        backward_pipeline_stages=0,
        backward_heavy_pipeline_stages=0,
    ):
        if torch.is_autocast_enabled():
            attention_weights = attention_weights.to(torch.get_autocast_gpu_dtype())
        schedule_id = resolve_schedule(schedule)
        head_dim = x_left.shape[2]
        (x_left, x_right, attention_weights), width = _pad_attention_rows(
            indices_forward.numel(), x_left.shape[1], x_left, x_right, attention_weights
        )

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
            pipeline_stages,
            heavy_pipeline_stages,
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
        ctx.backward_pipeline_stages = backward_pipeline_stages
        ctx.backward_heavy_pipeline_stages = backward_heavy_pipeline_stages
        ctx.is_directed = is_directed
        ctx.heads = x_left.shape[1]
        ctx.head_dim = head_dim
        ctx.width = width

        # The padded tensors, output included: the backward runs on the same rows.
        ctx.save_for_backward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
            output,
            fwd_light_nodes,
            fwd_heavy_nodes,
            bwd_light_nodes,
            bwd_heavy_nodes,
        )

        return output if width == head_dim else output[..., :head_dim].contiguous()

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
            output,
            fwd_light_nodes,
            fwd_heavy_nodes,
            bwd_light_nodes,
            bwd_heavy_nodes,
        ) = ctx.saved_tensors

        num_heads = ctx.heads
        head_dim = ctx.head_dim

        # The kernels read the gradient through its row strides but need unit stride over the
        # head dim; autograd may hand over any layout.
        grad_output = grad_output.reshape(-1, num_heads, head_dim)
        if ctx.width != head_dim:
            grad_output = _pad_rows(grad_output, ctx.width)
        elif grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()

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
            output,
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
            ctx.backward_pipeline_stages,
            ctx.backward_heavy_pipeline_stages,
        )

        if ctx.width != head_dim:
            grad_x_left, grad_x_right, grad_attention = (
                g[..., :head_dim] for g in (grad_x_left, grad_x_right, grad_attention)
            )

        # 16 trailing forward args, the 4 carrying the heavy-node edge-slice table, and
        # backward_pipeline_stages from the async-copy work.
        return (None, None, None, None, grad_x_left, grad_x_right, grad_attention) + (None,) * 28


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
        pipeline_stages=0,
        heavy_pipeline_stages=0,
        backward_pipeline_stages=0,
        backward_heavy_pipeline_stages=0,
    ):
        # Of the head dim as given: padding below must not change the scale.
        scale = scale or 1 / (Q.shape[-1] ** 0.5)
        schedule_id = resolve_schedule(schedule)
        empty = _empty_i32(Q.device)
        head_dim = Q.shape[2]
        (Q, K, V), width = _pad_attention_rows(edge_idx.numel(), Q.shape[1], Q, K, V)
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
            pipeline_stages,
            heavy_pipeline_stages,
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
        ctx.head_dim = head_dim
        ctx.width = width
        ctx.backward_light_warps = backward_light_warps
        ctx.backward_heavy_warps = backward_heavy_warps
        # The backward bucket has its own table: it slices the transpose CSR, not the forward one.
        ctx.backward_heavy_edge_slice = backward_heavy_edge_slice
        ctx.bwd_chunk_node = bwd_chunk_node if bwd_chunk_node is not None else empty
        ctx.bwd_chunk_start = bwd_chunk_start if bwd_chunk_start is not None else empty
        ctx.bwd_node_chunk_offset = bwd_node_chunk_offset if bwd_node_chunk_offset is not None else empty
        ctx.backward_pipeline_stages = backward_pipeline_stages
        ctx.backward_heavy_pipeline_stages = backward_heavy_pipeline_stages
        # The padded tensors, output included: the backward runs on the same rows.
        ctx.save_for_backward(
            edge_ptr, edge_idx, edge_ptr_T, edge_idx_T, Q, K, V, out, logsumexp, bwd_light_nodes, bwd_heavy_nodes
        )

        return out if width == head_dim else out[..., :head_dim].contiguous()

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
        grad_output = grad_output.reshape(-1, num_heads, head_dim)
        grad_output = _pad_rows(grad_output, ctx.width) if ctx.width != head_dim else grad_output.contiguous()

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
            ctx.backward_pipeline_stages,
            ctx.backward_heavy_pipeline_stages,
        )

        if ctx.width != head_dim:
            dQ, dK, dV = (g[..., :head_dim] for g in (dQ, dK, dV))

        # 15 trailing forward args, plus the 4 forward-table and 4 backward-table arguments.
        return (None,) * 4 + (dQ, dK, dV) + (None,) * 27


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


class GsddmmFunction(torch.autograd.Function):
    """Autograd wrapper around a resolved :class:`turbo_gnn._gsddmm.GsddmmLaunchPlan`.

    The plan decides which forward kernel runs and which backward variant pairs
    with it, and it is stashed on ``ctx`` as-is -- it is frozen, hashable and
    holds no tensors, so the backward reconstructs nothing.

    Only the operands whose partials actually need them are saved: ``add``,
    ``sub`` and ``copy`` have constant partials and so save no feature tensors at
    all (see ``GsddmmSpec.backward_needs_operands``), which is the difference
    between keeping one and two ``[E, D]`` activations alive per op in a layer.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, plan, graph, lhs, rhs):
        out = plan.launch(graph, lhs, rhs)
        needed = plan.spec.backward_needs_operands
        ctx.plan = plan
        # The graph is not an autograd input, so it rides on ctx rather than
        # through save_for_backward.
        ctx.graph = graph
        ctx.save_for_backward(lhs if "lhs" in needed else None, rhs if "rhs" in needed else None)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        lhs, rhs = ctx.saved_tensors
        d_lhs, d_rhs = ctx.plan.backward(ctx.graph, lhs, rhs, grad_out)
        # forward(plan, graph, lhs, rhs): the first two take no gradient.
        needs_lhs, needs_rhs = ctx.needs_input_grad[2], ctx.needs_input_grad[3]
        return None, None, (d_lhs if needs_lhs else None), (d_rhs if needs_rhs else None)


#: Bounds of the edges per slice of g-SpMM's walk (see ``gspmm_slice_size``).
GSPMM_MIN_SLICE = 16
GSPMM_SLICE_SIZE = 128


def gspmm_slice_size(graph) -> int:
    """Edges per slice of g-SpMM's walk on ``graph``: rows of more edges are cut into slices.

    A launch is over once its longest work item is, and a slice is a walk of its edges by one lane
    group or warp. On a graph with few edges per resident warp the whole launch is a fraction of a
    wave and that walk is all that is left: measured on H100, cora (1.6 edges per resident warp) ran
    1.6-2.3x faster at 16-edge slices than at 128. With many edges per warp, short slices only add
    items, merges and re-read ids: ogbn-arxiv and twitch-views (160-1600 per warp) ran 3-28% faster
    at 128 than at 32. In between, amazon-ratings (25 per warp) was best at 16-32 and tolokers-2
    (124) at 64. Hence the power of two at or below the edges per resident warp, within [16, 128].

    Args:
        graph: An ``AdjacencyForwardBackwardWithNodeBuckets`` on a CUDA device.

    Returns:
        The slice size.
    """
    props = torch.cuda.get_device_properties(graph.forward_indptr.device)
    resident_warps = props.multi_processor_count * (props.max_threads_per_multi_processor // 32)
    per_warp = graph.forward_indices.numel() // max(resident_warps, 1)
    return min(GSPMM_SLICE_SIZE, max(GSPMM_MIN_SLICE, 1 << max(per_warp.bit_length() - 1, 0)))


class GSpMMFunction(torch.autograd.Function):
    """Generalized SpMM: ``out[v] = reduce_{(u,e) in in(v)} op(lhs[u], rhs[e])``.

    Covers the ``dgl.ops.gspmm`` operator family -- ``op`` in
    {copy_u, copy_e, add, sub, mul, div} times ``reduce`` in {sum, min, max} -- for any feature
    width.

    Forward calls ``_C.gspmm_forward``: the rows walked whole and the slices of the longer ones
    (``fwd_walk``, the forward adjacency's ``WalkPartition`` at ``gspmm_slice_size(graph)``, see
    ``reduction_graph_args``) are work items of a lane group or a warp, which gathers 16-byte vectors
    where every row allows them. For min/max it records the winning CSR edge position per output
    element. Without a partition (None) the extension walks the graph's own light/heavy buckets and
    builds their slice table on every call.

    Backward has two shapes:

    - **min/max**: only the winning edge of each output element contributed.
      On a sparse graph ``_C.gspmm_backward_arg`` walks the *transposed* CSR:
      a source's node gradient is the sum over its out-edges of the ones that
      won, so it is written once per element with no atomics, and the edge
      gradient falls out of the same walk; ``graph.backward_edge_map`` tells
      which forward edge each transposed position is. On a dense graph that
      walk tests too many edges and the launcher scatters over the winning
      edges instead (one block per destination, atomics on the node gradient).
    - **sum**: every edge contributed, and the two gradients decouple. The node
      gradient is this same g-SpMM re-run on the *transposed* CSR (``bwd_walk``); the edge
      gradient is ``_C.gspmm_backward_edge`` over the forward walk's items. For ``mul``
      and ``div`` the transposed run still needs the edge values, which live in
      forward-CSR order, so ``graph.backward_edge_map`` goes to the kernel and
      the read is redirected there.

    ``features_per_block``, ``tiles_y`` and ``pipeline_stages`` shape the min/max backward only;
    ``bwd_max_degree`` gates its heavy-row slicing. ``max_degree`` is accepted for compatibility and
    unused.
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
        max_degree=-1,
        bwd_max_degree=-1,
        fwd_walk=None,
        bwd_walk=None,
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

        walk_light, walk_heavy = (fwd_walk.light, fwd_walk.heavy) if fwd_walk is not None else (light, heavy)
        out, arg_eid = _C.gspmm_forward(
            edge_ptr,
            edge_idx,
            lhs_t,
            rhs_t,
            walk_light,
            walk_heavy,
            op,
            reduce,
            warps_per_block,
            fwd_walk.slice_size if fwd_walk is not None else GSPMM_SLICE_SIZE,
            **_walk_kwargs(fwd_walk),
        )

        ctx.save_for_backward(
            lhs_t,
            rhs_t,
            arg_eid,
            edge_ptr,
            edge_idx,
            light,
            heavy,
            bwd_edge_ptr,
            bwd_edge_idx,
            bwd_light,
            bwd_heavy,
            bwd_edge_map,
        )
        ctx.op = op
        ctx.reduce = reduce
        ctx.warps_per_block = warps_per_block
        ctx.features_per_block = features_per_block
        ctx.tiles_y = tiles_y
        ctx.pipeline_stages = pipeline_stages
        ctx.max_degree = max_degree
        ctx.bwd_max_degree = bwd_max_degree
        ctx.fwd_walk = fwd_walk
        ctx.bwd_walk = bwd_walk
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
            light,
            heavy,
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
            if bwd_edge_ptr is None or bwd_edge_map is None:
                raise RuntimeError(
                    f"gspmm(op='{op}', reduce='{ctx.reduce}') backward walks the transposed CSR. "
                    "Pass the backward adjacency and graph.backward_edge_map through GSpMMFunction.apply."
                )
            # Both gradients come back in the operand dtype: the walk over the
            # transpose writes every element exactly once (see
            # gspmm_backward_arg), so nothing is staged in fp32 here.
            g_lhs, g_rhs = _C.gspmm_backward_arg(
                grad_out,
                arg_eid,
                edge_idx,
                bwd_edge_ptr,
                bwd_edge_idx,
                bwd_edge_map,
                lhs_t,
                rhs_t,
                bwd_light,
                bwd_heavy,
                op,
                ctx.warps_per_block,
                ctx.features_per_block,
                ctx.tiles_y,
                ctx.bwd_max_degree,
                ctx.pipeline_stages,
            )
            if lhs_needs_grad:
                grad_lhs = g_lhs
            if rhs_needs_grad:
                grad_rhs = g_rhs
        else:
            # The two gradients of a sum reduction are independent -- the node
            # one is this same g-SpMM walked on the transposed CSR, the edge one
            # an edge-parallel scatter over the forward CSR -- and they were once
            # launched on two streams for that reason.  Measured on a T4 at
            # d=64, that overlap is worth nothing: exactly 1.00x on tolokers-2,
            # ogbn-arxiv and city-reviews (both kernels are already
            # bandwidth-bound, so there is no idle capacity to fill), 0.85x on
            # cora where the stream sync outweighs a 0.3 ms backward, and 0.73x
            # on twitch-views, where the [E, d] edge gradient allocated against
            # a side stream stops the caching allocator from reusing its block
            # and every iteration pays a fresh 3.5 GB cudaMalloc.  So both run
            # on the caller's stream.
            if rhs_needs_grad:
                # Already in the operand dtype: this kernel writes every slot
                # exactly once, so it has no float staging buffer to cast back.
                walk = ctx.fwd_walk
                grad_rhs = _C.gspmm_backward_edge(
                    edge_ptr,
                    edge_idx,
                    grad_out,
                    lhs_t,
                    rhs_t,
                    walk.light if walk is not None else light,
                    walk.heavy if walk is not None else heavy,
                    op,
                    ctx.warps_per_block,
                    walk.slice_size if walk is not None else GSPMM_SLICE_SIZE,
                    **_slice_kwargs(walk.slices if walk is not None else None),
                )

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

                walk = ctx.bwd_walk
                grad_lhs, _ = _C.gspmm_forward(
                    bwd_edge_ptr,
                    bwd_edge_idx,
                    grad_out,
                    rhs_bwd,
                    walk.light if walk is not None else bwd_light,
                    walk.heavy if walk is not None else bwd_heavy,
                    bwd_op,
                    "sum",
                    ctx.warps_per_block,
                    walk.slice_size if walk is not None else GSPMM_SLICE_SIZE,
                    edge_map=edge_map,
                    **_walk_kwargs(walk),
                )

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
            None,  # max_degree
            None,  # bwd_max_degree
            None,  # fwd_walk
            None,  # bwd_walk
        )
