from typing import Any, Literal, NamedTuple

import torch
from torch import nn

from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

from ..base import BaseAggr, BaseBackend, BaseConvolution
from ..registry import BackendRegistry
from .gatv2_aggr.utils import GATv2AggrKernel, gatv2_aggr
from .gsddmm_aggr.utils import (
    _GSDDMM_EDGE_PREFILLED_OPS,
    _GSDDMM_NAME_TO_MEMBER,
    _GSDDMM_PREFILLED_OPS,
    GSDDMMEdgeKernel,
    GSDDMMKernel,
)
from .gt_aggr.utils import GraphTransformerAggrKernel, graph_transformer_aggr
from .reduction_aggr.utils import ReductionAggrKernel, reduction_aggr
from .spmm_aggr.utils import spmm_aggr

doc = """
CUDA backend: wraps cuda-written kernels .
"""


class _CudaSimpleAggrConv(BaseConvolution):
    def __init__(
        self,
        aggr_type: Literal["min", "max"] = "min",
        *,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=bias, **kwargs)
        self.aggr_type = aggr_type
        self.kernel = ReductionAggrKernel(reduce=aggr_type, **kwargs)
        self.register_kernel(self.kernel)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.kernel(graph, x)


class _CUDAGATv2Conv(BaseConvolution):
    """CUDA-backed GATv2Conv wrapper."""

    def __init__(
        self,
        feature_dim: int,
        bias: bool = False,
        heads: int = 1,
        negative_slope: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_heads=heads, bias=bias, **kwargs)
        self.left_right_projection = nn.Linear(feature_dim, 2 * feature_dim * heads, bias=bias)
        self._outer_proj = torch.nn.Linear(feature_dim * heads, feature_dim, bias=bias)

        self.negative_slope = negative_slope
        self.heads = heads

        self.feature_dim = feature_dim
        self.head_dim = feature_dim

        self.attn_weights = nn.Parameter(torch.FloatTensor(size=(heads, feature_dim)))

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_weights, gain=gain)

        self.kernel = GATv2AggrKernel(**kwargs)
        self.register_kernel(self.kernel)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        *,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x_left, x_right = self.left_right_projection(x).split(self.heads * self.head_dim, -1)
        x_left = x_left.view(-1, self.heads, self.head_dim)
        x_right = x_right.view(-1, self.heads, self.head_dim)

        out = self.kernel(
            graph,
            x_left,
            x_neighbors=x_right,
            attention_weights=self.attn_weights.data,
            negative_slope=self.negative_slope,
        ).view(-1, self.heads * self.head_dim)

        out = self._outer_proj(out)
        return out


class _CudaGraphTransformerConv(BaseConvolution):
    """CUDA-based Fused graph transformer"""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 8,
        **kwargs,
    ):
        super().__init__(bias=False, dropout=0.0)

        self.feature_dim = feature_dim
        self.num_heads = heads
        self.qkv_proj = nn.Linear(self.feature_dim, 3 * self.feature_dim)

        self.head_dim = self.feature_dim // self.num_heads

        self.attn_scores_multiplier = torch.rsqrt(torch.tensor(self.head_dim)).item()

        self.kernel = GraphTransformerAggrKernel()
        self.register_kernel(self.kernel)

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.split(self.feature_dim, -1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        return self.kernel(
            graph,
            x,
            Q=q,
            K=k,
            V=v,
            scale=self.attn_scores_multiplier,
        ).view(-1, self.feature_dim)


class _CudaSpMMConv(BaseConvolution):
    """cuSPARSE SpMM convolution using AdjacencyForwardBackwardWithNodeBuckets.

    Supports float32, float16, bfloat16 features via mixed-precision cuSPARSE.
    """

    def __init__(
        self,
        norm_type: str = "none",
        cu_sparse_algorithm_id: int = -1,
        block_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(bias=False, dropout=0.0)
        self.norm_type = norm_type
        self.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        self.block_dim = block_dim

    def forward(
        self,
        x: torch.Tensor,
        graph: AdjacencyForwardBackwardWithNodeBuckets,
        **kwargs: Any,
    ) -> torch.Tensor:
        return spmm_aggr(
            x,
            graph.forward_indptr,
            graph.forward_indices,
            self.norm_type,
            self.cu_sparse_algorithm_id,
            self.block_dim,
        )


class _CudaSimpleAggr(BaseAggr):
    """Aggregation-only min/max via turbo_gnn."""

    def __init__(self, reduce: str = "min", **kwargs: Any) -> None:
        super().__init__(conv_type=f"{reduce}_aggr", **kwargs)
        self.reduce = reduce

    def forward(self, x: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        return reduction_aggr(graph, x, reduce=self.reduce)


class _CudaGATv2Aggr(BaseAggr):
    """Aggregation-only GATv2 attention via turbo_gnn (no linear projections)."""

    def __init__(self, heads: int, head_dim: int, negative_slope: float = 0.2, **kwargs: Any) -> None:
        super().__init__(conv_type="gat_v2", **kwargs)
        self.heads = heads
        self.head_dim = head_dim
        self.negative_slope = negative_slope
        self.attn_weights = nn.Parameter(torch.empty(heads, head_dim))
        nn.init.xavier_normal_(self.attn_weights, gain=nn.init.calculate_gain("relu"))

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        return gatv2_aggr(graph, x_left, x_right, self.attn_weights.data, self.negative_slope)


class _CudaGTAggr(BaseAggr):
    """Aggregation-only graph transformer attention via turbo_gnn (no QKV projection)."""

    def __init__(self, heads: int, head_dim: int, **kwargs: Any) -> None:
        super().__init__(conv_type="gt", **kwargs)
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        x_dummy = Q.view(Q.shape[0], -1)
        return graph_transformer_aggr(graph, x_dummy, Q, K, V, self.scale)


class _CudaSpMMAggr(BaseAggr):
    """Aggregation-only SpMM via turbo_gnn."""

    def __init__(self, norm_type: str = "none", **kwargs: Any) -> None:
        super().__init__(conv_type=f"spmm_{norm_type}", **kwargs)
        self.norm_type = norm_type

    def forward(self, x: torch.Tensor, graph, **kwargs: Any) -> torch.Tensor:
        return spmm_aggr(
            x,
            graph.forward_indptr.int(),
            graph.forward_indices.int(),
            self.norm_type,
            -1,
            256,
        )


# ---------------------------------------------------------------------------
# Raw turbo_gnn GSDDMM ops
# ---------------------------------------------------------------------------

#: Suffix marking the edge-parallel (one warp per edge) kernel variant.
_GSDDMM_EDGE_SUFFIX = "_edge"


class _GsddmmOpSpec(NamedTuple):
    """Kernel arguments and operand layout behind a DGL-style gsddmm op name."""

    op: str
    lhs_target: str
    rhs_target: str
    edge_variant: bool
    operand_kinds: tuple[str, ...]


def is_gsddmm_op(name: str) -> bool:
    """True if *name* is a turbo_gnn gsddmm op name (incl. ``_edge`` variants)."""
    return name in _GSDDMM_PREFILLED_OPS or name in _GSDDMM_EDGE_PREFILLED_OPS


def gsddmm_op_spec(name: str) -> _GsddmmOpSpec:
    """Parse a turbo_gnn gsddmm op name into the kernel's constructor arguments.

    Recognizes exactly the names :mod:`turbo_gnn.ops` generates — the binary
    ``{lhs}_{op}_{rhs}`` ops over mixed member pairs (``u_add_v``, ``e_dot_v``,
    ...), the ``copy_u`` / ``copy_v`` copies, and each of those with an
    ``_edge`` suffix selecting the edge-parallel kernel. Membership is checked
    against turbo_gnn's own op tables, so an op the library does not provide
    is rejected here rather than at launch.

    Args:
        name (str): Op name, e.g. ``"u_sub_v"``, ``"copy_u"``, ``"e_mul_v_edge"``.

    Returns:
        _GsddmmOpSpec: Op, lhs/rhs targets ("src"/"dst"/"edge"), whether the
            edge-parallel variant was requested, and the operand kinds
            ("u"/"v": node features [N, D]; "e": edge features [E, D]) in the
            order the wrapper takes them.

    Raises:
        KeyError: If *name* is not a turbo_gnn gsddmm op.
    """
    if name in _GSDDMM_EDGE_PREFILLED_OPS:
        base, edge_variant = name[: -len(_GSDDMM_EDGE_SUFFIX)], True
    elif name in _GSDDMM_PREFILLED_OPS:
        base, edge_variant = name, False
    else:
        raise KeyError(f"Unknown turbo_gnn gsddmm op: {name!r}")

    parts = base.split("_")
    if parts[0] == "copy":
        # copy_u / copy_v take a single operand; rhs is allocated by the op
        # wrapper to satisfy the binding's shape check but never read.
        return _GsddmmOpSpec("copy", _GSDDMM_NAME_TO_MEMBER[parts[1]], "edge", edge_variant, (parts[1],))

    lhs, op, rhs = parts
    return _GsddmmOpSpec(op, _GSDDMM_NAME_TO_MEMBER[lhs], _GSDDMM_NAME_TO_MEMBER[rhs], edge_variant, (lhs, rhs))


class _CudaGsddmmOp(BaseAggr):
    """Launch a turbo_gnn GSDDMM kernel directly (no projections).

    ``forward(*operands, graph)`` mirrors the DGL raw-op wrapper so one
    benchmarking path drives both backends. ``operand_kinds`` describes each
    operand ("u"/"v": node features [N, D], "e": edge features [E, D]) so
    callers can generate matching inputs; ``copy_*`` takes a single operand.

    Feature dim D must be one of 32, 64, 128, 256, and both operands must
    share a dtype (float32, float16 or bfloat16) — the kernels dispatch on it.

    Forward-only: the kernels have no backward pass, so the returned tensor
    carries no ``grad_fn``.
    """

    def __init__(self, op: str, **kwargs: Any) -> None:
        super().__init__(conv_type=op)
        spec = gsddmm_op_spec(op)
        self.op = op
        self.operand_kinds = spec.operand_kinds
        self.edge_variant = spec.edge_variant
        kernel_cls = GSDDMMEdgeKernel if spec.edge_variant else GSDDMMKernel
        self.kernel = kernel_cls(
            op=spec.op,
            lhs_target=spec.lhs_target,
            rhs_target=spec.rhs_target,
            **kwargs,
        )

    def forward(self, *args: Any) -> torch.Tensor:
        """Run the op; the last positional argument must be the graph."""
        *operands, graph = args
        rhs = operands[1] if len(operands) > 1 else None
        return self.kernel(graph, operands[0], rhs=rhs)


@BackendRegistry.register_backend("cuda")
class CUDABackend(BaseBackend):
    """Backend that instantiates CUDA-based convolutions."""

    def create_conv(
        self,
        conv_type: str,
        **kwargs: Any,
    ):
        """Factory for CUDA convolution layers.

        Args:
            conv_type (str): 'gat_v2', 'min_aggr', 'max_aggr', 'gt', 'sum_aggr', 'mean_aggr', 'gcn'.
            feature_dim (int): Input (and output) feature size.
            **kwargs (Any): Extra arguments for CUDA layers.

        Returns:
            BaseConvolution: An instance of the requested CUDA conv.
        """
        autotune = kwargs.pop("autotune", False)
        autotune_config = kwargs.pop("autotune_config", None)

        feature_dim = kwargs.pop("feature_dim")

        ct = conv_type.lower()
        match ct:
            case "gat_v2":
                heads = kwargs.pop("heads")
                conv = _CUDAGATv2Conv(feature_dim=feature_dim, heads=heads, **kwargs)
            case "min_aggr":
                conv = _CudaSimpleAggrConv(
                    aggr_type="min",
                    **kwargs,
                )
            case "max_aggr":
                return _CudaSimpleAggrConv(
                    aggr_type="max",
                    **kwargs,
                )
            case "gt":
                heads = kwargs.pop("heads")
                conv = _CudaGraphTransformerConv(feature_dim=feature_dim, heads=heads, **kwargs)
            case "sum_aggr":
                return _CudaSpMMConv(
                    norm_type="none",
                    cu_sparse_algorithm_id=kwargs.get("cu_sparse_algorithm_id", -1),
                    block_dim=kwargs.get("block_dim", 256),
                )
            case "mean_aggr":
                return _CudaSpMMConv(
                    norm_type="right",
                    cu_sparse_algorithm_id=kwargs.get("cu_sparse_algorithm_id", -1),
                    block_dim=kwargs.get("block_dim", 256),
                )
            case "gcn":
                return _CudaSpMMConv(
                    norm_type="both",
                    cu_sparse_algorithm_id=kwargs.get("cu_sparse_algorithm_id", -1),
                    block_dim=kwargs.get("block_dim", 256),
                )
            case _:
                raise KeyError(f"Unsupported conv_type for CUDA backend: {conv_type}")

        if autotune:
            conv.enable_autotune(config=autotune_config)

        return conv

    def create_aggr(self, conv_type: str, **kwargs: Any) -> BaseAggr:
        """Factory for CUDA aggregation-only callables.

        Besides the named aggregations (min_aggr, gcn, gat_v2, ...), any
        turbo_gnn gsddmm op name (u_add_v, e_dot_v, copy_u, and their
        ``_edge`` edge-parallel variants) is launched directly.
        """
        feature_dim = kwargs.pop("feature_dim", None)
        ct = conv_type.lower()
        match ct:
            case "gat_v2":
                heads = kwargs.pop("heads", 1)
                return _CudaGATv2Aggr(heads=heads, head_dim=feature_dim, **kwargs)
            case "gt":
                heads = kwargs.pop("heads", 8)
                head_dim = feature_dim // heads
                return _CudaGTAggr(heads=heads, head_dim=head_dim, **kwargs)
            case "min_aggr":
                return _CudaSimpleAggr(reduce="min")
            case "max_aggr":
                return _CudaSimpleAggr(reduce="max")
            case "sum_aggr":
                return _CudaSpMMAggr(norm_type="none")
            case "mean_aggr":
                return _CudaSpMMAggr(norm_type="right")
            case "gcn":
                return _CudaSpMMAggr(norm_type="both")
            case _:
                # Raw turbo_gnn gsddmm ops, launched directly (no wrappers).
                if is_gsddmm_op(ct):
                    # gsddmm is head-agnostic: it works on the flat [*, D] rows.
                    kwargs.pop("heads", None)
                    return _CudaGsddmmOp(ct, **kwargs)
                raise KeyError(f"Unsupported conv_type for CUDA aggr: {conv_type}")
