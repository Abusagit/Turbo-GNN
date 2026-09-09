"""Correctness tests for the edge-parallel GSDDMM kernels (turbo_gnn.gsddmm_edge).

The edge kernel parallelizes one warp per edge over an explicit [E, 2] edge
list of (src, dst) pairs, built once from the graph's CSR and cached. The
grouping depends on the operands: edges are grouped by destination (forward
CSR, CSR edge order) when a "dst" operand is involved, and by source
(backward CSR, CSC edge order) otherwise. The output rows follow the grouping,
so the reference below reconstructs the matching order from the graph's own
CSR/CSC and the comparison is order-exact.
"""

import pytest
import torch

import turbo_gnn
from turbo_gnn import gsddmm, gsddmm_edge
from turbo_gnn._kernels import _graph_edge_list
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")

OPS = ["add", "sub", "mul", "div", "dot"]
MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("src", "edge"), ("edge", "src"), ("dst", "edge"), ("edge", "dst")]
# Pairs involving a dst operand: the edge kernel keeps CSR (dst-grouped) order
# for these, so its output is directly comparable to the CSR kernel's.
DST_MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("dst", "edge"), ("edge", "dst")]
# Pairs with no dst operand: the edge kernel switches to CSC (src-grouped) order.
SRC_MEMBER_PAIRS = [("src", "edge"), ("edge", "src")]
COPY_TARGETS = ["src", "dst"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FEATURE_DIMS = [32, 64, 128, 256]


def _make_graph(num_nodes: int = 200, num_edges: int = 1500, quantile: float = 0.99, index_dtype: torch.dtype = torch.int32, seed: int = 0) -> AdjacencyForwardBackwardWithNodeBuckets:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=DEVICE, generator=gen, dtype=torch.long)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=quantile, index_dtype=index_dtype
    ).to(DEVICE)

    return graph


def _make_unique_graph(num_nodes: int = 200, num_edges: int = 1500, index_dtype: torch.dtype = torch.int32, seed: int = 0) -> AdjacencyForwardBackwardWithNodeBuckets:
    """Graph with no duplicate (src, dst) pairs, so CSR and CSC edge orders are
    related by a well-defined permutation."""

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    perm = torch.randperm(num_nodes * num_nodes, device=DEVICE, generator=gen)[:num_edges]
    edge_index = torch.stack([perm // num_nodes, perm % num_nodes])
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, index_dtype=index_dtype, is_directed=True
    ).to(DEVICE)

    return graph


def _make_operands(lhs_target: str, rhs_target: str, op: str, num_nodes: int, num_edges: int, dim: int, dtype: torch.dtype, seed=1) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rows(target):
        n = num_edges if target == "edge" else num_nodes
        return torch.randn(n, dim, device=DEVICE, generator=gen, dtype=torch.float32).to(dtype)

    lhs = rows(lhs_target)
    rhs = rows(rhs_target)
    if op == "div":
        # Keep denominators away from zero so fp16/bf16 quotients stay sane.
        rhs = rhs.sign() * (rhs.abs() + 0.5)

    return lhs, rhs


def _grouped_src_dst(graph: AdjacencyForwardBackwardWithNodeBuckets, by_src: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(src, dst) index arrays in the kernel's edge order for the grouping."""
    if by_src:
        indptr = graph.backward_indptr.long()
        dst = graph.backward_indices.long()
    else:
        indptr = graph.forward_indptr.long()
        dst = None
    num_nodes = indptr.numel() - 1
    rows = torch.repeat_interleave(torch.arange(num_nodes, device=indptr.device), indptr[1:] - indptr[:-1])
    if by_src:
        return rows, dst  # CSC rows are sources

    return graph.forward_indices.long(), rows  # CSR rows are destinations


def _reference(graph: AdjacencyForwardBackwardWithNodeBuckets, lhs: torch.Tensor, rhs: torch.Tensor, op: str, lhs_target: str, rhs_target: str) -> torch.Tensor:
    if op == "copy":
        rhs_target = "edge"
    by_src = "dst" not in (lhs_target, rhs_target)
    src, dst = _grouped_src_dst(graph, by_src)

    def select(t: torch.Tensor, target: str) -> torch.Tensor:
        if target == "src":
            return t[src]
        if target == "dst":
            return t[dst]
        return t  # edge rows are already in the grouping's edge order

    left = select(lhs, lhs_target)
    if op == "copy":
        return left
    right = select(rhs, rhs_target)
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / right
    if op == "dot":
        # Same semantics as the CSR-kernel reference: products in cuda_t,
        # accumulation in fp32, result written in cuda_t.
        return (left * right).sum(-1, dtype=torch.float32).to(lhs.dtype)
    raise AssertionError(f"unknown op {op}")


def _tol(dtype: torch.dtype, op: str | None = None) -> dict[str, float]:
    if dtype == torch.float32:
        if op == "dot":
            return {"rtol": 1e-4, "atol": 1e-4}
        return {"rtol": 1e-5, "atol": 1e-6}
    return {"rtol": 1e-2, "atol": 1e-2}


# =============================================================================
# Full op x member-pair sweep (fp32)
# =============================================================================


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("lhs_target,rhs_target", MEMBER_PAIRS)
def test_gsddmm_edge_ops_members_fp32(op: str, lhs_target: str, rhs_target: str) -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim=64, dtype=torch.float32)

    out = gsddmm_edge(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)

    assert out.shape == ref.shape
    assert torch.allclose(out.double(), ref.double(), **_tol(torch.float32, op)), (
        f"max err {(out.double() - ref.double()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("lhs_target", COPY_TARGETS)
def test_gsddmm_edge_copy_fp32(lhs_target: str) -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)

    # rhs omitted entirely -- copy never reads it.
    out = gsddmm_edge(graph, lhs, None, op="copy", lhs_target=lhs_target)
    ref = _reference(graph, lhs, None, "copy", lhs_target, "edge")

    assert torch.allclose(out, ref, **_tol(torch.float32))


# =============================================================================
# Dtype / feature-dim sweeps
# =============================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", FEATURE_DIMS)
@pytest.mark.parametrize("op", ["mul", "dot", "copy"])
def test_gsddmm_edge_dtypes_dims(dtype: torch.dtype, dim: int, op: str) -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs_target, rhs_target = ("src", "dst") if op != "copy" else ("dst", "edge")
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim, dtype)

    out = gsddmm_edge(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)

    assert out.dtype == dtype
    assert torch.allclose(out.double(), ref.double(), **_tol(dtype, op)), (
        f"max err {(out.double() - ref.double()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gsddmm_edge_index_dtypes(index_dtype: torch.dtype) -> None:
    graph = _make_graph(index_dtype=index_dtype)
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)

    out = gsddmm_edge(graph, lhs, rhs, op="sub", lhs_target="src", rhs_target="dst")
    ref = _reference(graph, lhs, rhs, "sub", "src", "dst")
    assert torch.allclose(out, ref, **_tol(torch.float32))


# =============================================================================
# Grouping direction: dst operand -> CSR order, no dst operand -> CSC order
# =============================================================================


def test_gsddmm_edge_grouping_direction() -> None:
    graph = _make_unique_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)

    src_csr, dst_csr = _grouped_src_dst(graph, by_src=False)
    src_csc, dst_csc = _grouped_src_dst(graph, by_src=True)

    # dst operand -> CSR (dst-grouped) edge order
    out_dst = gsddmm_edge(graph, x, None, op="copy", lhs_target="dst")
    assert torch.allclose(out_dst, x[dst_csr], **_tol(torch.float32))

    # no dst operand -> CSC (src-grouped) edge order
    out_src = gsddmm_edge(graph, x, None, op="copy", lhs_target="src")
    assert torch.allclose(out_src, x[src_csc], **_tol(torch.float32))

    # The two groupings are genuine permutations of each other on a directed graph.
    assert not torch.equal(src_csr, src_csc)


# =============================================================================
# Equivalence with the CSR-based gsddmm kernel
# =============================================================================


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("lhs_target,rhs_target", DST_MEMBER_PAIRS)
def test_gsddmm_edge_matches_csr_kernel_dst_pairs(op: str, lhs_target: str, rhs_target: str) -> None:
    """dst-involving pairs keep CSR edge order: direct comparison with the CSR kernel."""

    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim=128, dtype=torch.float32)

    out_edge = gsddmm_edge(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    out_csr = gsddmm(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)

    assert out_edge.shape == out_csr.shape
    assert torch.allclose(out_edge, out_csr, **_tol(torch.float32, op)), (
        f"max err {(out_edge - out_csr).abs().max().item():.3e}"
    )


def _csr_csc_orders(graph: AdjacencyForwardBackwardWithNodeBuckets) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutations sorting the CSR and CSC edge orders by (src, dst) key.

    Requires a graph without duplicate edges (see _make_unique_graph).
    """

    num_nodes = graph.forward_indptr.numel() - 1
    src_csr, dst_csr = _grouped_src_dst(graph, by_src=False)
    src_csc, dst_csc = _grouped_src_dst(graph, by_src=True)
    key_csr = src_csr * num_nodes + dst_csr
    key_csc = src_csc * num_nodes + dst_csc
    order_csr = key_csr.argsort()
    order_csc = key_csc.argsort()
    # Unique edges: sorted keys must coincide.
    assert torch.equal(key_csr[order_csr], key_csc[order_csc])

    return order_csr, order_csc


def test_gsddmm_edge_src_grouped_permutation_of_csr() -> None:
    """copy_u has no dst and no edge operand: the CSC-grouped edge-kernel output
    must be a permutation of the CSR-kernel output."""

    graph = _make_unique_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)

    out_csr = gsddmm(graph, x, None, op="copy", lhs_target="src")  # CSR order
    out_edge = gsddmm_edge(graph, x, None, op="copy", lhs_target="src")  # CSC order

    order_csr, order_csc = _csr_csc_orders(graph)
    assert torch.equal(out_csr[order_csr], out_edge[order_csc])


@pytest.mark.parametrize("lhs_target,rhs_target", SRC_MEMBER_PAIRS)
def test_gsddmm_edge_src_grouped_matches_csr_with_edge_operand(lhs_target: str, rhs_target: str) -> None:
    """
    src-grouped pairs carry an edge-indexed operand; keying that operand by
    edge identity makes the two kernels' outputs comparable up to permutation.
    """

    graph = _make_unique_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    order_csr, order_csc = _csr_csc_orders(graph)

    x = torch.randn(num_nodes, 64, device=DEVICE)
    e_csr = torch.randn(num_edges, 64, device=DEVICE)
    # The edge operand row e means "e-th edge in the grouping's order"; rekey
    # it so both groupings see the same row for the same edge.
    e_csc = e_csr[order_csr][order_csc.argsort()]

    lhs_csr, rhs_csr = (e_csr, x) if lhs_target == "edge" else (x, e_csr)
    lhs_csc, rhs_csc = (e_csc, x) if lhs_target == "edge" else (x, e_csc)

    out_csr = gsddmm(graph, lhs_csr, rhs_csr, op="mul", lhs_target=lhs_target, rhs_target=rhs_target)
    out_edge = gsddmm_edge(graph, lhs_csc, rhs_csc, op="mul", lhs_target=lhs_target, rhs_target=rhs_target)

    assert torch.allclose(out_csr[order_csr], out_edge[order_csc], **_tol(torch.float32))


# =============================================================================
# Edge-list caching
# =============================================================================


def test_edge_list_cached_per_graph() -> None:
    graph = _make_graph()
    for by_src in (False, True):
        first = _graph_edge_list(graph, by_src=by_src)
        second = _graph_edge_list(graph, by_src=by_src)
        assert first is second  # same tensor object, not recomputed


def test_edge_list_values() -> None:
    graph = _make_graph()

    # dst-grouped: from the forward CSR
    edge_list = _graph_edge_list(graph, by_src=False)
    assert edge_list.dtype == torch.uint64
    assert edge_list.dim() == 2 and edge_list.size(1) == 2
    assert edge_list.is_contiguous()
    src, dst = _grouped_src_dst(graph, by_src=False)
    expected = torch.stack([src, dst], dim=1)
    # First element of each pair is the source, second is the destination.
    assert torch.equal(edge_list.view(torch.int64), expected)

    # src-grouped: from the backward CSR
    edge_list = _graph_edge_list(graph, by_src=True)
    assert edge_list.dtype == torch.uint64
    assert edge_list.is_contiguous()
    src, dst = _grouped_src_dst(graph, by_src=True)
    expected = torch.stack([src, dst], dim=1)
    assert torch.equal(edge_list.view(torch.int64), expected)


def test_edge_list_cache_repartitioned_graph() -> None:
    # repartition() shares the CSR tensors, so the stamp still matches and the
    # cached edge lists are reused (no rebuild, same values).
    graph = _make_graph()
    repartitioned = graph.repartition(forward_huge_degree_threshold_quantile=0.9)
    x = torch.randn(graph.forward_indptr.numel() - 1, 64, device=DEVICE)
    for lhs_target in ("src", "dst"):
        out_a = gsddmm_edge(graph, x, None, op="copy", lhs_target=lhs_target)
        out_b = gsddmm_edge(repartitioned, x, None, op="copy", lhs_target=lhs_target)
        assert torch.equal(out_a, out_b)


def test_edge_list_undirected_shares_single_list() -> None:
    # Undirected graphs alias backward CSR to forward CSR: both groupings must
    # return the same cached list.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=DEVICE)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes=3, index_dtype=torch.int32, is_directed=False
    ).to(DEVICE)
    assert _graph_edge_list(graph, by_src=True) is _graph_edge_list(graph, by_src=False)


# =============================================================================
# DGL-style prefilled aliases (*_edge variants)
# =============================================================================


@pytest.mark.parametrize(
    "alias_name,op,lhs_target,rhs_target",
    [
        ("u_add_v_edge", "add", "src", "dst"),
        ("v_sub_u_edge", "sub", "dst", "src"),
        ("u_mul_e_edge", "mul", "src", "edge"),
        ("e_div_v_edge", "div", "edge", "dst"),
        ("u_dot_v_edge", "dot", "src", "dst"),
        ("e_dot_u_edge", "dot", "edge", "src"),
    ],
)
def test_dgl_style_edge_alias_matches_generic(alias_name: str, op: str, lhs_target: str, rhs_target: str) -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim=64, dtype=torch.float32)

    alias = getattr(turbo_gnn, alias_name)
    out_alias = alias(graph, lhs, rhs)
    out_generic = gsddmm_edge(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    assert torch.equal(out_alias, out_generic)

    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)
    assert torch.allclose(out_alias.double(), ref.double(), **_tol(torch.float32, op))


def test_copy_edge_aliases() -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)

    out_u = turbo_gnn.copy_u_edge(graph, x)
    out_v = turbo_gnn.copy_v_edge(graph, x)

    assert torch.equal(out_u, gsddmm_edge(graph, x, None, op="copy", lhs_target="src"))
    assert torch.equal(out_v, gsddmm_edge(graph, x, None, op="copy", lhs_target="dst"))
    assert torch.allclose(out_u, _reference(graph, x, None, "copy", "src", "edge"), **_tol(torch.float32))
    assert torch.allclose(out_v, _reference(graph, x, None, "copy", "dst", "edge"), **_tol(torch.float32))


def test_all_prefilled_edge_ops_exported() -> None:
    # 6 ordered member pairs x 5 binary ops + 2 copy ops, mirrored for _edge.
    assert len(turbo_gnn.ops._GSDDMM_EDGE_PREFILLED_OPS) == 6 * 5 + 2
    for name in ("u_sub_v_edge", "v_dot_u_edge", "e_add_u_edge", "u_mul_e_edge", "copy_u_edge", "copy_v_edge"):
        assert callable(getattr(turbo_gnn, name))


# =============================================================================
# Autotune smoke test
# =============================================================================


def test_gsddmm_edge_autotune_matches_reference() -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)

    cfg = turbo_gnn.AutotuneConfig(warmup=1, iters=2)
    out = gsddmm_edge(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", autotune=True, autotune_config=cfg)
    ref = _reference(graph, lhs, rhs, "mul", "src", "dst")
    assert torch.allclose(out, ref, **_tol(torch.float32))


# =============================================================================
# Edge cases and error handling
# =============================================================================


def test_gsddmm_edge_empty_graph() -> None:
    edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(edge_index, num_nodes=4, index_dtype=torch.int32).to(
        DEVICE
    )
    x = torch.randn(4, 64, device=DEVICE)

    out = gsddmm_edge(graph, x, x, op="add", lhs_target="src", rhs_target="dst")
    assert out.shape == (0, 64)


def test_gsddmm_edge_isolated_nodes_produce_no_rows() -> None:
    # Node 2 is isolated (no incoming edges); only edges 0->1, 1->0 exist.
    edge_index = torch.tensor([[0, 1], [1, 0]], device=DEVICE)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(edge_index, num_nodes=3, index_dtype=torch.int32).to(
        DEVICE
    )
    x = torch.arange(1, 4, device=DEVICE).float().unsqueeze(1).repeat(1, 32)

    out = gsddmm_edge(graph, x, x, op="sub", lhs_target="src", rhs_target="dst")
    # CSR order: dst 0 has edge from 1; dst 1 has edge from 0.
    expected = torch.stack([x[1] - x[0], x[0] - x[1]])
    assert torch.allclose(out, expected, **_tol(torch.float32))


def test_gsddmm_edge_same_member_rejected() -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)
    with pytest.raises(RuntimeError):
        gsddmm_edge(graph, x, x, op="add", lhs_target="src", rhs_target="src")


def test_gsddmm_edge_copy_edge_target_rejected() -> None:
    graph = _make_graph()
    num_edges = graph.forward_indices.numel()
    e = torch.randn(num_edges, 64, device=DEVICE)
    with pytest.raises(RuntimeError):
        gsddmm_edge(graph, e, None, op="copy", lhs_target="edge")


def test_gsddmm_edge_unsupported_dim_rejected() -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 16, device=DEVICE)  # D must be in {32, 64, 128, 256}
    with pytest.raises(RuntimeError):
        gsddmm_edge(graph, x, x, op="add", lhs_target="src", rhs_target="dst")


def test_gsddmm_edge_rhs_required_for_binary_ops() -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)
    with pytest.raises(ValueError):
        gsddmm_edge(graph, x, None, op="mul", lhs_target="src", rhs_target="dst")
