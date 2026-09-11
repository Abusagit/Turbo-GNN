"""Correctness tests for the GSDDMM kernels (turbo_gnn.gsddmm and DGL-style aliases).

The kernel writes one output row per edge in CSR edge order (edges sorted by
destination node). The pure-PyTorch reference below reconstructs that exact
order from the graph's own CSR, so the comparison is order-exact.
"""

import pytest
import torch

import turbo_gnn
from turbo_gnn import gsddmm
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")

OPS = ["add", "sub", "mul", "div", "dot"]
MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("src", "edge"), ("edge", "src"), ("dst", "edge"), ("edge", "dst")]
COPY_TARGETS = ["src", "dst"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FEATURE_DIMS = [32, 64, 128, 256]


def _make_graph(num_nodes=200, num_edges=1500, quantile=0.99, index_dtype=torch.int32, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=DEVICE, generator=gen, dtype=torch.long)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=quantile, index_dtype=index_dtype
    ).to(DEVICE)
    return graph


def _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim, dtype, seed=1):
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


def _reference(graph, lhs, rhs, op, lhs_target, rhs_target):
    indptr = graph.forward_indptr.long()
    src = graph.forward_indices.long()
    num_nodes = indptr.numel() - 1
    dst = torch.repeat_interleave(torch.arange(num_nodes, device=indptr.device), indptr[1:] - indptr[:-1])

    def select(t, target):
        if target == "src":
            return t[src]
        if target == "dst":
            return t[dst]
        return t  # edge rows are already in CSR edge order

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
        # Emulate the kernel's exact semantics (verified bitwise vs the kernel):
        # products are formed and rounded in cuda_t, accumulated in fp32, and
        # the result is written in cuda_t. For half dtypes the cuda_t rounding
        # swallows the fp32 sum-ordering difference, so this matches the kernel
        # bit-for-bit; for fp32 only the tree-vs-pairwise ordering differs,
        # which the looser dot tolerances in _tol absorb.
        return (left * right).sum(-1, dtype=torch.float32).to(lhs.dtype)
    raise AssertionError(f"unknown op {op}")


def _tol(dtype, op=None):
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
def test_gsddmm_ops_members_fp32(op, lhs_target, rhs_target):
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim=64, dtype=torch.float32)

    out = gsddmm(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)

    assert out.shape == ref.shape
    assert torch.allclose(out.double(), ref.double(), **_tol(torch.float32, op)), (
        f"max err {(out.double() - ref.double()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("lhs_target", COPY_TARGETS)
def test_gsddmm_copy_fp32(lhs_target):
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)

    # rhs omitted entirely -- copy never reads it.
    out = gsddmm(graph, lhs, None, op="copy", lhs_target=lhs_target)
    ref = _reference(graph, lhs, None, "copy", lhs_target, "edge")

    assert torch.allclose(out, ref, **_tol(torch.float32))


# =============================================================================
# Dtype / feature-dim sweeps
# =============================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", FEATURE_DIMS)
@pytest.mark.parametrize("op", ["mul", "dot", "copy"])
def test_gsddmm_dtypes_dims(dtype, dim, op):
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs_target, rhs_target = ("src", "dst") if op != "copy" else ("dst", "edge")
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim, dtype)

    out = gsddmm(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)

    assert out.dtype == dtype
    assert torch.allclose(out.double(), ref.double(), **_tol(dtype, op)), (
        f"max err {(out.double() - ref.double()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gsddmm_index_dtypes(index_dtype):
    graph = _make_graph(index_dtype=index_dtype)
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)

    out = gsddmm(graph, lhs, rhs, op="sub", lhs_target="src", rhs_target="dst")
    ref = _reference(graph, lhs, rhs, "sub", "src", "dst")
    assert torch.allclose(out, ref, **_tol(torch.float32))


# =============================================================================
# Kernel-config equivalence (warps / pipeline stages / bucketing)
# =============================================================================


@pytest.mark.parametrize("pipeline_stages", [0, 1, 2, 3])
def test_gsddmm_pipeline_stages_match(pipeline_stages):
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs = torch.randn(num_nodes, 128, device=DEVICE)
    rhs = torch.randn(num_edges, 128, device=DEVICE)

    out = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="edge", pipeline_stages=pipeline_stages)
    ref = _reference(graph, lhs, rhs, "mul", "src", "edge")
    assert torch.allclose(out, ref, **_tol(torch.float32))


@pytest.mark.parametrize("light_warps", [4])
@pytest.mark.parametrize("heavy_warps", [32])
def test_gsddmm_warps_match(light_warps, heavy_warps):
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)

    out = gsddmm(
        graph,
        lhs,
        rhs,
        op="add",
        lhs_target="dst",
        rhs_target="src",
        light_warps_per_block=light_warps,
        heavy_warps_per_block=heavy_warps,
    )
    ref = _reference(graph, lhs, rhs, "add", "dst", "src")
    assert torch.allclose(out, ref, **_tol(torch.float32))


@pytest.mark.parametrize("overlap_buckets", [False, True])
@pytest.mark.parametrize("pipeline_stages", [0, 2])
@pytest.mark.parametrize("heavy_edges_per_block", [0, 1, 3, 8, 4096])
@pytest.mark.parametrize(
    "op,lhs_target,rhs_target", [("mul", "src", "dst"), ("dot", "edge", "src"), ("copy", "dst", "edge")]
)
def test_gsddmm_heavy_chunking_match(
    op, lhs_target, rhs_target, heavy_edges_per_block, pipeline_stages, overlap_buckets
):
    # quantile 0.5 puts half the nodes (degree ~7-20) in the heavy bucket, so
    # chunks of 1/3/8 edges split nodes into several blocks with a ragged tail,
    # and 4096 degenerates to one block per node.
    graph = _make_graph(quantile=0.5)
    assert graph.heavy_nodes.numel() > 0
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim=64, dtype=torch.float32)

    out = gsddmm(
        graph,
        lhs,
        rhs,
        op=op,
        lhs_target=lhs_target,
        rhs_target=rhs_target,
        pipeline_stages=pipeline_stages,
        heavy_edges_per_block=heavy_edges_per_block,
        overlap_buckets=overlap_buckets,
    )
    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)
    assert torch.allclose(out.double(), ref.double(), **_tol(torch.float32, op))


def test_gsddmm_overlap_buckets_stream_ordering():
    # The side-stream light launch must be joined back before anything the
    # caller enqueues next: overwrite the output right after the call and check
    # nothing from the light kernel lands afterwards.
    graph = _make_graph(quantile=0.5)
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)
    for _ in range(20):
        out = gsddmm(graph, lhs, rhs, op="add", lhs_target="src", rhs_target="dst", overlap_buckets=True)
        out.zero_()
        torch.cuda.synchronize()
        assert not out.any()


def test_gsddmm_heavy_blocks_descriptors():
    from turbo_gnn._kernels import _graph_heavy_blocks

    graph = _make_graph(quantile=0.5)
    indptr = graph.forward_indptr.long()
    heavy = graph.heavy_nodes.long()
    deg = indptr[heavy + 1] - indptr[heavy]
    nodes, parts = _graph_heavy_blocks(graph, 4)
    assert nodes.dtype == graph.heavy_nodes.dtype and parts.dtype == graph.heavy_nodes.dtype
    assert nodes.numel() == int(torch.clamp((deg + 3) // 4, min=1).sum())
    # every heavy node appears ceil(deg/4) times with chunk ids 0..k-1, and
    # the chunks cover the node's edge list exactly once
    for n in heavy[:10].tolist():
        sel = parts[nodes.long() == n].long().sort().values
        assert torch.equal(sel, torch.arange(sel.numel(), device=sel.device))
    # cached: same object on repeat, new object for a new chunk size
    assert _graph_heavy_blocks(graph, 4) is _graph_heavy_blocks(graph, 4)
    assert _graph_heavy_blocks(graph, 8) is not _graph_heavy_blocks(graph, 4)


@pytest.mark.parametrize("quantile", [-1, 0.5, 0.99])
def test_gsddmm_bucketing_match(quantile):
    graph = _make_graph(quantile=quantile)
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)

    out = gsddmm(graph, lhs, rhs, op="dot", lhs_target="src", rhs_target="dst")
    ref = _reference(graph, lhs, rhs, "dot", "src", "dst")
    assert torch.allclose(out.double(), ref.double(), **_tol(torch.float32, "dot"))


# =============================================================================
# DGL-style prefilled aliases
# =============================================================================


@pytest.mark.parametrize(
    "alias_name,op,lhs_target,rhs_target",
    [
        ("u_add_v", "add", "src", "dst"),
        ("v_sub_u", "sub", "dst", "src"),
        ("u_mul_e", "mul", "src", "edge"),
        ("e_div_v", "div", "edge", "dst"),
        ("u_dot_v", "dot", "src", "dst"),
        ("e_dot_u", "dot", "edge", "src"),
    ],
)
def test_dgl_style_alias_matches_generic(alias_name, op, lhs_target, rhs_target):
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    lhs, rhs = _make_operands(lhs_target, rhs_target, op, num_nodes, num_edges, dim=64, dtype=torch.float32)

    alias = getattr(turbo_gnn, alias_name)
    out_alias = alias(graph, lhs, rhs)
    out_generic = gsddmm(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target)
    assert torch.equal(out_alias, out_generic)

    ref = _reference(graph, lhs, rhs, op, lhs_target, rhs_target)
    assert torch.allclose(out_alias.double(), ref.double(), **_tol(torch.float32, op))


def test_copy_aliases():
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)

    out_u = turbo_gnn.copy_u(graph, x)
    out_v = turbo_gnn.copy_v(graph, x)

    assert torch.equal(out_u, gsddmm(graph, x, None, op="copy", lhs_target="src"))
    assert torch.equal(out_v, gsddmm(graph, x, None, op="copy", lhs_target="dst"))
    assert torch.allclose(out_u, _reference(graph, x, None, "copy", "src", "edge"), **_tol(torch.float32))
    assert torch.allclose(out_v, _reference(graph, x, None, "copy", "dst", "edge"), **_tol(torch.float32))


def test_all_prefilled_ops_exported():
    # 6 ordered member pairs x 5 binary ops + 2 copy ops.
    assert len(turbo_gnn.ops._GSDDMM_PREFILLED_OPS) == 6 * 5 + 2
    for name in ("u_sub_v", "v_dot_u", "e_add_u", "u_mul_e", "copy_u", "copy_v"):
        assert callable(getattr(turbo_gnn, name))


# =============================================================================
# Autotune smoke test
# =============================================================================


def test_gsddmm_autotune_matches_reference():
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    lhs = torch.randn(num_nodes, 64, device=DEVICE)
    rhs = torch.randn(num_nodes, 64, device=DEVICE)

    cfg = turbo_gnn.AutotuneConfig(warmup=1, iters=2)
    out = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", autotune=True, autotune_config=cfg)
    ref = _reference(graph, lhs, rhs, "mul", "src", "dst")
    assert torch.allclose(out, ref, **_tol(torch.float32))


# =============================================================================
# Edge cases and error handling
# =============================================================================


def test_gsddmm_empty_graph():
    edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(edge_index, num_nodes=4, index_dtype=torch.int32).to(
        DEVICE
    )
    x = torch.randn(4, 64, device=DEVICE)

    out = gsddmm(graph, x, x, op="add", lhs_target="src", rhs_target="dst")
    assert out.shape == (0, 64)


def test_gsddmm_isolated_nodes_produce_no_rows():
    # Node 2 is isolated (no incoming edges); only edges 0->1, 1->0 exist.
    edge_index = torch.tensor([[0, 1], [1, 0]], device=DEVICE)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(edge_index, num_nodes=3, index_dtype=torch.int32).to(
        DEVICE
    )
    x = torch.arange(1, 5, device=DEVICE).float().unsqueeze(1).repeat(1, 32)  # rows 0..3 -> values 1..4... use 3 nodes
    x = torch.arange(1, 4, device=DEVICE).float().unsqueeze(1).repeat(1, 32)

    out = gsddmm(graph, x, x, op="sub", lhs_target="src", rhs_target="dst")
    # CSR order: dst 0 has edge from 1; dst 1 has edge from 0.
    expected = torch.stack([x[1] - x[0], x[0] - x[1]])
    assert torch.allclose(out, expected, **_tol(torch.float32))


# These two operand combinations have no kernel. GsddmmSpec now rejects them in
# Python (ValueError) before anything is allocated or launched; before the launch
# plan existed they reached the CUDA dispatch and raised RuntimeError from a
# TORCH_CHECK. Either rejection satisfies the intent of these tests.
def test_gsddmm_same_member_rejected():
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError)):
        gsddmm(graph, x, x, op="add", lhs_target="src", rhs_target="src")


def test_gsddmm_copy_edge_target_rejected():
    graph = _make_graph()
    num_edges = graph.forward_indices.numel()
    e = torch.randn(num_edges, 64, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError)):
        gsddmm(graph, e, None, op="copy", lhs_target="edge")


def test_gsddmm_unsupported_dim_rejected():
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 16, device=DEVICE)  # D must be in {32, 64, 128, 256}
    with pytest.raises(RuntimeError):
        gsddmm(graph, x, x, op="add", lhs_target="src", rhs_target="dst")


def test_gsddmm_rhs_required_for_binary_ops():
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)
    with pytest.raises(ValueError):
        gsddmm(graph, x, None, op="mul", lhs_target="src", rhs_target="dst")
