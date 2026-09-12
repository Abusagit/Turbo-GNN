"""Correctness tests for the CUDA backend's raw GSDDMM op harness.

``scripts/benchmark.py --backend cuda --aggr`` drives every sweep row through
``_CudaGsddmmOp`` (``src/backends/cuda_backend/conv.py``) -- one instance per
op name, launched directly with no projections. These tests pin the harness
contracts the sweep depends on; the kernels themselves are covered by
``test_gsddmm.py`` / ``test_gsddmm_edge.py`` / ``test_gsddmm_backward.py``.

- Name -> kernel-family mapping: a bare name pins the CSR node-block family,
  ``_edge`` the edge-parallel family *in both directions*, ``_auto`` the forward
  dispatcher with the default node-parallel backward.
- ``forward`` routes through ``GSDDMMKernel.forward``, so the output carries a
  ``grad_fn`` (what ``--mode backward`` needs) and is in canonical forward-CSR
  edge order -- a ``_edge`` row computes the same rows as the bare one.
- Gradients through the conv match the fp32 autograd reference, so the sweep's
  backward rows time kernels computing the right math.
- The legacy traversal-order ``GSDDMMEdgeKernel`` refuses differentiability:
  its CSC-ordered output cannot serve as ``d_out`` for the backward kernels.

The reference (``index_select`` + ``.backward()`` in fp32) mirrors
``test_gsddmm_backward.py``: a low-precision kernel is compared against
something more accurate than itself, not against an equally lossy reference.
"""

import pytest
import torch

from src.backends.cuda_backend.conv import _CudaGsddmmOp, gsddmm_op_spec, is_gsddmm_op
from turbo_gnn import GSDDMMEdgeKernel, gsddmm
from turbo_gnn._gsddmm import _edge_endpoints
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")

#: The two backward variants accumulate differently (fp32 registers vs fp32
#: atomics), so the low-precision dtypes need room for storage rounding on top.
_TOL = {
    torch.float32: {"rtol": 2e-4, "atol": 2e-4},
    torch.float16: {"rtol": 2e-2, "atol": 2e-2},
    torch.bfloat16: {"rtol": 6e-2, "atol": 6e-2},
}


def _make_graph(num_nodes=256, num_edges=2000, seed=0, is_directed=True):
    """Random directed multigraph: duplicates and isolated nodes are expected."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=DEVICE, generator=gen, dtype=torch.long)
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=0.99, index_dtype=torch.int32, is_directed=is_directed
    ).to(DEVICE)


def _rows(graph, target: str) -> int:
    return graph.forward_indices.numel() if target == "edge" else graph.forward_indptr.numel() - 1


def _operands(graph, spec, feat_dim, dtype, seed=0):
    """Operands in the order _CudaGsddmmOp.forward takes them, plus a d_out."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_edges = graph.forward_indices.numel()
    operands = []
    for kind in spec.operand_kinds:
        n = num_edges if kind == "e" else graph.forward_indptr.numel() - 1
        operands.append(torch.randn(n, feat_dim, device=DEVICE, dtype=dtype, generator=gen))
    d_out_shape = (num_edges,) if spec.op == "dot" else (num_edges, feat_dim)
    d_out = torch.randn(d_out_shape, device=DEVICE, dtype=dtype, generator=gen)
    return *operands, d_out


def _reference_grads(graph, operands, d_out, spec):
    """Autograd gradients of the same math, computed in fp32 from upcast inputs."""
    src, dst = _edge_endpoints(graph, by_src=False)  # canonical (forward-CSR) edge order

    leaves = [o.detach().float().requires_grad_(True) for o in operands]

    def pick(t, target):
        if target == "src":
            return t.index_select(0, src)
        if target == "dst":
            return t.index_select(0, dst)
        return t

    a = pick(leaves[0], spec.lhs_target)
    b = pick(leaves[1], spec.rhs_target) if len(leaves) > 1 else None
    if spec.op == "add":
        out = a + b
    elif spec.op == "sub":
        out = a - b
    elif spec.op == "mul":
        out = a * b
    elif spec.op == "div":
        out = a / b
    elif spec.op == "dot":
        out = (a * b).sum(-1)
    elif spec.op == "copy":
        out = a
    else:
        raise AssertionError(f"unhandled op {spec.op!r}")

    out.backward(d_out.detach().float())
    return [leaf.grad for leaf in leaves]


def _assert_close(got, ref, dtype, what):
    assert got is not None, f"{what}: the conv produced no gradient"
    torch.testing.assert_close(got.float(), ref, **_TOL[dtype], msg=lambda m: f"{what}\n{m}")


# =============================================================================
# Name -> kernel family (what the sweep's conv_type strings promise)
# =============================================================================


@pytest.mark.parametrize(
    "name,variant,backward_variant",
    [
        ("u_add_v", "node", "node"),
        ("copy_v", "node", "node"),
        ("u_add_v_edge", "edge", "edge"),
        ("copy_v_edge", "edge", "edge"),
        ("u_add_v_auto", "auto", "node"),
    ],
)
def test_op_name_pins_the_kernel_family(name, variant, backward_variant):
    """The suffix selects the family in BOTH directions: the forward row and the
    backward row of a sweep run must time the same family."""
    op = _CudaGsddmmOp(name)
    assert op.kernel.variant == variant
    assert op.kernel.backward_variant == backward_variant


@pytest.mark.parametrize(
    "name,spec_fields",
    [
        ("u_add_v", ("add", "src", "dst", False, ("u", "v"), "node")),
        ("copy_u", ("copy", "src", "edge", False, ("u",), "node")),
        ("e_dot_v_edge", ("dot", "edge", "dst", True, ("e", "v"), "edge")),
        ("u_mul_e_edge", ("mul", "src", "edge", True, ("u", "e"), "edge")),
        ("v_sub_u_auto", ("sub", "dst", "src", False, ("v", "u"), "auto")),
        ("copy_v_auto", ("copy", "dst", "edge", False, ("v",), "auto")),
    ],
)
def test_gsddmm_op_spec_parses_names(name, spec_fields):
    op, lhs_target, rhs_target, edge_variant, operand_kinds, variant = spec_fields
    spec = gsddmm_op_spec(name)
    assert (
        spec.op,
        spec.lhs_target,
        spec.rhs_target,
        spec.edge_variant,
        spec.operand_kinds,
        spec.variant,
    ) == spec_fields
    assert is_gsddmm_op(name)


def test_gsddmm_op_spec_rejects_bad_names():
    with pytest.raises(KeyError, match="Unknown turbo_gnn gsddmm op"):
        gsddmm_op_spec("u_add_w")
    # '_edge' pins a kernel while '_auto' chooses one; asking for both is ambiguous.
    with pytest.raises(KeyError, match="pins a kernel"):
        gsddmm_op_spec("u_add_v_edge_auto")
    assert not is_gsddmm_op("u_add_w")


# =============================================================================
# Forward through the conv: canonical rows, and the family equivalence
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["u_mul_e", "copy_u", "e_dot_v"])
def test_conv_forward_is_canonical_and_family_equivalent(name, dtype):
    """A `_edge` row computes the same rows as the bare one: both variants
    renumber to forward-CSR edge order, so only the launch differs. Checked
    against turbo_gnn's own `gsddmm` with the variant pinned the same way."""
    graph = _make_graph()
    spec = gsddmm_op_spec(name)
    *operands, _ = _operands(graph, spec, 64, dtype)

    out_bare = _CudaGsddmmOp(name)(*operands, graph)
    out_edge = _CudaGsddmmOp(f"{name}_edge")(*operands, graph)
    out_ref = gsddmm(
        graph,
        *operands,
        op=spec.op,
        lhs_target=spec.lhs_target,
        rhs_target=spec.rhs_target,
        variant="node",
    )
    torch.testing.assert_close(out_edge, out_bare, **_TOL[dtype], msg=lambda m: f"{name}: _edge != bare\n{m}")
    torch.testing.assert_close(out_bare, out_ref, **_TOL[dtype], msg=lambda m: f"{name}: conv != gsddmm\n{m}")


def test_conv_output_requires_grad():
    """The wiring `--mode backward` depends on: the conv routes through
    GSDDMMKernel.forward, so the output carries a grad_fn."""
    graph = _make_graph()
    spec = gsddmm_op_spec("u_add_v")
    *operands, _ = _operands(graph, spec, 64, torch.float32)
    operands = [o.requires_grad_(True) for o in operands]
    out = _CudaGsddmmOp("u_add_v")(*operands, graph)
    assert out.requires_grad and out.grad_fn is not None


# =============================================================================
# Backward through the conv vs the fp32 autograd reference
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("suffix", ["", "_edge"])
@pytest.mark.parametrize("name", ["u_add_v", "u_sub_v", "u_mul_e", "e_div_v", "v_dot_u", "copy_u", "e_mul_v"])
def test_conv_backward_matches_autograd(name, suffix, dtype):
    """Gradients through the harness path, for the backward family the name
    pins (bare -> node-parallel, `_edge` -> edge-parallel). An `e` on the LEFT
    is the per-edge map case; a u/v operand is the reduction case."""
    graph = _make_graph()
    spec = gsddmm_op_spec(name)
    *operands, d_out = _operands(graph, spec, 64, dtype)

    ref_grads = _reference_grads(graph, operands, d_out, spec)
    leaves = [o.detach().requires_grad_(True) for o in operands]
    out = _CudaGsddmmOp(name + suffix)(*leaves, graph)
    out.backward(d_out)

    for i, (got, ref) in enumerate(zip(leaves, ref_grads)):
        kind = spec.operand_kinds[i]
        _assert_close(got.grad, ref, dtype, f"d_{kind} for {name}{suffix} {dtype}")


def test_copy_conv_takes_a_single_operand():
    """copy has one operand, so the conv is called as forward(lhs, graph) --
    the rhs must stay None rather than becoming a misread lhs."""
    graph = _make_graph()
    spec = gsddmm_op_spec("copy_v")
    (lhs,), d_out = _operands(graph, spec, 64, torch.float32)
    lhs = lhs.requires_grad_(True)
    out = _CudaGsddmmOp("copy_v")(lhs, graph)
    out.backward(d_out)

    ref_grads = _reference_grads(graph, [lhs.detach()], d_out, spec)
    _assert_close(lhs.grad, ref_grads[0], torch.float32, "d_v for copy_v through the conv")


# =============================================================================
# The legacy traversal-order kernel must refuse differentiability
# =============================================================================


def test_gsddmm_edge_kernel_refuses_differentiable_forward():
    """GSDDMMEdgeKernel numbers its output by traversal order (CSC when no
    operand reads dst), so its d_out cannot feed the backward kernels -- a
    backward through it would be silently wrong, so forward() refuses."""
    graph = _make_graph()
    lhs = torch.randn(graph.forward_indptr.numel() - 1, 64, device=DEVICE)
    with pytest.raises(NotImplementedError, match="traversal order"):
        GSDDMMEdgeKernel(op="mul", lhs_target="src", rhs_target="dst").forward(graph, lhs)
