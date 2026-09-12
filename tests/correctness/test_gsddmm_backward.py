"""Correctness tests for the two GSDDMM backward kernels.

The forward is a per-edge map, but the backward is a *reduction*: a gradient for
an operand read at a node sums over that node's incident edges, while an operand
read per edge gets a plain per-edge gradient. ``csrc/gsddmm/`` implements that
reduction twice --

- ``GSDDMM_backward_normal``: one block per bucketed node, fp32 register
  accumulation, one store per row; no atomics, so it is deterministic;
- ``GSDDMM_backward_edge_block``: one warp per edge chunk with fp32 atomic
  accumulation, load balanced regardless of the degree distribution

-- and both must agree with autograd on the same math. The reference here is
plain PyTorch (``index_select`` + the op + ``.backward()``), built in fp32 so a
low-precision kernel is compared against something more accurate than itself
rather than against an equally lossy reference.

A gradient for a source-side operand sums over each node's *outgoing* edges, so
it walks the backward CSR, whose slots are CSC positions; the canonical-id
permutation built for the forward is what maps those back onto ``dO``'s
numbering. That holds for undirected graphs too, even though they alias their two
CSRs: a symmetric sparsity pattern does not make the CSC slot order equal the CSR
edge order, since row ``u`` then lists ``u``'s *incoming* edges while this pass
sums over the outgoing ones -- different rows of ``dO``. Both graph kinds are
covered here, and duplicate ``(src, dst)`` pairs exercise the stable-sort
bijection the permutation is built from.
"""

import pytest
import torch

from turbo_gnn import gsddmm
from turbo_gnn._gsddmm import EdgeBlockParams, GsddmmLaunchPlan, GsddmmSpec, NodeBlockParams, _edge_endpoints
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")

OPS = ["add", "sub", "mul", "div", "dot"]
MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("src", "edge"), ("edge", "src"), ("dst", "edge"), ("edge", "dst")]
BACKWARD_VARIANTS = ["node", "edge"]
FEATURE_DIMS = [32, 64, 128, 256]

#: The node-parallel variant reduces in fp32 registers and the reference in fp32,
#: so fp32 inputs agree tightly. The edge-parallel one accumulates atomically, so
#: its sum order is arbitrary; low-precision dtypes then need room for both the
#: storage rounding and that reordering.
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


def _make_symmetric_graph(num_nodes=256, num_edges=1000, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=0.99, index_dtype=torch.int32, is_directed=False
    ).to(DEVICE)


def _rows(graph, target: str) -> int:
    return graph.forward_indices.numel() if target == "edge" else graph.forward_indptr.numel() - 1


def _operands(graph, op, lhs_target, rhs_target, feat_dim, dtype, seed=0):
    """Forward operands plus an upstream gradient of the matching shape."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_edges = graph.forward_indices.numel()
    lhs = torch.randn(_rows(graph, lhs_target), feat_dim, device=DEVICE, dtype=dtype, generator=gen)
    # Keep divisors away from zero: div's gradient carries 1/R^2, which amplifies
    # anything close to it and would make the comparison measure noise.
    rhs = torch.randn(_rows(graph, rhs_target), feat_dim, device=DEVICE, dtype=dtype, generator=gen).abs() + 1.0
    d_out_shape = (num_edges,) if op == "dot" else (num_edges, feat_dim)
    d_out = torch.randn(d_out_shape, device=DEVICE, dtype=dtype, generator=gen)
    return lhs, rhs, d_out


def _reference_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target):
    """Autograd gradients of the same math, computed in fp32 from upcast inputs."""
    src, dst = _edge_endpoints(graph, by_src=False)  # canonical (forward-CSR) edge order

    lhs32 = lhs.detach().float().requires_grad_(True)
    rhs32 = rhs.detach().float().requires_grad_(True)

    def pick(t, target):
        if target == "src":
            return t.index_select(0, src)
        if target == "dst":
            return t.index_select(0, dst)
        return t

    a, b = pick(lhs32, lhs_target), pick(rhs32, rhs_target)
    if op == "add":
        out = a + b
    elif op == "sub":
        out = a - b
    elif op == "mul":
        out = a * b
    elif op == "div":
        out = a / b
    elif op == "dot":
        out = (a * b).sum(-1)
    elif op == "copy":
        out = a
    else:
        raise AssertionError(f"unhandled op {op!r}")

    out.backward(d_out.detach().float())
    return lhs32.grad, (rhs32.grad if op != "copy" else None)


def _kernel_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target, backward_variant, variant="node"):
    lhs = lhs.detach().clone().requires_grad_(True)
    rhs_arg = None
    if op != "copy":
        rhs_arg = rhs.detach().clone().requires_grad_(True)
    out = gsddmm(
        graph,
        lhs,
        rhs_arg,
        op=op,
        lhs_target=lhs_target,
        rhs_target=rhs_target,
        variant=variant,
        backward_variant=backward_variant,
    )
    out.backward(d_out)
    return lhs.grad, (rhs_arg.grad if rhs_arg is not None else None), out


def _assert_close(got, ref, dtype, what):
    assert got is not None, f"{what}: kernel produced no gradient"
    torch.testing.assert_close(got.float(), ref, **_TOL[dtype], msg=lambda m: f"{what}\n{m}")


# =============================================================================
# Both variants against autograd, over every op and operand pair
# =============================================================================


@pytest.mark.parametrize("backward_variant", BACKWARD_VARIANTS)
@pytest.mark.parametrize("lhs_target,rhs_target", MEMBER_PAIRS)
@pytest.mark.parametrize("op", OPS)
def test_backward_matches_autograd(op, lhs_target, rhs_target, backward_variant):
    graph = _make_graph()
    lhs, rhs, d_out = _operands(graph, op, lhs_target, rhs_target, 64, torch.float32)
    d_lhs_ref, d_rhs_ref = _reference_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target)
    d_lhs, d_rhs, _ = _kernel_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target, backward_variant)

    _assert_close(d_lhs, d_lhs_ref, torch.float32, f"d_lhs for {op}({lhs_target},{rhs_target}) [{backward_variant}]")
    _assert_close(d_rhs, d_rhs_ref, torch.float32, f"d_rhs for {op}({lhs_target},{rhs_target}) [{backward_variant}]")


@pytest.mark.parametrize("backward_variant", BACKWARD_VARIANTS)
@pytest.mark.parametrize("lhs_target", ["src", "dst"])
def test_copy_backward_matches_autograd(lhs_target, backward_variant):
    """copy has one operand, so exactly one gradient and no reduction to share."""
    graph = _make_graph()
    lhs, rhs, d_out = _operands(graph, "copy", lhs_target, "edge", 64, torch.float32)
    d_lhs_ref, d_rhs_ref = _reference_grads(graph, lhs, rhs, d_out, "copy", lhs_target, "edge")
    assert d_rhs_ref is None
    d_lhs, d_rhs, _ = _kernel_grads(graph, lhs, rhs, d_out, "copy", lhs_target, "edge", backward_variant)
    assert d_rhs is None, "copy never reads a right operand, so it has no gradient for one"
    _assert_close(d_lhs, d_lhs_ref, torch.float32, f"d_lhs for copy({lhs_target}) [{backward_variant}]")


@pytest.mark.parametrize("backward_variant", BACKWARD_VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("feat_dim", FEATURE_DIMS)
def test_backward_across_dims_and_dtypes(feat_dim, dtype, backward_variant):
    """Every instantiated feature width and dtype, on a src/dst pair (both
    gradients reduce) and a src/edge pair (one reduces, one is a map)."""
    graph = _make_graph()
    for lhs_target, rhs_target in (("src", "dst"), ("src", "edge")):
        lhs, rhs, d_out = _operands(graph, "mul", lhs_target, rhs_target, feat_dim, dtype)
        d_lhs_ref, d_rhs_ref = _reference_grads(graph, lhs, rhs, d_out, "mul", lhs_target, rhs_target)
        d_lhs, d_rhs, _ = _kernel_grads(graph, lhs, rhs, d_out, "mul", lhs_target, rhs_target, backward_variant)
        tag = f"mul({lhs_target},{rhs_target}) D={feat_dim} {dtype} [{backward_variant}]"
        _assert_close(d_lhs, d_lhs_ref, dtype, f"d_lhs for {tag}")
        _assert_close(d_rhs, d_rhs_ref, dtype, f"d_rhs for {tag}")


@pytest.mark.parametrize("backward_variant", BACKWARD_VARIANTS)
@pytest.mark.parametrize("op", ["mul", "dot", "copy"])
def test_backward_on_undirected_graph(op, backward_variant):
    """An undirected graph aliases its two CSRs, so the source-side pass runs
    without a canonical permutation -- the orders already coincide."""
    graph = _make_symmetric_graph()
    lhs_target, rhs_target = ("src", "edge") if op == "copy" else ("src", "dst")
    lhs, rhs, d_out = _operands(graph, op, lhs_target, rhs_target, 64, torch.float32)
    d_lhs_ref, d_rhs_ref = _reference_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target)
    d_lhs, d_rhs, _ = _kernel_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target, backward_variant)
    _assert_close(d_lhs, d_lhs_ref, torch.float32, f"d_lhs for {op} undirected [{backward_variant}]")
    if d_rhs_ref is not None:
        _assert_close(d_rhs, d_rhs_ref, torch.float32, f"d_rhs for {op} undirected [{backward_variant}]")


@pytest.mark.parametrize("backward_variant", BACKWARD_VARIANTS)
def test_isolated_nodes_get_zero_gradient(backward_variant):
    """A node with no incident edges contributes to nothing, so its gradient row
    must be zero -- and the node-parallel variant only ever writes it once, so
    an unwritten row would show up as garbage rather than as zeros."""
    # Two nodes, one edge: node 2 and node 3 are isolated.
    edge_index = torch.tensor([[0], [1]], device=DEVICE)
    graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, 4, quantile=0.99, index_dtype=torch.int32, is_directed=True
    ).to(DEVICE)
    lhs = torch.randn(4, 32, device=DEVICE, requires_grad=True)
    rhs = (torch.randn(4, 32, device=DEVICE).abs() + 1).requires_grad_(True)
    out = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", backward_variant=backward_variant)
    out.backward(torch.randn_like(out))
    assert torch.equal(lhs.grad[2], torch.zeros(32, device=DEVICE)), "isolated node got a nonzero src gradient"
    assert torch.equal(rhs.grad[3], torch.zeros(32, device=DEVICE)), "isolated node got a nonzero dst gradient"


# =============================================================================
# The two variants against each other, and the memory contract
# =============================================================================


@pytest.mark.parametrize("lhs_target,rhs_target", MEMBER_PAIRS)
@pytest.mark.parametrize("op", OPS)
def test_backward_variants_agree(op, lhs_target, rhs_target):
    """The two decompositions differ only in how they accumulate, so up to fp32
    atomic reordering they must produce the same gradients."""
    graph = _make_graph()
    lhs, rhs, d_out = _operands(graph, op, lhs_target, rhs_target, 64, torch.float32)
    d_lhs_node, d_rhs_node, _ = _kernel_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target, "node")
    d_lhs_edge, d_rhs_edge, _ = _kernel_grads(graph, lhs, rhs, d_out, op, lhs_target, rhs_target, "edge")
    torch.testing.assert_close(d_lhs_node, d_lhs_edge, **_TOL[torch.float32])
    if d_rhs_node is not None:
        torch.testing.assert_close(d_rhs_node, d_rhs_edge, **_TOL[torch.float32])


@pytest.mark.parametrize(
    "op,expect_saved", [("add", False), ("sub", False), ("copy", False), ("mul", True), ("div", True), ("dot", True)]
)
def test_only_ops_with_nonconstant_partials_save_operands(op, expect_saved):
    """add/sub/copy have constant partials, so their backward keeps no feature
    tensors alive -- the difference between one and two [E, D] activations per op
    in a layer. This checks the forward actually honours that."""
    graph = _make_graph()
    rhs_target = "edge" if op == "copy" else "dst"
    lhs, rhs, d_out = _operands(graph, op, "src", rhs_target, 64, torch.float32)
    lhs = lhs.detach().requires_grad_(True)
    rhs_arg = None if op == "copy" else rhs.detach().requires_grad_(True)
    out = gsddmm(graph, lhs, rhs_arg, op=op, lhs_target="src", rhs_target=rhs_target)

    saved = out.grad_fn.saved_tensors
    any_saved = any(t is not None for t in saved)
    assert any_saved == expect_saved, f"{op}: saved={[None if t is None else tuple(t.shape) for t in saved]}"
    # Whatever it saved (or did not), the gradient still has to be right.
    out.backward(d_out)
    d_lhs_ref, _ = _reference_grads(graph, lhs, rhs, d_out, op, "src", rhs_target)
    _assert_close(lhs.grad, d_lhs_ref, torch.float32, f"d_lhs for {op} after the save check")


def test_backward_variant_is_validated():
    graph = _make_graph()
    lhs, rhs, _ = _operands(graph, "mul", "src", "dst", 64, torch.float32)
    with pytest.raises(ValueError, match="backward_variant"):
        gsddmm(graph, lhs, rhs, op="mul", backward_variant="nope")


def test_plan_carries_an_independent_backward_variant():
    """The forward and backward kernels are chosen separately: the backward is a
    reduction, so the forward's winner says nothing about it."""
    spec = GsddmmSpec("mul", "src", "dst")
    plan = GsddmmLaunchPlan(spec, "edge", EdgeBlockParams())
    assert plan.backward_variant == "node", "the deterministic backward is the default"
    plan_node = GsddmmLaunchPlan(spec, "node", NodeBlockParams(), backward_variant="edge")
    assert plan_node.variant == "node" and plan_node.backward_variant == "edge"


@pytest.mark.parametrize("forward_variant", ["node", "edge", "auto"])
def test_backward_is_independent_of_the_forward_kernel(forward_variant):
    """Whichever forward kernel ran, the gradients must match -- the forward's
    output numbering is canonical, so the backward reads the same dO either way."""
    graph = _make_graph()
    lhs, rhs, d_out = _operands(graph, "mul", "src", "edge", 64, torch.float32)
    d_lhs_ref, d_rhs_ref = _reference_grads(graph, lhs, rhs, d_out, "mul", "src", "edge")
    d_lhs, d_rhs, _ = _kernel_grads(graph, lhs, rhs, d_out, "mul", "src", "edge", "node", variant=forward_variant)
    _assert_close(d_lhs, d_lhs_ref, torch.float32, f"d_lhs with forward={forward_variant}")
    _assert_close(d_rhs, d_rhs_ref, torch.float32, f"d_rhs with forward={forward_variant}")


def test_prefilled_aliases_are_differentiable():
    """The public op family goes through the same autograd path."""
    import turbo_gnn

    graph = _make_graph()
    lhs, rhs, d_out = _operands(graph, "add", "src", "dst", 64, torch.float32)
    lhs = lhs.detach().requires_grad_(True)
    rhs = rhs.detach().requires_grad_(True)
    out = turbo_gnn.u_add_v(graph, lhs, rhs)
    assert out.requires_grad, "the prefilled ops must be differentiable"
    out.backward(d_out)
    d_lhs_ref, d_rhs_ref = _reference_grads(graph, lhs, rhs, d_out, "add", "src", "dst")
    _assert_close(lhs.grad, d_lhs_ref, torch.float32, "d_lhs via u_add_v")
    _assert_close(rhs.grad, d_rhs_ref, torch.float32, "d_rhs via u_add_v")
