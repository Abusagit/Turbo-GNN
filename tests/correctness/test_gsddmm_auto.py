"""Correctness tests for automatic GSDDMM kernel selection (``gsddmm(variant=...)``).

The two forward kernels in ``csrc/gsddmm/`` decompose the work differently
(one block per bucketed CSR row vs one warp per edge chunk), and ``gsddmm``
picks between them by measuring once per graph. That is only safe if the choice
cannot be observed in the result, so the central assertion here is that
``variant="node"``, ``variant="edge"`` and ``variant="auto"`` agree
*elementwise* -- same values, same row numbering -- for every op, operand pair,
feature width and dtype.

The interesting case is an op whose operands never read the destination vertex
(``u_mul_e``, ``copy_u``, ...): there the edge kernel traverses a source-grouped
edge list for locality, and only the canonical-edge-id indirection added to
``GSDDMM_forward_edge_block`` keeps its ``Edge`` operand reads and output rows
numbered by forward-CSR position. Duplicate ``(src, dst)`` pairs exercise the
stable-sort bijection that builds that mapping.
"""

import pytest
import torch

import turbo_gnn
from turbo_gnn import gsddmm, gsddmm_edge
from turbo_gnn._autotune import AutotuneConfig
from turbo_gnn._gsddmm import (
    EdgeBlockParams,
    GsddmmPlan,
    GsddmmSpec,
    NodeBlockParams,
    TraversalOrder,
    _geometry_prefers_node,
    _graph_canonical_edge_idx,
    _graph_edge_list,
    _plan_cache_keys,
    select_variant,
)
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE = torch.device("cuda")

OPS = ["add", "sub", "mul", "div", "dot"]
MEMBER_PAIRS = [("src", "dst"), ("dst", "src"), ("src", "edge"), ("edge", "src"), ("dst", "edge"), ("edge", "dst")]
#: Pairs with no dst operand: the edge kernel traverses source-grouped and must
#: remap its edge ids to stay comparable with the CSR kernel.
SRC_ONLY_PAIRS = [("src", "edge"), ("edge", "src")]
VARIANTS = ["node", "edge", "auto"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FEATURE_DIMS = [32, 64, 128, 256]


def _make_graph(
    num_nodes: int = 512,
    num_edges: int = 4000,
    index_dtype: torch.dtype = torch.int32,
    seed: int = 0,
    is_directed: bool = True,
) -> AdjacencyForwardBackwardWithNodeBuckets:
    """Random directed multigraph: duplicate (src, dst) pairs are expected, and
    are exactly what the canonical-id bijection has to handle."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=DEVICE, generator=gen, dtype=torch.long)
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=0.99, index_dtype=index_dtype, is_directed=is_directed
    ).to(DEVICE)


def _make_symmetric_graph(num_nodes: int = 512, num_edges: int = 2000, seed: int = 0):
    """Undirected graph: the backward CSR is aliased to the forward one, so the
    source-grouped edge list *is* the canonical order and no remap is built."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), device=DEVICE, generator=gen)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, num_nodes, quantile=0.99, index_dtype=torch.int32, is_directed=False
    ).to(DEVICE)


def _operands(graph, lhs_target: str, rhs_target: str, feat_dim: int, dtype: torch.dtype, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    rows = lambda target: num_edges if target == "edge" else num_nodes  # noqa: E731
    lhs = torch.randn(rows(lhs_target), feat_dim, device=DEVICE, dtype=dtype, generator=gen)
    # Keep the divisor away from zero so div stays well conditioned in fp16.
    rhs = torch.randn(rows(rhs_target), feat_dim, device=DEVICE, dtype=dtype, generator=gen).abs() + 1
    return lhs, rhs


def _same(reference: torch.Tensor, other: torch.Tensor, op: str) -> bool:
    """Elementwise equality; ``dot`` reduces over D, so allow rounding slack.

    Both kernels reduce the same products in the same per-lane tile order, but
    they do not have to emit identical rounding for a fused multiply-add mix.
    """
    if op != "dot":
        return torch.equal(reference, other)
    return torch.allclose(reference, other, rtol=2e-2, atol=2e-2)


# =============================================================================
# The central guarantee: the kernel choice is invisible in the result
# =============================================================================


@pytest.mark.parametrize("lhs_target,rhs_target", MEMBER_PAIRS)
@pytest.mark.parametrize("op", OPS)
def test_variants_agree_elementwise(op: str, lhs_target: str, rhs_target: str) -> None:
    graph = _make_graph()
    lhs, rhs = _operands(graph, lhs_target, rhs_target, 64, torch.float32)
    outs = {
        v: gsddmm(graph, lhs, rhs, op=op, lhs_target=lhs_target, rhs_target=rhs_target, variant=v) for v in VARIANTS
    }
    for variant in ("edge", "auto"):
        assert _same(outs["node"], outs[variant], op), (
            f"{variant} disagrees with node for {op}({lhs_target},{rhs_target})"
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("feat_dim", FEATURE_DIMS)
@pytest.mark.parametrize("lhs_target,rhs_target", SRC_ONLY_PAIRS)
def test_src_grouped_traversal_agrees_across_dims_and_dtypes(
    lhs_target: str, rhs_target: str, feat_dim: int, dtype: torch.dtype
) -> None:
    """The remapped path, over every instantiated feature width and dtype."""
    graph = _make_graph()
    lhs, rhs = _operands(graph, lhs_target, rhs_target, feat_dim, dtype)
    reference = gsddmm(graph, lhs, rhs, op="mul", lhs_target=lhs_target, rhs_target=rhs_target, variant="node")
    got = gsddmm(graph, lhs, rhs, op="mul", lhs_target=lhs_target, rhs_target=rhs_target, variant="edge")
    assert torch.equal(reference, got)


@pytest.mark.parametrize(
    "stages,edges_per_warp,warps_per_block", [(0, 1, 1), (0, 4, 4), (0, 32, 8), (2, 8, 2), (3, 32, 8)]
)
def test_remap_holds_for_every_edge_kernel_shape(stages: int, edges_per_warp: int, warps_per_block: int) -> None:
    """Both the direct and the cp.async path must apply the canonical ids.

    The direct path additionally caches the shared operand row in registers
    after a warp ballot, so a whole chunk can take a different code path.
    """
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "edge", 64, torch.float32)
    reference = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="edge", variant="node")
    got = gsddmm(
        graph,
        lhs,
        rhs,
        op="mul",
        lhs_target="src",
        rhs_target="edge",
        variant="edge",
        pipeline_stages=stages,
        edges_per_warp=edges_per_warp,
        warps_per_block=warps_per_block,
    )
    assert torch.equal(reference, got)


@pytest.mark.parametrize("op", ["mul", "copy"])
def test_variants_agree_on_undirected_graph(op: str) -> None:
    """Undirected graphs alias the two CSRs, so the remap must be skipped."""
    graph = _make_symmetric_graph()
    lhs, rhs = _operands(graph, "src", "edge", 64, torch.float32)
    rhs_arg = None if op == "copy" else rhs
    reference = gsddmm(graph, lhs, rhs_arg, op=op, lhs_target="src", rhs_target="edge", variant="node")
    for variant in ("edge", "auto"):
        got = gsddmm(graph, lhs, rhs_arg, op=op, lhs_target="src", rhs_target="edge", variant=variant)
        assert torch.equal(reference, got)
    plan = GsddmmPlan(GsddmmSpec(op, "src", "edge"), "edge", EdgeBlockParams(), TraversalOrder.CSC)
    assert not plan.remaps_edge_ids(graph), "aliased CSRs need no canonical-id array"


@pytest.mark.parametrize("lhs_target", ["src", "dst"])
def test_copy_variants_agree(lhs_target: str) -> None:
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 128, device=DEVICE)
    outs = [gsddmm(graph, x, None, op="copy", lhs_target=lhs_target, variant=v) for v in VARIANTS]
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0], outs[2])


def test_public_aliases_dispatch_automatically() -> None:
    """The prefilled family forwards ``variant``, and defaults to auto."""
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    assert torch.equal(turbo_gnn.u_add_v(graph, lhs, rhs), turbo_gnn.u_add_v(graph, lhs, rhs, variant="node"))
    assert torch.equal(turbo_gnn.u_add_v(graph, lhs, rhs), turbo_gnn.u_add_v(graph, lhs, rhs, variant="edge"))
    # One op per operation is public; the *_edge spellings stay importable.
    assert "u_add_v" in turbo_gnn.__all__
    assert "u_add_v_edge" not in turbo_gnn.__all__
    assert "gsddmm_edge" not in turbo_gnn.__all__
    assert callable(turbo_gnn.u_add_v_edge)


# =============================================================================
# The canonical-id mapping itself
# =============================================================================


def test_canonical_edge_idx_is_a_permutation() -> None:
    graph = _make_graph()
    num_edges = graph.forward_indices.numel()
    canonical = _graph_canonical_edge_idx(graph).view(torch.int64)
    assert canonical.numel() == num_edges
    assert torch.equal(torch.sort(canonical).values, torch.arange(num_edges, device=DEVICE))


def test_canonical_edge_idx_maps_endpoints_consistently() -> None:
    """Slot k and canonical id ``canonical[k]`` must be the same edge."""
    graph = _make_graph()
    canonical = _graph_canonical_edge_idx(graph).view(torch.int64)
    csc = _graph_edge_list(graph, by_src=True).view(torch.int64)
    csr = _graph_edge_list(graph, by_src=False).view(torch.int64)
    assert torch.equal(csc, csr[canonical]), "canonical ids must preserve each edge's (src, dst)"


def test_canonical_edge_idx_is_cached_per_graph() -> None:
    graph = _make_graph()
    first = _graph_canonical_edge_idx(graph)
    assert _graph_canonical_edge_idx(graph) is first


def test_legacy_edge_op_keeps_traversal_order() -> None:
    """``gsddmm_edge`` is the internal, unrenumbered spelling.

    With only node operands its rows are exactly a permutation of the canonical
    ones (same values, traversal order).
    """
    graph = _make_graph()
    num_nodes = graph.forward_indptr.numel() - 1
    x = torch.randn(num_nodes, 64, device=DEVICE)
    canonical = gsddmm(graph, x, None, op="copy", lhs_target="src")
    legacy = gsddmm_edge(graph, x, None, op="copy", lhs_target="src")
    assert not torch.equal(canonical, legacy), "legacy op should follow traversal order"
    canonical_idx = _graph_canonical_edge_idx(graph).view(torch.int64)
    assert torch.equal(canonical[canonical_idx], legacy)


def test_legacy_edge_op_indexes_edge_operands_by_traversal_slot() -> None:
    """With an ``Edge`` operand the legacy op is not even a permutation of the
    canonical result: it pairs slot ``k`` with ``rhs[k]``, i.e. it assumes the
    caller's edge features are already in traversal (CSC) order, whereas
    ``gsddmm`` pairs each edge with its forward-CSR row. This is exactly the
    ambiguity the canonical ids remove, and why only ``gsddmm`` may auto-dispatch.
    """
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "edge", 64, torch.float32)
    canonical = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="edge")
    legacy = gsddmm_edge(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="edge")
    canonical_idx = _graph_canonical_edge_idx(graph).view(torch.int64)
    assert not torch.equal(canonical[canonical_idx], legacy)
    # Re-index the caller's edge rows into traversal order and they agree again.
    reordered = gsddmm_edge(graph, lhs, rhs[canonical_idx], op="mul", lhs_target="src", rhs_target="edge")
    assert torch.equal(canonical[canonical_idx], reordered)


# =============================================================================
# Selection: measured once per graph, keyed by dtype, and cheap afterwards
# =============================================================================


def _probe_memo(graph) -> dict:
    """The (stamp, {key: choice}) memo the probe keeps on the graph."""
    entry = graph.__dict__.get("_gsddmm_variant")
    return entry[1] if entry else {}


def _probe_count(graph) -> int:
    return len(_probe_memo(graph))


def test_probe_runs_once_per_graph_and_op() -> None:
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    for _ in range(5):
        gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst")
    assert _probe_count(graph) == 1, "repeat calls must reuse the memoized verdict"
    gsddmm(graph, lhs, rhs, op="add", lhs_target="src", rhs_target="dst")
    assert _probe_count(graph) == 2, "a different op is a different workload"


def test_probe_key_includes_dtype_and_feature_dim() -> None:
    """dtype is part of the key, never a searched axis: a graph probed in fp16
    must not hand that verdict to an fp32 call."""
    graph = _make_graph()
    for dtype in (torch.float16, torch.float32):
        lhs, rhs = _operands(graph, "src", "dst", 64, dtype)
        gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst")
    assert _probe_count(graph) == 2
    lhs, rhs = _operands(graph, "src", "dst", 128, torch.float32)
    gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst")
    assert _probe_count(graph) == 3

    keys = list(_probe_memo(graph))
    assert {(key[1], key[2]) for key in keys} == {(64, torch.float16), (64, torch.float32), (128, torch.float32)}


@pytest.mark.parametrize("variant", ["node", "edge"])
def test_pinned_variants_never_measure(variant: str) -> None:
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", variant=variant)
    assert _probe_count(graph) == 0


@pytest.mark.parametrize("num_nodes,num_edges", [(64, 128), (512, 4000)])
def test_probe_keeps_only_what_the_winner_reads(num_nodes: int, num_edges: int) -> None:
    """The probe builds both edge lists (16 B/edge) and the canonical index
    (8 B/edge) to time its candidates; whatever the winner does not read must be
    released rather than left pinned to the graph."""
    graph = _make_graph(num_nodes=num_nodes, num_edges=num_edges)
    lhs, rhs = _operands(graph, "src", "edge", 64, torch.float32)
    plan = select_variant(GsddmmSpec("mul", "src", "edge"), graph, lhs, rhs)

    all_scratch = {"_gsddmm_edge_list_dst", "_gsddmm_edge_list_src", "_gsddmm_canonical_edge_idx"}
    kept = {key for key in all_scratch if key in graph.__dict__}
    assert kept == set(_plan_cache_keys(plan, graph)), f"{plan.variant}/{plan.traversal.value} kept {kept}"
    # Relaunching must not need anything that was released.
    plan.launch(graph, lhs, rhs)


def test_staged_autotune_searches_only_the_winning_variant() -> None:
    """Stage B must never time the losing kernel, nor vary its parameters.

    That is the whole point of staging the search: a cartesian product over both
    variants would spend most of its budget on axes the running kernel ignores.
    """
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    kernel = turbo_gnn.GSDDMMKernel(op="mul", lhs_target="src", rhs_target="dst", variant="auto")

    launched: list[str] = []
    original_plan = kernel._plan

    def spy_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        launched.append(plan.variant)
        return plan

    configured: list[frozenset] = []
    original_configure = kernel.configure

    def spy_configure(**kwargs):
        configured.append(frozenset(kwargs))
        original_configure(**kwargs)

    kernel._plan = spy_plan
    kernel.configure = spy_configure
    result = kernel._inline_autotune(lhs, graph, AutotuneConfig(warmup=1, iters=2), rhs=rhs)

    winner = result["kernel_config"]["forward_variant"]
    assert winner == kernel.forward_variant
    assert set(launched) == {winner}, f"stage B timed the losing kernel: {set(launched)}"

    allowed = {param.name for param in kernel._variant_kernel_params(winner)}
    for names in configured:
        assert names <= allowed, f"searched an axis the {winner} kernel does not read: {names - allowed}"


def test_measurement_free_fallback_uses_geometry() -> None:
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    config = AutotuneConfig(measure_variant=False)
    plan = select_variant(GsddmmSpec("mul", "src", "dst"), graph, lhs, rhs, config=config)
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    expected = "node" if _geometry_prefers_node(num_nodes, num_edges, 64) else "edge"
    assert plan.variant == expected


def test_shared_probe_is_opt_in() -> None:
    """Off by default, so a per-op measurement never depends on call order."""
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    shared = AutotuneConfig(share_variant_probe=True)
    select_variant(GsddmmSpec("mul", "src", "dst"), graph, lhs, rhs, config=shared)
    # add/sub/mul/div over (src, dst) all have the same operand shape class.
    select_variant(GsddmmSpec("add", "src", "dst"), graph, lhs, rhs, config=shared)
    assert _probe_count(graph) == 1

    other = _make_graph(seed=1)
    lhs2, rhs2 = _operands(other, "src", "dst", 64, torch.float32)
    select_variant(GsddmmSpec("mul", "src", "dst"), other, lhs2, rhs2)
    select_variant(GsddmmSpec("add", "src", "dst"), other, lhs2, rhs2)
    assert _probe_count(other) == 2


def test_autotune_matches_reference_and_reports_its_choice() -> None:
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    reference = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", variant="node")
    config = AutotuneConfig(warmup=1, iters=2)
    tuned = gsddmm(graph, lhs, rhs, op="mul", lhs_target="src", rhs_target="dst", autotune=True, autotune_config=config)
    assert torch.equal(reference, tuned)

    kernel = turbo_gnn.GSDDMMKernel._get_or_create(op="mul", lhs_target="src", rhs_target="dst", variant="auto")
    assert kernel.forward_variant in ("node", "edge")
    cached = kernel._inline_cache.lookup(graph, 64, torch.float32)
    assert cached is not None and cached["kernel_config"]["forward_variant"] == kernel.forward_variant


# =============================================================================
# A resolved plan carries only the selected kernel's parameters and context
# =============================================================================


@pytest.mark.parametrize("variant,params_cls", [("node", NodeBlockParams), ("edge", EdgeBlockParams)])
def test_plan_carries_only_its_own_parameters(variant: str, params_cls: type) -> None:
    plan = GsddmmPlan(GsddmmSpec("mul", "src", "dst"), variant, params_cls())
    assert isinstance(plan.params, params_cls)
    other = EdgeBlockParams if params_cls is NodeBlockParams else NodeBlockParams
    with pytest.raises(TypeError):
        GsddmmPlan(GsddmmSpec("mul", "src", "dst"), variant, other())


def test_cross_variant_parameters_are_rejected_when_pinned() -> None:
    graph = _make_graph()
    lhs, rhs = _operands(graph, "src", "dst", 64, torch.float32)
    with pytest.raises(ValueError, match="edges_per_warp"):
        gsddmm(graph, lhs, rhs, op="mul", variant="node", edges_per_warp=8)
    with pytest.raises(ValueError, match="overlap_buckets"):
        gsddmm(graph, lhs, rhs, op="mul", variant="edge", overlap_buckets=True)
    # Under "auto" either kernel may run, so both are accepted as its candidate's
    # starting point.
    gsddmm(graph, lhs, rhs, op="mul", variant="auto", edges_per_warp=8, overlap_buckets=True)


@pytest.mark.parametrize(
    "op,expected",
    [("add", ()), ("sub", ()), ("copy", ()), ("mul", ("rhs",)), ("div", ("rhs",)), ("dot", ("lhs", "rhs"))],
)
def test_backward_needs_only_the_operands_with_nonconstant_partials(op: str, expected: tuple) -> None:
    """Preparation for the backward pass: add/sub/copy save no feature tensors."""
    rhs_target = "edge" if op == "copy" else "dst"
    assert GsddmmSpec(op, "src", rhs_target).backward_needs_operands == expected


def test_backward_context_holds_only_the_chosen_kernel_inputs() -> None:
    graph = _make_graph()
    spec = GsddmmSpec("mul", "src", "dst")

    node_ctx = GsddmmPlan(spec, "node", NodeBlockParams()).backward_context(graph)
    assert node_ctx["forward_indptr"] is graph.forward_indptr
    # src operand -> the gradient reduces over outgoing edges -> backward CSR.
    assert "backward_indptr" in node_ctx
    assert not any(key.startswith("edge_list") for key in node_ctx)

    edge_plan = GsddmmPlan(spec, "edge", EdgeBlockParams(), TraversalOrder.CSR)
    edge_ctx = edge_plan.backward_context(graph)
    assert set(edge_ctx) == {"edge_list"}, "the edge kernel reads no CSR"

    # A source-grouped traversal also needs the permutation back to canonical ids.
    remapped = GsddmmPlan(GsddmmSpec("mul", "src", "edge"), "edge", EdgeBlockParams(), TraversalOrder.CSC)
    assert set(remapped.backward_context(graph)) == {"edge_list", "canonical_edge_idx"}


def test_spec_normalizes_and_rejects_impossible_operands() -> None:
    assert GsddmmSpec("copy", "src", "dst").rhs_target == "edge", "copy never reads rhs"
    assert not GsddmmSpec("copy", "src", "edge").uses_rhs
    assert GsddmmSpec("mul", "src", "dst").preferred_traversal is TraversalOrder.CSR
    assert GsddmmSpec("mul", "src", "edge").preferred_traversal is TraversalOrder.CSC
    with pytest.raises(ValueError):
        GsddmmSpec("mul", "src", "src")
    with pytest.raises(ValueError):
        GsddmmSpec("copy", "edge", "edge")
    with pytest.raises(ValueError):
        GsddmmSpec("nope", "src", "dst")
