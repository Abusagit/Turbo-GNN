"""
GATv2 correctness tests: CUDA backend vs pure-torch scatter reference, plus low-precision tests.

Tests:
  - fp32: CUDA vs torch_native forward & backward
  - fp16/bf16: CUDA (low-precision) vs torch_native (fp32) forward & backward
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from src.backends.registry import BackendRegistry
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets, build_csr_as_is

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_undirected_graph(N: int, E_approx: int, device: str = "cuda", seed: int = 42):
    """Random undirected graph with self-loops, deduplicated.

    Returns edge_index [2, E] on *device*.
    """
    src = torch.randint(0, N, (E_approx,), device=device)
    dst = torch.randint(0, N, (E_approx,), device=device)
    # Make undirected
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    # Add self-loops
    self_nodes = torch.arange(N, device=device)
    src_all = torch.cat([src_all, self_nodes])
    dst_all = torch.cat([dst_all, self_nodes])
    edge_index = torch.stack([src_all, dst_all], dim=0)
    # Deduplicate
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    row = flat_unique // N
    col = flat_unique % N
    return torch.stack([row, col], dim=0)


def build_cuda_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    heavy_degree_threshold: int | None = None,
):
    """Build AdjacencyForwardBackwardWithNodeBuckets from edge_index [2, E].

    If *heavy_degree_threshold* is None, all nodes go to the light bucket
    (exercises only the light-warp kernel instantiation, W=1).  If set,
    nodes with forward degree > threshold go to the heavy bucket, which
    exercises the W=8 instantiation with maximal shared-memory pressure --
    the configuration where pipeline bugs are most likely to hide.
    """
    fwd_indptr, fwd_indices, _, _ = build_csr_as_is(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        do_transpose=True,
    )
    bwd_indptr, bwd_indices, _, _ = build_csr_as_is(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        do_transpose=False,
    )
    device = edge_index.device

    def _split(indptr: torch.Tensor):
        all_nodes = torch.arange(num_nodes, device=device, dtype=torch.int32)
        if heavy_degree_threshold is None:
            empty = torch.tensor([], dtype=torch.int32, device=device)
            return all_nodes, empty
        degrees = indptr[1:] - indptr[:-1]
        heavy_mask = degrees > heavy_degree_threshold
        light = all_nodes[~heavy_mask]
        heavy = all_nodes[heavy_mask]
        return light, heavy

    fwd_light, fwd_heavy = _split(fwd_indptr)
    bwd_light, bwd_heavy = _split(bwd_indptr)

    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=fwd_indptr.int(),
        forward_indices=fwd_indices.int(),
        backward_indptr=bwd_indptr.int(),
        backward_indices=bwd_indices.int(),
        forward_light_nodes=fwd_light,
        forward_heavy_nodes=fwd_heavy,
        backward_light_nodes=bwd_light,
        backward_heavy_nodes=bwd_heavy,
    )


def make_hub_graph(N: int, num_hubs: int = 8, hub_degree: int = 200, device: str = "cuda"):
    """Graph with a few high-degree hubs: guarantees a non-empty heavy bucket
    and neighbor lists much longer than any pipeline stage count."""
    src_bg = torch.randint(0, N, (N * 3,), device=device)
    dst_bg = torch.randint(0, N, (N * 3,), device=device)
    hubs = torch.arange(num_hubs, device=device)
    src_hub = torch.randint(0, N, (num_hubs * hub_degree,), device=device)
    dst_hub = hubs.repeat_interleave(hub_degree)
    src_all = torch.cat([src_bg, dst_bg, src_hub, torch.arange(N, device=device)])
    dst_all = torch.cat([dst_bg, src_bg, dst_hub, torch.arange(N, device=device)])
    edge_index = torch.stack([src_all, dst_all], dim=0)
    flat = torch.unique(edge_index[0] * N + edge_index[1])
    return torch.stack([flat // N, flat % N], dim=0)


def build_coo_graph(edge_index: torch.Tensor, num_nodes: int, device: str = "cuda"):
    """Build COO graph tuple for torch_native backend: (edge_index, edge_weight, num_nodes)."""
    return (edge_index.to(device), None, num_nodes)


def share_gatv2_weights(cuda_layer, ref_layer):
    """Copy weights from torch_native GATv2 layer to CUDA GATv2 layer.

    Weight mapping:
      cuda.left_right_projection.weight[:H*D] = ref.fc_dst.weight  (left = dst)
      cuda.left_right_projection.weight[H*D:] = ref.fc_src.weight  (right = src)
      cuda.attn_weights = ref.attn.squeeze(0)  ([1,H,D] -> [H,D])
      cuda._outer_proj.weight = ref._outer_proj.weight
    """
    H = cuda_layer.heads
    D = cuda_layer.head_dim

    with torch.no_grad():
        cuda_layer.left_right_projection.weight.data[: H * D].copy_(ref_layer.fc_dst.weight.data)
        cuda_layer.left_right_projection.weight.data[H * D :].copy_(ref_layer.fc_src.weight.data)
        if cuda_layer.left_right_projection.bias is not None:
            cuda_layer.left_right_projection.bias.data[: H * D].copy_(ref_layer.fc_dst.bias.data)
            cuda_layer.left_right_projection.bias.data[H * D :].copy_(ref_layer.fc_src.bias.data)

        cuda_layer.attn_weights.data.copy_(ref_layer.attn.data.squeeze(0))

        cuda_layer._outer_proj.weight.data.copy_(ref_layer._outer_proj.weight.data)
        if cuda_layer._outer_proj.bias is not None:
            cuda_layer._outer_proj.bias.data.copy_(ref_layer._outer_proj.bias.data)


def _max_mean_diff(a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs()
    return f"max|diff|={d.max().item():.3e}, mean|diff|={d.mean().item():.3e}"


# ---------------------------------------------------------------------------
# fp32: CUDA vs torch_native
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [32, 128])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_gatv2_cuda_vs_torch_native_forward(num_nodes, feature_dim, heads):
    """fp32 forward: CUDA GATv2 vs torch_native GATv2."""
    device = "cuda"

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    ref_backend = BackendRegistry.get_backend("torch_native")

    cuda_layer = cuda_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)
    ref_layer = ref_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)

    share_gatv2_weights(cuda_layer, ref_layer)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x = torch.randn(num_nodes, feature_dim, device=device)

    cuda_out = cuda_layer(x, cuda_graph)
    ref_out = ref_layer(x, ref_graph)

    assert not cuda_out.isnan().any(), "CUDA output contains NaN"
    assert not ref_out.isnan().any(), "Reference output contains NaN"
    assert_close(
        cuda_out,
        ref_out,
        rtol=1e-4,
        atol=1e-4,
        msg=lambda m: f"CUDA vs torch_native forward mismatch: {_max_mean_diff(cuda_out, ref_out)}\n{m}",
    )


@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [32, 128])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_gatv2_cuda_vs_torch_native_backward(num_nodes, feature_dim, heads):
    """fp32 backward: compare input gradients between CUDA and torch_native."""
    device = "cuda"

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    ref_backend = BackendRegistry.get_backend("torch_native")

    cuda_layer = cuda_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)
    ref_layer = ref_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)

    share_gatv2_weights(cuda_layer, ref_layer)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x_cuda = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
    x_ref = x_cuda.detach().clone().requires_grad_(True)

    cuda_out = cuda_layer(x_cuda, cuda_graph)
    ref_out = ref_layer(x_ref, ref_graph)

    cuda_out.sum().backward()
    ref_out.sum().backward()

    assert x_cuda.grad is not None, "No CUDA gradient"
    assert x_ref.grad is not None, "No reference gradient"
    assert not x_cuda.grad.isnan().any(), "CUDA grad contains NaN"
    assert_close(
        x_cuda.grad,
        x_ref.grad,
        rtol=1e-4,
        atol=1e-4,
        msg=lambda m: f"CUDA vs torch_native backward mismatch: {_max_mean_diff(x_cuda.grad, x_ref.grad)}\n{m}",
    )


# ---------------------------------------------------------------------------
# Low-precision: CUDA (fp16/bf16) vs torch_native (fp32)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [32, 128])
@pytest.mark.parametrize("heads", [2, 4])
def test_gatv2_cuda_low_precision_forward(dtype, num_nodes, feature_dim, heads):
    """Low-precision forward: CUDA at fp16/bf16 vs torch_native fp32 reference."""
    device = "cuda"

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    ref_backend = BackendRegistry.get_backend("torch_native")

    # torch_native layer in fp32 as reference
    ref_layer = ref_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)

    # CUDA layer -- share weights from ref, then cast to low precision
    cuda_layer = cuda_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)
    share_gatv2_weights(cuda_layer, ref_layer)
    cuda_layer = cuda_layer.to(dtype)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x = torch.randn(num_nodes, feature_dim, device=device)

    cuda_out = cuda_layer(x.to(dtype), cuda_graph)
    ref_out = ref_layer(x, ref_graph)

    cuda_f32 = cuda_out.float()
    ref_f32 = ref_out.float()

    assert not cuda_f32.isnan().any(), "CUDA output contains NaN"
    assert_close(
        cuda_f32,
        ref_f32,
        rtol=5e-2,
        atol=5e-2,
        msg=lambda m: f"Low-precision forward mismatch ({dtype}): {_max_mean_diff(cuda_f32, ref_f32)}\n{m}",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [32, 128])
@pytest.mark.parametrize("heads", [2, 4])
def test_gatv2_cuda_low_precision_backward(dtype, num_nodes, feature_dim, heads):
    """Low-precision backward: CUDA at fp16/bf16 vs torch_native fp32 reference gradients."""
    device = "cuda"
    torch.manual_seed(94)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    ref_backend = BackendRegistry.get_backend("torch_native")

    ref_layer = ref_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)

    cuda_layer = cuda_backend.create_conv(
        "gat_v2",
        feature_dim=feature_dim,
        heads=heads,
        bias=False,
    ).to(device)
    share_gatv2_weights(cuda_layer, ref_layer)
    cuda_layer = cuda_layer.to(dtype)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x_cuda = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype, requires_grad=True)
    x_ref = x_cuda.detach().float().clone().requires_grad_(True)

    cuda_out = cuda_layer(x_cuda, cuda_graph)
    ref_out = ref_layer(x_ref, ref_graph)

    cuda_out.sum().backward()
    ref_out.sum().backward()

    assert x_cuda.grad is not None, "No CUDA gradient"
    assert x_ref.grad is not None, "No reference gradient"
    assert not x_cuda.grad.isnan().any(), "CUDA grad contains NaN"

    g_cuda = x_cuda.grad.float()
    g_ref = x_ref.grad.float()
    assert_close(
        g_cuda,
        g_ref,
        rtol=1e-5,
        atol=2e-1,
        msg=lambda m: f"Low-precision backward mismatch ({dtype}): {_max_mean_diff(g_cuda, g_ref)}\n{m}",
    )


# ---------------------------------------------------------------------------
# Pipeline correctness: USE_PIPELINE=true vs USE_PIPELINE=false
# ---------------------------------------------------------------------------


_PIPE_TOL = {
    torch.float32: {"rtol": 1e-6, "atol": 1e-6},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
}

# (graph_kind, heavy_degree_threshold): light-only covers the W=1 kernel
# instantiation; hub graph with bucketing covers W=8 heavy path where
# shared-memory pressure (and pipeline bugs) are maximal.
_PIPE_GRAPHS = ["light_only", "with_heavy"]


def _make_pipeline_graph(kind: str, num_nodes: int, device: str = "cuda"):
    if kind == "light_only":
        edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)
        return build_cuda_graph(edge_index, num_nodes)
    edge_index = make_hub_graph(num_nodes, device=device)
    return build_cuda_graph(edge_index, num_nodes, heavy_degree_threshold=32)


@pytest.mark.parametrize("graph_kind", _PIPE_GRAPHS)
@pytest.mark.parametrize("num_stages", [1, 2, 4])
@pytest.mark.parametrize("feature_dim", [64, 256])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gatv2_pipeline_vs_baseline_forward(graph_kind, num_stages, feature_dim, heads, dtype):
    """Forward: pipeline (any stage count) must match no-pipeline baseline."""
    device = "cuda"
    torch.manual_seed(42)
    num_nodes = 1000

    cuda_graph = _make_pipeline_graph(graph_kind, num_nodes, device)
    cuda_backend = BackendRegistry.get_backend("cuda")

    layer = (
        cuda_backend.create_conv(
            "gat_v2",
            feature_dim=feature_dim,
            heads=heads,
            bias=False,
        )
        .to(device)
        .to(dtype)
    )

    x = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype)

    layer.use_pipeline = False
    out_baseline = layer(x, cuda_graph)

    layer.use_pipeline = True
    layer.num_stages = num_stages
    out_pipeline = layer(x, cuda_graph)

    assert not out_pipeline.isnan().any(), "Pipeline output contains NaN"
    assert_close(
        out_pipeline,
        out_baseline,
        **_PIPE_TOL[dtype],
        msg=lambda m: (
            f"Pipeline(s={num_stages}, {graph_kind}) vs baseline forward mismatch ({dtype}): "
            f"{_max_mean_diff(out_baseline, out_pipeline)}\n{m}"
        ),
    )


@pytest.mark.parametrize("graph_kind", _PIPE_GRAPHS)
@pytest.mark.parametrize("num_stages", [1, 2, 4])
@pytest.mark.parametrize("feature_dim", [64, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gatv2_pipeline_vs_baseline_backward(graph_kind, num_stages, feature_dim, dtype):
    """Backward: gradients must match between pipeline and baseline.

    The backward kernel itself has no pipeline, but its inputs include the
    forward output, and the +2 forward arguments changed the autograd
    contract (number of returned grads) -- both are exercised here.
    """
    device = "cuda"
    torch.manual_seed(42)
    num_nodes = 1000
    heads = 2

    cuda_graph = _make_pipeline_graph(graph_kind, num_nodes, device)
    cuda_backend = BackendRegistry.get_backend("cuda")

    layer = (
        cuda_backend.create_conv(
            "gat_v2",
            feature_dim=feature_dim,
            heads=heads,
            bias=False,
        )
        .to(device)
        .to(dtype)
    )

    x_base = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype, requires_grad=True)
    x_pipe = x_base.detach().clone().requires_grad_(True)

    layer.use_pipeline = False
    out_base = layer(x_base, cuda_graph)
    out_base.sum().backward()

    layer.use_pipeline = True
    layer.num_stages = num_stages
    out_pipe = layer(x_pipe, cuda_graph)
    out_pipe.sum().backward()

    assert x_base.grad is not None and x_pipe.grad is not None
    assert not x_pipe.grad.isnan().any(), "Pipeline grad contains NaN"

    assert_close(
        out_pipe,
        out_base,
        **_PIPE_TOL[dtype],
        msg=lambda m: (
            f"Pipeline(s={num_stages}, {graph_kind}) forward mismatch in backward test ({dtype}): "
            f"{_max_mean_diff(out_base, out_pipe)}\n{m}"
        ),
    )
    assert_close(
        x_pipe.grad,
        x_base.grad,
        **_PIPE_TOL[dtype],
        msg=lambda m: (
            f"Pipeline(s={num_stages}, {graph_kind}) backward mismatch ({dtype}): "
            f"{_max_mean_diff(x_base.grad, x_pipe.grad)}\n{m}"
        ),
    )
