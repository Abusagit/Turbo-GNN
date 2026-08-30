"""
Graph Transformer correctness tests:
  - CUDA backend vs torch_native scatter reference (fp32)
  - Triton backend vs torch_native scatter reference (fp32, noting Triton internally uses fp16)
  - Low-precision (fp16/bf16) variants against fp32 reference

"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.backends.registry import BackendRegistry  # noqa: E402
from src.data.converters import (  # noqa: E402
    AdjacencyForwardBackwardWithNodeBuckets,
    WSBFormat,
    build_csr_as_is,
    normalize_adj,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_undirected_graph(N: int, E_approx: int, device: str = "cuda", seed: int = 42):
    """Random undirected graph with self-loops, deduplicated."""
    gen = torch.Generator(device=device).manual_seed(seed)
    src = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    dst = torch.randint(0, N, (E_approx,), device=device, generator=gen)
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    self_nodes = torch.arange(N, device=device)
    src_all = torch.cat([src_all, self_nodes])
    dst_all = torch.cat([dst_all, self_nodes])
    edge_index = torch.stack([src_all, dst_all], dim=0)
    flat = edge_index[0] * N + edge_index[1]
    flat_unique = torch.unique(flat)
    row = flat_unique // N
    col = flat_unique % N
    return torch.stack([row, col], dim=0)


def build_cuda_graph(edge_index: torch.Tensor, num_nodes: int):
    """Build AdjacencyForwardBackwardWithNodeBuckets for CUDA GT backend."""
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
    all_nodes = torch.arange(num_nodes, device=edge_index.device, dtype=torch.int32)
    empty_nodes = torch.tensor([], dtype=torch.int32, device=edge_index.device)
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=fwd_indptr.int(),
        forward_indices=fwd_indices.int(),
        backward_indptr=bwd_indptr.int(),
        backward_indices=bwd_indices.int(),
        forward_light_nodes=all_nodes,
        forward_heavy_nodes=empty_nodes,
        backward_light_nodes=all_nodes,
        backward_heavy_nodes=empty_nodes,
    )


def make_hub_graph(N: int, num_hubs: int = 8, hub_degree: int = 200, device: str = "cuda"):
    """Directed graph with a few high-degree hubs (asymmetric hub edges):
    guarantees a non-empty heavy bucket and exercises the directed backward
    kernel (graph_attn_backward_csrT_kernel_D), unlike make_undirected_graph."""
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


def build_cuda_graph_bucketed(edge_index: torch.Tensor, num_nodes: int, heavy_degree_threshold: int | None = None):
    """Like build_cuda_graph, but optionally buckets high-degree nodes into
    the heavy partition (exercises W=8/16/32 kernel instantiations)."""
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
        return all_nodes[~heavy_mask], all_nodes[heavy_mask]

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


def build_wsb_graph(edge_index: torch.Tensor, num_nodes: int):
    """Build WSBFormat graph for Triton backend."""
    adj_sparse = normalize_adj(
        edge_index.cpu(),
        num_nodes=num_nodes,
        how="both",
        add_self_loops=False,
    )
    adj_csr = adj_sparse.to_sparse_csr()
    wsb = WSBFormat.build_wsb_format(adj_csr)
    return wsb.cuda()


def build_coo_graph(edge_index: torch.Tensor, num_nodes: int, device: str = "cuda"):
    """Build COO graph tuple for torch_native backend: (edge_index, edge_weight, num_nodes)."""
    return (edge_index.to(device), None, num_nodes)


def share_gt_weights(target_layer, ref_layer):
    """Copy weights from reference GT layer to target (CUDA/Triton) GT layer."""
    with torch.no_grad():
        ref_w = ref_layer.qkv_proj.weight.data
        ref_b = ref_layer.qkv_proj.bias.data
        target_layer.qkv_proj.weight.data.copy_(ref_w.clone())
        target_layer.qkv_proj.bias.data.copy_(ref_b.clone())


# ---------------------------------------------------------------------------
# fp32: CUDA vs torch_native
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [256, 1024])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_cuda_vs_torch_native_forward(num_nodes, feature_dim, heads):
    """fp32 forward: CUDA GT vs torch_native GT."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    ref_backend = BackendRegistry.get_backend("torch_native")

    cuda_layer = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)
    ref_layer = ref_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)

    share_gt_weights(cuda_layer, ref_layer)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x = torch.randn(num_nodes, feature_dim, device=device)

    cuda_out = cuda_layer(x, cuda_graph)
    ref_out = ref_layer(x, ref_graph)

    assert not cuda_out.isnan().any(), "CUDA output contains NaN"
    assert not ref_out.isnan().any(), "Reference output contains NaN"
    assert torch.allclose(cuda_out, ref_out, atol=1e-4, rtol=1e-4), (
        f"CUDA vs torch_native forward mismatch: "
        f"max|diff|={(cuda_out - ref_out).abs().max().item():.3e}, "
        f"mean|diff|={(cuda_out - ref_out).abs().mean().item():.3e}"
    )


@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [256, 1024])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_cuda_vs_torch_native_backward(num_nodes, feature_dim, heads):
    """fp32 backward: compare input gradients CUDA GT vs torch_native GT."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    ref_backend = BackendRegistry.get_backend("torch_native")

    cuda_layer = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)
    ref_layer = ref_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)

    share_gt_weights(cuda_layer, ref_layer)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x_cuda = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
    x_ref = x_cuda.detach().clone().requires_grad_(True)

    cuda_out = cuda_layer(x_cuda, cuda_graph)
    ref_out = ref_layer(x_ref, ref_graph)

    grad_output = torch.randn_like(cuda_out)

    cuda_out.backward(grad_output)
    ref_out.backward(grad_output)

    assert x_cuda.grad is not None, "No CUDA gradient"
    assert x_ref.grad is not None, "No reference gradient"
    assert not x_cuda.grad.isnan().any(), "CUDA grad contains NaN"
    assert torch.allclose(x_cuda.grad, x_ref.grad, atol=1e-3, rtol=1e-3), (
        f"CUDA vs torch_native backward mismatch: "
        f"max|diff|={(x_cuda.grad - x_ref.grad).abs().max().item():.3e}, "
        f"mean|diff|={(x_cuda.grad - x_ref.grad).abs().mean().item():.3e}"
    )


# ---------------------------------------------------------------------------
# fp32: Triton vs torch_native
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes", [48, 128])
@pytest.mark.parametrize("feature_dim", [256, 1024])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_triton_vs_torch_native_forward(num_nodes, feature_dim, heads):
    """fp32 forward: Triton GT vs torch_native GT.

    Triton layer has NO layer_norm -> pre-apply F.layer_norm before Triton.
    Triton internally converts fp32 -> fp16, so outputs have fp16-level error.
    """
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    triton_backend = BackendRegistry.get_backend("triton_block_sparse")
    ref_backend = BackendRegistry.get_backend("torch_native")

    triton_layer = triton_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)
    ref_layer = ref_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)

    share_gt_weights(triton_layer, ref_layer)

    wsb_graph = build_wsb_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x = torch.randn(num_nodes, feature_dim, device=device)

    # Triton has no layer_norm, so pre-apply it
    x_normed = F.layer_norm(x, (feature_dim,))
    triton_out = triton_layer(x_normed, wsb_graph)
    ref_out = ref_layer(x, ref_graph)

    assert not triton_out.isnan().any(), "Triton output contains NaN"
    assert not ref_out.isnan().any(), "Reference output contains NaN"
    # Relaxed tolerance due to Triton's internal fp16 conversion
    assert torch.allclose(triton_out, ref_out, atol=5e-3, rtol=5e-3), (
        f"Triton vs torch_native forward mismatch: "
        f"max|diff|={(triton_out - ref_out).abs().max().item():.3e}, "
        f"mean|diff|={(triton_out - ref_out).abs().mean().item():.3e}"
    )


@pytest.mark.parametrize("num_nodes", [48, 128])
@pytest.mark.parametrize("feature_dim", [256, 1024])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_triton_vs_torch_native_backward(num_nodes, feature_dim, heads):
    """fp32 backward: compare input gradients Triton GT vs torch_native GT."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    triton_backend = BackendRegistry.get_backend("triton_block_sparse")
    ref_backend = BackendRegistry.get_backend("torch_native")

    triton_layer = triton_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)
    ref_layer = ref_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)

    share_gt_weights(triton_layer, ref_layer)

    wsb_graph = build_wsb_graph(edge_index, num_nodes)
    ref_graph = build_coo_graph(edge_index, num_nodes, device)

    x_triton = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
    x_ref = x_triton.detach().clone().requires_grad_(True)

    x_normed = F.layer_norm(x_triton, (feature_dim,))
    triton_out = triton_layer(x_normed, wsb_graph)
    ref_out = ref_layer(x_ref, ref_graph)
    grad_output = torch.randn_like(triton_out)

    triton_out.backward(grad_output)
    ref_out.backward(grad_output)

    assert x_triton.grad is not None, "No Triton gradient"
    assert x_ref.grad is not None, "No reference gradient"
    assert not x_triton.grad.isnan().any(), "Triton grad contains NaN"
    # Relaxed tolerance due to fp16 internal computation
    assert torch.allclose(x_triton.grad, x_ref.grad, atol=5e-3, rtol=5e-3), (
        f"Triton vs torch_native backward mismatch: "
        f"max|diff|={(x_triton.grad - x_ref.grad).abs().max().item():.3e}, "
        f"mean|diff|={(x_triton.grad - x_ref.grad).abs().mean().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Low-precision: CUDA vs CUDA fp32 / Triton vs Triton fp32
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["cuda", "triton_block_sparse"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [512])
@pytest.mark.parametrize("heads", [2, 8])
def test_gt_low_precision_forward(backend, dtype, num_nodes, feature_dim, heads):
    """Low-precision forward: backend at fp16/bf16 vs same backend fp32."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend(backend)

    cuda_layer = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)
    cuda_layer_full_precision = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)

    share_gt_weights(cuda_layer, cuda_layer_full_precision)
    cuda_layer = cuda_layer.to(dtype)

    if backend == "cuda":
        cuda_graph = build_cuda_graph(edge_index, num_nodes)
    else:
        cuda_graph = build_wsb_graph(edge_index, num_nodes)

    x = torch.randn(num_nodes, feature_dim, device=device)

    cuda_out_full_precision = cuda_layer_full_precision(x, cuda_graph)
    cuda_out = cuda_layer(x.to(dtype), cuda_graph)

    cuda_f32 = cuda_out_full_precision.float()
    cuda_casted_from_low_to_fp32 = cuda_out.float()

    assert not cuda_f32.isnan().any(), "CUDA output contains NaN"
    assert torch.allclose(cuda_f32, cuda_casted_from_low_to_fp32, atol=5e-2, rtol=5e-2), (
        f"Low-precision CUDA forward mismatch ({dtype}): "
        f"max|diff|={(cuda_f32 - cuda_casted_from_low_to_fp32).abs().max().item():.3e}, "
        f"mean|diff|={(cuda_f32 - cuda_casted_from_low_to_fp32).abs().mean().item():.3e}"
    )


@pytest.mark.parametrize("backend", ["cuda", "triton_block_sparse"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [512])
@pytest.mark.parametrize("heads", [2, 8])
def test_gt_low_precision_backward(backend, dtype, num_nodes, feature_dim, heads):
    """Low-precision backward: backend at fp16/bf16 vs same backend fp32 gradients."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend(backend)

    cuda_layer = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)
    cuda_layer_full_precision = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device)

    share_gt_weights(cuda_layer, cuda_layer_full_precision)
    cuda_layer = cuda_layer.to(dtype)

    if backend == "cuda":
        cuda_graph = build_cuda_graph(edge_index, num_nodes)
    else:
        cuda_graph = build_wsb_graph(edge_index, num_nodes)

    x_cuda = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype, requires_grad=True).to(dtype)
    x_cuda_full_precision = x_cuda.detach().float().clone().requires_grad_(True)

    cuda_out = cuda_layer(x_cuda, cuda_graph)
    cuda_out_full_precision = cuda_layer_full_precision(x_cuda_full_precision, cuda_graph)

    grad_output = torch.randn_like(cuda_out)
    grad_output_fp = grad_output.detach().clone().float()

    cuda_out.backward(grad_output)
    cuda_out_full_precision.backward(grad_output_fp)

    assert x_cuda.grad is not None, "No low precision gradient"
    assert x_cuda_full_precision.grad is not None, "No full precision gradient"
    assert torch.allclose(x_cuda.grad.float(), x_cuda_full_precision.grad.float(), atol=1e-1, rtol=1e-1), (
        f"Low-precision CUDA backward mismatch ({dtype}): "
        f"max|diff|={(x_cuda.grad.float() - x_cuda_full_precision.grad.float()).abs().max().item():.3e}, "
        f"mean|diff|={(x_cuda.grad.float() - x_cuda_full_precision.grad.float()).abs().mean().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Pipeline correctness: pipeline_stages > 0 vs pipeline_stages == 0 (baseline)
# ---------------------------------------------------------------------------

_GT_PIPE_TOL = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
}

# "light_only" is undirected (make_undirected_graph): exercises the
# undirected backward kernel and the W<=4 light-bucket forward path.
# "with_heavy" is directed (make_hub_graph): exercises the directed AL/R-like
# backward kernel and the W=8/16/32 heavy-bucket forward path.
_GT_PIPE_GRAPHS = ["light_only", "with_heavy"]


def _make_gt_pipeline_graph(kind: str, num_nodes: int, device: str = "cuda"):
    if kind == "light_only":
        edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)
        return build_cuda_graph_bucketed(edge_index, num_nodes)
    edge_index = make_hub_graph(num_nodes, device=device)
    return build_cuda_graph_bucketed(edge_index, num_nodes, heavy_degree_threshold=32)


@pytest.mark.parametrize("graph_kind", _GT_PIPE_GRAPHS)
@pytest.mark.parametrize("pipeline_stages", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gt_pipeline_vs_baseline_forward(graph_kind, pipeline_stages, dtype):
    """Forward: pipeline (any stage count) must match the pipeline_stages=0 baseline."""
    device = "cuda"
    torch.manual_seed(42)
    num_nodes = 1000
    feature_dim = 256
    heads = 4

    cuda_graph = _make_gt_pipeline_graph(graph_kind, num_nodes, device)
    cuda_backend = BackendRegistry.get_backend("cuda")

    layer = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device).to(dtype)

    x = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype)

    layer.configure(forward_pipeline_stages=0)
    out_baseline = layer(x, cuda_graph)

    layer.configure(forward_pipeline_stages=pipeline_stages)
    out_pipeline = layer(x, cuda_graph)

    assert not out_pipeline.isnan().any(), "Pipeline output contains NaN"
    assert torch.allclose(out_pipeline, out_baseline, **_GT_PIPE_TOL[dtype]), (
        f"GT pipeline(stages={pipeline_stages}, {graph_kind}) forward mismatch ({dtype}): "
        f"max|diff|={(out_pipeline - out_baseline).abs().max().item():.3e}, "
        f"mean|diff|={(out_pipeline - out_baseline).abs().mean().item():.3e}"
    )


@pytest.mark.parametrize("graph_kind", _GT_PIPE_GRAPHS)
@pytest.mark.parametrize("pipeline_stages", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gt_backward_pipeline_vs_baseline(graph_kind, pipeline_stages, dtype):
    """Backward kernels' own pipeline: gradients must match between
    backward_pipeline_stages > 0 and the backward_pipeline_stages=0 baseline.
    """
    device = "cuda"
    torch.manual_seed(42)
    num_nodes = 1000
    feature_dim = 256
    heads = 4

    cuda_graph = _make_gt_pipeline_graph(graph_kind, num_nodes, device)
    cuda_backend = BackendRegistry.get_backend("cuda")

    layer = cuda_backend.create_conv("gt", feature_dim=feature_dim, heads=heads).to(device).to(dtype)

    x_base = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype, requires_grad=True)
    x_pipe = x_base.detach().clone().requires_grad_(True)

    layer.configure(backward_pipeline_stages=0)
    out_base = layer(x_base, cuda_graph)
    out_base.sum().backward()

    layer.configure(backward_pipeline_stages=pipeline_stages)
    out_pipe = layer(x_pipe, cuda_graph)
    out_pipe.sum().backward()

    assert x_base.grad is not None and x_pipe.grad is not None
    assert not x_pipe.grad.isnan().any(), "Pipeline grad contains NaN"
    assert torch.allclose(out_pipe, out_base, **_GT_PIPE_TOL[dtype]), (
        f"GT backward pipeline(stages={pipeline_stages}, {graph_kind}) forward mismatch ({dtype}): "
        f"max|diff|={(out_pipe - out_base).abs().max().item():.3e}"
    )
    assert torch.allclose(x_pipe.grad, x_base.grad, **_GT_PIPE_TOL[dtype]), (
        f"GT backward pipeline(stages={pipeline_stages}, {graph_kind}) grad mismatch ({dtype}): "
        f"max|diff|={(x_pipe.grad - x_base.grad).abs().max().item():.3e}"
    )
