"""
Graph Transformer correctness tests:
  - CUDA backend vs DGL reference (fp32)
  - Triton backend vs DGL reference (fp32, noting Triton internally uses fp16)
  - Low-precision (fp16/bf16) variants against fp32 DGL reference

"""

import sys
from pathlib import Path

import dgl
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


def build_wsb_graph(edge_index: torch.Tensor, num_nodes: int):
    """Build WSBFormat graph for Triton backend.

    Steps: normalize_adj -> to_sparse_csr -> WSBFormat.build_wsb_format -> .cuda()
    """
    adj_sparse = normalize_adj(
        edge_index.cpu(),
        num_nodes=num_nodes,
        how="both",
        add_self_loops=False,
    )
    adj_csr = adj_sparse.to_sparse_csr()
    wsb = WSBFormat.build_wsb_format(adj_csr)
    return wsb.cuda()


def build_dgl_graph(edge_index: torch.Tensor, num_nodes: int, device: str = "cuda"):
    """Build a DGL graph from edge_index [2, E]."""
    src, dst = edge_index[0], edge_index[1]
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g.to(device)


def share_gt_weights(target_layer, dgl_layer):
    """Copy weights from DGL GT layer to target (CUDA/Triton) GT layer."""
    with torch.no_grad():
        dgl_w = dgl_layer.qkv_proj.weight.data
        dgl_b = dgl_layer.qkv_proj.bias.data

        new_w = dgl_w.clone()
        new_b = dgl_b.clone()

        target_layer.qkv_proj.weight.data.copy_(new_w)
        target_layer.qkv_proj.bias.data.copy_(new_b)


# ---------------------------------------------------------------------------
# fp32: CUDA vs DGL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [256, 512])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_cuda_vs_dgl_forward(num_nodes, feature_dim, heads):
    """fp32 forward: CUDA GT vs DGL GT (with Q/K swap)."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    dgl_backend = BackendRegistry.get_backend("dgl")

    cuda_layer = cuda_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)
    dgl_layer = dgl_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)

    share_gt_weights(cuda_layer, dgl_layer)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    dgl_graph = build_dgl_graph(edge_index, num_nodes, device)

    x = torch.randn(num_nodes, feature_dim, device=device)

    cuda_out = cuda_layer(x, cuda_graph)
    dgl_out = dgl_layer(x, dgl_graph)

    assert not cuda_out.isnan().any(), "CUDA output contains NaN"
    assert not dgl_out.isnan().any(), "DGL output contains NaN"
    assert torch.allclose(cuda_out, dgl_out, atol=1e-4, rtol=1e-4), (
        f"CUDA vs DGL forward mismatch: "
        f"max|diff|={(cuda_out - dgl_out).abs().max().item():.3e}, "
        f"mean|diff|={(cuda_out - dgl_out).abs().mean().item():.3e}"
    )


@pytest.mark.parametrize("num_nodes", [64, 200])
@pytest.mark.parametrize("feature_dim", [256, 512])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_cuda_vs_dgl_backward(num_nodes, feature_dim, heads):
    """fp32 backward: compare input gradients CUDA GT vs DGL GT."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend("cuda")
    dgl_backend = BackendRegistry.get_backend("dgl")

    cuda_layer = cuda_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)
    dgl_layer = dgl_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)

    share_gt_weights(cuda_layer, dgl_layer)

    cuda_graph = build_cuda_graph(edge_index, num_nodes)
    dgl_graph = build_dgl_graph(edge_index, num_nodes, device)

    x_cuda = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
    x_dgl = x_cuda.detach().clone().requires_grad_(True)

    cuda_out = cuda_layer(x_cuda, cuda_graph)
    dgl_out = dgl_layer(x_dgl, dgl_graph)

    grad_output = torch.randn_like(cuda_out)

    cuda_out.backward(grad_output)
    dgl_out.backward(grad_output)

    assert x_cuda.grad is not None, "No CUDA gradient"
    assert x_dgl.grad is not None, "No DGL gradient"
    assert not x_cuda.grad.isnan().any(), "CUDA grad contains NaN"
    # Slightly relaxed vs forward (1e-4) due to error accumulation in backward:
    # __expf fast math, atomic dK adds, and chain rule amplification
    assert torch.allclose(x_cuda.grad, x_dgl.grad, atol=1e-3, rtol=1e-3), (
        f"CUDA vs DGL backward mismatch: "
        f"max|diff|={(x_cuda.grad - x_dgl.grad).abs().max().item():.3e}, "
        f"mean|diff|={(x_cuda.grad - x_dgl.grad).abs().mean().item():.3e}"
    )


# # ---------------------------------------------------------------------------
# # fp32: Triton vs DGL (Triton internally converts fp32 -> fp16)
# # ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes", [48, 128])
@pytest.mark.parametrize("feature_dim", [256, 512])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_triton_vs_dgl_forward(num_nodes, feature_dim, heads):
    """fp32 forward: Triton GT vs DGL GT.

    Triton layer has NO layer_norm -> pre-apply F.layer_norm before Triton.
    Triton internally converts fp32 -> fp16, so outputs have fp16-level error.
    """
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    triton_backend = BackendRegistry.get_backend("triton_block_sparse")
    dgl_backend = BackendRegistry.get_backend("dgl")

    triton_layer = triton_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)
    dgl_layer = dgl_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)

    share_gt_weights(triton_layer, dgl_layer)

    wsb_graph = build_wsb_graph(edge_index, num_nodes)
    dgl_graph = build_dgl_graph(edge_index, num_nodes, device)

    x = torch.randn(num_nodes, feature_dim, device=device)

    # Triton has no layer_norm, so pre-apply it
    x_normed = F.layer_norm(x, (feature_dim,))
    triton_out = triton_layer(x_normed, wsb_graph)
    dgl_out = dgl_layer(x, dgl_graph)

    assert not triton_out.isnan().any(), "Triton output contains NaN"
    assert not dgl_out.isnan().any(), "DGL output contains NaN"
    # Relaxed tolerance due to Triton's internal fp16 conversion
    assert torch.allclose(triton_out, dgl_out, atol=5e-3, rtol=5e-3), (
        f"Triton vs DGL forward mismatch: "
        f"max|diff|={(triton_out - dgl_out).abs().max().item():.3e}, "
        f"mean|diff|={(triton_out - dgl_out).abs().mean().item():.3e}"
    )


@pytest.mark.parametrize("num_nodes", [48, 128])
@pytest.mark.parametrize("feature_dim", [256, 512])
@pytest.mark.parametrize("heads", [4, 8])
def test_gt_triton_vs_dgl_backward(num_nodes, feature_dim, heads):
    """fp32 backward: compare input gradients Triton GT vs DGL GT."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    triton_backend = BackendRegistry.get_backend("triton_block_sparse")
    dgl_backend = BackendRegistry.get_backend("dgl")

    triton_layer = triton_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)
    dgl_layer = dgl_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)

    share_gt_weights(triton_layer, dgl_layer)

    wsb_graph = build_wsb_graph(edge_index, num_nodes)
    dgl_graph = build_dgl_graph(edge_index, num_nodes, device)

    x_triton = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
    x_dgl = x_triton.detach().clone().requires_grad_(True)

    x_normed = F.layer_norm(x_triton, (feature_dim,))
    triton_out = triton_layer(x_normed, wsb_graph)
    dgl_out = dgl_layer(x_dgl, dgl_graph)
    grad_output = torch.randn_like(triton_out)

    triton_out.backward(grad_output)
    dgl_out.backward(grad_output)

    assert x_triton.grad is not None, "No Triton gradient"
    assert x_dgl.grad is not None, "No DGL gradient"
    assert not x_triton.grad.isnan().any(), "Triton grad contains NaN"
    # Relaxed tolerance due to fp16 internal computation
    assert torch.allclose(x_triton.grad, x_dgl.grad, atol=5e-3, rtol=5e-3), (
        f"Triton vs DGL backward mismatch: "
        f"max|diff|={(x_triton.grad - x_dgl.grad).abs().max().item():.3e}, "
        f"mean|diff|={(x_triton.grad - x_dgl.grad).abs().mean().item():.3e}"
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
    """Low-precision forward: CUDA GT at fp16/bf16 vs DGL fp32."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend(backend)

    cuda_layer = cuda_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)
    cuda_layer_full_precision = cuda_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)

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
    """Low-precision backward: CUDA GT at fp16/bf16 vs DGL fp32 gradients."""
    device = "cuda"
    torch.manual_seed(42)

    edge_index = make_undirected_graph(num_nodes, num_nodes * 5, device=device)

    cuda_backend = BackendRegistry.get_backend(backend)

    cuda_layer = cuda_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)
    cuda_layer_full_precision = cuda_backend.create_conv(
        "gt",
        feature_dim=feature_dim,
        heads=heads,
    ).to(device)

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
