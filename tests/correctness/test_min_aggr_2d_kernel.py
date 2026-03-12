import pytest
import torch

from src.backends.cuda_backend.reduction_aggr.utils import reduction_aggr, reduction_aggr_forward_partitioned
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets


def zero_inf(x):
    return torch.where(torch.isinf(x), torch.zeros_like(x), x)


def create_simple_graph(device, num_nodes=100, num_edges=500, index_dtype=torch.int32):
    src = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int64)

    indptr = torch.zeros(num_nodes + 1, device=device, dtype=torch.int64)
    for i in range(num_edges):
        indptr[dst[i].item() + 1] += 1
    indptr = torch.cumsum(indptr, dim=0)

    sorted_idx = torch.argsort(dst)
    indices = src[sorted_idx]

    return indptr.to(index_dtype), indices.to(index_dtype)


def partition_nodes(indptr: torch.Tensor, threshold=100):
    deg = indptr[1:] - indptr[:-1]
    index_dtype = indptr.dtype
    light = torch.nonzero(deg <= threshold, as_tuple=True)[0].to(index_dtype).to(indptr.device)
    heavy = torch.nonzero(deg > threshold, as_tuple=True)[0].to(index_dtype).to(indptr.device)
    return light, heavy


def make_graph_repr(indptr, indices, light, heavy):
    """Wrap raw tensors into AdjacencyForwardBackwardWithNodeBuckets."""
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=indptr,
        forward_indices=indices,
        backward_indptr=indptr,
        backward_indices=indices,
        forward_light_nodes=light,
        forward_heavy_nodes=heavy,
        backward_light_nodes=light,
        backward_heavy_nodes=heavy,
    )


def run_forward(indptr, indices, x, light, heavy, warps=8, epb=128, use_2d=False, fpb=32, tiles=8):
    out, argmin = reduction_aggr_forward_partitioned(
        indptr, indices, x, light, heavy, warps, epb, use_2d_kernel=use_2d, features_per_block=fpb, tiles_y=tiles
    )
    out = zero_inf(out)
    return out, argmin


def run_backward(indptr, indices, x, light, heavy, warps=8, epb=128, use_2d=False, fpb=32, tiles=8):
    graph = make_graph_repr(indptr, indices, light, heavy)
    out = reduction_aggr(
        graph,
        x,
        warps_per_block=warps,
        edges_per_block_heavy_nodes=epb,
        use_2d_kernel=use_2d,
        features_per_block=fpb,
        tiles_y=tiles,
    )
    out = zero_inf(out)

    grad_out = torch.ones_like(out)
    out.backward(grad_out)
    return x.grad.detach()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_features", [32, 64, 128])
@pytest.mark.parametrize("fpb,tiles", [(32, 4), (32, 8), (64, 8)])
def test_2d_kernel_vs_atomic_kernel(dtype, num_features, fpb, tiles):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 300
    E = 2000
    indptr, indices = create_simple_graph(device, N, E)
    light, heavy = partition_nodes(indptr, threshold=50)

    x_fwd = torch.randn(N, num_features, device=device, dtype=dtype)

    out_2d, argmin_2d = run_forward(indptr, indices, x_fwd, light, heavy, use_2d=True, fpb=fpb, tiles=tiles)
    out_atomic, argmin_atomic = run_forward(indptr, indices, x_fwd, light, heavy, use_2d=False)

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    else:
        atol, rtol = 1e-2, 1e-2

    torch.testing.assert_close(
        out_2d,
        out_atomic,
        atol=atol,
        rtol=rtol,
        msg=f"2D kernel vs atomic forward mismatch for dtype={dtype}, fpb={fpb}, tiles={tiles}",
    )

    matches = (argmin_2d == argmin_atomic).float().mean()
    assert matches > 0.95, f"Argmin match rate too low: {matches:.2%}"

    x_2d = torch.randn(N, num_features, device=device, dtype=dtype, requires_grad=True)
    x_atomic = x_2d.detach().clone().requires_grad_(True)

    graph = make_graph_repr(indptr, indices, light, heavy)
    out_2d_grad = reduction_aggr(
        graph,
        x_2d,
        warps_per_block=8,
        edges_per_block_heavy_nodes=128,
        use_2d_kernel=True,
        features_per_block=fpb,
        tiles_y=tiles,
    )
    out_2d_grad = zero_inf(out_2d_grad)

    out_atomic_grad = reduction_aggr(
        graph,
        x_atomic,
        warps_per_block=8,
        edges_per_block_heavy_nodes=128,
        use_2d_kernel=False,
        features_per_block=fpb,
        tiles_y=tiles,
    )
    out_atomic_grad = zero_inf(out_atomic_grad)

    grad_out = torch.ones_like(out_2d_grad)
    out_2d_grad.backward(grad_out)
    out_atomic_grad.backward(grad_out.clone())

    torch.testing.assert_close(
        x_2d.grad,
        x_atomic.grad,
        atol=atol,
        rtol=rtol,
        msg=f"2D kernel vs atomic backward mismatch for dtype={dtype}, fpb={fpb}, tiles={tiles}",
    )


@pytest.mark.parametrize("tiles_y", [1, 2, 4, 8, 16])
def test_2d_kernel_tiles_y_validation(tiles_y):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 100
    E = 500
    F = 64

    indptr, indices = create_simple_graph(device, N, E)
    light, heavy = partition_nodes(indptr, threshold=30)

    x = torch.randn(N, F, device=device, dtype=torch.float32)

    out, argmin = run_forward(indptr, indices, x, light, heavy, use_2d=True, fpb=32, tiles=tiles_y)

    assert out.shape == (N, F), "Output shape mismatch"
    assert argmin.shape == (N, F), "Argmin shape mismatch"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
