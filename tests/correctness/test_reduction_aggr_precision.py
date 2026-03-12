import pytest
import torch

from src.backends.cuda_backend.reduction_aggr.utils import reduction_aggr, reduction_aggr_forward_partitioned
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets
from src.data.datasets import load_pyg_single_graph


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


def run_forward(indptr, indices, x, light, heavy, warps=8, epb=128, reduce="min", use_2d=False, fpb=32, tiles=8):
    out, arg_idx = reduction_aggr_forward_partitioned(
        indptr,
        indices,
        x,
        light,
        heavy,
        warps,
        epb,
        use_2d_kernel=use_2d,
        features_per_block=fpb,
        tiles_y=tiles,
        reduce=reduce,
    )
    out = zero_inf(out)
    return out, arg_idx


def run_backward(indptr, indices, x, light, heavy, warps=8, epb=128, reduce="min", use_2d=False, fpb=32, tiles=8):
    graph = make_graph_repr(indptr, indices, light, heavy)
    out = reduction_aggr(
        graph,
        x,
        warps_per_block=warps,
        edges_per_block_heavy_nodes=epb,
        use_2d_kernel=use_2d,
        features_per_block=fpb,
        tiles_y=tiles,
        reduce=reduce,
    )
    out = zero_inf(out)

    grad_out = torch.ones_like(out)
    out.backward(grad_out)
    return x.grad.detach()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_features", [16, 64, 128])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("use_2d_kernel", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_forward_matches_fp32_reference(dtype, num_features, reduce, use_2d_kernel, index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if use_2d_kernel and dtype == torch.float64:
        pytest.skip("2D kernel does not support float64")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 200
    E = 1000
    indptr, indices = create_simple_graph(device, N, E, index_dtype=index_dtype)
    light, heavy = partition_nodes(indptr)

    x = torch.randn(N, num_features, device=device, dtype=dtype)
    x_ref = x.float()

    out, _ = run_forward(indptr, indices, x, light, heavy, reduce=reduce, use_2d=use_2d_kernel)
    out_ref, _ = run_forward(indptr, indices, x_ref, light, heavy, reduce=reduce, use_2d=use_2d_kernel)

    a = out.float()
    b = out_ref.float()

    if dtype == torch.float64:
        atol, rtol = 1e-6, 1e-5
    elif dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    else:
        atol, rtol = 1e-2, 1e-2

    kernel_type = "2D" if use_2d_kernel else "atomic"
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
        msg=f"Forward mismatch vs fp32 ref for dtype {dtype}, reduce={reduce}, kernel={kernel_type}",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_features", [16, 64, 128])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("use_2d_kernel", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_backward_matches_fp32_reference(dtype, num_features, reduce, use_2d_kernel, index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 200
    E = 1000
    indptr, indices = create_simple_graph(device, N, E, index_dtype=index_dtype)
    light, heavy = partition_nodes(indptr)

    x = torch.randn(N, num_features, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().float().requires_grad_(True)

    grad_x = run_backward(indptr, indices, x, light, heavy, reduce=reduce, use_2d=use_2d_kernel)
    grad_x_ref = run_backward(indptr, indices, x_ref, light, heavy, reduce=reduce, use_2d=use_2d_kernel)

    a = grad_x.float()
    b = grad_x_ref.float()

    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-3
    else:
        atol, rtol = 1e-2, 1e-2

    kernel_type = "2D" if use_2d_kernel else "atomic"
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
        msg=f"Backward mismatch vs fp32 ref for dtype {dtype}, reduce={reduce}, kernel={kernel_type}",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("use_2d_kernel", [False, True])
def test_real_dataset_matches_fp32_reference(dtype, reduce, use_2d_kernel):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    sample = load_pyg_single_graph(name="cora", graph_backend="csr", root="data", allow_random_split=True)

    indptr, indices, _ = sample.graph_repr
    indptr = indptr.to(device).to(torch.int32)
    indices = indices.to(device).to(torch.int32)

    N = sample.num_nodes
    F = 128

    light, heavy = partition_nodes(indptr, threshold=50)

    x = torch.randn(N, F, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().float().requires_grad_(True)

    graph = make_graph_repr(indptr, indices, light, heavy)
    out = reduction_aggr(
        graph,
        x,
        warps_per_block=8,
        edges_per_block_heavy_nodes=128,
        use_2d_kernel=use_2d_kernel,
        features_per_block=32,
        tiles_y=8,
        reduce=reduce,
    )
    out_ref = reduction_aggr(
        graph,
        x_ref,
        warps_per_block=8,
        edges_per_block_heavy_nodes=128,
        use_2d_kernel=use_2d_kernel,
        features_per_block=32,
        tiles_y=8,
        reduce=reduce,
    )

    out = zero_inf(out).float()
    out_ref = zero_inf(out_ref).float()

    if dtype == torch.float32:
        atol_fwd, rtol_fwd = 1e-5, 1e-4
        atol_bwd, rtol_bwd = 1e-3, 1e-2
    else:
        atol_fwd, rtol_fwd = 1e-2, 1e-2
        atol_bwd, rtol_bwd = 5e-2, 5e-2

    kernel_type = "2D" if use_2d_kernel else "atomic"
    torch.testing.assert_close(
        out,
        out_ref,
        atol=atol_fwd,
        rtol=rtol_fwd,
        msg=f"Forward mismatch on Cora vs fp32 ref for {dtype}, reduce={reduce}, kernel={kernel_type}",
    )

    grad_out = torch.ones_like(out_ref, device=device, dtype=torch.float32)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    gx = x.grad.detach().float()
    gx_ref = x_ref.grad.detach().float()

    torch.testing.assert_close(
        gx,
        gx_ref,
        atol=atol_bwd,
        rtol=rtol_bwd,
        msg=f"Backward mismatch on Cora vs fp32 ref for {dtype}, reduce={reduce}, kernel={kernel_type}",
    )


def _reinterpret_graph_unsigned(indptr, indices, light, heavy, unsigned_dtype):
    """Reinterpret signed int32 graph tensors as unsigned (zero-copy .view())."""
    return (
        indptr.view(unsigned_dtype),
        indices.view(unsigned_dtype),
        light.view(unsigned_dtype),
        heavy.view(unsigned_dtype),
    )


@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("use_2d_kernel", [False, True])
@pytest.mark.parametrize("unsigned_dtype", [torch.uint32])
def test_forward_unsigned_index(reduce, use_2d_kernel, unsigned_dtype):
    """Verify that unsigned index types produce the same results as signed int32."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N, E, F = 200, 1000, 64
    indptr, indices = create_simple_graph(device, N, E, index_dtype=torch.int32)
    light, heavy = partition_nodes(indptr)

    x = torch.randn(N, F, device=device, dtype=torch.float32)

    out_signed, _ = run_forward(indptr, indices, x, light, heavy, reduce=reduce, use_2d=use_2d_kernel)

    u_indptr, u_indices, u_light, u_heavy = _reinterpret_graph_unsigned(indptr, indices, light, heavy, unsigned_dtype)
    out_unsigned, _ = run_forward(u_indptr, u_indices, x, u_light, u_heavy, reduce=reduce, use_2d=use_2d_kernel)

    torch.testing.assert_close(
        zero_inf(out_unsigned),
        zero_inf(out_signed),
        atol=1e-5,
        rtol=1e-4,
        msg=f"Unsigned {unsigned_dtype} output differs from int32 for reduce={reduce}, 2d={use_2d_kernel}",
    )


@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("unsigned_dtype", [torch.uint32])
def test_backward_unsigned_index(reduce, unsigned_dtype):
    """Verify backward pass with unsigned index types."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N, E, F = 200, 1000, 64
    indptr, indices = create_simple_graph(device, N, E, index_dtype=torch.int32)
    light, heavy = partition_nodes(indptr)

    u_indptr, u_indices, u_light, u_heavy = _reinterpret_graph_unsigned(indptr, indices, light, heavy, unsigned_dtype)

    x = torch.randn(N, F, device=device, dtype=torch.float32, requires_grad=True)
    grad_x = run_backward(u_indptr, u_indices, x, u_light, u_heavy, reduce=reduce)

    assert grad_x is not None, "Gradient should be computed"
    assert grad_x.shape == x.shape, "Gradient shape mismatch"
    assert not torch.isnan(grad_x).any(), "Gradient contains NaN"


@pytest.mark.parametrize("warps", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("use_2d_kernel", [False, True])
def test_forward_block_sizes_vs_fp32_reference(warps, dtype, reduce, use_2d_kernel):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 256
    E = 2000
    F = 64

    indptr, indices = create_simple_graph(device, N, E)
    light, heavy = partition_nodes(indptr)

    x = torch.randn(N, F, device=device, dtype=dtype)
    x_ref = x.float()

    out, _ = run_forward(indptr, indices, x, light, heavy, warps=warps, epb=128, reduce=reduce, use_2d=use_2d_kernel)
    out_ref, _ = run_forward(
        indptr, indices, x_ref, light, heavy, warps=warps, epb=128, reduce=reduce, use_2d=use_2d_kernel
    )

    a = out.float()
    b = out_ref.float()

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2

    kernel_type = "2D" if use_2d_kernel else "atomic"
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
        msg=f"Forward mismatch vs fp32 ref for warps={warps}, dtype={dtype}, reduce={reduce}, kernel={kernel_type}",
    )
