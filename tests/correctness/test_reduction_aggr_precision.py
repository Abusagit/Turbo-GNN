import pytest
import torch

from src.backends.cuda_backend.reduction_aggr.utils import reduction_aggr, reduction_aggr_forward_partitioned
from src.data.datasets import load_pyg_single_graph


def zero_inf(x):
    return torch.where(torch.isinf(x), torch.zeros_like(x), x)


def create_simple_graph(device, num_nodes=100, num_edges=500):
    src = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int32)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.int32)

    indptr = torch.zeros(num_nodes + 1, device=device, dtype=torch.int32)
    for i in range(num_edges):
        indptr[dst[i].item() + 1] += 1
    indptr = torch.cumsum(indptr, dim=0).to(torch.int32)

    sorted_idx = torch.argsort(dst.long()).to(torch.int32)
    indices = src[sorted_idx].to(torch.int32)

    return indptr, indices


def partition_nodes(indptr: torch.Tensor, threshold=100):
    deg = indptr[1:] - indptr[:-1]
    light = torch.nonzero(deg <= threshold, as_tuple=True)[0].to(torch.int32).to(indptr.device)
    heavy = torch.nonzero(deg > threshold, as_tuple=True)[0].to(torch.int32).to(indptr.device)
    return light, heavy


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
    out = reduction_aggr(
        indptr,
        indices,
        x,
        light,
        heavy,
        131070,
        warps,
        epb,
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
def test_forward_matches_fp32_reference(dtype, num_features, reduce, use_2d_kernel):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if use_2d_kernel and dtype == torch.float64:
        pytest.skip("2D kernel does not support float64")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 200
    E = 1000
    indptr, indices = create_simple_graph(device, N, E)
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
def test_backward_matches_fp32_reference(dtype, num_features, reduce, use_2d_kernel):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    N = 200
    E = 1000
    indptr, indices = create_simple_graph(device, N, E)
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

    out = reduction_aggr(
        indptr,
        indices,
        x,
        light,
        heavy,
        131070,
        8,
        128,
        use_2d_kernel=use_2d_kernel,
        features_per_block=32,
        tiles_y=8,
        reduce=reduce,
    )
    out_ref = reduction_aggr(
        indptr,
        indices,
        x_ref,
        light,
        heavy,
        131070,
        8,
        128,
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
