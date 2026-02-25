import os

if os.environ.get("CUDA_HOME") is None:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

import glob
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = __file__.replace("utils.py", "")
sources = ["cusparse_spmm.cpp", "edge_norm_kernels.cu"]

repo_root_path = Path(__file__).parent.parent.parent.parent.parent
build_path = repo_root_path / "build/spmm_aggr"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

cuda_kernels = load(
    name="cuda_kernels",
    build_directory=str(build_path),
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_80",
        "--generate-line-info",
        "-lcusparse",
    ],
    extra_include_paths=[
        # *glob.glob(str(repo_root_path / ".venv/lib/python3.11/site-packages/**/include"), recursive=True),
        os.environ["CUDA_HOME"]
    ],
    sources=[path + s for s in sources],
    verbose=True,
)


def csr_SPMM_normalized(
    indptr,
    indices,
    features,
    edge_weights=None,
    norm="none",
    algorithm=-1,
    use_cache=True,
    do_transpose_a=False,
    block_dim=256,
):
    """
    Normalized SpMM operation supporting different GCN normalization schemes.
    Supports float32, float16, and bfloat16 features (mixed-precision: sparse
    values and compute always use float32 for accuracy).

    Args:
        indptr: CSR row pointers (int32)
        indices: CSR column indices (int32)
        features: Feature matrix (float32/float16/bfloat16)
        edge_weights: Optional edge weights tensor (float32). If None, uses all 1s.
        norm: Normalization type. One of:
            - 'none': No normalization (default)
            - 'right': Divide by in-degrees (averaging)
            - 'left': Divide by out-degrees (random walk)
            - 'both': Symmetric normalization (GCN paper)
        algorithm: cuSPARSE algorithm ID (-1 for auto)
        use_cache: Whether to use caching
        do_transpose_a: Whether to transpose A matrix before matmul.
        block_dim: Block dimension for the kernel.
    Returns:
        Normalized result of A @ features (same dtype as features)
    """
    # Edge weights are always float32 (for sparse matrix normalization values)
    if edge_weights is None:
        edge_weights_gpu = torch.empty(0, device=features.device, dtype=torch.float32)
    else:
        edge_weights_gpu = edge_weights.to(device=features.device, dtype=torch.float32)

    out = cuda_kernels.csr_SPMM_normalized(
        indptr, indices, features.contiguous(), edge_weights_gpu, norm, algorithm, use_cache, do_transpose_a, block_dim
    )

    return out


class _CudaSpMMConvFn(torch.autograd.Function):
    """cuSPARSE SpMM with AdjacencyForwardBackwardWithNodeBuckets graph format.

    Forward: forward_indptr/forward_indices (transposed CSR), do_transpose_a=False
    Backward: forward_indptr/forward_indices (transposed CSR), do_transpose_a=True

    Supports float32, float16, bfloat16 features (mixed-precision SpMM).
    """

    @staticmethod
    def forward(ctx, x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim):
        ctx.save_for_backward(forward_indptr, forward_indices)
        ctx.norm_type = norm_type
        ctx.cu_sparse_algorithm_id = cu_sparse_algorithm_id
        ctx.block_dim = block_dim

        return csr_SPMM_normalized(
            indptr=forward_indptr,
            indices=forward_indices,
            features=x,
            edge_weights=None,
            norm=norm_type,
            algorithm=cu_sparse_algorithm_id,
            do_transpose_a=False,
            block_dim=block_dim,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        forward_indptr, forward_indices = ctx.saved_tensors
        grad_x = csr_SPMM_normalized(
            indptr=forward_indptr,
            indices=forward_indices,
            features=grad_outputs[0],
            edge_weights=None,
            norm=ctx.norm_type,
            algorithm=ctx.cu_sparse_algorithm_id,
            do_transpose_a=True,
            block_dim=ctx.block_dim,
        )
        return grad_x, None, None, None, None, None


def spmm_aggr(x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim):
    return _CudaSpMMConvFn.apply(x, forward_indptr, forward_indices, norm_type, cu_sparse_algorithm_id, block_dim)
