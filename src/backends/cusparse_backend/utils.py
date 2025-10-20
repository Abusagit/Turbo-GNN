import glob
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = __file__.replace("utils.py", "")
sources = ["cusparse_spmm.cpp", "edge_norm_kernels.cu"]

repo_root_path = Path(__file__).parent.parent.parent.parent

cuda_kernels = load(
    name="cuda_kernels",
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80", "--generate-line-info", "-lcusparse"],
    extra_include_paths=glob.glob(
        str(repo_root_path / ".venv/lib/python3.11/site-packages/**/include"), recursive=True
    ),
    sources=[path + s for s in sources],
    verbose=True,
)


def csr_SPMM_normalized(
    indptr, indices, features, edge_weights=None, norm="none", algorithm=-1, use_cache=True, do_transpose_a=False
):
    """
    Normalized SpMM operation supporting different GCN normalization schemes.

    Args:
        indptr: CSR row pointers (int32)
        indices: CSR column indices (int32)
        features: Feature matrix (float32)
        edge_weights: Optional edge weights tensor. If None, uses all 1s.
        norm: Normalization type. One of:
            - 'none': No normalization (default)
            - 'right': Divide by in-degrees (averaging)
            - 'left': Divide by out-degrees (random walk)
            - 'both': Symmetric normalization (GCN paper)
        algorithm: cuSPARSE algorithm ID (-1 for auto)
        use_cache: Whether to use caching
        do_transpose_a: Whether to transpose A matrix before matmul.

    Returns:
        Normalized result of A @ features
    """
    # Handle edge weights
    if edge_weights is None:
        edge_weights_gpu = torch.empty(0, device=features.device, dtype=torch.float32)
    else:
        edge_weights_gpu = edge_weights.to(features.device).to(torch.float32)

    out = cuda_kernels.csr_SPMM_normalized(
        indptr, indices, features.contiguous(), edge_weights_gpu, norm, algorithm, use_cache, do_transpose_a
    )

    return out
