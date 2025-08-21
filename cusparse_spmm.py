import torch
from torch.utils.cpp_extension import load
import os

path = __file__.replace('cusparse_spmm.py', '')

sources = ["cusparse_spmm.cpp", "edge_norm_kernels.cu.cu"]

cuda_kernels = load(name="cuda_kernels", extra_cflags=["-O3"], extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "-arch=sm_80", "--generate-line-info", "-lcusparse"],
                    extra_include_paths=[
                        "/usr/local/cuda-12.1/include/",
                    ],
                    sources=[path + s for s in sources],
                    verbose=True)


def csr_SPMM(indptr, indices, features, algorithm=-1, use_cache=True):
    """
    Backward compatibility function for unnormalized SpMM.
    
    Args:
        indptr: CSR row pointers
        indices: CSR column indices  
        features: Feature matrix
        algorithm: cuSPARSE algorithm ID (-1 for auto)
        use_cache: Whether to use caching
        
    Returns:
        Result of A @ features where A is the adjacency matrix
    """
    out = cuda_kernels.csr_SPMM(indptr, indices, features, algorithm, use_cache)

    return out


def csr_SPMM_normalized(indptr, indices, features, edge_weights=None, norm='none', 
                       algorithm=-1, use_cache=True):
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
        
    Returns:
        Normalized result of A @ features
    """
    # Handle edge weights
    if edge_weights is None:
        edge_weights_gpu = torch.empty(0, device=features.device, dtype=torch.float32)
    else:
        edge_weights_gpu = edge_weights.to(features.device).to(torch.float32)
    
    out = cuda_kernels.csr_SPMM_normalized(
        indptr, indices, features, edge_weights_gpu, 
        norm, algorithm, use_cache)

    return out


def find_best_algorithm(indptr, indices, features):
    """
    Backward compatibility function to find best algorithm for unnormalized SpMM.
    """
    indptr_gpu = indptr.to(features.device).to(torch.int32)
    indices_gpu = indices.to(features.device).to(torch.int32)
    features = features.to(torch.float32)
    
    return cuda_kernels.find_best_algorithm(indptr_gpu, indices_gpu, features)


def find_best_algorithm_normalized(indptr, indices, features, edge_weights=None, norm='none'):
    """
    Find the best cuSPARSE algorithm for a given graph structure with normalization.
    
    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        features: Feature matrix
        edge_weights: Optional edge weights
        norm: Normalization type
        
    Returns:
        Best algorithm ID
    """

    if edge_weights is None:
        edge_weights_gpu = torch.empty(0, device=features.device, dtype=torch.float32)
    else:
        edge_weights_gpu = edge_weights.to(features.device).to(torch.float32)
    
    return cuda_kernels.find_best_algorithm_normalized(
        indptr, indices, features, edge_weights_gpu, norm)


def clear_cache():
    """Clear the internal graph structure cache."""
    cuda_kernels.clear_graph_cache()
