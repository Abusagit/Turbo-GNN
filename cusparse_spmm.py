import torch
from torch.utils.cpp_extension import load
import os

# CUTLASS_PATH = '/workspace/cutlass'

path = __file__.replace('cusparse_spmm.py', '')

sources = ["cusparse_spmm.cpp"]
cuda_kernels = load(name="cuda_kernels", extra_cflags=["-O3"], extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "-arch=sm_80", "--generate-line-info", "-lcusparse"],
                    # extra_include_paths=[
                    #     CUTLASS_PATH + '/include', CUTLASS_PATH + '/tools/util/include'],
                    extra_include_paths=[
                        # os.path.join(os.environ['CONDA_PREFIX'], 'include'),
                        # os.path.join(os.environ['CONDA_PREFIX'], 'lib'),
                        "/usr/local/cuda-12.1/include/",
                    ],
                    sources=[path + s for s in sources],
                    # sources="",
                    verbose=True)


def csr_SPMM(indptr, indices, features, algorithm=-1, use_cache=True):

    # print(f"{indptr.shape=} {indices.shape=} {features.shape=}")

    out = cuda_kernels.csr_SPMM(indptr, indices, features, algorithm, use_cache)
    torch.cuda.synchronize()

    # print("AFTER", features.shape, out.shape)
    return out


def find_best_algorithm(indptr, indices, features):
    """Find the best cuSPARSE algorithm for a given graph structure."""
    indptr_gpu = indptr.to(features.device).to(torch.int32)
    indices_gpu = indices.to(features.device).to(torch.int32)
    features = features.to(torch.float32)

    return cuda_kernels.find_best_algorithm(indptr_gpu, indices_gpu, features)


def clear_cache():
    """Clear the internal graph structure cache."""
    cuda_kernels.clear_graph_cache()
