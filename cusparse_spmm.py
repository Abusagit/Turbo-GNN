import torch
from torch.utils.cpp_extension import load
import os

# CUTLASS_PATH = '/workspace/cutlass'

path = __file__.replace('cusparse_spmm.py', '')

sources = ["cusparse_spmm.cpp"]
cuda_kernels = load(name="cuda_kernels", extra_cflags=["-O3"], extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "--generate-line-info", "-lnvrtc", "-lcusparse", "-lcutlass"],
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

def csr_SPMM(indptr, indices, features):
    print(f"{indptr.shape=} {indices.shape=} {features.shape=}")
    out = torch.empty_like(features)

    cuda_kernels.csr_SPMM(out, indptr, indices, features)
    torch.cuda.synchronize()

    print("AFTER", features.shape, out.shape)
    return out
