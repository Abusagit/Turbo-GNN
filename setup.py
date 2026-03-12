import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="turbo-gnn",
    packages=find_packages(include=["turbo_gnn*", "src*", "scripts*"]),
    ext_modules=[
        CUDAExtension(
            name="turbo_gnn._C",
            sources=[
                "csrc/turbo_gnn.cpp",
                "csrc/reduction/reduction_aggr.cu",
                "csrc/reduction/reduction_aggr_base.cu",
                "csrc/gatv2/gatv2_kernel.cu",
                "csrc/gt/graph_transformer.cu",
                "csrc/spmm/cusparse_spmm.cpp",
                "csrc/spmm/edge_norm_kernels.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            libraries=["cusparse"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "--generate-line-info"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
