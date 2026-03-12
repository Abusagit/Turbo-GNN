import os
import re

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Read version from pyproject.toml
# ---------------------------------------------------------------------------
_pyproject = os.path.join(this_dir, "pyproject.toml")
try:
    import tomllib

    with open(_pyproject, "rb") as f:
        version = tomllib.load(f)["project"]["version"]
except Exception:
    # Fallback for Python 3.10 (no tomllib)
    with open(_pyproject) as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    version = m.group(1) if m else "0.0.0"

# ---------------------------------------------------------------------------
# Append local version tag (e.g. 0.1.0+torch2.4.1.cu121)
# ---------------------------------------------------------------------------
_local = os.environ.get("TURBO_GNN_LOCAL_VERSION")
if _local:
    version = f"{version}+{_local}"

# ---------------------------------------------------------------------------
# Default CUDA architectures if not already set (V100 → H100)
# ---------------------------------------------------------------------------
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0 7.5 8.0 8.6 8.9 9.0"

setup(
    name="turbo-gnn",
    version=version,
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
