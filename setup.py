import os
import re

from setuptools import find_packages, setup

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
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.0 8.6 8.9 9.0"

# ---------------------------------------------------------------------------
# CUDA extension — gracefully degrade when CUDA is unavailable (e.g. sdist)
# ---------------------------------------------------------------------------
ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Find cusparse headers/libs from pip-installed nvidia packages
    # (avoids requiring system CUDA cusparse-dev)
    _extra_include = []
    _extra_libdir = []
    try:
        import nvidia.cusparse as _nv_cusparse

        _nv_root = os.path.dirname(_nv_cusparse.__file__)
        _inc = os.path.join(_nv_root, "include")
        _lib = os.path.join(_nv_root, "lib")
        if os.path.isdir(_inc):
            _extra_include.append(_inc)
        if os.path.isdir(_lib):
            _extra_libdir.append(_lib)
    except ImportError:
        pass

    ext_modules = [
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
            include_dirs=[os.path.join(this_dir, "csrc")] + _extra_include,
            library_dirs=_extra_libdir,
            libraries=["cusparse"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "--generate-line-info"],
            },
        ),
    ]
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}
except (ImportError, OSError):
    # No CUDA toolkit — sdist / metadata queries still work
    pass

setup(
    name="turbo-gnn",
    version=version,
    packages=find_packages(include=["turbo_gnn*", "src*", "scripts*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
