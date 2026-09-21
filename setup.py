import os
import re
import sys
from pathlib import Path

from packaging.version import Version, parse
from setuptools import find_packages, setup

PACKAGE_NAME = "skewgnn"
this_dir = os.path.dirname(os.path.abspath(__file__))

FORCE_BUILD = os.getenv("SKEWGNN_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("SKEWGNN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("SKEWGNN_FORCE_CXX11_ABI", "FALSE") == "TRUE"

UNDEFINE_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_ninja_on_path():
    """Make the `ninja` build requirement visible to torch."""
    import shutil

    if shutil.which("ninja"):
        return True

    candidates = []
    try:
        import ninja

        candidates.append(ninja.BIN_DIR)
    except (ImportError, AttributeError):
        pass
    candidates.append(os.path.dirname(sys.executable))

    for bindir in candidates:
        if bindir and os.path.isfile(os.path.join(bindir, "ninja")):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            return True

    print(
        "WARNING: ninja was not found. The extension will be compiled with the "
        "serial distutils backend (MAX_JOBS is ignored) and the build will be "
        "several times slower. Run `pip install ninja` into the build environment.",
        file=sys.stderr,
    )
    return False


def get_package_version():
    """Read version from pyproject.toml."""
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
    return version


# ---------------------------------------------------------------------------
# CUDA extension setup
# ---------------------------------------------------------------------------
ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

        ensure_ninja_on_path()

        # NinjaBuildExtension — auto-calculates MAX_JOBS to prevent OOM
        class NinjaBuildExtension(BuildExtension):
            def __init__(self, *args, **kwargs):
                if not os.environ.get("MAX_JOBS"):
                    import psutil

                    max_num_jobs_cores = max(1, os.cpu_count() // 2)  # type: ignore
                    free_memory_gb = psutil.virtual_memory().available / (1024**3)
                    # ~5GB per NVCC thread, assume 2 NVCC threads
                    max_num_jobs_memory = max(1, int(free_memory_gb / (5 * 2)))
                    max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                    print(
                        f"Auto set MAX_JOBS to `{max_jobs}`. "
                        "If you see memory pressure, use a lower MAX_JOBS=N value."
                    )
                    os.environ["MAX_JOBS"] = str(max_jobs)
                super().__init__(*args, **kwargs)
                # BuildExtension flips use_ninja off with only a log warning if
                # it cannot find the binary; surface that instead of shipping a
                # silently serial build.
                print(f"skewgnn: ninja backend {'ENABLED' if self.use_ninja else 'DISABLED (serial build!)'}")

        if FORCE_CXX11_ABI:
            torch._C._GLIBCXX_USE_CXX11_ABI = True

        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 8.6 8.9 9.0"

        # Find headers/libs from pip-installed nvidia packages
        _extra_include = []
        _extra_libdir = []
        try:
            import nvidia

            _nvidia_root = nvidia.__path__[0]
            for _pkg in os.listdir(_nvidia_root):
                _inc = os.path.join(_nvidia_root, _pkg, "include")
                _lib = os.path.join(_nvidia_root, _pkg, "lib")
                if os.path.isdir(_inc):
                    _extra_include.append(_inc)
                if os.path.isdir(_lib):
                    _extra_libdir.append(_lib)
        except ImportError:
            pass

        # parallelise inside nvcc too:
        # need nvcc >= 11.2; gate on CUDA 12+ so older toolchains are safe.
        _nvcc_flags = [
            "-O3", "--use_fast_math", "--generate-line-info", "-std=c++20",
        ] + UNDEFINE_FLAGS
        _nvcc_threads = os.getenv("SKEWGNN_NVCC_THREADS", "4")
        if torch.version.cuda and parse(torch.version.cuda).major >= 12:
            _nvcc_flags += [f"--threads={_nvcc_threads}", f"--split-compile={_nvcc_threads}"]

        ext_modules = [
            CUDAExtension(
                name="skewgnn._C",
                sources=[
                    "csrc/skewgnn.cpp",
                    "csrc/reduction/reduction_aggr.cu",
                    "csrc/reduction/reduction_aggr_base.cu",
                    # The GSDDMM dispatch grid is sharded by op and dtype, one
                    # translation unit each, so nvcc compiles them in
                    # parallel (see csrc/gsddmm/gsddmm_launch.cuh).
                    "csrc/gsddmm/gsddmm_binding.cu",
                    "csrc/gsddmm/gsddmm_launch_dtype.cu",
                    "csrc/gsddmm/gsddmm_launch_add_f32.cu",
                    "csrc/gsddmm/gsddmm_launch_add_f16.cu",
                    "csrc/gsddmm/gsddmm_launch_add_bf16.cu",
                    "csrc/gsddmm/gsddmm_launch_sub_f32.cu",
                    "csrc/gsddmm/gsddmm_launch_sub_f16.cu",
                    "csrc/gsddmm/gsddmm_launch_sub_bf16.cu",
                    "csrc/gsddmm/gsddmm_launch_mul_f32.cu",
                    "csrc/gsddmm/gsddmm_launch_mul_f16.cu",
                    "csrc/gsddmm/gsddmm_launch_mul_bf16.cu",
                    "csrc/gsddmm/gsddmm_launch_div_f32.cu",
                    "csrc/gsddmm/gsddmm_launch_div_f16.cu",
                    "csrc/gsddmm/gsddmm_launch_div_bf16.cu",
                    "csrc/gsddmm/gsddmm_launch_dot_f32.cu",
                    "csrc/gsddmm/gsddmm_launch_dot_f16.cu",
                    "csrc/gsddmm/gsddmm_launch_dot_bf16.cu",
                    "csrc/gsddmm/gsddmm_launch_copy_f32.cu",
                    "csrc/gsddmm/gsddmm_launch_copy_f16.cu",
                    "csrc/gsddmm/gsddmm_launch_copy_bf16.cu",
                    # The attention kernels instantiate every head dim x warp count x
                    # pipeline depth per dtype; one translation unit per dtype lets
                    # nvcc compile them in parallel (see gatv2_kernel.cu).
                    "csrc/gatv2/gatv2_dispatch.cpp",
                    "csrc/gatv2/gatv2_shard_f32.cu",
                    "csrc/gatv2/gatv2_shard_f16.cu",
                    "csrc/gatv2/gatv2_shard_bf16.cu",
                    "csrc/gt/gt_dispatch.cpp",
                    "csrc/gt/gt_shard_f32.cu",
                    "csrc/gt/gt_shard_f16.cu",
                    "csrc/gt/gt_shard_bf16.cu",
                    "csrc/spmm/cusparse_spmm.cpp",
                    "csrc/spmm/gspmm.cu",
                    "csrc/spmm/edge_norm_kernels.cu",
                ],
                include_dirs=[os.path.join(this_dir, "csrc")] + _extra_include,
                library_dirs=_extra_libdir,
                libraries=["cusparse"],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++20"] + UNDEFINE_FLAGS,
                    "nvcc": _nvcc_flags,
                },
            ),
        ]
        cmdclass = {"build_ext": NinjaBuildExtension.with_options(use_ninja=True)}
    except (ImportError, OSError):
        if FORCE_BUILD:
            raise
        # No CUDA toolkit — sdist / metadata queries still work
        cmdclass = {}

setup(
    name="skewgnn",
    version=get_package_version(),
    packages=find_packages(include=["skewgnn*", "src*", "scripts*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
