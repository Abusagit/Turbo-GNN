import os
import platform
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import Version, parse
from setuptools import find_packages, setup

PACKAGE_NAME = "turbo_gnn"
this_dir = os.path.dirname(os.path.abspath(__file__))

FORCE_BUILD = os.getenv("TURBO_GNN_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("TURBO_GNN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("TURBO_GNN_FORCE_CXX11_ABI", "FALSE") == "TRUE"

BASE_WHEEL_URL = "https://github.com/Abusagit/Turbo-GNN/releases/download/{tag_name}/{wheel_name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_platform():
    """Returns the platform name as used in wheel filenames."""
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError(f"Unsupported platform: {sys.platform}")


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


def get_wheel_url():
    """Construct the GitHub Release URL for a matching pre-built wheel."""
    import torch

    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    pkg_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    torch_cuda_version = parse(torch.version.cuda)
    cuda_version = f"{torch_cuda_version.major}"

    wheel_filename = (
        f"{PACKAGE_NAME}-{pkg_version}"
        f"+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}"
        f"-{python_version}-{python_version}-{platform_name}.whl"
    )
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{pkg_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


# ---------------------------------------------------------------------------
# CachedWheelsCommand — downloads pre-built wheel from GitHub Releases
# ---------------------------------------------------------------------------
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class CachedWheelsCommand(_bdist_wheel):
    """
    When pip install runs, instead of compiling from source, first try to
    download a pre-built wheel from GitHub Releases.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL:", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            import shutil
            shutil.move(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            super().run()


# ---------------------------------------------------------------------------
# CUDA extension setup
# ---------------------------------------------------------------------------
ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

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
        cmdclass = {
            "bdist_wheel": CachedWheelsCommand,
            "build_ext": NinjaBuildExtension.with_options(use_ninja=True),
        }
    except (ImportError, OSError):
        if FORCE_BUILD:
            raise
        # No CUDA toolkit — sdist / metadata queries still work
        cmdclass = {"bdist_wheel": CachedWheelsCommand}
else:
    cmdclass = {"bdist_wheel": CachedWheelsCommand}

setup(
    name="turbo-gnn",
    version=get_package_version(),
    packages=find_packages(include=["turbo_gnn*", "src*", "scripts*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
