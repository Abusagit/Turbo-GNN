import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

path = Path(__file__).parent

all_sources = [
    "cuda/bindings.cpp",  # Only this file compiles full torch headers
    "cuda/aggregate_kernel.cu",
    "cuda/format_kernel.cu",
    "cuda/gcn_kernel.cu",
    "cuda/gat_kernel.cu",
]

repo_root_path = Path(__file__).parent.parent.parent.parent
build_path = repo_root_path / "build/fuseGNN"

extra_cflags = ["-O3", "-fPIC"]
extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-arch=sm_80",
    "-Xcompiler",
    "-fPIC",
    "--expt-relaxed-constexpr",
]
extra_ldflags = ["-lcusparse"]

_fgnn_ops = None


def _get_fgnn_ops():
    global _fgnn_ops
    if _fgnn_ops is None:
        build_path.mkdir(parents=True, exist_ok=True)
        _fgnn_ops = load(
            name="fgnn_ops",
            build_directory=str(build_path),
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=[str(path / "cuda"), "/usr/local/cuda/include"],
            sources=[str(path / s) for s in all_sources],
            verbose=True,
            with_cuda=True,
        )
    return _fgnn_ops


class _LazyOps:
    """Proxy that defers JIT compilation until first attribute access."""

    def __getattr__(self, name):
        return getattr(_get_fgnn_ops(), name)


fgnn_ops = _LazyOps()
fgnn_agg = fgnn_ops
fgnn_format = fgnn_ops
fgnn_gcn = fgnn_ops
fgnn_gat = fgnn_ops

__all__ = ["fgnn_ops", "fgnn_agg", "fgnn_format", "fgnn_gcn", "fgnn_gat"]
