import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = Path(__file__).parent
build_path = path.parent.parent.parent / "build/dfgnn"
build_path.mkdir(parents=True, exist_ok=True)

if os.environ.get("CUDA_HOME") is None:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

cuda_path = os.environ["CUDA_PATH"]

sources = [
    "csrc/fused_gtconv/fused_gtconv.cpp",
    "csrc/fused_gtconv/fused_gtconv_backward.cu",
    "csrc/fused_gtconv/fused_gtconv_tiling.cu",
    "csrc/util/indicator.cc",
    "csrc/util/indicator.cu",
]

extra_include_path = ["csrc/util/"]

dfgnn_tiling_ops = load(
    # NOTE: C++ sources use `PYBIND11_MODULE(fused_gtconv, m)` (and `ind`).
    # `load(name=...)` must match the exported `PyInit_<name>` symbol.
    name="fused_gtconv_tiling",
    build_directory=str(build_path),
    # Don't override `_GLIBCXX_USE_CXX11_ABI`; PyTorch already provides the correct one.
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
    extra_include_paths=[f"{cuda_path}/include", str(path)] + extra_include_path,
    sources=[str(path / s) for s in sources],
    verbose=True,
)
