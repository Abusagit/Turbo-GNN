import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

import glob
import typing as tp
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = __file__.replace("utils.py", "")

sources = ["dot_aggr_base.cu"]
repo_root_path = Path(__file__).parent.parent.parent.parent.parent
build_path = repo_root_path / "build/dot_aggr_test_reordering"

_dot_aggr_cuda = None


def _get_dot_aggr_cuda():
    global _dot_aggr_cuda
    if _dot_aggr_cuda is None:
        if not build_path.is_dir():
            build_path.mkdir(parents=True)
        _dot_aggr_cuda = load(
            name="dot_aggr_cuda",
            build_directory=str(build_path),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--generate-line-info",
            ],
            extra_include_paths=["/usr/local/cuda/include"],
            sources=[path + s for s in sources],
            verbose=True,
        )
    return _dot_aggr_cuda


class DotAggrFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        edge_ptr: torch.Tensor,
        edge_idx: torch.Tensor,
        X: torch.Tensor,
        kernel_kind,
        use_second_access,
        use_vectorized_loads,
    ):
        out = _get_dot_aggr_cuda().dot_aggr_forward(
            edge_ptr,
            edge_idx,
            X,
            kernel_kind,
            use_second_access,
            use_vectorized_loads,
        )
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out: torch.Tensor):
        raise NotImplementedError


def dot_aggr(
    edge_ptr: torch.Tensor,
    edge_idx: torch.Tensor,
    X: torch.Tensor,
    kernel_kind: tp.Literal[0, 1] = 0,
    use_second_access: bool = False,
    use_vectorized_loads: bool = False,
):
    return DotAggrFunction.apply(
        edge_ptr,
        edge_idx,
        X,
        kernel_kind,
        use_second_access,
        use_vectorized_loads,
    )
