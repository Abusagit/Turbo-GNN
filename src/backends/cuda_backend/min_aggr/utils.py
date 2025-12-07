import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

import glob
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = __file__.replace("utils.py", "")
sources = ["min_aggr.cu", "min_aggr_base.cu"]

repo_root_path = Path(__file__).parent.parent.parent.parent
build_path = repo_root_path / "build/min_aggr"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

min_aggr_cuda = load(
    name="min_aggr_cuda",
    build_directory=str(build_path),
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        # "-arch=sm_80",
        "--generate-line-info",
        "-lcusparse",
    ],
    extra_include_paths=[
        # *glob.glob(str(repo_root_path / ".venv/lib/python3.11/site-packages/**/include"), recursive=True),
        "/usr/local/cuda/include"
    ],
    sources=[path + s for s in sources],
    verbose=True,
)


def min_aggr_forward(edge_ptr: torch.Tensor, edge_idx: torch.Tensor, X: torch.Tensor):
    return min_aggr_cuda.min_aggr_forward(edge_ptr, edge_idx, X)


class MinAggrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_ptr: torch.Tensor, edge_idx: torch.Tensor, X: torch.Tensor):
        out, argmin = min_aggr_cuda.min_aggr_forward(edge_ptr, edge_idx, X)
        ctx.save_for_backward(argmin)
        ctx.num_src_nodes = X.size(0)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (argmin,) = ctx.saved_tensors
        num_src_nodes = ctx.num_src_nodes
        grad_x = min_aggr_cuda.min_aggr_backward(grad_out, argmin, num_src_nodes)
        return None, None, grad_x


def min_aggr(edge_ptr: torch.Tensor, edge_idx: torch.Tensor, X: torch.Tensor):
    return MinAggrFunction.apply(edge_ptr, edge_idx, X)


class MinAggr(torch.nn.Module):
    """DGL-backed MinAggregation wrapper."""

    def __init__(self, bias: bool = True, **kwargs) -> None:
        """Initialize a MinAggr layer using DGL.

        Args:
            bias (bool): Include bias.
            **kwargs (Any): Reserved for future options.
        """
        super().__init__(**kwargs)

    def forward(self, edge_ptr: torch.Tensor, edge_idx: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return min_aggr(edge_ptr, edge_idx, X)
