import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

import glob
import warnings
from math import ceil
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = __file__.replace("utils.py", "")

sources = ["min_aggr.cu", "min_aggr_base.cu"]
repo_root_path = Path(__file__).parent.parent.parent.parent.parent
build_path = repo_root_path / "build/min_aggr"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

WARP_SIZE = 32
FOUR_BYTES_CONSTANT = 4

min_aggr_cuda = load(
    name="min_aggr_cuda",
    build_directory=str(build_path),
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--generate-line-info",
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


def min_aggr_forward_partitioned(edge_ptr, edge_idx, X, light, heavy, warps_per_block, edges_per_block_heavy_nodes):
    return min_aggr_cuda.min_aggr_forward_partitioned(
        edge_ptr, edge_idx, X, light, heavy, 131070, warps_per_block, edges_per_block_heavy_nodes
    )


def next_power_of_two(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x


class MinAggrFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        edge_ptr: torch.Tensor,
        edge_idx: torch.Tensor,
        X: torch.Tensor,
        light,
        heavy,
        max_degree,
        warps_per_block,
        edges_per_block_heavy_nodes,
    ):
        if torch.is_autocast_enabled():
            X = X.to(torch.get_autocast_gpu_dtype())

        num_of_threads_invoked = WARP_SIZE * warps_per_block
        num_features_per_thread = FOUR_BYTES_CONSTANT // X.dtype.itemsize  # 1, 2, 4, ...

        num_threads_needed = ceil(X.shape[-1] / num_features_per_thread)
        warps_per_block_needed = ceil(num_threads_needed / WARP_SIZE)

        if num_threads_needed < num_of_threads_invoked:
            warnings.warn(
                f"Number of threads involved for MinAggr is {num_of_threads_invoked} "
                f"({warps_per_block} warps per thread block requested). "
                f"However, number of threads needed is {num_threads_needed} "
                f"({warps_per_block_needed} warps). Setting this value instead."
            )

            warps_per_block = warps_per_block_needed
            if warps_per_block not in {1, 2, 4, 8, 16, 32, 64}:
                warps_per_block = next_power_of_two(warps_per_block)

        out, argmin = min_aggr_cuda.min_aggr_forward_partitioned(
            edge_ptr, edge_idx, X, light, heavy, max_degree, warps_per_block, edges_per_block_heavy_nodes
        )
        ctx.save_for_backward(argmin)
        ctx.num_src_nodes = X.size(0)
        ctx.warps_per_block = warps_per_block
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out: torch.Tensor):
        (argmin,) = ctx.saved_tensors
        num_src_nodes = ctx.num_src_nodes
        grad_x = min_aggr_cuda.min_aggr_backward(grad_out, argmin, num_src_nodes, ctx.warps_per_block)
        return None, None, grad_x, None, None, None, None, None


def min_aggr(
    edge_ptr: torch.Tensor,
    edge_idx: torch.Tensor,
    X: torch.Tensor,
    light,
    heavy,
    max_degree: int,
    warps_per_block: int,
    edges_per_block_heavy_nodes: int,
):
    return MinAggrFunction.apply(
        edge_ptr, edge_idx, X, light, heavy, max_degree, warps_per_block, edges_per_block_heavy_nodes
    )
