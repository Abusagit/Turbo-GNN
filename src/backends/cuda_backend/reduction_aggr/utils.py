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

from src.backends.base import TunableKernel, TunableParam, with_autotune
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

path = __file__.replace("utils.py", "")

sources = ["reduction_aggr.cu", "reduction_aggr_base.cu"]
repo_root_path = Path(__file__).parent.parent.parent.parent.parent
build_path = repo_root_path / "build/reduction_aggr"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

WARP_SIZE = 32
FOUR_BYTES_CONSTANT = 4

reduction_aggr_cuda = load(
    name="reduction_aggr_cuda",
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


def reduction_aggr_forward_partitioned(
    edge_ptr,
    edge_idx,
    X,
    light,
    heavy,
    warps_per_block,
    edges_per_block_heavy_nodes,
    use_2d_kernel=False,
    features_per_block=32,
    tiles_y=8,
    reduce="min",
):
    return reduction_aggr_cuda.reduction_aggr_forward_partitioned(
        edge_ptr,
        edge_idx,
        X,
        light,
        heavy,
        131070,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel,
        features_per_block,
        tiles_y,
        reduce,
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


class ReductionAggrFunction(torch.autograd.Function):
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
        use_2d_kernel=False,
        features_per_block=32,
        tiles_y=8,
        reduce="min",
    ):
        if torch.is_autocast_enabled():
            X = X.to(torch.get_autocast_gpu_dtype())

        num_of_threads_invoked = WARP_SIZE * warps_per_block
        num_features_per_thread = FOUR_BYTES_CONSTANT // X.dtype.itemsize  # 1, 2, 4, ...

        num_threads_needed = ceil(X.shape[-1] / num_features_per_thread)

        if num_threads_needed < num_of_threads_invoked:
            warps_per_block_needed = ceil(num_threads_needed / WARP_SIZE)
            warnings.warn(
                f"Number of threads involved for ReductionAggr is {num_of_threads_invoked} "
                f"({warps_per_block} warps per thread block requested). "
                f"However, number of threads needed is {num_threads_needed} "
                f"({warps_per_block_needed} warps). Setting this value instead."
            )

            warps_per_block = warps_per_block_needed
            if warps_per_block not in {1, 2, 4, 8, 16, 32, 64}:
                warps_per_block = next_power_of_two(warps_per_block)

        out, arg_idx = reduction_aggr_cuda.reduction_aggr_forward_partitioned(
            edge_ptr,
            edge_idx,
            X,
            light,
            heavy,
            max_degree,
            warps_per_block,
            edges_per_block_heavy_nodes,
            use_2d_kernel,
            features_per_block,
            tiles_y,
            reduce,
        )
        ctx.save_for_backward(arg_idx)
        ctx.num_src_nodes = X.size(0)
        ctx.warps_per_block = warps_per_block
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out: torch.Tensor):
        (arg_idx,) = ctx.saved_tensors
        num_src_nodes = ctx.num_src_nodes
        grad_x = reduction_aggr_cuda.reduction_aggr_backward(grad_out, arg_idx, num_src_nodes, ctx.warps_per_block)
        return None, None, grad_x, None, None, None, None, None, None, None, None, None


class ReductionAggrKernel(TunableKernel):
    """Tunable kernel callable for reduction aggregation (min/max)."""

    def __init__(self, reduce: str = "min", **kwargs):
        super().__init__()
        self.reduce = reduce
        self.forward_warps_per_block = kwargs.get("warps_per_block", 8)
        self.forward_edges_per_block_heavy_nodes = kwargs.get("edges_per_block_heavy_nodes", 128)
        self.forward_use_2d_kernel = kwargs.get("use_2d_kernel", False)
        self.forward_features_per_block = kwargs.get("features_per_block", 32)
        self.forward_tiles_y = kwargs.get("tiles_y", 8)

    def _execute(self, graph, x, **kwargs):
        return ReductionAggrFunction.apply(
            graph.forward_indptr,
            graph.forward_indices,
            x,
            graph.light_nodes,
            graph.heavy_nodes,
            graph.max_degree,
            self.forward_warps_per_block,
            self.forward_edges_per_block_heavy_nodes,
            self.forward_use_2d_kernel,
            self.forward_features_per_block,
            self.forward_tiles_y,
            self.reduce,
        )

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_warps_per_block", [1, 2, 4, 8, 16, 32], default=8),
            TunableParam("forward_edges_per_block_heavy_nodes", [32, 64, 128, 256, 512, 1024, 2048], default=128),
            TunableParam("forward_use_2d_kernel", [True, False], default=False),
            TunableParam("forward_features_per_block", [32, 64, 128, 256], default=32),
            TunableParam("forward_tiles_y", [2, 4, 8, 16], default=128),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99, 0.999], default=-1),
        ]


@with_autotune(ReductionAggrKernel, init_params=("reduce",))
def reduction_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    X: torch.Tensor,
    warps_per_block: int = 8,
    edges_per_block_heavy_nodes: int = 128,
    use_2d_kernel: bool = False,
    features_per_block: int = 32,
    tiles_y: int = 8,
    reduce: str = "min",
) -> torch.Tensor:
    return ReductionAggrFunction.apply(
        graph.forward_indptr,
        graph.forward_indices,
        X,
        graph.light_nodes,
        graph.heavy_nodes,
        graph.max_degree,
        warps_per_block,
        edges_per_block_heavy_nodes,
        use_2d_kernel,
        features_per_block,
        tiles_y,
        reduce,
    )
