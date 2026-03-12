import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

import glob
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from src.backends.base import TunableKernel, TunableParam, with_autotune
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

path = __file__.replace("utils.py", "")
sources = ["gatv2_kernel.cu"]

repo_root_path = Path(__file__).parent.parent.parent.parent.parent
build_path = repo_root_path / "build/gatv2_backend"
if not build_path.is_dir():
    build_path.mkdir(parents=True)


gatv2_kernels = load(
    name="gatv2_kernels",
    build_directory=str(build_path),
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_80",
        "--generate-line-info",
    ],
    extra_include_paths=[
        # *glob.glob(str(repo_root_path / ".venv/lib/python3.11/site-packages/**/include"), recursive=True),
        "/usr/local/cuda/include"
    ],
    sources=[path + s for s in sources],
    verbose=True,
)


class gatv2_function(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        indptr_forward: torch.Tensor,
        indices_forward: torch.Tensor,
        indptr_backward: torch.Tensor,
        indices_backward: torch.Tensor,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        attention_weights: torch.Tensor,
        negative_slope: float,
        grad_A_reduce_row_chunk_size: int,
    ):
        if torch.is_autocast_enabled():
            attention_weights = attention_weights.to(torch.get_autocast_gpu_dtype())

        output, logsumexp = gatv2_kernels.forward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            attention_weights,
            negative_slope,
        )
        ctx.negative_slope = negative_slope
        ctx.grad_A_reduce_row_chunk_size = grad_A_reduce_row_chunk_size
        ctx.heads = x_left.shape[1]
        ctx.head_dim = x_left.shape[2]

        ctx.save_for_backward(
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
        )

        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
        ) = ctx.saved_tensors

        num_heads = ctx.heads
        head_dim = ctx.head_dim

        grad_output = grad_output.view(-1, num_heads, head_dim)

        negative_slope = ctx.negative_slope
        grad_A_reduce_row_chunk_size = ctx.grad_A_reduce_row_chunk_size
        grad_x_left, grad_x_right, grad_attention = gatv2_kernels.backward(
            grad_output,
            x_left,
            x_right,
            indptr_forward,
            indices_forward,
            indptr_backward,
            indices_backward,
            attention_weights,
            logsumexp,
            negative_slope,
            grad_A_reduce_row_chunk_size,
        )

        return None, None, None, None, grad_x_left, grad_x_right, grad_attention, None, None


class GATv2AggrKernel(TunableKernel):
    """Tunable kernel callable for GATv2 aggregation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.backward_grad_A_reduce_row_chunk_size = kwargs.get("grad_A_reduce_row_chunk_size", 512)

    def _execute(self, graph, x, *, x_neighbors=None, attention_weights=None, negative_slope=None, **kwargs):
        return gatv2_function.apply(
            graph.forward_indptr,
            graph.forward_indices,
            graph.backward_indptr,
            graph.backward_indices,
            x,
            x_neighbors,
            attention_weights,
            negative_slope,
            self.backward_grad_A_reduce_row_chunk_size,
        )

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_grad_A_reduce_row_chunk_size", [16, 32, 64, 128, 256, 512, 1024, 2048], default=512),
        ]

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("backward_huge_degree_threshold_quantile", [-1, 0.9, 0.95, 0.99], default=-1),
        ]

    def make_forward_bench_fn(self, x, graph_repr, **kwargs):
        x_neighbors = kwargs["x_neighbors"]
        attention_weights = kwargs["attention_weights"]
        negative_slope = kwargs["negative_slope"]

        def _bench():
            return self._execute(
                graph_repr,
                x,
                x_neighbors=x_neighbors,
                attention_weights=attention_weights,
                negative_slope=negative_slope,
            )

        return _bench


@with_autotune(GATv2AggrKernel)
def gatv2_aggr(
    graph: AdjacencyForwardBackwardWithNodeBuckets,
    x: torch.Tensor,
    x_neighbors: torch.Tensor,
    attention_weights: torch.Tensor,
    negative_slope: float,
    grad_A_reduce_row_chunk_size: int = 512,
) -> torch.Tensor:
    return gatv2_function.apply(
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        x,
        x_neighbors,
        attention_weights,
        negative_slope,
        grad_A_reduce_row_chunk_size,
    )
