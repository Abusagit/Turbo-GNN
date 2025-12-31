import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"

import glob
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

path = __file__.replace("utils.py", "")
sources = ["graph_transformer.cu"]

repo_root_path = Path(__file__).parent.parent.parent.parent.parent
build_path = repo_root_path / "build/graph_transformer_backend"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

graph_transformer_kernels = load(
    name="graph_transformer_kernels",
    build_directory=str(build_path),
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_80",
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


class _FusedGraphAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_ptr, edge_idx, edge_ptr_T, edge_idx_T, Q, K, V, scale):
        """
        Forward pass wrapper.
        Returns: (output, logsumexp)
        """
        out, logsumexp = graph_transformer_kernels.gt_forward_csr_mh(edge_ptr, edge_idx, Q, K, V, scale)

        # Save for backward
        ctx.scale = scale
        ctx.num_heads = Q.shape[1]
        ctx.head_dim = Q.shape[2]
        ctx.save_for_backward(edge_ptr_T, edge_idx_T, Q, K, V, out, logsumexp)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass wrapper.
        Returns: (None, None, None, None, dQ, dK, dV)
        """
        edge_ptr_T, edge_idx_T, Q, K, V, out, logsumexp = ctx.saved_tensors
        scale = ctx.scale
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        grad_output = grad_output.view(-1, num_heads, head_dim)

        dQ, dK, dV = graph_transformer_kernels.gt_backward_csr_mh(
            edge_ptr_T, edge_idx_T, Q, K, V, out, grad_output, logsumexp, scale
        )

        return None, None, None, None, dQ, dK, dV, None


def graph_transformer_aggr(edge_ptr, edge_idx, edge_ptr_T, edge_idx_T, Q, K, V, scale):
    return _FusedGraphAttention.apply(edge_ptr, edge_idx, edge_ptr_T, edge_idx_T, Q, K, V, scale)
