import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

if os.environ.get("CUDA_HOME") is None:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"


path = Path(__file__).parent
sources = ["F3S.cpp", "F3S_kernel.cu", "utils.cu"]

cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_path is None:
    raise ValueError("CUDA_HOME or CUDA_PATH is not set")

repo_root_path = Path(__file__).parent.parent.parent.parent
build_path = repo_root_path / "build/fused3s"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

# Get PyTorch's ABI setting to match it
torch_abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

f3s_ops = load(
    name="f3s_ops",
    build_directory=str(build_path),
    extra_cflags=["-O3", f"-D_GLIBCXX_USE_CXX11_ABI={torch_abi}"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_80",
        "--generate-line-info",
        "-Xcompiler",
        f"-D_GLIBCXX_USE_CXX11_ABI={torch_abi}",  # Use -Xcompiler for nvcc
    ],
    extra_include_paths=[
        f"{cuda_path}/include",
        str(path),  # Add directory containing config.h and ptx.h
    ],
    sources=[path / s for s in sources],
    verbose=True,
)


def f3s_preprocess(edge_index, block_h=16, block_w=8):
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_sparse_csr()
    size = edge_index.shape[0]
    num_row_windows = (size + block_h - 1) // block_h
    edgeToColumn = torch.zeros(size, dtype=torch.int, device="cuda")
    edgeToRow = torch.zeros(size, dtype=torch.int, device="cuda")
    block_partition = torch.zeros(num_row_windows, dtype=torch.int, device="cuda")
    indices = A.crow_indices().int().cuda()
    indptr = A.col_indices().int().cuda()
    row_window_offset, sorted_row_windows, tcblock_rowid, _, _, sparse_a_to_index, tcblock_bit_map, _ = (
        f3s_ops.preprocess_gpu(indices, indptr, size, block_h, block_w, block_partition, edgeToColumn, edgeToRow)
    )

    return sorted_row_windows, row_window_offset, sparse_a_to_index, tcblock_bit_map, edge_index.shape[1]


def F3S_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, graph, block_h=16, block_w=8, n_warps_per_block=8):
    sorted_row_windows, row_window_offset, sparse_a_to_index, tcblock_bit_map, size = graph
    x = f3s_ops.f3s_1tb1rw_scheduled_permuteV(
        row_window_offset, sorted_row_windows, sparse_a_to_index, tcblock_bit_map, size, q, k, v, n_warps_per_block
    )

    return x
