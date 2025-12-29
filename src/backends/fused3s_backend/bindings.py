import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

if os.environ.get("CUDA_HOME") is None:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"


path = __file__.replace("utils.py", "")
sources = ["F3S_kernel.cu", "F3S_kernel.cpp"]

cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_path is None:
    raise ValueError("CUDA_HOME or CUDA_PATH is not set")

repo_root_path = Path(__file__).parent.parent.parent.parent
build_path = repo_root_path / "build/fused3s"
if not build_path.is_dir():
    build_path.mkdir(parents=True)

f3s_ops = load(
    name="f3s_ops",
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
        f"{cuda_path}/include"
    ],
    sources=[path + s for s in sources],
    verbose=True,
)


def f3s_preprocess(row_index, column_index, block_h=16, block_w=8):
    size = row_index.size(0)
    num_row_windows = (size + block_h - 1) // block_h
    edgeToColumn = torch.zeros(size, dtype=torch.int, device="cuda")
    edgeToRow = torch.zeros(size, dtype=torch.int, device="cuda")
    block_partition = torch.zeros(num_row_windows, dtype=torch.int, device="cuda")
    indices = torch.IntTensor(row_index).cuda()
    indptr = torch.IntTensor(column_index).cuda()
    row_window_offset, sorted_row_windows, tcblock_rowid, _, _, sparse_a_to_index, tcblock_bit_map, _ = (
        f3s_ops.preprocess_gpu(indices, indptr, size, block_h, block_w, block_partition, edgeToColumn, edgeToRow)
    )

    return sorted_row_windows, row_window_offset, sparse_a_to_index, tcblock_bit_map


def F3S_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, graph, block_h=16, block_w=8, n_warps_per_block=8):
    sorted_row_windows, row_window_offset, sparse_a_to_index, tcblock_bit_map, size = graph
    time, fusedR = f3s_ops.f3s_1tb1rw_scheduled_permuteV(
        row_window_offset, sorted_row_windows, sparse_a_to_index, tcblock_bit_map, size, q, k, v, n_warps_per_block
    )

    return time, fusedR
