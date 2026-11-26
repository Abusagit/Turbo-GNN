"""C++17 + OpenMP backend for `CPUNeighborSampler` and `CPUBatchMaker`, loaded via ctypes.

Uses a plain ``extern "C"`` ABI so a single compiled shared library works across any Python
and PyTorch version (e.g. both the main env and the torch-2.4 DGL baseline env) with zero
PyTorch C++ header or ninja dependency. Falls back cleanly to pure-PyTorch if ``g++`` is
unavailable or ``TURBO_GNN_CPU_SAMPLER_PYTORCH=1`` is set.
"""

from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "_cpu_sampler.cpp"
_LIB: ctypes.CDLL | None = None
_TRIED = False
_LOAD_LOCK = threading.Lock()


def _so_path() -> Path:
    default = _DIR / "_cpu_sampler_native.so"
    if os.access(_DIR, os.W_OK):
        return default
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbo_gnn"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "_cpu_sampler_native.so"


def _compile_if_needed(so_path: Path) -> bool:
    if so_path.exists() and so_path.stat().st_mtime >= _SRC.stat().st_mtime:
        return True
    cxx_env = os.environ.get("CXX")
    cxx_cmd = shlex.split(cxx_env) if cxx_env else ([c] if (c := shutil.which("g++") or shutil.which("clang++")) else [])
    if not cxx_cmd:
        return False
    so_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".cpu_sampler_", suffix=".so", dir=str(so_path.parent))
    os.close(fd)
    try:
        flags = ["-O3", "-std=c++17", "-march=native", "-fopenmp", "-shared", "-fPIC", str(_SRC), "-o", tmp_name]
        try:
            res = subprocess.run([*cxx_cmd, *flags], capture_output=True, text=True, check=False)
            if res.returncode != 0:
                flags.remove("-march=native")
                res = subprocess.run([*cxx_cmd, *flags], capture_output=True, text=True, check=False)
                if res.returncode != 0:
                    return False
        except OSError:
            return False
        os.replace(tmp_name, so_path)
        return True
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_native_lib() -> ctypes.CDLL | None:
    """Return the compiled C++ OpenMP library, or None if disabled / unavailable."""
    global _LIB, _TRIED
    if os.environ.get("TURBO_GNN_CPU_SAMPLER_PYTORCH", "").lower() in ("1", "true", "yes"):
        return None
    if _TRIED:
        return _LIB
    with _LOAD_LOCK:
        if _TRIED:
            return _LIB
        if not _SRC.exists():
            _TRIED = True
            return None
        so_path = _so_path()
        if not _compile_if_needed(so_path):
            _TRIED = True
            return None
        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError:
            _TRIED = True
            return None

        c_vp = ctypes.c_void_p
        c_i64 = ctypes.c_int64
        c_u64 = ctypes.c_uint64
        c_i32 = ctypes.c_int

        lib.cpu_sampler_state_create.argtypes = [c_i64]
        lib.cpu_sampler_state_create.restype = c_vp

        lib.cpu_sampler_state_destroy.argtypes = [c_vp]
        lib.cpu_sampler_state_destroy.restype = None

        lib.cpu_sample_layer_indptr.argtypes = [c_vp, c_vp, c_i64, c_i64, c_vp, c_i32, c_vp, c_i32]
        lib.cpu_sample_layer_indptr.restype = c_i64

        lib.cpu_sample_layer_edges.argtypes = [
            c_vp, c_vp, c_vp, c_i32, c_vp, c_i64, c_i64, c_vp, c_vp, c_i32, c_u64, c_i32
        ]
        lib.cpu_sample_layer_edges.restype = c_i64

        lib.cpu_copy_src_nodes.argtypes = [c_vp, c_vp, c_i64]
        lib.cpu_copy_src_nodes.restype = None

        lib.cpu_transpose_and_bucket_prep.argtypes = [
            c_vp, c_vp, c_vp, c_i64, c_i64, c_i64, c_i32, c_vp, c_vp, c_i64, c_i64, c_i32, c_i32, c_i32, c_vp
        ]
        lib.cpu_transpose_and_bucket_prep.restype = None

        lib.cpu_copy_buckets.argtypes = [c_vp, c_vp, c_vp, c_vp, c_vp, c_i32]
        lib.cpu_copy_buckets.restype = None

        lib.cpu_gather_rows.argtypes = [c_vp, c_vp, c_vp, c_i64, c_i64, c_i32]
        lib.cpu_gather_rows.restype = None

        _LIB = lib
        _TRIED = True
        return _LIB


class NativeSamplerState:
    """RAII wrapper around C++ ``CpuSamplerState`` (reusable generation-stamped scratch)."""

    def __init__(self, num_nodes: int) -> None:
        self._lib = load_native_lib()
        self._ptr: int | None = None
        self.num_nodes = int(num_nodes)
        if self._lib is not None:
            self._ptr = self._lib.cpu_sampler_state_create(self.num_nodes)

    @property
    def available(self) -> bool:
        return self._lib is not None and self._ptr is not None

    def close(self) -> None:
        if self._ptr is not None and self._lib is not None:
            self._lib.cpu_sampler_state_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def sample_layer(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        dst: torch.Tensor,
        fanout: int,
        index_dtype: torch.dtype,
        seed: int,
        num_threads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._lib is not None and self._ptr is not None
        n = dst.numel()
        out_is_i32 = 1 if index_dtype == torch.int32 else 0
        out_indptr = torch.empty(n + 1, dtype=index_dtype)
        indptr_i64_buf = torch.empty(n + 1, dtype=torch.int64)

        total_edges = int(
            self._lib.cpu_sample_layer_indptr(
                indptr.data_ptr(),
                dst.data_ptr(),
                n,
                fanout,
                out_indptr.data_ptr(),
                out_is_i32,
                indptr_i64_buf.data_ptr(),
                num_threads,
            )
        )
        out_col = torch.empty(total_edges, dtype=index_dtype)
        indices_is_i32 = 1 if indices.dtype == torch.int32 else 0
        num_src = int(
            self._lib.cpu_sample_layer_edges(
                self._ptr,
                indptr.data_ptr(),
                indices.data_ptr(),
                indices_is_i32,
                dst.data_ptr(),
                n,
                fanout,
                indptr_i64_buf.data_ptr(),
                out_col.data_ptr(),
                out_is_i32,
                seed & 0xFFFFFFFFFFFFFFFF,
                num_threads,
            )
        )
        src_nodes = torch.empty(num_src, dtype=torch.int64)
        self._lib.cpu_copy_src_nodes(self._ptr, src_nodes.data_ptr(), num_src)
        return out_indptr, out_col, src_nodes

    def transpose_and_bucket(
        self,
        fwd_indptr: torch.Tensor,
        fwd_indices: torch.Tensor,
        num_src: int,
        *,
        fwd_threshold: int | None,
        bwd_threshold: int | None,
        fwd_order: str,
        bwd_order: str,
        num_threads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._lib is not None and self._ptr is not None
        idx_dt = fwd_indptr.dtype
        idx_is_i32 = 1 if idx_dt == torch.int32 else 0
        num_dst = fwd_indptr.numel() - 1
        num_edges = fwd_indices.numel()

        bwd_indptr = torch.empty(num_src + 1, dtype=idx_dt)
        bwd_indices = torch.empty(num_edges, dtype=idx_dt)
        sizes = (ctypes.c_int64 * 4)()

        self._lib.cpu_transpose_and_bucket_prep(
            self._ptr,
            fwd_indptr.data_ptr(),
            fwd_indices.data_ptr(),
            num_dst,
            num_src,
            num_edges,
            idx_is_i32,
            bwd_indptr.data_ptr(),
            bwd_indices.data_ptr(),
            -1 if fwd_threshold is None else int(fwd_threshold),
            -1 if bwd_threshold is None else int(bwd_threshold),
            1 if fwd_order == "degree" else 0,
            1 if bwd_order == "degree" else 0,
            num_threads,
            ctypes.byref(sizes),
        )
        fwd_light = torch.empty(int(sizes[0]), dtype=idx_dt)
        fwd_heavy = torch.empty(int(sizes[1]), dtype=idx_dt)
        bwd_light = torch.empty(int(sizes[2]), dtype=idx_dt)
        bwd_heavy = torch.empty(int(sizes[3]), dtype=idx_dt)
        self._lib.cpu_copy_buckets(
            self._ptr,
            fwd_light.data_ptr(),
            fwd_heavy.data_ptr(),
            bwd_light.data_ptr(),
            bwd_heavy.data_ptr(),
            idx_is_i32,
        )
        return bwd_indptr, bwd_indices, fwd_light, fwd_heavy, bwd_light, bwd_heavy


def native_gather_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, num_threads: int) -> bool:
    """Gather ``dst[i] = src[idx[i]]`` via OpenMP memcpy when native backend is available."""
    lib = load_native_lib()
    if (
        lib is None
        or src.device.type != "cpu"
        or idx.device.type != "cpu"
        or dst.device.type != "cpu"
        or src.dtype != dst.dtype
        or not src.is_contiguous()
        or not dst.is_contiguous()
    ):
        return False
    n = idx.numel()
    if n == 0:
        return True
    if src.ndim < 1 or tuple(dst.shape) != (n, *src.shape[1:]):
        return False
    idx_c = idx if (idx.dtype == torch.int64 and idx.is_contiguous()) else idx.to(torch.int64).contiguous()
    row_elems = src.numel() // src.shape[0] if src.shape[0] > 0 else 0
    row_bytes = row_elems * src.element_size()
    lib.cpu_gather_rows(src.data_ptr(), idx_c.data_ptr(), dst.data_ptr(), n, row_bytes, num_threads)
    return True
