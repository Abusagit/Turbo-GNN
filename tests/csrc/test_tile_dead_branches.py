"""Compile-only coverage for the code paths no runtime test on this hardware can reach.

``can_be_packed_new`` admits float only from ``kCudaArch >= 1000``, and
``TORCH_CUDA_ARCH_LIST`` defaults to ``8.0 8.6 8.9 9.0`` -- so on every GPU this project
currently builds for, the float packed branches of ``packed_add`` / ``packed_mul`` /
``packed_fma`` are dead. Dead code still ships, and it still breaks builds the day someone
adds ``10.0`` to the arch list. Compiling one small translation unit for ``sm_100a`` is the
only way to see it.

Marked slow: it shells out to nvcc and is not part of the normal loop.
"""

from __future__ import annotations

import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest
from conftest import _NVCC_FLAGS, _UNDEFINE_FLAGS, CSRC_DIR

pytestmark = [pytest.mark.cuda, pytest.mark.csrc, pytest.mark.slow]

# Instantiating these for float is what pulls in the kCudaArch >= 1000 branches.
PROBE_SOURCE = """
#include "common/tile.cuh"

using Ops = VecOpsFloatBase<4, float>;

__global__ void probe(Vec<4, float>* a, Vec<4, float>* b, Vec<4, float>* c) {
    Ops::add_(a, b);
    Ops::mul_(a, b);
    Ops::fmam_(a, b, c);
    Ops::sub_(a, b);
    float acc = 0.0f;
    Ops::sum(&acc, a);
    Ops::prod(&acc, a);
    (*c)[0] = acc;
}
"""

ARCH = "sm_100a"


def _nvcc() -> str:
    for candidate in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    pytest.skip("nvcc not found")


def _include_flags() -> list[str]:
    from torch.utils.cpp_extension import include_paths

    flags = [f"-I{CSRC_DIR}"]
    flags += [f"-I{p}" for p in include_paths(device_type="cuda")]
    flags.append(f"-I{sysconfig.get_paths()['include']}")
    return flags


def _compile_for(arch: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    src = tmp_path / "probe.cu"
    src.write_text(PROBE_SOURCE)
    cmd = [
        _nvcc(),
        "-x",
        "cu",
        f"-arch={arch}",
        "--expt-relaxed-constexpr",
        *_NVCC_FLAGS,
        *_UNDEFINE_FLAGS,
        *_include_flags(),
        "-c",
        str(src),
        "-o",
        str(tmp_path / "probe.o"),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def sm100_available(tmp_path_factory) -> bool:
    """Whether this toolkit can target sm_100a at all."""
    probe = tmp_path_factory.mktemp("archcheck")
    (probe / "empty.cu").write_text("__global__ void k() {}\n")
    res = subprocess.run(
        [_nvcc(), "-x", "cu", f"-arch={ARCH}", "-c", str(probe / "empty.cu"), "-o", str(probe / "empty.o")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if res.returncode != 0:
        pytest.skip(f"this CUDA toolkit cannot target {ARCH}")
    return True


def test_packed_float_branches_compile_for_sm100(sm100_available, tmp_path):
    """The float packed branches must be buildable, not just unreachable.

    Guards against the whole ``kCudaArch >= 1000`` ladder silently rotting: it is never
    instantiated on the supported arch list, so ordinary CI cannot tell whether it would
    compile at all.
    """
    res = _compile_for(ARCH, tmp_path)
    assert res.returncode == 0, f"tile.cuh does not compile for {ARCH}:\n{res.stdout}\n{res.stderr}"


def test_packed_float_branches_all_return_a_value(sm100_available, tmp_path):
    """No 'missing return statement' on the sm_100+ float paths.

    ``packed_add`` / ``packed_mul`` / ``packed_fma`` call ``__fadd2_rn`` / ``__fmul2_rn`` /
    ``__ffma2_rn`` and then discard the result, falling off the end of a non-void function.
    That is UB, and it is invisible at sm_80/90 because the branch is discarded before
    codegen -- nvcc only warns once the branch is live.
    """
    res = _compile_for(ARCH, tmp_path)
    offenders = sorted(
        {line.strip() for line in (res.stdout + res.stderr).splitlines() if "missing return statement" in line}
    )
    assert not offenders, "functions fall off the end without returning on " + ARCH + ":\n" + "\n".join(offenders)
