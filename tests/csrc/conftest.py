"""JIT-built bridge modules for the ``csrc/common/tile.cuh`` test suite.

``tile_test_bridge.cu`` is compiled once per (group, dtype, pass) triple. The split keeps
a combination that currently fails to instantiate from taking down unrelated rows: a
build failure is captured here and re-reported by whichever tests needed it, rather than
aborting collection.

Modules are built lazily on first use and cached under ``build/tile_tests/``, so a
filtered run only pays for the groups it touches.

**This suite is opt-in.** Pass ``--csrc`` (or set ``TGNN_RUN_CSRC=1``) to run it. The
default is to skip, for two reasons: ``make test`` is wired into the pre-commit hook, and
these tests both take minutes to JIT-compile and are expected red until the header issues
they document are fixed -- neither belongs on the commit path. Drop
``pytest_collection_modifyitems`` below to make them run by default.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SRC = Path(__file__).resolve().parent / "tile_test_bridge.cu"
BUILD_ROOT = REPO_ROOT / "build" / "tile_tests"

# Normally the suite compiles against the repo's own headers. Point
# TGNN_TILE_TEST_CSRC at a copy to try a candidate fix without editing csrc/ --
# useful while working through the blockers, since one non-instantiable overload
# takes the whole group's module with it.
CSRC_DIR = Path(os.environ.get("TGNN_TILE_TEST_CSRC", REPO_ROOT / "csrc")).resolve()

# Keep builds against different header trees in separate directories, so switching
# TGNN_TILE_TEST_CSRC cannot serve a stale module.
_CSRC_TAG = "" if CSRC_DIR == (REPO_ROOT / "csrc") else "_" + hashlib.sha1(str(CSRC_DIR).encode()).hexdigest()[:8]

# name -> (TGNN_DTYPE code, torch dtype)
DTYPES: dict[str, tuple[int, torch.dtype]] = {
    "float": (0, torch.float32),
    "half": (1, torch.float16),
    "bf16": (2, torch.bfloat16),
}

# name -> TGNN_GROUP code
GROUPS: dict[str, int] = {"data": 0, "ops": 1, "cvt": 2, "grad": 3}

# Largest N that Vec<N, T> allows, per dtype (Vec caps at 16 bytes).
MAX_N: dict[str, int] = {"float": 4, "half": 8, "bf16": 8}

# All (N, dtype) combinations the suite covers: N in {1,2,4,8} with N*sizeof(T) <= 16.
VALID_NS: dict[str, tuple[int, ...]] = {
    "float": (1, 2, 4),
    "half": (1, 2, 4, 8),
    "bf16": (1, 2, 4, 8),
}


def all_combos() -> list[tuple[int, str]]:
    """Every (N, dtype_name) pair under test, in a stable order."""
    return [(n, dt) for dt in DTYPES for n in VALID_NS[dt]]


# torch's cpp_extension prepends -D__CUDA_NO_HALF_OPERATORS__ and friends, which would
# block the raw operators and conversions that tile.cuh's scalar paths use on
# num_type. extra_cuda_cflags is appended *after* COMMON_NVCC_FLAGS
# (torch/utils/cpp_extension.py), so these -U flags win.
_UNDEFINE_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
]

# Mirror setup.py's nvcc flags so the suite exercises what actually ships.
_NVCC_FLAGS = ["-O3", "--use_fast_math", "--generate-line-info", "-std=c++20"]


class BridgeBuildError(RuntimeError):
    """A (group, dtype) module failed to compile. Carries the nvcc diagnostics."""


def _trim_diagnostics(text: str, keep: int = 25) -> str:
    """Keep the distinct lines that name errors, so failures stay readable.

    nvcc repeats the same diagnostic once per arch pass and once per instantiation,
    so the raw output is mostly duplicates; dedupe while preserving order.
    """
    lines = text.splitlines()
    errors = [ln.strip() for ln in lines if "error" in ln.lower()]
    body = errors if errors else [ln.strip() for ln in lines]

    seen: set[str] = set()
    unique = [ln for ln in body if not (ln in seen or seen.add(ln))]
    if len(unique) > keep:
        unique = unique[:keep] + [f"... ({len(unique) - keep} more distinct errors suppressed)"]
    return "\n".join(unique)


class BridgeLoader:
    """Builds and caches the bridge modules for one test session."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, bool], Any] = {}
        self._errors: dict[tuple[str, str, bool], str] = {}

    def _build(self, group: str, dtype: str, on_host: bool) -> Any:
        from torch.utils.cpp_extension import load

        try:  # keeps ninja discoverable the same way the src/ backends do
            from src._ninja import ensure_ninja_on_path

            ensure_ninja_on_path()
        except Exception:  # pragma: no cover - optional helper
            pass

        defines = [
            f"-DTGNN_DTYPE={DTYPES[dtype][0]}",
            f"-DTGNN_GROUP={GROUPS[group]}",
            f"-DTGNN_HOST={int(on_host)}",
        ]
        name = f"tile_test_{group}_{dtype}_{'host' if on_host else 'dev'}{_CSRC_TAG}"
        build_dir = BUILD_ROOT / name
        build_dir.mkdir(parents=True, exist_ok=True)

        return load(
            name=name,
            sources=[str(BRIDGE_SRC)],
            build_directory=str(build_dir),
            extra_include_paths=[str(CSRC_DIR)],
            extra_cflags=["-O3", "-std=c++20", *defines],
            extra_cuda_cflags=[*_NVCC_FLAGS, *_UNDEFINE_FLAGS, *defines],
            verbose=False,
        )

    def get(self, group: str, dtype: str, on_host: bool = False) -> Any:
        """Return the module, or fail the calling test with the compiler diagnostics."""
        key = (group, dtype, on_host)
        label = f"{group}/{dtype}/{'host' if on_host else 'device'}"

        if key not in self._errors and key not in self._cache:
            try:
                self._cache[key] = self._build(group, dtype, on_host)
            except Exception as exc:
                self._errors[key] = _trim_diagnostics(str(exc))

        if key in self._errors:
            pytest.fail(
                f"bridge module {label} does not compile against {CSRC_DIR}/common/tile.cuh:\n{self._errors[key]}",
                pytrace=False,
            )
        return self._cache[key]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--csrc",
        action="store_true",
        default=False,
        help="run the csrc/common/tile.cuh tests (JIT-compiles CUDA; needs a GPU)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip this directory unless explicitly asked for. See the module docstring."""
    if config.getoption("--csrc") or os.environ.get("TGNN_RUN_CSRC") == "1":
        return
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="tile.cuh tests are opt-in: pass --csrc or set TGNN_RUN_CSRC=1")
    for item in items:
        if item.path is not None and item.path.parent == here:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def bridge() -> BridgeLoader:
    """Session-wide loader: ``bridge.get("ops", "half")``."""
    if not torch.cuda.is_available():
        pytest.skip("tile.cuh tests need a CUDA device")
    return BridgeLoader()


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("tile.cuh tests need a CUDA device")
    return torch.device("cuda")
