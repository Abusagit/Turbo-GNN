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
import sys
import time
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


# A build that dies (Ctrl-C, OOM-killer, a killed CI job) leaves torch's FileBaton
# lock behind. The next load() then blocks in baton.wait() forever, printing nothing
# -- indistinguishable from a hang. Anything older than this is treated as abandoned.
_LOCK_STALE_SECONDS = 600


_config: Any = None

# (group, dtype) pairs the selected tests will need, filled in during collection and
# compiled concurrently by the `bridge` fixture before the first test runs.
_PREBUILD_TARGETS: set[tuple[str, str]] = set()

# The session's loader, created once the prebuild targets are known.
_LOADER: Any = None


def _note(message: str) -> None:
    """Write a progress line that survives pytest's output capture.

    The reporter is resolved lazily, not cached in pytest_configure: conftest plugins
    are registered after the builtins, so pluggy calls their pytest_configure *first*
    and "terminalreporter" does not exist yet at that point.

    Capture also has to be suspended explicitly -- pytest redirects at the file
    descriptor level, so the reporter's own stream otherwise lands in the capture pipe
    and is discarded.
    """
    if _config is None:
        return
    reporter = _config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    capman = _config.pluginmanager.get_plugin("capturemanager")
    if capman is None:
        reporter.write_line(message)
        return
    with capman.global_and_fixture_disabled():
        reporter.write_line(message)


def _clear_stale_locks(build_dir: Path) -> None:
    for lock in (build_dir / "lock", build_dir / ".ninja_lock"):
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            continue
        if age > _LOCK_STALE_SECONDS:
            _note(f"[tile.cuh] removing stale build lock {lock} (age {age / 60:.0f} min)")
            lock.unlink(missing_ok=True)


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


def _module_name(group: str, dtype: str) -> str:
    return f"tile_test_{group}_{dtype}_dev{_CSRC_TAG}"


def _newest_source_mtime() -> float:
    """Most recent mtime across the bridge and every header it can reach."""
    newest = BRIDGE_SRC.stat().st_mtime
    for header in CSRC_DIR.rglob("*.cuh"):
        try:
            newest = max(newest, header.stat().st_mtime)
        except OSError:  # pragma: no cover - racing with an editor
            continue
    return newest


def _is_fresh(name: str, cutoff: float) -> bool:
    """True if this module's .so is newer than every source that feeds it."""
    try:
        return (BUILD_ROOT / name / f"{name}.so").stat().st_mtime >= cutoff
    except OSError:
        return False


def _max_build_workers(n_targets: int) -> int:
    """Same shape as setup.py's MAX_JOBS heuristic, but counting whole nvcc processes."""
    if n_targets <= 1:
        return 1
    by_cores = max(1, (os.cpu_count() or 2) // 2)
    by_memory = by_cores
    try:
        import psutil

        # ~5GB per nvcc thread, and --threads=4 below, so budget generously.
        by_memory = max(1, int(psutil.virtual_memory().available / (1024**3) / 20))
    except Exception:  # pragma: no cover - psutil is optional here
        pass
    return max(1, min(n_targets, by_cores, by_memory))


# Compiling one module in a fresh interpreter. Deliberately a subprocess rather than a
# thread: torch's load() mutates process-global state (the JIT extension versioner,
# sys.modules) and is not documented as thread-safe. Separate build directories mean
# separate FileBatons and separate ninja invocations, so these cannot collide.
_PREBUILD_SNIPPET = """
import sys
sys.path.insert(0, {tests_dir!r})
import conftest
conftest.BridgeLoader()._build({group!r}, {dtype!r})
"""


class BridgeLoader:
    """Builds and caches the bridge modules for one test session."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}
        self._errors: dict[tuple[str, str], str] = {}

    def prebuild(self, targets: list[tuple[str, str]]) -> None:
        """Compile several modules concurrently, so the session pays one build, not N.

        Each module is a single translation unit, so ninja's own MAX_JOBS cannot help --
        the only parallelism available is across modules. Afterwards ``get()`` finds
        every .so up to date and merely imports it.
        """
        import subprocess
        from concurrent.futures import ThreadPoolExecutor

        cutoff = _newest_source_mtime()
        stale = [(g, d) for g, d in targets if not _is_fresh(_module_name(g, d), cutoff)]
        if not stale:
            return

        workers = _max_build_workers(len(stale))
        _note(f"[tile.cuh] compiling {len(stale)} bridge module(s), {workers} at a time")
        started = time.monotonic()

        def run_one(target: tuple[str, str]) -> tuple[tuple[str, str], int, str]:
            group, dtype = target
            build_dir = BUILD_ROOT / _module_name(group, dtype)
            build_dir.mkdir(parents=True, exist_ok=True)
            _clear_stale_locks(build_dir)
            code = _PREBUILD_SNIPPET.format(tests_dir=str(Path(__file__).resolve().parent), group=group, dtype=dtype)
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            return target, proc.returncode, proc.stdout + proc.stderr

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for target, rc, output in pool.map(run_one, stale):
                if rc != 0:
                    # Record it now so get() reports the diagnostics instead of
                    # spending another minute reproducing the same failure.
                    self._errors[target] = _trim_diagnostics(output)
                    _note(f"[tile.cuh]   {'/'.join(target)} FAILED to compile")

        ok = len(stale) - sum(1 for t in stale if t in self._errors)
        _note(f"[tile.cuh] {ok}/{len(stale)} module(s) built in {time.monotonic() - started:.0f}s")

    def _build(self, group: str, dtype: str) -> Any:
        from torch.utils.cpp_extension import load

        try:  # keeps ninja discoverable the same way the src/ backends do
            from src._ninja import ensure_ninja_on_path

            ensure_ninja_on_path()
        except Exception:  # pragma: no cover - optional helper
            pass

        defines = [
            f"-DTGNN_DTYPE={DTYPES[dtype][0]}",
            f"-DTGNN_GROUP={GROUPS[group]}",
        ]
        name = f"tile_test_{group}_{dtype}_dev{_CSRC_TAG}"
        build_dir = BUILD_ROOT / name
        build_dir.mkdir(parents=True, exist_ok=True)
        _clear_stale_locks(build_dir)

        # Usually a no-op: prebuild() has already compiled this concurrently, so load()
        # just imports the cached .so. Only announce when there is real work to do.
        started = time.monotonic()
        cold = not _is_fresh(name, _newest_source_mtime())
        if cold:
            _note(f"[tile.cuh] compiling {name} (~1 min) ...")
        try:
            return load(
                name=name,
                sources=[str(BRIDGE_SRC)],
                build_directory=str(build_dir),
                extra_include_paths=[str(CSRC_DIR)],
                extra_cflags=["-O3", "-std=c++20", *defines],
                extra_cuda_cflags=[*_NVCC_FLAGS, *_UNDEFINE_FLAGS, *defines],
                verbose=False,
            )
        finally:
            if cold:
                _note(f"[tile.cuh] {name} finished in {time.monotonic() - started:.0f}s")

    def get(self, group: str, dtype: str) -> Any:
        """Return the module, or fail the calling test with the compiler diagnostics."""
        key = (group, dtype)
        label = f"{group}/{dtype}/device"

        if key not in self._errors and key not in self._cache:
            try:
                self._cache[key] = self._build(group, dtype)
            except Exception as exc:
                self._errors[key] = _trim_diagnostics(str(exc))

        if key in self._errors:
            pytest.fail(
                f"bridge module {label} does not compile against {CSRC_DIR}/common/tile.cuh:\n{self._errors[key]}",
                pytrace=False,
            )
        return self._cache[key]


def pytest_configure(config: pytest.Config) -> None:
    global _config
    _config = config


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--csrc",
        action="store_true",
        default=False,
        help="run the csrc/common/tile.cuh tests (JIT-compiles CUDA; needs a GPU)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip this directory unless explicitly asked for, else plan the parallel prebuild."""
    here = Path(__file__).parent

    if not (config.getoption("--csrc") or os.environ.get("TGNN_RUN_CSRC") == "1"):
        skip = pytest.mark.skip(reason="tile.cuh tests are opt-in: pass --csrc or set TGNN_RUN_CSRC=1")
        for item in items:
            if item.path is not None and item.path.parent == here:
                item.add_marker(skip)
        return

    # Work out which (group, dtype) modules the *selected* tests actually touch, so
    # `-k half` does not pay for the float and bf16 builds. Groups are scraped from the
    # bridge.get() calls in each test file rather than hardcoded, so this keeps working
    # as files move around.
    import re

    groups_in_file: dict[Path, set[str]] = {}
    for item in items:
        path = item.path
        if path is None or path.parent != here:
            continue
        if path not in groups_in_file:
            try:
                groups_in_file[path] = set(re.findall(r'bridge\.get\(\s*"(\w+)"', path.read_text()))
            except OSError:  # pragma: no cover
                groups_in_file[path] = set()
        dtypes = [d for d in DTYPES if d in item.name] or list(DTYPES)
        for group in groups_in_file[path]:
            for dtype in dtypes:
                _PREBUILD_TARGETS.add((group, dtype))


def pytest_collection_finish(session: pytest.Session) -> None:
    """Compile the needed modules concurrently, before the first test runs.

    Doing it here rather than lazily in the fixture makes the build a visible phase of
    its own instead of several minutes of apparent silence inside the first test.
    """
    global _LOADER
    if not _PREBUILD_TARGETS or not torch.cuda.is_available():
        return
    _LOADER = BridgeLoader()
    _LOADER.prebuild(sorted(_PREBUILD_TARGETS))


@pytest.fixture(scope="session")
def bridge() -> BridgeLoader:
    """Session-wide loader: ``bridge.get("ops", "half")``."""
    if not torch.cuda.is_available():
        pytest.skip("tile.cuh tests need a CUDA device")
    return _LOADER if _LOADER is not None else BridgeLoader()


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("tile.cuh tests need a CUDA device")
    return torch.device("cuda")
