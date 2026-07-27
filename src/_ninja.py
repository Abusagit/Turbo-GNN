"""Make the `ninja` build requirement visible to torch's JIT extension loader."""

import os
import shutil
import sys
import warnings

_RESOLVED = False


def ensure_ninja_on_path() -> bool:
    """Prepend the directory holding the ``ninja`` binary to PATH.

    Idempotent and safe to call from module scope. Returns True if a ninja
    binary is reachable afterwards.
    """
    global _RESOLVED
    if _RESOLVED:
        return True

    if shutil.which("ninja"):
        _RESOLVED = True
        return True

    candidates = []
    try:
        import ninja

        candidates.append(ninja.BIN_DIR)
    except (ImportError, AttributeError):
        pass
    # Fallback for a venv whose `ninja` python package is absent but whose
    # scripts directory still holds the binary.
    candidates.append(os.path.dirname(sys.executable))

    for bindir in candidates:
        if bindir and os.path.isfile(os.path.join(bindir, "ninja")):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            _RESOLVED = True
            return True

    warnings.warn(
        "ninja was not found on PATH and could not be located next to "
        f"{sys.executable}. JIT-compiled CUDA extensions will fail to load. "
        "Install it with `pip install ninja`.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False
