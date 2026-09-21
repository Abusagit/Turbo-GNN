"""Loads the compiled extension, or a stub for environments that only run the baselines.

The DGL baselines need PyTorch 2.4, which cannot load an extension built for a newer PyTorch,
yet the benchmark harness imports this package for every backend. With
``SKEWGNN_ALLOW_MISSING_EXTENSION=1`` a missing extension is replaced by a stub whose every
attribute raises, so the package imports and the baseline backends run, while any attempt to
launch one of our kernels fails loudly instead of silently measuring something else.
"""

import os

try:
    import skewgnn._C as _C
except ImportError:
    if os.environ.get("SKEWGNN_ALLOW_MISSING_EXTENSION") != "1":
        raise

    class _MissingExtension:
        def __getattr__(self, name):
            raise RuntimeError(
                f"skewgnn._C.{name} is unavailable: the extension is not built for this "
                "environment (SKEWGNN_ALLOW_MISSING_EXTENSION=1). Build it to run our kernels."
            )

    _C = _MissingExtension()

__all__ = ["_C"]
