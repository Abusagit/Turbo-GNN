"""
Backends package: register/import specific backend implementations (PyG, DGL, etc.).

Each backend sub-package registers itself with BackendRegistry on import.
JIT-compiled backends use lazy loading internally so that importing here
does NOT trigger expensive CUDA kernel compilation.
"""

import logging as _logging

_log = _logging.getLogger(__name__)

# Eagerly import every backend so that @register_backend decorators execute
# and the registry is populated.  Each import is wrapped in try/except so
# that a missing optional dependency (e.g. DGL, triton, cugraph) does not
# prevent the rest of the backends from registering.

_backend_modules = [
    "cuda_backend",
    "cuda_test_backend",
    "cugraph_backend",
    "cusparse_backend",
    "dfgnn_backend",
    "dgl_backend",
    "fusegnn_backend",
    "pyg_backend",
    "tcgnn_backend",
    "torch_native_backend",
    "triton_backend",
]

for _name in _backend_modules:
    try:
        __import__(f"{__name__}.{_name}")
    except Exception as _exc:  # noqa: BLE001
        _log.debug("Backend %s not available: %s", _name, _exc)
