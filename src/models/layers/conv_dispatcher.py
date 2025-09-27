import inspect
import re
from typing import Any, Callable, Dict, Iterable, Optional

from ...backends.registry import BackendRegistry

doc = """
Dispatcher that instantiates a convolution for (conv_type, backend) via BackendRegistry.

This version is signature-aware:
1) If the backend exposes `get_conv_signature(conv_type) -> Iterable[str]`, we
   filter kwargs using that allowlist before constructing the conv.
2) Otherwise, we robustly retry on TypeError by pruning kwargs that appear in
   "got an unexpected keyword argument '...'" until construction succeeds.

This avoids hard-coding per-layer knobs (e.g., 'heads', 'concat'), making the
system resilient as backends evolve.
"""


# ------------------------------ helpers ---------------------------------------

def _filter_kwargs_by_signature(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by a callable's signature.

    If the callable has **kwargs (VAR_KEYWORD), we return the original kwargs.

    Args:
        func (Callable[..., Any]): Target callable (e.g., class __init__).
        kwargs (Dict[str, Any]): Candidate keyword arguments.

    Returns:
        Dict[str, Any]: Filtered kwargs.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):  # builtins or C funcs without signature
        return dict(kwargs)

    params = sig.parameters.values()
    # if **kwargs present, keep everything — constructor will handle specifics
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


_UNEXPECTED_KW_RE = re.compile(r"unexpected keyword argument '([^']+)'")


def _retry_create_pruning_kwargs(creator: Callable[..., Any], *, kwargs: Dict[str, Any], max_retries: int = 10) -> Any:
    """Call `creator(**kwargs)`, pruning unknown kwargs on TypeError and retrying.

    Args:
        creator (Callable[..., Any]): Zero-arg/kwargs-only callable that constructs the conv.
        kwargs (Dict[str, Any]): Initial kwargs to try.
        max_retries (int): Max number of prune-retry cycles.

    Returns:
        Any: The constructed object.

    Raises:
        Exception: Re-raises non-signature TypeErrors or after exhausting retries.
    """
    remaining = dict(kwargs)
    for _ in range(max_retries):
        try:
            return creator(**remaining)
        except TypeError as e:
            # Only handle unexpected-kwarg errors; otherwise re-raise
            m = _UNEXPECTED_KW_RE.search(str(e))
            if not m:
                raise
            bad = m.group(1)
            if bad in remaining:
                remaining.pop(bad)
            else:
                # Couldn't find the kw to drop; bubble up
                raise
    # Too many rounds — surface last error
    return creator(**remaining)

# ------------------------------- dispatcher -----------------------------------

def create_conv_layer(
    conv_type: str,
    backend: str,
    in_channels: int,
    out_channels: int,
    **kwargs: Any,
):
    """Create a convolution layer via the backend registry, filtering kwargs by signature.

    Args:
        conv_type (str): Convolution type ("gcn", "gat", "sage", "gin", ...).
        backend (str): Backend name ("pyg", "dgl", "torch_native", ...).
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        **kwargs (Any): Additional layer params (e.g., heads, bias, aggr, concat, ...).

    Returns:
        torch.nn.Module: A backend-specific convolution layer instance.

    Behavior:
        - If backend provides `get_conv_signature(conv_type) -> Iterable[str]`, we keep only
          those kw names before calling `create_conv`.
        - Otherwise, we call `create_conv(...)` and, on TypeError due to an unexpected kw,
          prune it and retry until success.
    """
    backend_inst = BackendRegistry.get_backend(backend)

    # 1) Backend-provided signature (preferred if available)
    allowlist: Optional[Iterable[str]] = None
    get_sig = getattr(backend_inst, "get_conv_signature", None)
    if callable(get_sig):
        try:
            allowlist = list(get_sig(conv_type))  # type: ignore[misc]
        except Exception:
            allowlist = None

    if allowlist is not None:
        filtered = {k: v for k, v in kwargs.items() if k in set(allowlist)}
        return backend_inst.create_conv(conv_type, in_channels, out_channels, **filtered)

    # 2) Fallback: try to filter using the signature of the backend's `create_conv`
    #    (helps if backend declares explicit kwargs instead of **kwargs)
    try:
        filtered_once = _filter_kwargs_by_signature(backend_inst.create_conv, kwargs)
    except Exception:
        filtered_once = dict(kwargs)

    # 3) Robust fallback: attempt construction; if a TypeError reports an unexpected kw,
    #    prune it and retry. This handles deeply wrapped convs (e.g., framework -> superclass).
    def _creator(**kw: Any) -> Any:
        return backend_inst.create_conv(conv_type, in_channels, out_channels, **kw)

    return _retry_create_pruning_kwargs(_creator, kwargs=filtered_once)
