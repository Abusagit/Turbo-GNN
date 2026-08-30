from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarking.microbench import MicrobenchResult, get_gpu_info, time_callable
from turbo_gnn import AdjacencyForwardBackwardWithNodeBuckets, AutotuneConfig

doc = """
Kernel microbenchmark launcher.

Times a backend's graph convolution alone -- the projection-free aggregation
from src/backends -- on a CSR graph and randomly generated pre-projected
inputs. No linear/QKV projections, no bias, no framework dispatch.

`--backend cuda` reaches the turbo_gnn kernels; every other backend runs its
own aggregation, so implementations compare at the same level.

Autotuning is OFF by default so that a run measures exactly the configuration
you asked for; pass --autotune to grid-search first. Kernel parameters are
settable with repeated `-K name=value` flags (see --help for names/defaults).
"""


INDEX_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}

FEATURE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _parse_bool(raw: str) -> bool:
    """Parse a boolean kernel argument.

    Args:
        raw (str): Value as typed on the command line.

    Returns:
        bool: Parsed value.

    Raises:
        ValueError: If the value is not a recognized boolean spelling.
    """
    low = raw.strip().lower()
    if low in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {raw!r}")


def _parse_optional_float(raw: str) -> Optional[float]:
    """Parse a float kernel argument that also accepts 'none'/'auto'.

    Args:
        raw (str): Value as typed on the command line.

    Returns:
        Optional[float]: Parsed value, or None when the kernel should pick its own.
    """
    if raw.strip().lower() in {"none", "auto", ""}:
        return None
    return float(raw)


@dataclass(frozen=True)
class KernelParam:
    """A non-tensor kernel argument exposed on the command line.

    Attributes:
        name: Argument name as accepted by the kernel entry point.
        parse: Callable turning the raw CLI string into the argument value.
        default: Value used when the argument is not passed.
        help: One-line description shown in the --help epilog.
        choices: Optional allowed values, validated after parsing.
    """

    name: str
    parse: Callable[[str], Any]
    default: Any
    help: str
    choices: Optional[tuple] = None


@dataclass
class InputContext:
    """Shapes and dtypes needed to synthesize kernel inputs.

    Attributes:
        num_nodes: Node count of the benchmarked graph.
        feature_dim: Total feature width (``heads * head_dim`` for attention kernels).
        heads: Attention head count (1 for non-attention kernels).
        head_dim: Per-head feature width.
        device: Torch device.
        dtype: Feature dtype.
    """

    num_nodes: int
    feature_dim: int
    heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype


# --------------------------------------------------------------------------
# Input builders
# --------------------------------------------------------------------------


def _build_flat_features(ctx: InputContext) -> dict[str, torch.Tensor]:
    """Build a single ``[N, F]`` feature matrix.

    Args:
        ctx (InputContext): Shape/dtype context.

    Returns:
        dict[str, torch.Tensor]: Mapping with the single input ``x``.
    """
    x = torch.randn(ctx.num_nodes, ctx.feature_dim, device=ctx.device, dtype=ctx.dtype, requires_grad=True)
    return {"x": x}


# --------------------------------------------------------------------------
# Backend aggregations (src/backends), projection-free
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvFamily:
    """Calling convention shared by every backend's aggregation for a conv type.

    ``BaseAggr`` subclasses take already-projected tensors, so the benchmark
    synthesizes them directly and the linear/QKV projections never run.

    Attributes:
        name: Family label reported in the JSON payload.
        conv_types: ``--conv`` values dispatching to this family.
        build_inputs: Callable creating the pre-projected tensors.
        arg_order: Input names, in the order the aggregation's forward() takes them.
    """

    name: str
    conv_types: tuple[str, ...]
    build_inputs: Callable[[InputContext], dict[str, torch.Tensor]]
    arg_order: tuple[str, ...]


def _build_gat_v2_aggr_inputs(ctx: InputContext) -> dict[str, torch.Tensor]:
    """Build the pre-projected GATv2 tensors a backend aggregation expects.

    Backends build their GATv2 aggregation with ``head_dim=feature_dim`` (see
    each ``create_aggr``), so the per-head width here is --feature-dim and the
    concatenated width is ``heads * feature_dim`` -- unlike ``gt``, which splits
    --feature-dim across heads.

    Args:
        ctx (InputContext): Shape/dtype context.

    Returns:
        dict[str, torch.Tensor]: Mapping with ``x_left`` and ``x_right``.
    """
    shape = (ctx.num_nodes, ctx.heads, ctx.feature_dim)
    return {
        "x_left": torch.randn(*shape, device=ctx.device, dtype=ctx.dtype, requires_grad=True),
        "x_right": torch.randn(*shape, device=ctx.device, dtype=ctx.dtype, requires_grad=True),
    }


def _build_gt_aggr_inputs(ctx: InputContext) -> dict[str, torch.Tensor]:
    """Build the pre-projected Q/K/V a backend graph-transformer aggregation expects.

    Args:
        ctx (InputContext): Shape/dtype context.

    Returns:
        dict[str, torch.Tensor]: Mapping with ``Q``, ``K`` and ``V``.
    """
    shape = (ctx.num_nodes, ctx.heads, ctx.head_dim)
    return {
        name: torch.randn(*shape, device=ctx.device, dtype=ctx.dtype, requires_grad=True) for name in ("Q", "K", "V")
    }


CONV_FAMILIES: tuple[ConvFamily, ...] = (
    ConvFamily(
        name="simple",
        conv_types=("min_aggr", "max_aggr", "sum_aggr", "mean_aggr", "gcn"),
        build_inputs=_build_flat_features,
        arg_order=("x",),
    ),
    ConvFamily(
        name="gat_v2",
        conv_types=("gat_v2",),
        build_inputs=_build_gat_v2_aggr_inputs,
        arg_order=("x_left", "x_right"),
    ),
    ConvFamily(
        name="gt",
        conv_types=("gt",),
        build_inputs=_build_gt_aggr_inputs,
        arg_order=("Q", "K", "V"),
    ),
)

CONV_TYPES: dict[str, ConvFamily] = {ct: family for family in CONV_FAMILIES for ct in family.conv_types}

ATTENTION_CONVS = frozenset(ct for ct in CONV_TYPES if CONV_TYPES[ct].name in {"gat_v2", "gt"})

#: turbo_gnn kernel parameters, per conv type, settable with ``-K name=value``.
#: These describe the ``cuda`` backend's kernels, so they are validated only for
#: that backend; every other backend takes -K as unvalidated pass-through.
#: Node->block scheduling, shared by the convs whose kernels map nodes onto blocks.
#: The SpMM convs are cuSPARSE and have no such mapping, so they do not take these.
_SCHEDULE_PARAMS = (
    KernelParam(
        "schedule",
        str,
        "one_per_block",
        "Node->block policy: one_per_block (grid.x == node count) | grid_stride | precomputed | dynamic.",
        choices=("one_per_block", "grid_stride", "precomputed", "dynamic"),
    ),
    KernelParam("blocks_per_sm", int, 1024, "Resident blocks per SM targeted by the persistent policies."),
    KernelParam("sched_chunk", int, 4, "Work items claimed per atomic (dynamic policy only)."),
    KernelParam(
        "forward_bucket_launch",
        str,
        "sequential",
        "Forward light/heavy bucket launch: sequential (one stream) | concurrent (separate "
        "streams, heavy issued first).",
        choices=("sequential", "concurrent"),
    ),
    KernelParam(
        "backward_bucket_launch",
        str,
        "sequential",
        "Backward light/heavy bucket launch. Concurrency helps forward and hurts backward, so "
        "the two are set independently.",
        choices=("sequential", "concurrent"),
    ),
)

_REDUCTION_PARAMS = (
    KernelParam("warps_per_block", int, 8, "Warps per block for the light-node atomic kernel."),
    KernelParam("edges_per_block_heavy_nodes", int, 128, "Edges per block for the heavy-node tiled kernel."),
    KernelParam(
        "forward_heavy_slice_blocks_per_sm",
        float,
        0.0,
        "Heavy slice as min_heavy_degree/divisor; 0 disables. Overridden by forward_heavy_edge_slice.",
    ),
    KernelParam(
        "forward_heavy_edge_slice",
        int,
        0,
        "Flat edge-slice size for the heavy bucket; 0 keeps the rectangular tiled grid.",
    ),
    KernelParam("use_2d_kernel", _parse_bool, False, "Use the 2-D tiled heavy-node kernel variant."),
    KernelParam("features_per_block", int, 32, "Feature tile size (2-D kernel only)."),
    KernelParam("tiles_y", int, 8, "Row tile count (2-D kernel only)."),
) + _SCHEDULE_PARAMS

_SPMM_PARAMS = (
    KernelParam("cu_sparse_algorithm_id", int, -1, "cuSPARSE algorithm id (-1 = library default)."),
    KernelParam("block_dim", int, 256, "Block size of the normalization pre-pass."),
)

_GATV2_PARAMS = (
    KernelParam("negative_slope", float, 0.2, "LeakyReLU negative slope."),
    KernelParam("grad_A_reduce_row_chunk_size", int, 512, "Row chunk size for the backward attention reduction."),
    KernelParam("forward_light_warps", int, 1, "Warps per block, forward light-node kernel."),
    KernelParam("forward_heavy_warps", int, 8, "Warps per block, forward heavy-node kernel."),
    KernelParam("backward_light_warps", int, 1, "Warps per block, backward light-node kernel."),
    KernelParam("backward_heavy_warps", int, 8, "Warps per block, backward heavy-node kernel."),
    KernelParam(
        "forward_heavy_slice_blocks_per_sm",
        float,
        0.0,
        "Heavy slice as min_heavy_degree/divisor; 0 disables. Overridden by forward_heavy_edge_slice.",
    ),
    KernelParam(
        "forward_heavy_edge_slice",
        int,
        0,
        "Edges per block in the forward heavy bucket; 0 keeps one block per heavy node.",
    ),
) + _SCHEDULE_PARAMS

_GT_PARAMS = (
    KernelParam("scale", _parse_optional_float, None, "Score scale; 'none' means 1/sqrt(head_dim)."),
    KernelParam("forward_light_warps", int, 4, "Warps per block, forward light-node kernel."),
    KernelParam("forward_heavy_warps", int, 8, "Warps per block, forward heavy-node kernel."),
    KernelParam("backward_light_warps", int, 1, "Warps per block, backward light-node kernel."),
    KernelParam("backward_heavy_warps", int, 8, "Warps per block, backward heavy-node kernel."),
    KernelParam(
        "forward_heavy_slice_blocks_per_sm",
        float,
        0.0,
        "Heavy slice as min_heavy_degree/divisor; 0 disables. Overridden by forward_heavy_edge_slice.",
    ),
    KernelParam(
        "forward_heavy_edge_slice",
        int,
        0,
        "Edges per block in the forward heavy bucket; 0 keeps one block per heavy node.",
    ),
    KernelParam(
        "backward_heavy_slice_blocks_per_sm",
        float,
        0.0,
        "Heavy slice as min_heavy_degree/divisor; 0 disables. Overridden by backward_heavy_edge_slice.",
    ),
    KernelParam(
        "backward_heavy_edge_slice",
        int,
        0,
        "Edges per block in the backward heavy bucket; 0 keeps one block per heavy node.",
    ),
) + _SCHEDULE_PARAMS

CUDA_CONV_PARAMS: dict[str, tuple[KernelParam, ...]] = {
    "min_aggr": _REDUCTION_PARAMS,
    "max_aggr": _REDUCTION_PARAMS,
    "sum_aggr": _SPMM_PARAMS,
    "mean_aggr": _SPMM_PARAMS,
    "gcn": _SPMM_PARAMS,
    "gat_v2": _GATV2_PARAMS,
    "gt": _GT_PARAMS,
}

#: Convs whose turbo_gnn entry point accepts ``autotune=True``. The SpMM op has
#: no autotuning path, so --autotune is rejected for the norm-based convs.
AUTOTUNABLE_CONVS = frozenset({"min_aggr", "max_aggr", "gat_v2", "gt"})


def _parse_scalar(raw: str) -> Any:
    """Parse a ``-K`` value in backend mode, where there is no declared schema.

    Args:
        raw (str): Value as typed on the command line.

    Returns:
        Any: bool, int, float or the original string.
    """
    text = raw.strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low == "none":
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def resolve_backend_kwargs(raw_args: Sequence[str] | None) -> dict[str, Any]:
    """Parse ``-K name=value`` overrides forwarded to ``create_aggr``.

    Backend aggregations have no declared parameter schema, so values are
    inferred rather than validated; a backend that does not use a given name
    will silently ignore it.

    Args:
        raw_args (Sequence[str] | None): Raw ``name=value`` strings from the CLI.

    Returns:
        dict[str, Any]: Keyword arguments for ``create_aggr``.

    Raises:
        SystemExit: If an entry is not of the form ``name=value``.
    """
    kwargs: dict[str, Any] = {}
    for item in raw_args or ():
        name, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"Kernel argument {item!r} is not of the form name=value.")
        kwargs[name.strip()] = _parse_scalar(raw)
    return kwargs


def _format_default(value: Any) -> str:
    """Render a parameter default for help output.

    Args:
        value (Any): Default value.

    Returns:
        str: Printable representation.
    """
    return "none" if value is None else str(value)


def format_kernel_catalog() -> str:
    """Render the conv catalog, cuda -K arguments, and available backends.

    Returns:
        str: Catalog text, used verbatim as the --help epilog.
    """
    lines = [
        "--conv values, and the tensors each aggregation receives:",
        "",
    ]
    for family in CONV_FAMILIES:
        args_sig = ", ".join(family.arg_order)
        lines.append(f"  {', '.join(family.conv_types):<46} takes ({args_sig})")
    lines += [
        "",
        "  For gat_v2 the per-head width is --feature-dim (total heads*feature-dim);",
        "  for gt, --feature-dim is split across --heads. This follows create_aggr.",
        "",
        "-K arguments for --backend cuda (turbo_gnn kernel parameters). Other backends",
        "take -K as unvalidated pass-through to create_aggr:",
        "",
    ]
    for conv in sorted(CUDA_CONV_PARAMS):
        tunable = " (supports --autotune)" if conv in AUTOTUNABLE_CONVS else ""
        lines.append(f"  {conv}{tunable}")
        for param in CUDA_CONV_PARAMS[conv]:
            choices = f" one of {{{', '.join(map(str, param.choices))}}};" if param.choices else ""
            lines.append(f"        {param.name:<32} default={_format_default(param.default):<8}{choices} {param.help}")
        lines.append("")

    lines += ["Backends registered in this environment:"]
    for name in _available_aggr_backends():
        lines.append(f"  {name}")
    return "\n".join(lines)


@contextlib.contextmanager
def stdout_to_stderr():
    """Send everything written to stdout to stderr for the duration of the block.

    Backends chatter while importing ("TCGNN is not found!") and while JIT
    compiling ("ninja: no work to do."). The latter comes from a *subprocess*
    writing to file descriptor 1, which ``contextlib.redirect_stdout`` cannot
    intercept, so fd 1 is duplicated onto fd 2 as well. This keeps stdout
    carrying the JSON result and nothing else.

    Yields:
        None: For the duration of the redirect.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def import_backends() -> None:
    """Import src.backends so the registry is populated."""
    with stdout_to_stderr():
        import src.backends  # noqa: F401  -- registers every importable backend


def _available_aggr_backends() -> list[str]:
    """List registered backends that implement a projection-free aggregation.

    Importing backends is best-effort: optional dependencies (DGL, cuGraph,
    TC-GNN, ...) are simply absent from the result when not installed.

    Returns:
        list[str]: Sorted backend names, or a single explanatory entry on failure.
    """
    try:
        import_backends()

        from src.backends.base import BaseBackend
        from src.backends.registry import BackendRegistry

        return sorted(
            name
            for name in BackendRegistry.list_backends()
            if BackendRegistry._backends[name].create_aggr is not BaseBackend.create_aggr
        )
    except Exception:  # noqa: BLE001 -- help text must render even if src/ cannot import
        return ["(could not import src.backends)"]


def resolve_kernel_params(
    conv: str, params: tuple[KernelParam, ...], raw_args: Sequence[str] | None
) -> tuple[dict[str, Any], list[str]]:
    """Merge ``-K name=value`` overrides onto a conv's kernel-parameter defaults.

    Args:
        conv (str): Conv type being benchmarked, for error messages.
        params (tuple[KernelParam, ...]): Declared parameters for that conv.
        raw_args (Sequence[str] | None): Raw ``name=value`` strings from the CLI.

    Returns:
        tuple[dict[str, Any], list[str]]: Resolved parameters and the names set explicitly.

    Raises:
        SystemExit: On an unknown name, unparseable value, or value outside `choices`.
    """
    by_name = {param.name: param for param in params}
    values: dict[str, Any] = {param.name: param.default for param in params}
    explicit: list[str] = []

    for item in raw_args or ():
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep:
            raise SystemExit(f"Kernel argument {item!r} is not of the form name=value.")
        if name not in by_name:
            valid = ", ".join(by_name) or "(none)"
            raise SystemExit(f"Unknown kernel argument {name!r} for conv {conv!r}. Valid names: {valid}")

        param = by_name[name]
        try:
            value = param.parse(raw.strip())
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Cannot parse value for kernel argument {name!r}: {exc}") from exc
        if param.choices is not None and value not in param.choices:
            allowed = ", ".join(map(str, param.choices))
            raise SystemExit(f"Kernel argument {name!r} must be one of {{{allowed}}}, got {value!r}.")

        values[name] = value
        explicit.append(name)

    return values, explicit


# --------------------------------------------------------------------------
# Graph-construction parameter sweep
# --------------------------------------------------------------------------

#: Graph-construction parameters that ``--sweep`` can vary, mapped to their
#: value parsers. ``quantile`` sets the light/heavy split for both directions;
#: the directional names override it per direction. -1 disables bucketing, so
#: every node lands in the light bucket.
SWEEPABLE: dict[str, Callable[[str], Any]] = {
    "quantile": float,
    "forward_quantile": float,
    "backward_quantile": float,
    "index_dtype": str,
    "node_order": str,
}

#: Order in which the light/heavy buckets are walked. This only changes *which node a block
#: visits*, never where its result is written, so every order is bit-exact -- but it is the
#: largest single performance parameter on these kernels, because cache reuse depends on
#: whether nodes processed close together share neighbours.
NODE_ORDERS = ("natural", "degree", "locality")

QUANTILE_KEYS = ("quantile", "forward_quantile", "backward_quantile")

#: Kernel parameters that ``--sweep-kernel`` can vary *within a single process*, so a whole
#: grid is measured against one loaded graph instead of one subprocess per point. Only
#: parameters the op accepts at call time belong here.
SWEEPABLE_KERNEL: dict[str, Callable[[str], Any]] = {
    "schedule": str,
    "forward_bucket_launch": str,
    "backward_bucket_launch": str,
    "blocks_per_sm": int,
    "sched_chunk": int,
    # Absolute heavy-slice sizes, so the raw-edge-count form can be compared head to head
    # against the device-relative blocks-per-SM form under identical pins.
    "forward_heavy_edge_slice": int,
    "backward_heavy_edge_slice": int,
}


def parse_sweep(raw_args: Sequence[str] | None) -> dict[str, list[Any]]:
    """Parse ``--sweep name=v1,v2,...`` entries into candidate lists.

    Args:
        raw_args (Sequence[str] | None): Raw ``name=v1,v2`` strings from the CLI.

    Returns:
        dict[str, list[Any]]: Candidate values per parameter, empty if none given.

    Raises:
        SystemExit: On a malformed entry, unknown name, or unparseable value.
    """
    sweep: dict[str, list[Any]] = {}
    for item in raw_args or ():
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep:
            raise SystemExit(f"--sweep entry {item!r} is not of the form name=v1,v2,...")
        if name not in SWEEPABLE:
            raise SystemExit(f"Unknown --sweep parameter {name!r}. Valid names: {', '.join(SWEEPABLE)}")
        parse = SWEEPABLE[name]
        values: list[Any] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(parse(token))
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Cannot parse --sweep value {token!r} for {name!r}: {exc}") from exc
        if not values:
            raise SystemExit(f"--sweep {name!r} lists no values.")
        if name == "index_dtype":
            unknown = [v for v in values if v not in INDEX_DTYPES]
            if unknown:
                raise SystemExit(
                    f"--sweep index_dtype has unknown value(s) {unknown}; valid: {', '.join(INDEX_DTYPES)}"
                )
        if name == "node_order":
            unknown = [v for v in values if v not in NODE_ORDERS]
            if unknown:
                raise SystemExit(f"--sweep node_order has unknown value(s) {unknown}; valid: {', '.join(NODE_ORDERS)}")
        sweep[name] = values
    return sweep


KERNEL_SWEEP_PREFIX = "kernel:"


def parse_kernel_sweep(raw_args: Sequence[str] | None) -> dict[str, list[Any]]:
    """Parse ``--sweep-kernel NAME=v1,v2`` entries.

    Raises:
        SystemExit: On a malformed entry or a parameter that cannot be set at call time.
    """
    out: dict[str, list[Any]] = {}
    for item in raw_args or ():
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep:
            raise SystemExit(f"--sweep-kernel entry {item!r} is not of the form name=v1,v2,...")
        if name not in SWEEPABLE_KERNEL:
            raise SystemExit(f"Unknown --sweep-kernel parameter {name!r}. Valid: {', '.join(SWEEPABLE_KERNEL)}")
        parse = SWEEPABLE_KERNEL[name]
        values = [parse(tok.strip()) for tok in raw.split(",") if tok.strip()]
        if not values:
            raise SystemExit(f"--sweep-kernel {name!r} lists no values.")
        out[name] = values
    return out


def split_point(point: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate a sweep point into (graph parameters, kernel overrides)."""
    graph_cfg = {k: v for k, v in point.items() if not k.startswith(KERNEL_SWEEP_PREFIX)}
    kernel_cfg = {k[len(KERNEL_SWEEP_PREFIX) :]: v for k, v in point.items() if k.startswith(KERNEL_SWEEP_PREFIX)}
    return graph_cfg, kernel_cfg


def sweep_points(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a sweep into the list of configurations to measure.

    Args:
        sweep (dict[str, list[Any]]): Candidate values per parameter.

    Returns:
        list[dict[str, Any]]: One dict per point; a single empty dict when not sweeping.
    """
    if not sweep:
        return [{}]
    names = list(sweep)
    return [dict(zip(names, combo)) for combo in itertools.product(*(sweep[n] for n in names))]


def apply_sweep_point(args: argparse.Namespace, point: dict[str, Any]) -> argparse.Namespace:
    """Copy ``args`` with one sweep configuration applied.

    Args:
        args (argparse.Namespace): Base CLI args.
        point (dict[str, Any]): One sweep configuration.

    Returns:
        argparse.Namespace: Copy carrying the swept values.
    """
    merged = argparse.Namespace(**vars(args))
    if "quantile" in point:
        merged.quantile = point["quantile"]
    if "index_dtype" in point:
        merged.index_dtype = point["index_dtype"]
    if "node_order" in point:
        merged.node_order = point["node_order"]
    return merged


def reorder_nodes(graph: BenchGraph, order: str) -> BenchGraph:
    """Return ``graph`` with its node buckets walked in ``order``.

    Reordering shares the CSR arrays and rebuilds only the bucket index arrays, so it costs a
    sort rather than a graph rebuild -- except ``locality``, which runs reverse Cuthill-McKee
    on the host and is the one order worth measuring the preparation cost of separately.

    Args:
        graph (BenchGraph): Base graph.
        order (str): One of ``natural``, ``degree``, ``locality``.

    Returns:
        BenchGraph: Reordered graph, or ``graph`` itself for ``natural``.

    Raises:
        SystemExit: On an unknown order, or if ``locality`` is asked for without scipy.
    """
    if order == "natural":
        return graph
    if not isinstance(graph.repr, AdjacencyForwardBackwardWithNodeBuckets):
        raise SystemExit(f"--node-order {order!r} needs the turbo_gnn CSR representation (--backend cuda).")
    if order == "degree":
        reordered = graph.repr.sorted_by_degree()
    elif order == "locality":
        try:
            reordered = graph.repr.sorted_by_locality()
        except ImportError as exc:
            raise SystemExit(f"--node-order locality needs scipy: {exc}") from exc
    else:
        raise SystemExit(f"Unknown --node-order {order!r}; valid: {', '.join(NODE_ORDERS)}")
    return BenchGraph(reordered, graph.num_nodes, _csr_stats(reordered), _collect_tensors(reordered))


def repartition_for(graph: BenchGraph, point: dict[str, Any]) -> BenchGraph | None:
    """Re-bucket an already-built CSR graph for a sweep point, if that suffices.

    Re-bucketing shares the CSR arrays, so sweeping the heavy-node threshold
    costs a quantile computation rather than a full graph rebuild. Returns None
    when the point changes something bucketing cannot express (a different index
    dtype), or when the representation has no node buckets at all.

    Args:
        graph (BenchGraph): Base graph.
        point (dict[str, Any]): One sweep configuration.

    Returns:
        BenchGraph | None: Re-bucketed graph, or None if a rebuild is required.
    """
    if not isinstance(graph.repr, AdjacencyForwardBackwardWithNodeBuckets):
        return None
    if "index_dtype" in point:
        return None

    base = point.get("quantile")
    fwd = point.get("forward_quantile", base)
    bwd = point.get("backward_quantile", base)
    if fwd is None and bwd is None:
        return None

    kwargs: dict[str, Any] = {}
    if fwd is not None:
        kwargs["forward_huge_degree_threshold_quantile"] = fwd
    if bwd is not None:
        kwargs["backward_huge_degree_threshold_quantile"] = bwd

    repartitioned = graph.repr.repartition(**kwargs)
    return BenchGraph(
        repartitioned,
        graph.num_nodes,
        _csr_stats(repartitioned),
        _collect_tensors(repartitioned),
    )


def make_random_graph(
    num_nodes: int,
    avg_degree: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Generate an Erdos-Renyi-like ``edge_index`` with approximately ``avg_degree``.

    Args:
        num_nodes (int): Number of nodes.
        avg_degree (int): Approximate average out-degree.
        device (torch.device): Torch device.

    Returns:
        torch.Tensor: edge_index of shape ``[2, E]``.
    """
    num_edges = max(1, num_nodes * max(1, avg_degree))
    src = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


@dataclass
class BenchGraph:
    """A graph in one backend's representation, plus everything reported about it.

    Attributes:
        repr: Backend-specific graph object handed to the kernel or aggregation.
        num_nodes: Node count.
        stats: Summary written to the JSON payload.
        tensors: Device tensors held by the representation, for memory accounting.
            Empty when the representation is opaque (e.g. a DGL graph).
    """

    repr: Any
    num_nodes: int
    stats: dict[str, Any]
    tensors: list[torch.Tensor]


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append one self-loop per node to a COO edge list.

    Args:
        edge_index (torch.Tensor): Edge list of shape ``[2, E]``.
        num_nodes (int): Node count.

    Returns:
        torch.Tensor: Edge list of shape ``[2, E + num_nodes]``.
    """
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype).expand(2, num_nodes)
    return torch.cat([edge_index, loops], dim=1)


def _csr_stats(graph: AdjacencyForwardBackwardWithNodeBuckets) -> dict[str, Any]:
    """Summarize a turbo_gnn dual-CSR graph.

    Args:
        graph (AdjacencyForwardBackwardWithNodeBuckets): Graph under test.

    Returns:
        dict[str, Any]: Node/edge counts, degree stats and bucket sizes.
    """
    num_nodes = graph.forward_indptr.numel() - 1
    num_edges = graph.forward_indices.numel()
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": num_edges / num_nodes if num_nodes else 0.0,
        "max_degree": graph.max_degree,
        "is_directed": bool(graph.is_directed),
        "index_dtype": str(graph.index_dtype).removeprefix("torch."),
        "forward_heavy_nodes": graph.forward_heavy_nodes.numel(),
        "backward_heavy_nodes": graph.backward_heavy_nodes.numel(),
    }


def _collect_tensors(obj: Any) -> list[torch.Tensor]:
    """Best-effort collection of device tensors inside a graph representation.

    Representations vary per backend: the turbo_gnn CSR object, plain tuples of
    tensors, or opaque objects (DGL graphs) that expose none. Returns an empty
    list for the opaque case, which surfaces as a null ``graph_mb``.

    Args:
        obj (Any): Backend-specific graph representation.

    Returns:
        list[torch.Tensor]: Tensors found, possibly empty.
    """
    if isinstance(obj, AdjacencyForwardBackwardWithNodeBuckets):
        return graph_tensors(obj)
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        found: list[torch.Tensor] = []
        for item in obj:
            found.extend(_collect_tensors(item))
        return found
    return []


def load_graph(args: argparse.Namespace, device: torch.device, conv_backend: str) -> BenchGraph:
    """Build the graph in ``conv_backend``'s representation, from a dataset or at random.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        device (torch.device): Torch device.
        conv_backend (str): Backend name selecting the graph representation.

    Returns:
        BenchGraph: Representation plus reported stats.
    """
    index_dtype = INDEX_DTYPES[args.index_dtype]
    kernel_kwargs = {
        "index_dtype": args.index_dtype,
        "huge_degree_threshold_quantile": args.quantile,
    }

    # Fast path: the turbo_gnn CSR can be built directly, keeping the dataset
    # loaders -- and their heavyweight deps (PyG / DGL / OGB) -- unimported.
    if args.dataset is None and conv_backend == "cuda":
        edge_index = make_random_graph(args.num_nodes, args.avg_degree, device=device)
        if args.self_loops:
            edge_index = _add_self_loops(edge_index, args.num_nodes)
        graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
            edge_index,
            num_nodes=args.num_nodes,
            quantile=args.quantile,
            index_dtype=index_dtype,
        ).to(device)
        return BenchGraph(graph, args.num_nodes, _csr_stats(graph), graph_tensors(graph))

    from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR, DatasetConfig, GraphSample, load_single_graph

    if args.dataset is None:
        num_nodes = args.num_nodes
        edge_index = make_random_graph(num_nodes, args.avg_degree, device=device)
        sample = GraphSample(
            backend=MODEL_BACKEND_TO_GRAPH_REPR[conv_backend],
            # Only the node count is read off these; keep them zero-width so the
            # benchmark does not carry feature/label buffers it never uses.
            x=torch.empty(num_nodes, 0, device=device),
            y=torch.empty(num_nodes, 0, device=device),
            edge_index=edge_index,
            add_self_loops=args.self_loops,
            kernel_related_kwargs=kernel_kwargs,
        )
    else:
        import yaml

        with open(args.dataset, encoding="utf-8") as handle:
            dataset_cfg = yaml.safe_load(handle)["dataset"]

        sample = load_single_graph(
            DatasetConfig(
                source=dataset_cfg["source"],
                name=dataset_cfg["name"],
                root=dataset_cfg["root"],
                conv_backend=conv_backend,
                kernel_related_kwargs=kernel_kwargs,
            )
        )

    graph = sample.graph_repr
    if hasattr(graph, "to"):
        graph = graph.to(device)

    num_nodes = sample.num_nodes
    if isinstance(graph, AdjacencyForwardBackwardWithNodeBuckets):
        stats = _csr_stats(graph)
    else:
        num_edges = sample.edge_index.shape[1]  # type: ignore
        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": num_edges / num_nodes if num_nodes else 0.0,
            "graph_repr": MODEL_BACKEND_TO_GRAPH_REPR[conv_backend],
        }
    return BenchGraph(graph, num_nodes, stats, _collect_tensors(graph))


def tensor_megabytes(tensors: Sequence[torch.Tensor]) -> float:
    """Total size of ``tensors`` in MiB, counting shared storage only once.

    Undirected graphs alias their backward CSR onto the forward one, so naive
    summing would double-count it.

    Args:
        tensors (Sequence[torch.Tensor]): Tensors to measure.

    Returns:
        float: Combined size in MiB.
    """
    by_storage = {t.data_ptr(): t.numel() * t.element_size() for t in tensors}
    return sum(by_storage.values()) / 1024**2  # type: ignore


def graph_tensors(graph: AdjacencyForwardBackwardWithNodeBuckets) -> list[torch.Tensor]:
    """Collect every device tensor held by the graph representation.

    Args:
        graph (AdjacencyForwardBackwardWithNodeBuckets): Graph under test.

    Returns:
        list[torch.Tensor]: The dual CSR arrays plus the light/heavy bucket indices.
    """
    return [
        graph.forward_indptr,
        graph.forward_indices,
        graph.backward_indptr,
        graph.backward_indices,
        graph.forward_light_nodes,
        graph.forward_heavy_nodes,
        graph.backward_light_nodes,
        graph.backward_heavy_nodes,
    ]


def build_timed_callable(
    forward: Callable[[], torch.Tensor],
    mode: str,
    differentiable: list[torch.Tensor],
) -> Callable[[], Any]:
    """Wrap the kernel call into the zero-arg callable the timer measures.

    ``forward`` is always invoked once up front. That primes lazily built state
    (cuSPARSE descriptors, and the autotuning grid search when it is enabled) so
    the search cost never lands inside the measured region.

    Args:
        forward (Callable[[], torch.Tensor]): Runs the kernel's forward pass.
        mode (str): One of ``forward``, ``backward``, ``forward_backward``.
        differentiable (list[torch.Tensor]): Kernel inputs that require gradients.

    Returns:
        Callable[[], Any]: Callable to hand to the timer.

    Raises:
        SystemExit: If a backward pass is requested but no input requires gradients.
    """
    if mode == "forward":
        forward()
        return forward

    if not differentiable:
        raise SystemExit(f"Mode {mode!r} needs at least one differentiable input, but none require grad.")

    out = forward()
    grad_output = torch.randn_like(out)

    if mode == "backward":
        # Time the backward kernel alone: the forward graph above is reused every
        # iteration, so only the gradient kernels are inside the measured region.
        def _timed() -> None:
            out.backward(grad_output, retain_graph=True)

        return _timed

    def _timed_fwd_bwd() -> None:
        forward().backward(grad_output)

    return _timed_fwd_bwd


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: Parsed args.
    """

    class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        """Show argument defaults while leaving the kernel catalog epilog unwrapped."""

        def _get_help_string(self, action: argparse.Action) -> str:
            """Suppress the '(default: ...)' suffix for required arguments.

            Args:
                action (argparse.Action): Argument being rendered.

            Returns:
                str: Help text for the argument.
            """
            if action.required:
                return action.help or ""
            return super()._get_help_string(action)  # type: ignore

    class _Parser(argparse.ArgumentParser):
        """Builds the kernel/backend catalog only when help is actually rendered.

        The catalog enumerates installed backends, which means importing
        src.backends. Doing that eagerly would slow every run and pull in
        PyG/DGL even for a plain turbo_gnn kernel benchmark.
        """

        def format_help(self) -> str:
            self.epilog = format_kernel_catalog()
            return super().format_help()

    p = _Parser(
        description="Microbenchmark projection-free graph convolutions (turbo_gnn kernels and other backends).",
        formatter_class=_HelpFormatter,
    )
    p.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Backend whose projection-free aggregation to benchmark: cuda, pyg, dgl, cusparse, ...",
    )
    p.add_argument(
        "--conv",
        type=str,
        required=True,
        choices=sorted(CONV_TYPES),
        help="Convolution type to benchmark.",
    )
    p.add_argument("--device", type=int, default=0, help="CUDA device ordinal.")

    graph_group = p.add_argument_group("graph")
    graph_group.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset YAML. If omitted, a random graph with --num-nodes and --avg-degree is generated.",
    )
    graph_group.add_argument("--num-nodes", type=int, default=20000, help="Nodes in the generated random graph.")
    graph_group.add_argument(
        "--avg-degree", type=int, default=10, help="Average out-degree of the generated random graph."
    )
    graph_group.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Degree quantile splitting light from heavy nodes (-1 puts every node in the light bucket).",
    )
    graph_group.add_argument(
        "--index-dtype",
        type=str,
        default="int32",
        choices=sorted(INDEX_DTYPES),
        help="Dtype of the CSR index arrays.",
    )
    graph_group.add_argument(
        "--sweep",
        action="append",
        metavar="NAME=V1,V2,...",
        help="Sweep a graph-construction parameter and report every value, e.g. "
        "--sweep quantile=-1,0.9,0.99. Repeatable; multiple entries form a grid. "
        f"Valid names: {', '.join(SWEEPABLE)}.",
    )
    graph_group.add_argument(
        "--sweep-kernel",
        action="append",
        metavar="NAME=V1,V2,...",
        help="Sweep a kernel parameter inside one process, e.g. --sweep-kernel schedule=one_per_block,dynamic. "
        "Combines with --sweep as one grid. Repeatable. Valid names: " + ", ".join(SWEEPABLE_KERNEL) + ".",
    )
    graph_group.add_argument(
        "--node-order",
        type=str,
        default="natural",
        choices=list(NODE_ORDERS),
        help="Order the light/heavy buckets are walked in. 'degree' is descending degree "
        "(balances; cost tracks degree). 'locality' is reverse Cuthill-McKee (clusters "
        "connected nodes so neighbouring feature rows stay in L2; needs scipy). Bit-exact "
        "either way -- this changes only which node a block visits.",
    )
    graph_group.add_argument(
        "--no-self-loops",
        dest="self_loops",
        action="store_false",
        help="Skip adding one self-loop per node. Self-loops are added by default so that "
        "random graphs and datasets, and every backend, see the same edge set.",
    )

    input_group = p.add_argument_group("inputs")
    input_group.add_argument("--feature-dim", type=int, default=128, help="Total feature width (heads * head_dim).")
    input_group.add_argument("--heads", type=int, default=1, help="Attention heads (attention kernels only).")
    input_group.add_argument("--dtype", type=str, default="fp32", choices=sorted(FEATURE_DTYPES), help="Feature dtype.")
    input_group.add_argument("--seed", type=int, default=0, help="Seed for the generated graph and inputs.")

    run_group = p.add_argument_group("measurement")
    run_group.add_argument(
        "--mode",
        type=str,
        default="forward",
        choices=["forward", "backward", "forward_backward"],
        help="Which pass to time. 'backward' reuses one forward graph and times only the gradient kernels.",
    )
    run_group.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed budget in MILLISECONDS of repetition (triton.testing.do_bench). "
        "Ignored when --launch-ncu-override-iters is given.",
    )
    run_group.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup budget in MILLISECONDS. Ignored when --launch-ncu-override-iters is given.",
    )
    run_group.add_argument(
        "--launch-ncu-override-iters",
        type=int,
        metavar="N",
        help="Issue exactly N timed calls instead of timing against the --iters millisecond "
        "budget, giving 1 + M + N launches in total (one priming call, M warmup, N timed). "
        "do_bench derives its own repeat counts and adds 6 calibration calls, so it cannot "
        "produce a small deterministic launch count; this can. Use it under a kernel profiler "
        "such as ncu. Overrides --iters and --warmup.",
    )
    run_group.add_argument(
        "--launch-ncu-override-warmup",
        type=int,
        metavar="M",
        help="Exact number of warmup calls under --launch-ncu-override-iters (default: a quarter of N, at least 1).",
    )

    tune_group = p.add_argument_group("autotuning (disabled by default)")
    tune_group.add_argument(
        "--autotune",
        action="store_true",
        help="Grid-search the tunable parameters before timing. Off by default, so a run measures "
        "exactly the configuration given by -K.",
    )
    tune_group.add_argument("--autotune-warmup", type=int, default=10, help="Warmup iters per autotuning trial.")
    tune_group.add_argument("--autotune-iters", type=int, default=50, help="Timed iters per autotuning trial.")
    tune_group.add_argument(
        "--autotune-exclude",
        type=str,
        default=None,
        metavar="NAME[,NAME...]",
        help="Hold these tunable parameters fixed instead of searching them; they keep whatever "
        "value -K gave them. This is what makes an ablation expressible -- 'autotune everything "
        "except the scheduler' needs the scheduler pinned while the rest is searched.",
    )

    p.add_argument(
        "-K",
        "--kernel-arg",
        action="append",
        metavar="NAME=VALUE",
        help="Set a kernel argument, e.g. -K warps_per_block=16. Repeatable.",
    )
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON result.")
    return p.parse_args()


@dataclass(frozen=True)
class BenchTarget:
    """What is being benchmarked, resolved from either --kernel or --backend.

    Attributes:
        conv_backend: Backend name selecting which graph representation to build.
        heads: Attention head count (1 for non-attention targets).
        build_inputs: Callable creating the input tensors.
        make_forward: Given the graph and inputs, returns the zero-arg forward call.
        payload: Descriptive fields merged into the JSON result.
    """

    conv_backend: str
    heads: int
    build_inputs: Callable[[InputContext], dict[str, torch.Tensor]]
    make_forward: Callable[..., Callable[[], torch.Tensor]]
    payload: dict[str, Any]


def prepare_target(args: argparse.Namespace, device: torch.device) -> BenchTarget:
    """Resolve the run: a backend's projection-free aggregation.

    Uses ``BaseBackend.create_aggr``, which returns the backend's aggregation
    with the linear / QKV projections stripped, so the measurement covers the
    graph convolution alone. For the ``cuda`` backend the aggregation forwards
    every kernel parameter, so ``-K`` and ``--autotune`` reach the turbo_gnn
    kernel exactly as a direct call would.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        device (torch.device): Torch device.

    Returns:
        BenchTarget: Resolved target.

    Raises:
        SystemExit: If the backend is unknown, has no aggregation for this conv
            type, rejects the arguments, or cannot autotune.
    """
    import_backends()

    from src.backends.registry import BackendRegistry

    try:
        backend = BackendRegistry.get_backend(args.backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    family = CONV_TYPES[args.conv]
    heads = args.heads if args.conv in ATTENTION_CONVS else 1
    if args.conv == "gt" and args.feature_dim % heads:
        raise SystemExit(f"--feature-dim {args.feature_dim} is not divisible by --heads {heads}.")

    is_cuda = args.backend == "cuda"
    if is_cuda:
        # The turbo_gnn kernel parameters are known, so -K is validated here.
        kernel_params, explicit_params = resolve_kernel_params(args.conv, CUDA_CONV_PARAMS[args.conv], args.kernel_arg)
    else:
        # Other backends declare no schema; pass -K through untouched.
        kernel_params, explicit_params = resolve_backend_kwargs(args.kernel_arg), []

    call_kwargs: dict[str, Any] = {}
    if args.autotune:
        if not is_cuda:
            raise SystemExit("--autotune requires --backend cuda; other backends' aggregations are not tunable.")
        if args.conv not in AUTOTUNABLE_CONVS:
            tunable = ", ".join(sorted(AUTOTUNABLE_CONVS))
            raise SystemExit(f"Conv {args.conv!r} has no autotuning path. Autotunable convs: {tunable}.")
        call_kwargs = {
            "autotune": True,
            "autotune_config": AutotuneConfig(
                warmup=args.autotune_warmup,
                iters=args.autotune_iters,
                # Without this the backward parameters are never searched, so a --mode backward
                # run would report the *default* backward configuration under an "autotuned"
                # label. Tie it to what is actually being timed.
                # NOTE: the inline autotuner never reads this and never searches the backward
                # parameters -- it optimises forward time only, whatever --mode says. Set here
                # so the intent survives if that is ever fixed; see reports/ for what it means
                # for the backward numbers.
                tune_backward=args.mode in ("backward", "forward_backward"),
                exclude=tuple(n.strip() for n in (args.autotune_exclude or "").split(",") if n.strip()),
            ),
        }

    aggr_kwargs: dict[str, Any] = {"feature_dim": args.feature_dim}
    # Only attention convs take `heads`; several backends build their simple
    # convs through create_conv, which rejects the unexpected keyword.
    if args.conv in ATTENTION_CONVS:
        aggr_kwargs["heads"] = heads
    aggr_kwargs |= kernel_params

    try:
        aggr = backend.create_aggr(args.conv, **aggr_kwargs)
    except (KeyError, NotImplementedError) as exc:
        raise SystemExit(
            f"Backend {args.backend!r} has no projection-free aggregation for conv {args.conv!r}: {exc}"
        ) from exc
    except TypeError as exc:
        raise SystemExit(f"Backend {args.backend!r} rejected the arguments for conv {args.conv!r}: {exc}") from exc
    aggr = aggr.to(device)

    def _make_forward(
        graph: BenchGraph, inputs: dict[str, torch.Tensor], overrides: dict[str, Any] | None = None
    ) -> Callable[[], torch.Tensor]:
        ordered = [inputs[name] for name in family.arg_order]
        # Call-time kwargs win over the ones the aggregation was constructed with, so a sweep
        # can vary a kernel parameter without rebuilding the aggregation or reloading the graph.
        merged = {**call_kwargs, **(overrides or {})}

        def _forward() -> torch.Tensor:
            return aggr(*ordered, graph.repr, **merged)

        return _forward

    return BenchTarget(
        conv_backend=args.backend,
        heads=heads,
        build_inputs=family.build_inputs,
        make_forward=_make_forward,
        payload={
            "backend": args.backend,
            "conv": args.conv,
            "conv_family": family.name,
            "autotune": args.autotune,
            "kernel_params": {k: _format_default(v) if v is None else v for k, v in kernel_params.items()},
            "kernel_params_set_explicitly": explicit_params,
        },
    )


@dataclass(frozen=True)
class TimingPlan:
    """How the measurement loop is bounded.

    Attributes:
        exact: True to issue exactly ``iters`` timed calls; False to time
            against a millisecond budget via ``triton.testing.do_bench``.
        iters: Timed call count when *exact*, else the timed budget in ms.
        warmup: Warmup call count when *exact*, else the warmup budget in ms.
    """

    exact: bool
    iters: int
    warmup: int


def resolve_timing(args: argparse.Namespace) -> TimingPlan:
    """Decide between millisecond budgets and exact launch counts.

    ``--iters``/``--warmup`` are always millisecond budgets;
    ``--launch-ncu-override-iters`` replaces them with exact call counts so a
    profiler sees a small, deterministic number of launches.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Returns:
        TimingPlan: Resolved bounds for the timing loop.

    Raises:
        SystemExit: On a non-positive count, or a warmup override with no iters override.
    """
    override = args.launch_ncu_override_iters
    if override is None:
        if args.launch_ncu_override_warmup is not None:
            raise SystemExit("--launch-ncu-override-warmup requires --launch-ncu-override-iters.")
        return TimingPlan(exact=False, iters=args.iters, warmup=args.warmup)

    if override < 1:
        raise SystemExit(f"--launch-ncu-override-iters must be at least 1, got {override}.")
    warmup = args.launch_ncu_override_warmup
    if warmup is None:
        warmup = max(1, override // 4)
    elif warmup < 0:
        raise SystemExit(f"--launch-ncu-override-warmup cannot be negative, got {warmup}.")
    return TimingPlan(exact=True, iters=override, warmup=warmup)


def autotune_selection() -> dict[str, Any] | None:
    """Report what the inline autotuner picked, including the graph partitioning.

    ``--autotune`` grid-searches kernel parameters *and* the light/heavy node
    threshold, then swaps in the repartitioned graph. Without this the chosen
    threshold is invisible: the caller only ever sees the resulting graph.

    Reads the kernel singleton's private cache, so it degrades to None rather
    than failing if that internal layout changes.

    Returns:
        dict[str, Any] | None: Selected configuration, or None if unavailable.
    """
    try:
        from turbo_gnn._autotune import TunableKernel

        for kernel in TunableKernel._shared_instances.values():
            for by_csr in kernel._inline_cache._cache.values():
                for by_shape in by_csr.values():
                    for entry in by_shape.values():
                        return {
                            "kernel_config": entry.get("kernel_config") or {},
                            "graph_config": entry.get("graph_config") or {},
                            "ms_per_iter": entry.get("ms_per_iter"),
                        }
    except Exception:  # noqa: BLE001 -- reporting must never fail the benchmark
        return None
    return None


def main() -> int:
    """Entry: run the kernel microbenchmark.

    Returns:
        int: Exit code.
    """
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("turbo_gnn kernels are CUDA-only and no CUDA device is available.")

    device = torch.device("cuda", args.device)
    torch.set_default_device(device)
    torch.manual_seed(args.seed)

    timing = resolve_timing(args)
    sweep = parse_sweep(args.sweep)
    kernel_sweep = parse_kernel_sweep(args.sweep_kernel)
    # One grid over both kinds. Kernel keys are prefixed so `repartition_for` and
    # `apply_sweep_point`, which only understand graph parameters, cannot misread them.
    combined = {**sweep, **{f"{KERNEL_SWEEP_PREFIX}{k}": v for k, v in kernel_sweep.items()}}
    points = sweep_points(combined)
    trials: list[dict[str, Any]] = []

    with stdout_to_stderr():
        target = prepare_target(args, device)

        base_graph = load_graph(args, device, target.conv_backend)

        ctx = InputContext(
            num_nodes=base_graph.num_nodes,
            feature_dim=args.feature_dim,
            heads=target.heads,
            head_dim=args.feature_dim // target.heads,
            device=device,
            dtype=FEATURE_DTYPES[args.dtype],
        )
        # Inputs depend only on node count, which no sweep parameter changes,
        # so they are built once and shared across every point.
        inputs = target.build_inputs(ctx)
        differentiable = [t for t in inputs.values() if t.requires_grad]

        for point in points:
            graph_cfg, kernel_cfg = split_point(point)
            if not graph_cfg or set(graph_cfg) == {"node_order"}:
                graph = base_graph
            else:
                graph = repartition_for(base_graph, graph_cfg)  # type: ignore
                if graph is None:
                    graph = load_graph(apply_sweep_point(args, graph_cfg), device, target.conv_backend)  # type: ignore
            # Applied last, so it composes with a swept quantile or index dtype: bucketing
            # decides bucket membership, ordering decides the walk within each bucket.
            graph = reorder_nodes(graph, graph_cfg.get("node_order", args.node_order))

            _forward = target.make_forward(graph, inputs, kernel_cfg)
            timed = build_timed_callable(_forward, args.mode, differentiable)

            if timing.exact:
                # No do_bench, so the allocator's high-water mark across the timing
                # loop *is* the kernel's peak. Reading it costs nothing and keeps
                # the launch count at exactly 1 + warmup + iters, which is the
                # whole point of the override.
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                res: MicrobenchResult = time_callable(
                    timed,
                    warmup=timing.warmup,
                    iters=timing.iters,
                    do_memory_profile=False,
                    exact_iters=True,
                )
                torch.cuda.synchronize(device)
                peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            else:
                # do_bench allocates a ~256 MB buffer to flush L2 between reps, which
                # would swamp a loop-wide high-water mark (291 MB vs 35 MB measured).
                # So peak comes from one isolated call outside the loop instead.
                res = time_callable(
                    timed,
                    warmup=timing.warmup,
                    iters=timing.iters,
                    do_memory_profile=True,
                    exact_iters=False,
                )
                peak_mb = res.memory_allocated

            trials.append({"graph_config": point, "graph": graph, "res": res, "inputs": inputs, "peak_mb": peak_mb})

    best = min(trials, key=lambda t: t["res"].ms_per_iter)
    graph, res = best["graph"], best["res"]

    torch.cuda.synchronize(device)
    # Steady state once the loop has settled: graph + inputs + whatever the
    # mode keeps alive (accumulated .grad buffers, a retained autograd graph).
    resident_mb = torch.cuda.memory_allocated(device) / 1024**2
    peak_mb = best["peak_mb"]
    memory = {
        "peak_mb": peak_mb,
        "resident_mb": resident_mb,
        # What one call transiently adds on top of the resident set -- the
        # figure to watch when comparing kernel configurations.
        "kernel_transient_mb": peak_mb - resident_mb if peak_mb is not None else None,
        # Null when the representation is opaque and holds no reachable tensors.
        "graph_mb": tensor_megabytes(graph.tensors) if graph.tensors else None,
        "inputs_mb": tensor_megabytes(list(inputs.values())),
    }

    ms_per_iter = res.ms_per_iter
    num_edges = graph.stats["num_edges"]
    result = (
        target.payload
        | {
            "mode": args.mode,
            "feature_dim": args.feature_dim,
            "heads": target.heads,
            "head_dim": ctx.head_dim,
            "dtype": args.dtype,
            "dataset": args.dataset,
            "self_loops": args.self_loops,
            "node_order": best["graph_config"].get("node_order", args.node_order),
            "graph": graph.stats,
            "iters_are_exact": timing.exact,
            # Exactly one pair is populated: launch counts, or millisecond budgets.
            "iters": timing.iters if timing.exact else None,
            "warmup_calls": timing.warmup if timing.exact else None,
            "timed_budget_ms": None if timing.exact else timing.iters,
            "warmup_budget_ms": None if timing.exact else timing.warmup,
            "ms_per_iter": ms_per_iter,
            "giga_edges_per_second": num_edges / (ms_per_iter * 1e6) if ms_per_iter > 0 else None,
            "device": res.device,
            "memory": memory,
        }
        | get_gpu_info(device)
    )

    if args.autotune:
        result["autotune_selected"] = autotune_selection()

    if sweep:
        # Reported values are the winning point; `sweep` carries every trial.
        result["graph_config"] = best["graph_config"]
        result["sweep"] = [
            {
                "graph_config": t["graph_config"],
                "ms_per_iter": t["res"].ms_per_iter,
                "heavy_nodes": t["graph"].stats.get("forward_heavy_nodes"),
            }
            for t in sorted(trials, key=lambda t: t["res"].ms_per_iter)
        ]

    print(json.dumps(result, indent=4))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
