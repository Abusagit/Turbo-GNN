import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .microbench import MicrobenchResult, time_callable

doc = """
Simple grid-search autotuner for backend-specific convolution configs.
"""


@dataclass
class TuningResult:
    """Autotuning result with best config and complete trial log."""

    best_config: dict[str, Any]
    best_result: MicrobenchResult
    trials: list[tuple[dict[str, Any], MicrobenchResult]]


def grid_autotune(
    target: Any,
    param_space: Mapping[str, Iterable[Any]],
    measure: Callable[[], Any],
    *,
    warmup: int = 10,
    iters: int = 50,
) -> TuningResult:
    """Run a grid search over `param_space` for `target.configure(**params)`.

    Args:
        target (Any): Object exposing optional `configure(**params)`.
        param_space (Mapping[str, Iterable[Any]]): Dict of param→candidates.
        measure (Callable[[], Any]): Zero-arg callable that runs the kernel.
        warmup (int): Warmup iterations.
        iters (int): Timed iterations.

    Returns:
        TuningResult: Best configuration and detailed trials.
    """
    keys = list(param_space.keys())
    trials: list[tuple[dict[str, Any], MicrobenchResult]] = []
    best: tuple[dict[str, Any], MicrobenchResult] | None = None

    for values in itertools.product(*(param_space[k] for k in keys)):
        cfg = dict(zip(keys, values, strict=False))
        if hasattr(target, "configure") and callable(target.configure):
            target.configure(**cfg)
        res = time_callable(measure, warmup=warmup, iters=iters)
        trials.append((cfg, res))
        if best is None or res.ms_per_iter < best[1].ms_per_iter:
            best = (cfg, res)

    assert best is not None
    return TuningResult(best_config=best[0], best_result=best[1], trials=trials)
