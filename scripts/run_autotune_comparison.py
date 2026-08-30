"""Autotuned kernels versus the untouched baseline, across every benchmark graph.

Two configurations per cell, and only two, because the question is no longer "which knob
setting wins" but "does turning the autotuner loose beat what the library did before any of
this work":

* **baseline** -- `schedule=one_per_block`, both bucket launches sequential, autotuning off.
  This is the launch these kernels used before the scheduler and stream work existed.
* **autotuned** -- `--autotune`, which grid-searches the declared `TunableParam`s. That now
  includes `forward_bucket_launch` and `backward_bucket_launch`, so the search can take
  concurrency on the forward pass and decline it on the backward one, per graph.

The autotuning search itself runs inside a priming call and is excluded from the timed region
by `benchmark_kernels.py`, so what is compared is the chosen configuration's steady-state cost,
not the cost of finding it.

    python scripts/run_autotune_comparison.py --out reports/autotune-comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_kernels import CUDA_CONV_PARAMS  # noqa: E402
from free_gpus import is_free, wait_until_free  # noqa: E402
from run_kernel_benchmark_matrix import GRAPHS, free_gpus  # noqa: E402

#: Only these have an autotuning path; the SpMM convs are cuSPARSE wrappers.
CONVS = ["min_aggr", "gat_v2", "gt"]

#: The ablation arms. `baseline` and `autotuned` bracket the whole change; `autotuned-baseline`
#: sits between them, tuning only the parameters that existed *before* this work while the
#: scheduler and stream knobs stay pinned at their historical values. Comparing that arm with
#: `autotuned` separates "the autotuner was already leaving performance on the table" from
#: "the new knobs added something".
_PIN_BASELINE = [
    "-K",
    "schedule=one_per_block",
    "-K",
    "forward_bucket_launch=sequential",
    "-K",
    "backward_bucket_launch=sequential",
]
#: `baseline-replay` takes no search at all: it replays the kernel parameters a previous
#: autotune run already selected, with the scheduler and streams forced back to their historical
#: values. That isolates the two new features against a baseline that is otherwise *tuned*,
#: rather than against an untuned one -- comparing tuned-new against untuned-old would credit
#: the new work with gains that belong to parameters which already existed.
REPLAY_SOURCE = "reports/autotune-comparison/grid.json"


# Per-conv pins for the "slice-bps" setup: everything we already know the optimum for is fixed,
# so the search spends its whole budget on the one new axis. Values are the modal argmax from the
# completed 192-cell sweep (reports/autotune-edge-slice). Kept per-conv because the parameter sets
# differ -- passing min_aggr's -K names to gt would be rejected outright.
SLICE_BPS_PINS = {
    "gt": [
        "-K",
        "forward_light_warps=4",
        "-K",
        "forward_heavy_warps=8",
        "-K",
        "backward_light_warps=4",
        "-K",
        "backward_heavy_warps=16",
    ],
    "gat_v2": [
        "-K",
        "forward_light_warps=4",
        "-K",
        "forward_heavy_warps=8",
        "-K",
        "backward_light_warps=1",
        "-K",
        "backward_heavy_warps=8",
        "-K",
        "grad_A_reduce_row_chunk_size=512",
    ],
    "min_aggr": [
        "-K",
        "warps_per_block=1",
        "-K",
        "use_2d_kernel=False",
        "-K",
        "features_per_block=32",
        "-K",
        "tiles_y=8",
        "-K",
        "edges_per_block_heavy_nodes=128",
    ],
    "max_aggr": [
        "-K",
        "warps_per_block=1",
        "-K",
        "use_2d_kernel=False",
        "-K",
        "features_per_block=32",
        "-K",
        "tiles_y=8",
        "-K",
        "edges_per_block_heavy_nodes=128",
    ],
}

SETUPS = {
    "baseline": _PIN_BASELINE,
    "baseline-replay": _PIN_BASELINE,  # per-cell -K args are appended in run_one
    "autotuned-baseline": [
        *_PIN_BASELINE,
        "--autotune",
        "--autotune-exclude",
        "forward_bucket_launch,backward_bucket_launch",
    ],
    # Arm 1: the scheduler is searched, concurrency is not available.
    "sched-seq": [
        "-K",
        "forward_bucket_launch=sequential",
        "-K",
        "backward_bucket_launch=sequential",
        "--autotune",
        "--autotune-exclude",
        "forward_bucket_launch,backward_bucket_launch",
    ],
    # Concurrency switched on outright rather than searched, with everything else -- including
    # the schedule -- left to the autotuner. Answers "what do we get if we simply enable the
    # streams and tune around them", as opposed to letting the search decide per cell.
    # Only the new device-relative slice sizing is searched. Node order is pinned to degree,
    # buckets to concurrent, and the bucketing quantile to 0.99 -- the autotuner's own modal
    # choice for the quantile was -1, which disables bucketing entirely and would leave no heavy
    # bucket for the slice to act on, making the measurement vacuous.
    "slice-bps": [
        "-K",
        "schedule=one_per_block",
        "-K",
        "forward_bucket_launch=concurrent",
        "-K",
        "backward_bucket_launch=concurrent",
        # The bucketing quantile is a *graph* parameter set with --quantile, not a -K kernel
        # argument. Pinned to 0.99: the autotuner's own modal choice was -1, which disables
        # bucketing outright and leaves no heavy bucket for the slice to act on.
        "--quantile",
        "0.99",
        # Both orderings are searched rather than assumed. Locality (reverse Cuthill-McKee) beat
        # degree on the median in the isolated ablation (1.069x vs 1.056x), but which one wins is
        # graph-dependent.
        "--sweep",
        "node_order=degree,locality",
        "--autotune",
        "--autotune-exclude",
        "schedule,forward_bucket_launch,backward_bucket_launch,"
        "forward_light_warps,forward_heavy_warps,backward_light_warps,backward_heavy_warps,"
        "forward_warps_per_block,forward_use_2d_kernel,forward_features_per_block,forward_tiles_y,"
        "forward_edges_per_block_heavy_nodes,backward_grad_A_reduce_row_chunk_size,"
        "forward_huge_degree_threshold_quantile,backward_huge_degree_threshold_quantile",
    ],
    "slice-abs": [
        "-K",
        "schedule=one_per_block",
        "-K",
        "forward_bucket_launch=concurrent",
        "-K",
        "backward_bucket_launch=concurrent",
        # The bucketing quantile is a *graph* parameter set with --quantile, not a -K kernel
        # argument. Pinned to 0.99: the autotuner's own modal choice was -1, which disables
        # bucketing outright and leaves no heavy bucket for the slice to act on.
        "--quantile",
        "0.99",
        # Both orderings are searched rather than assumed. Locality (reverse Cuthill-McKee) beat
        # degree on the median in the isolated ablation (1.069x vs 1.056x), but which one wins is
        # graph-dependent.
        "--sweep",
        "node_order=degree,locality",
        # Absolute edge counts, five values to match the five blocks-per-SM settings, so the
        # two spellings get the same search budget and the comparison isolates the
        # parameterisation rather than the amount of tuning.
        "--sweep-kernel",
        "forward_heavy_edge_slice=0,64,256,1024,4096",
        "--autotune",
        "--autotune-exclude",
        "schedule,forward_bucket_launch,backward_bucket_launch,"
        "forward_light_warps,forward_heavy_warps,backward_light_warps,backward_heavy_warps,"
        "forward_warps_per_block,forward_use_2d_kernel,forward_features_per_block,forward_tiles_y,"
        "forward_edges_per_block_heavy_nodes,backward_grad_A_reduce_row_chunk_size,"
        "forward_huge_degree_threshold_quantile,backward_huge_degree_threshold_quantile,"
        "forward_heavy_slice_blocks_per_sm,backward_heavy_slice_blocks_per_sm",
    ],
    "autotuned-concurrent": [
        "-K",
        "forward_bucket_launch=concurrent",
        "-K",
        "backward_bucket_launch=concurrent",
        "--autotune",
        "--autotune-exclude",
        "forward_bucket_launch,backward_bucket_launch,forward_features_per_block,forward_tiles_y,forward_edges_per_block_heavy_nodes",
    ],
    # Arm 2: the scheduler is searched and concurrency is available. The difference between
    # this and `sched-seq` is exactly what concurrent bucket launches buy.
    "sched-concurrent": ["--autotune"],
    "autotuned": ["--autotune"],
}


def _replayed_args(graph: str, conv: str, dim: int, mode: str) -> list[str] | None:
    """`-K` flags reproducing a previously autotuned configuration, minus the new knobs.

    Scheduler and stream parameters are dropped rather than replayed: `_PIN_BASELINE` forces
    them back to `one_per_block` / sequential, which is the whole point of this arm.
    """
    src = REPO / REPLAY_SOURCE
    if not src.exists():
        return None
    cells = json.loads(src.read_text())["cells"]
    match = next(
        (c for c in cells if c["graph"] == graph and c["conv"] == conv and c["head_dim"] == dim and c["mode"] == mode),
        None,
    )
    if match is None:
        return None
    drop = {"schedule", "blocks_per_sm", "sched_chunk", "forward_bucket_launch", "backward_bucket_launch"}
    # Tunable names and CLI names do not correspond one-to-one: the reduction declares
    # `forward_warps_per_block` for a CLI parameter called `warps_per_block`, while GATv2 and GT
    # declare `forward_light_warps` for a CLI parameter of exactly that name. Resolve against the
    # real schema rather than assuming, or the replay silently emits flags the CLI rejects.
    valid = {param.name for param in CUDA_CONV_PARAMS[conv]}
    out: list[str] = []
    for k, v in (match.get("kernel_config") or {}).items():
        if k in drop:
            continue
        name = k if k in valid else k.removeprefix("forward_").removeprefix("backward_")
        if name not in valid:
            continue  # a tunable with no CLI equivalent; the default already applies
        out += ["-K", f"{name}={v}"]
    q = match.get("quantile")
    if q is not None:
        out += ["--quantile", str(q)]
    return out


def _error_summary(stderr: str, returncode: int) -> str:
    """Last meaningful stderr line, and a full copy on disk.

    Taking the literal last line loses the error whenever a Python warning is the final thing
    printed: a warning's continuation line ("  warnings.warn(") is what surfaces, and the real
    traceback scrolls past. Skip warning noise, and keep the whole stream so a failure is
    diagnosable after the fact instead of needing a re-run to reproduce.
    """
    lines = (stderr or "").strip().splitlines()
    noise = ("warnings.warn(", "warnings.warn", "UserWarning", "FutureWarning", "DeprecationWarning")
    for ln in reversed(lines):
        t = ln.strip()
        if t and not any(nz in t for nz in noise) and not t.startswith(('File "', "return func")):
            return t[:200]
    return lines[-1][:200] if lines else f"exit {returncode}"


def run_one(
    gpu: int, graph: str, cfg: str, conv: str, dim: int, mode: str, setup: str, out_dir: Path, args
) -> tuple[str, bool, str]:
    tag = f"{graph}__{conv}__d{dim}__{mode}__{setup}"
    dest = out_dir / f"{tag}.json"
    if dest.exists() and not args.force:
        return tag, True, "cached"

    cmd = [
        sys.executable,
        str(REPO / "scripts/benchmark_kernels.py"),
        "--backend",
        "cuda",
        "--conv",
        conv,
        "--dataset",
        cfg,
        "--feature-dim",
        str(dim),
        "--heads",
        "1",
        "--mode",
        mode,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--json-out",
        str(dest),
        *SETUPS[setup],
    ]
    if setup in ("slice-bps", "slice-abs"):
        cmd += SLICE_BPS_PINS.get(conv, [])
    if setup == "slice-abs" and conv == "gt" and mode == "backward":
        # Only GT's backward is split (the directed CSR^T path); GATv2's backward and min_aggr's
        # node-by-feature backward have no slice parameter at all, and passing one is an error.
        cmd += ["--sweep-kernel", "backward_heavy_edge_slice=0,64,256,1024,4096"]
    if setup == "baseline-replay":
        replay = _replayed_args(graph, conv, dim, mode)
        if not replay:
            return tag, False, "no previously-selected configuration to replay for this cell"
        cmd += replay
    # Any setup that actually autotunes needs the trial budget, not just the one literally
    # named "autotuned" -- matching on that name meant every other autotuning arm silently ran
    # at benchmark_kernels' defaults and ignored --autotune-warmup/--autotune-iters.
    if "--autotune" in SETUPS[setup]:
        cmd += ["--autotune-warmup", str(args.autotune_warmup), "--autotune-iters", str(args.autotune_iters)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))

    # Verifying the GPU is free once at launch is not enough for a multi-hour matrix: another
    # tenant arriving midway produces numbers that look like real regressions. Check before,
    # check after, and throw the result away if the GPU was shared for any part of it.
    if args.allow_shared_gpu:
        # Deliberately co-resident with our own other arms: the contention guard would reject
        # every run, since a sibling arm on the same device is indistinguishable from a foreign
        # tenant. Absolute times inflate under sharing -- only use this when the comparison of
        # interest is between arms that share the device symmetrically.
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env, timeout=args.timeout)
        if proc.returncode != 0 or not dest.exists():
            fail_log = dest.with_suffix(".stderr.txt")
            fail_log.write_text(proc.stderr or "")
            return tag, False, _error_summary(proc.stderr, proc.returncode)
        return tag, True, f"{time.time() - t0:.0f}s"

    for _attempt in range(args.contention_retries + 1):
        if not wait_until_free(gpu, timeout_s=args.wait_free_s, max_mib=args.free_max_mib):
            return tag, False, f"gpu {gpu} still busy after {args.wait_free_s:.0f}s"

        # Checking only before and after is not enough: a tenant that arrives *and leaves*
        # inside one run is invisible to those two checks, and the run silently absorbs the
        # contention. One cell in an earlier matrix took 320s against 28s clean for exactly this
        # reason, which looked convincingly like a code regression until it was re-timed. Poll
        # for the whole duration instead.
        clean = threading.Event()
        clean.set()
        stop = threading.Event()
        t0 = time.time()
        # Named apart from `proc` below: that one is a CompletedProcess, and reusing the name
        # for the live handle makes the two types collide.
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO, env=env)
        # Our own child is a compute process on this GPU; ignore it, or every poll would
        # report contention against ourselves.
        mine = {popen.pid}

        def _watch() -> None:
            while not stop.wait(args.contention_poll_s):
                if not is_free(gpu, max_mib=args.free_max_mib, ignore_pids=mine):
                    clean.clear()
                    return

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        try:
            out, err = popen.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            popen.kill()
            popen.communicate()
            raise
        finally:
            stop.set()
            watcher.join(timeout=2 * args.contention_poll_s)
        proc = subprocess.CompletedProcess(cmd, popen.returncode, out, err)

        if proc.returncode != 0 or not dest.exists():
            fail_log = dest.with_suffix(".stderr.txt")
            fail_log.write_text(proc.stderr or "")
            return tag, False, _error_summary(proc.stderr, proc.returncode)
        if clean.is_set() and is_free(gpu, max_mib=args.free_max_mib):
            return tag, True, f"{time.time() - t0:.0f}s"
        dest.unlink(missing_ok=True)  # shared at some point: the timing is not usable
    return tag, False, f"gpu {gpu} contended on every attempt"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="reports/autotune-comparison")
    p.add_argument("--graphs", nargs="+", default=[g for g, _ in GRAPHS])
    p.add_argument("--conv", nargs="+", default=CONVS)
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--modes", nargs="+", default=["forward", "backward"])
    p.add_argument("--setups", nargs="+", default=list(SETUPS))
    # Default 1, unlike the kernel matrix runner. The autotuning search is dominated by Python
    # overhead across thousands of trials, not by the kernels it launches -- roughly 4 tiny
    # launches per trial. Running several in parallel therefore contends for CPU rather than
    # using idle GPUs: measured 28s standalone against 319s for the same cell with three workers.
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument(
        "--gpu-ids",
        default=None,
        help="explicit GPU indices (comma-separated) instead of auto-selecting free ones; "
        "use when a GPU is idle but holds residual memory from another user's stale context",
    )
    p.add_argument(
        "--free-max-mib",
        type=int,
        default=64,
        help="used memory above this counts as busy in the exclusivity checks",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--autotune-warmup", type=int, default=5, help="warmup iters per autotuning trial")
    p.add_argument("--autotune-iters", type=int, default=20, help="timed iters per autotuning trial")
    p.add_argument("--timeout", type=int, default=14400, help="per-run timeout; autotuning is slow")
    p.add_argument("--wait-free-s", type=float, default=3600.0, help="how long to wait for a busy GPU")
    p.add_argument("--contention-retries", type=int, default=2, help="re-runs allowed when a GPU is invaded mid-run")
    p.add_argument(
        "--contention-poll-s", type=float, default=5.0, help="how often to check for other tenants during a run"
    )
    p.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="skip the exclusivity checks and let this run share a GPU (e.g. with a sibling arm)",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else free_gpus(args.gpus)
    print(f"free GPUs: {gpus}  ->  {out_dir.relative_to(REPO)}", flush=True)

    by_graph = dict(GRAPHS)
    jobs: dict[int, list[tuple]] = {g: [] for g in gpus}
    for i, graph in enumerate([g for g in args.graphs if g in by_graph]):
        gpu = gpus[i % len(gpus)]
        for conv in args.conv:
            for dim in args.head_dims:
                for mode in args.modes:
                    for setup in args.setups:
                        jobs[gpu].append((graph, by_graph[graph], conv, dim, mode, setup))

    total = sum(len(v) for v in jobs.values())
    print(f"{total} runs ({len(args.setups)} setups per cell)", flush=True)
    done = failed = 0

    def worker(gpu: int) -> None:
        nonlocal done, failed
        for graph, cfg, conv, dim, mode, setup in jobs[gpu]:
            try:
                tag, ok, note = run_one(gpu, graph, cfg, conv, dim, mode, setup, out_dir, args)
            except subprocess.TimeoutExpired:
                tag, ok, note = f"{graph}__{conv}__d{dim}__{mode}__{setup}", False, "timeout"
            done += 1
            if not ok:
                failed += 1
                print(f"  [{done}/{total}] FAIL {tag}: {note}", file=sys.stderr, flush=True)
            elif note != "cached":
                print(f"  [{done}/{total}] {tag}  {note}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))

    print(f"\n{len(list(out_dir.glob('*.json')))} JSON files in {out_dir.relative_to(REPO)}; {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
