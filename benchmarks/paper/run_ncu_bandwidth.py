#!/usr/bin/env python3
"""Hardware counters behind Section 4.1: DRAM traffic, achieved bandwidth and active warps of
the GATv2 kernels at the two ends of the ablation ladder (prior kernel and all techniques).

    sudo -E python benchmarks/paper/run_ncu_bandwidth.py --gpu 0 --graphs ogbn-arxiv web-fraud

Measure *actual* DRAM traffic for the attention kernels, to check the roofline claim.

The ablation report quotes an effective bandwidth: compulsory traffic (each input, output and
index tensor counted once) divided by kernel time. That is a lower bound on what really crosses
the bus, because a gather re-reads source rows that miss in L2. If the true DRAM utilisation is
near peak, the "latency- and imbalance-bound, not bandwidth-bound" claim in Section 3 is wrong.
So we read dram__bytes.sum directly.

ncu 2026.1 emits `--page raw --csv` in WIDE form (one row per launch, one column per metric);

Needs Nsight Compute (ncu on PATH or NCU=...) and, on most drivers, root for counter access.
"""
import argparse, csv, io, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphs import ATTENTION_GRAPHS as GRAPHS  # noqa: E402

METRICS = [
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "launch__grid_size",
]
OURS = ("GATv2", "GraphAttention", "graph_attn", "reduction_aggr")

# The two ends of the ablation ladder, as -K overrides.
RUNGS = {
    "baseline": dict(q=-1.0, order="natural", launch="sequential", slice=0, pipe=0),
    "best":     dict(q=0.99, order="degree",  launch="concurrent", slice=1024, pipe=0),
}


def profile(graph, cfg, rung, dim, heads, mode, gpu, out_dir):
    r = RUNGS[rung]
    bench = [
        sys.executable, str(REPO / "scripts/benchmark_kernels.py"),
        "--backend", "cuda", "--conv", "gat_v2", "--dataset", cfg,
        "--feature-dim", str(dim), "--heads", str(heads), "--dtype", "fp16",
        "--mode", mode, "--quantile", str(r["q"]), "--node-order", r["order"],
        # ncu replays each kernel; one timed call is enough and keeps the run short.
        "--launch-ncu-override-iters", "1", "--launch-ncu-override-warmup", "1",
        "-K", f"forward_bucket_launch={r['launch']}",
        "-K", f"backward_bucket_launch={r['launch']}",
        "-K", f"forward_heavy_edge_slice={r['slice']}",
        "-K", f"pipeline_stages={r['pipe']}",
        "-K", f"backward_pipeline_stages={r['pipe']}",
        "-K", f"heavy_pipeline_stages={r['pipe']}",
    ]
    cmd = [os.environ.get("NCU", "ncu"), "--target-processes", "all", "--csv", "--page", "raw",
           "--metrics", ",".join(METRICS),
           "--kernel-name", "regex:" + "|".join(OURS), "--kernel-name-base", "demangled", *bench]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=5400)
    tag = f"{graph}__{rung}__{mode}"
    (out_dir / f"{tag}.log").write_text(p.stdout + "\n" + p.stderr)
    if '"ID"' not in p.stdout:
        print(f"    !! no CSV ({p.returncode}) -- see {tag}.log", flush=True)
        return []
    rows = []
    for rec in csv.DictReader(io.StringIO(p.stdout[p.stdout.index('"ID"'):])):
        if not any(t in rec.get("Kernel Name", "") for t in OURS):
            continue
        f = lambda k: float(rec[k].replace(",", "")) if rec.get(k) not in (None, "", "n/a") else 0.0
        rows.append(dict(
            graph=graph, rung=rung, mode=mode, dim=dim, heads=heads,
            kernel=rec["Kernel Name"].split("(")[0][:70],
            dram_bytes=f("dram__bytes.sum"),
            dram_pct=f("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            ns=f("gpu__time_duration.sum"),
            warps_pct=f("sm__warps_active.avg.pct_of_peak_sustained_active"),
            stall_long=f("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
            stall_mio=f("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"),
            ld_sectors=f("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
            grid=f("launch__grid_size")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--rungs", nargs="+", default=["baseline", "best"])
    ap.add_argument("--modes", nargs="+", default=["forward", "backward"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", default=str(REPO / "results/ncu_bandwidth"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    by = dict(GRAPHS)
    todo = [(g, r, m) for g in a.graphs for r in a.rungs for m in a.modes]
    allr = []
    for i, (g, r, m) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {g} {r} {m}", flush=True)
        try:
            rows = profile(g, by[g], r, a.dim, a.heads, m, a.gpu, out)
        except subprocess.TimeoutExpired:
            print("    timed out", flush=True); continue
        # Each launch is a separate row; report the aggregate that matters.
        tb, tn = sum(x["dram_bytes"] for x in rows), sum(x["ns"] for x in rows)
        if tn:
            print(f"    {len(rows)} launches  {tb/1e6:.1f} MB  {tn/1e3:.1f} us"
                  f"  -> {tb/tn:.1f} GB/s = {tb/tn/2039*100:.1f}% of peak", flush=True)
        allr.extend(rows)
        (out / "metrics.json").write_text(json.dumps(allr, indent=1))
    print(f"\n{len(allr)} launch rows -> {out/'metrics.json'}")


if __name__ == "__main__":
    main()
