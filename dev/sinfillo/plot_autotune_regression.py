"""Compare autotuned GATv2 configs against the manual baseline to confirm the
autotuner never regresses.

Baseline runs (--pipeline-stages 0, i.e. manual warps/no-pipeline defaults) live
in results/pipeline_sweep_after_tile_ops/*_baseline.json (from run_pipeline_sweep.py).
Autotuned runs (conv.autotune()) live in results/pipeline_sweep_after_tile_ops_2/
*_autotuned.json (from run_pipeline_sweep_autotune.py). This joins the two on
(dataset, amp, feature_dim, heads, head_dim) and plots speedup = baseline_ms /
autotuned_ms per config, colored green (>=1, no regression) or red (<1, regression).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.plot_pipeline_results import BASELINE, build_comparison, load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("results/pipeline_sweep_after_tile_ops")
    )
    parser.add_argument(
        "--autotuned-dir", type=Path, default=Path("results/pipeline_sweep_after_tile_ops_2")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/pipeline_sweep_after_tile_ops_2/plots")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_raw = load_results(args.baseline_dir)
    baseline_raw = baseline_raw[baseline_raw["variant"] == BASELINE]

    autotuned_raw = load_results(args.autotuned_dir)
    autotuned_raw = autotuned_raw[autotuned_raw["variant"] == "autotuned"]

    raw = pd.concat([baseline_raw, autotuned_raw], ignore_index=True)
    comparison, variants = build_comparison(raw)

    if "autotuned" not in variants:
        raise RuntimeError(f"No autotuned variant found (variants={variants})")

    sub = comparison.dropna(subset=[BASELINE, "autotuned"]).copy()
    if sub.empty:
        raise RuntimeError("No configs with both baseline and autotuned results -- nothing to compare")

    # build_comparison's pivot only keeps ms_per_iter -- pull the chosen
    # pipeline_stages back in from the raw autotuned rows (flattened onto the
    # JSON payload by benchmark.py as "pipeline_stages").
    index_cols = ["dataset", "amp", "feature_dim", "heads", "head_dim"]
    chosen_stages = autotuned_raw[[*index_cols, "pipeline_stages"]].drop_duplicates(index_cols)
    sub = sub.merge(chosen_stages, on=index_cols, how="left")

    sub["label"] = (
        sub["dataset"] + " | " + sub["amp"] + " | " + sub["config"]
        + " | stage=" + sub["pipeline_stages"].astype("Int64").astype(str)
    )
    sub = sub.sort_values("speedup_autotuned")

    colors = ["#d62728" if s < 1.0 else "#2ca02c" for s in sub["speedup_autotuned"]]

    fig, ax = plt.subplots(figsize=(11, max(6, len(sub) * 0.28)))
    ax.barh(sub["label"], sub["speedup_autotuned"], color=colors)
    ax.axvline(1.0, linewidth=1, linestyle="--", color="black")
    ax.set_title("Autotuned vs baseline: speedup = baseline_ms / autotuned_ms\n(green = no regression, red = regression)")
    ax.set_xlabel("Speedup")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    for y, speedup in enumerate(sub["speedup_autotuned"]):
        ax.text(speedup, y, f" {speedup:.3f}x", va="center", fontsize=7)

    plt.tight_layout()
    output_path = args.output_dir / "autotune_vs_baseline.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    sub.to_csv(args.output_dir / "autotune_vs_baseline.csv", index=False)

    regressions = sub[sub["speedup_autotuned"] < 1.0]
    print(f"Compared configs: {len(sub)}")
    print(f"Min speedup: {sub['speedup_autotuned'].min():.3f}x  ({sub.iloc[0]['label']})")
    print(f"Max speedup: {sub['speedup_autotuned'].max():.3f}x  ({sub.iloc[-1]['label']})")
    print(f"Mean speedup: {sub['speedup_autotuned'].mean():.3f}x")
    print(f"Regressions (speedup < 1.0): {len(regressions)}")
    if not regressions.empty:
        for _, row in regressions.iterrows():
            print(f"  {row['label']}: {row['speedup_autotuned']:.3f}x")
    print(f"Plot written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
