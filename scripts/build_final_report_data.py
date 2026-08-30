"""Join every completed measurement into one per-cell record.

Each stage lives in its own directory because each was measured separately; this collects them
so a single cell can be followed from the untouched launch through to the fully tuned one.

    untuned      one_per_block, natural order, sequential, no tuning at all
    tuned_base   kernel parameters at their autotuned argmax, new features still off
    sched        + best (schedule x node order), buckets still sequential
    streams      + concurrent buckets where they win
    full_auto    autotuner free to choose everything, including the new knobs

    python scripts/build_final_report_data.py --out reports/final-report/cells.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _by_cell(directory: str, suffix: str | None = None) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    d = REPO / directory
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        if f.name == "grid.json":
            continue
        if suffix and not f.stem.endswith(suffix):
            continue
        j = json.loads(f.read_text())
        out[(f.name.split("__")[0], j["conv"], j["feature_dim"], j["mode"])] = j
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="reports/final-report/cells.json")
    args = p.parse_args()

    untuned = _by_cell("reports/autotune-comparison", "__baseline")
    tuned = _by_cell("reports/final-ablation", "__baseline-replay")
    full = _by_cell("reports/autotune-comparison", "__autotuned")
    conc = _by_cell("reports/autotune-concurrent", "__autotuned-concurrent")
    abl = _by_cell("reports/scheduler-ablation")

    rows: list[dict] = []
    facts: dict[str, dict] = {}
    for key, a in sorted(abl.items()):
        pts = {}
        for s in a["sweep"]:
            c = s["graph_config"]
            pts[
                (
                    c.get("kernel:schedule"),
                    c.get("kernel:forward_bucket_launch") or c.get("kernel:backward_bucket_launch"),
                    c.get("node_order", "natural"),
                )
            ] = s["ms_per_iter"]
        base_cfg = pts.get(("one_per_block", "sequential", "natural"))
        seq = [v for k, v in pts.items() if k[1] == "sequential"]
        con = [v for k, v in pts.items() if k[1] == "concurrent"]
        if base_cfg is None or not seq or not con:
            continue
        b, c_ = min(seq), min(min(seq), min(con))
        best = min(pts.items(), key=lambda kv: kv[1])[0]
        graph, conv, dim, mode = key
        facts.setdefault(graph, a["graph"])
        rows.append(
            {
                "graph": graph,
                "conv": conv,
                "head_dim": dim,
                "mode": mode,
                # stages, all milliseconds per iteration
                "untuned_ms": untuned[key]["ms_per_iter"] if key in untuned else None,
                "tuned_base_ms": tuned[key]["ms_per_iter"] if key in tuned else None,
                "abl_A_ms": base_cfg,
                "abl_B_ms": b,
                "abl_C_ms": c_,
                "full_auto_ms": full[key]["ms_per_iter"] if key in full else None,
                "auto_concurrent_ms": conc[key]["ms_per_iter"] if key in conc else None,
                # what the ablation picked
                "best_schedule": best[0],
                "best_bucket": best[1],
                "best_order": best[2],
                "B_over_A": base_cfg / b,
                "C_over_B": b / c_,
                "C_over_A": base_cfg / c_,
            }
        )

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": rows, "graphs": facts}, indent=1))
    have: defaultdict[str, int] = defaultdict(int)
    for r in rows:
        for k in ("untuned_ms", "tuned_base_ms", "full_auto_ms", "auto_concurrent_ms"):
            have[k] += r[k] is not None
    print(f"wrote {out.relative_to(REPO)}: {len(rows)} cells across {len(facts)} graphs")
    for k, v in have.items():
        print(f"  {k:<22}{v}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
