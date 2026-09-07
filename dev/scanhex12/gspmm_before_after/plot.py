"""Bar chart of before/after speedup from bench.py results.

    python plot.py --kind bwd -o ../fp16_bwd_before_after.png \
        --pair results/before_uniform.json results/after_uniform.json \
        --pair results/before_skewed.json  results/after_skewed.json

One bar per (graph, op/reduce).  Cells the change could not touch stay at
1.00x and are the control: if they move, the two files were not measured in
the same thermal state and nothing else on the chart means anything either.
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND_LABEL = {"fwd": "forward", "bwd": "backward"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, action="append", metavar=("BEFORE", "AFTER"), required=True)
    p.add_argument("--kind", default="bwd", choices=["fwd", "bwd"])
    p.add_argument("-o", "--out", default="before_after.png")
    p.add_argument("--title", default=None)
    return p.parse_args()


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    args = parse_args()
    key = f"{args.kind}_ms"

    rows = []
    subtitle_bits = []
    labels_before = set()
    labels_after = set()

    for before_path, after_path in args.pair:
        before, after = load(before_path), load(after_path)
        mb, ma = before["meta"], after["meta"]

        for field in ("graph", "N", "E", "d", "dtype", "stages"):
            if mb[field] != ma[field]:
                raise SystemExit(f"{before_path} and {after_path} disagree on {field}: {mb[field]} vs {ma[field]}")
        if mb["label"]:
            labels_before.add(mb["label"])
        if ma["label"]:
            labels_after.add(ma["label"])

        subtitle_bits.append(
            f"{mb['graph']} (N={mb['N']:,}, E={mb['E']:,}, макс. степень {mb['max_degree']:,})"
        )

        after_by_cell = {(c["op"], c["reduce"]): c for c in after["cells"]}
        for cell in before["cells"]:
            twin = after_by_cell.get((cell["op"], cell["reduce"]))
            if twin is None:
                continue
            rows.append({
                "label": f"{mb['graph']} | {cell['op']}/{cell['reduce']}",
                "speedup": cell[key] / twin[key],
                "before": cell[key],
                "after": twin[key],
            })

    if not rows:
        raise SystemExit("no cells matched")

    rows.sort(key=lambda r: r["speedup"])
    vals = [r["speedup"] for r in rows]
    labels = [f"{r['label']}  ({r['before']:.2f} → {r['after']:.2f} мс)" for r in rows]

    fig, ax = plt.subplots(figsize=(12.8, max(4.0, 0.34 * len(rows) + 1.6)))
    # +-3% is the measured spread between two builds of the same sources, so
    # anything inside it is grey (untouched, or too small to claim) and only a
    # genuine loss goes red.
    colors = ["#c44e52" if v < 0.97 else "#999999" if v < 1.03 else "#1f77b4" for v in vals]
    ax.barh(range(len(rows)), vals, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel(
        f"Ускорение: до / после ({KIND_LABEL[args.kind]}, медиана из 5 прогонов по 30 запусков)\n"
        "Серое — в пределах ±3%, то есть контроль: клетки, которых правка не касается"
    )
    ax.set_title(args.title or f"g-SpMM {KIND_LABEL[args.kind]}: до и после")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.008, i, f"{v:.2f}x", va="center", fontsize=8)
    ax.margins(y=0.005)

    footer = " · ".join(sorted(set(subtitle_bits)))
    if labels_before or labels_after:
        footer += f"   |   до: {', '.join(sorted(labels_before))} → после: {', '.join(sorted(labels_after))}"
    fig.text(0.01, 0.005, footer, fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  ({len(rows)} bars)")

    print(f"\nover all {len(vals)} cells: geomean {st.geometric_mean(vals):.2f}x "
          f"| median {st.median(vals):.2f}x | min {min(vals):.2f}x | max {max(vals):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
