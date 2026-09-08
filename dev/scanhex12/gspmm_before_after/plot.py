"""Bar chart of the speedup between two bench.py result files.

    # a change: same config, two commits
    python plot.py --kind bwd --dim 64 -o ../out.png \
        --pair results/before_random_float16_st0.json results/after_random_float16_st0.json \
        --pair results/before_skewed_float16_st0.json results/after_skewed_float16_st0.json

    # pipelining: same commit, two stage counts
    python plot.py --kind fwd --dim 64 --compare stages -o ../out.png \
        --pair results/after_random_float16_st0.json results/after_random_float16_st2.json

One bar per (graph, op/reduce).  Cells the change could not touch stay inside
+-3% and are drawn grey: that band is the measured spread between two builds of
the same sources, so if a control bar leaves it, the two files were not measured
in the same thermal state and nothing else on the chart means anything either.
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND_LABEL = {"fwd": "forward", "bwd": "backward"}

# What the two files are allowed to differ in.  Everything else must match, or
# the ratio is comparing two different measurements rather than one change.
VARYING = {
    "commit": ("label",),
    "stages": ("label", "stages"),
    "dtype": ("label", "dtype"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, action="append", metavar=("BEFORE", "AFTER"), required=True)
    p.add_argument("--kind", default="bwd", choices=["fwd", "bwd"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--compare", default="commit", choices=sorted(VARYING), help="which field the pair varies")
    p.add_argument("-o", "--out", default="before_after.png")
    p.add_argument("--title", default=None)
    p.add_argument("--xlabel", default=None)
    return p.parse_args()


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def describe(meta: dict, compare: str) -> str:
    """How this side of the pair differs from the other."""
    if compare == "stages":
        return f"stages={meta['stages']}"
    if compare == "dtype":
        return meta["dtype"]
    return meta["label"] or "?"


def main() -> int:
    args = parse_args()
    key = f"{args.kind}_ms"
    fixed = [f for f in ("graph", "N", "E", "dtype", "stages") if f not in VARYING[args.compare]]

    rows: list[dict] = []
    subtitle_bits: list[str] = []
    before_desc: set[str] = set()
    after_desc: set[str] = set()
    skipped: list[str] = []

    for before_path, after_path in args.pair:
        before, after = load(before_path), load(after_path)
        mb, ma = before["meta"], after["meta"]

        for field in fixed:
            if mb[field] != ma[field]:
                raise SystemExit(f"{before_path} and {after_path} disagree on {field}: {mb[field]} vs {ma[field]}")
        before_desc.add(describe(mb, args.compare))
        after_desc.add(describe(ma, args.compare))

        subtitle_bits.append(
            f"{mb['graph']} (N={mb['N']:,}, E={mb['E']:,}, макс. степень {mb['max_degree']:,})"
        )

        twins = {(c["op"], c["reduce"], c["d"]): c for c in after["cells"]}
        for cell in before["cells"]:
            if cell["d"] != args.dim:
                continue
            twin = twins.get((cell["op"], cell["reduce"], cell["d"]))
            name = f"{mb['graph']} | {cell['op']}/{cell['reduce']}"
            if args.compare != "commit":
                # Several pairs can share a graph and a cell when the sweep
                # varies something else, so the varying field goes in the label.
                name += f" | {describe(mb, args.compare)}→{describe(ma, args.compare)}"
            if twin is None or "error" in cell or "error" in twin:
                skipped.append(name)
                continue
            rows.append({
                "label": name,
                "speedup": cell[key] / twin[key],
                "before": cell[key],
                "after": twin[key],
            })

    if not rows:
        raise SystemExit(f"no cells matched (kind={args.kind}, d={args.dim})")

    rows.sort(key=lambda r: r["speedup"])
    vals = [r["speedup"] for r in rows]
    labels = [f"{r['label']}  ({r['before']:.2f} → {r['after']:.2f} мс)" for r in rows]

    fig, ax = plt.subplots(figsize=(12.8, max(4.0, 0.34 * len(rows) + 1.8)))
    # +-3% is the measured spread between two builds of the same sources, so
    # anything inside it is grey and only a genuine loss goes red.
    colors = ["#c44e52" if v < 0.97 else "#999999" if v < 1.03 else "#1f77b4" for v in vals]
    ax.barh(range(len(rows)), vals, color=colors, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    # Behind everything: at the default z-order it was drawn over the value
    # labels, and a dashed line through "0.98x" is easy to misread.
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, zorder=0.5)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel(
        args.xlabel
        or f"Ускорение: до / после ({KIND_LABEL[args.kind]}, медиана из 5 прогонов по 30 запусков)\n"
        "Серое — в пределах ±3%, то есть контроль: клетки, которых правка не касается"
    )
    ax.set_title(args.title or f"g-SpMM {KIND_LABEL[args.kind]}, d={args.dim}: до и после")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, f"{v:.2f}x", va="center", fontsize=8, zorder=3)
    ax.margins(y=0.005)

    footer = " · ".join(sorted(set(subtitle_bits)))
    footer += f"   |   до: {', '.join(sorted(before_desc))} → после: {', '.join(sorted(after_desc))}"
    if skipped:
        footer += f"   |   не измерено: {len(skipped)} клеток"
    fig.text(0.01, 0.005, footer, fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  ({len(rows)} bars"
          + (f", {len(skipped)} skipped)" if skipped else ")"))

    print(f"over all {len(vals)} cells: geomean {st.geometric_mean(vals):.2f}x "
          f"| median {st.median(vals):.2f}x | min {min(vals):.2f}x | max {max(vals):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
