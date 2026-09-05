from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="results_<graph>.json files")
    p.add_argument("-o", "--out", default="speedup.png")
    p.add_argument("--dim", type=int, default=None, help="show only this feature width (default: all)")
    p.add_argument("--reducers", default="sum,min,max")
    p.add_argument("--ops", default=None, help="comma-separated subset of ops")
    p.add_argument("--title", default=None)
    p.add_argument("--max-bars", type=int, default=64, help="keep the chart legible; drops the middle of the ranking")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    keep_red = args.reducers.split(",")
    keep_ops = args.ops.split(",") if args.ops else None

    rows = []
    subtitle_bits = []
    for path in args.results:
        with open(path) as fh:
            blob = json.load(fh)
        meta = blob["meta"]
        subtitle_bits.append(f"{meta['graph']} (N={meta['N']:,}, E={meta['E']:,})")
        for cell in blob["cells"]:
            if args.dim is not None and cell["d"] != args.dim:
                continue
            if cell["reduce"] not in keep_red:
                continue
            if keep_ops and cell["op"] not in keep_ops:
                continue
            rows.append({
                "label": f"{meta['graph']} | {cell['op']}/{cell['reduce']} | d={cell['d']}",
                "speedup": cell["dgl_ms"] / cell["turbo_ms"],
            })

    if not rows:
        raise SystemExit("no cells matched the filters")

    rows.sort(key=lambda r: r["speedup"])
    all_vals = [r["speedup"] for r in rows]
    shown = rows
    if len(rows) > args.max_bars:
        half = args.max_bars // 2
        shown = rows[:half] + rows[-half:]
        print(f"note: chart draws the {args.max_bars} extreme bars of {len(rows)}; "
              f"the statistics below still cover all {len(rows)}")

    vals = [r["speedup"] for r in shown]
    labels = [r["label"] for r in shown]

    fig, ax = plt.subplots(figsize=(12.8, max(4.0, 0.30 * len(shown) + 1.4)))
    colors = ["#c44e52" if v < 1.0 else "#1f77b4" for v in vals]  # losses stand out
    ax.barh(range(len(shown)), vals, color=colors)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("Ускорение: dgl / turbo_gnn (forward, медиана из 7 прогонов по 30 запусков)")
    ax.set_title(args.title or "g-SpMM: turbo_gnn против DGL по конфигурациям"
                 + (f" (d={args.dim})" if args.dim else ""))
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.008, i, f"{v:.2f}x", va="center", fontsize=8)
    ax.margins(y=0.005)
    fig.text(0.01, 0.005, " · ".join(sorted(set(subtitle_bits))), fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  ({len(shown)} bars)")

    print(f"\nover all {len(all_vals)} cells: geomean {st.geometric_mean(all_vals):.2f}x "
          f"| median {st.median(all_vals):.2f}x | min {min(all_vals):.2f}x | max {max(all_vals):.2f}x")
    losses = [(r["label"], r["speedup"]) for r in rows if r["speedup"] < 1.0]
    print(f"cells where DGL wins: {len(losses)}")
    for lab, v in losses:
        print(f"  {v:.2f}x  {lab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
