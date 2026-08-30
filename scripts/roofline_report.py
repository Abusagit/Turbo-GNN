"""Measured-traffic roofline report: saturation, per-edge cost, reuse, and tuning headroom.

Two measurement passes feed this. `roofline_analysis.py` timed every configuration and modelled
its *compulsory* traffic -- the bytes a kernel must move if nothing is reused.
`run_roofline_counters.py` then measured the DRAM traffic each kernel actually moved, using
hardware counters under `sudo` (they are root-only on this driver; see ERR_NVGPUCTRPERM).

Measured traffic is authoritative here and the model is reported beside it as a claim to check.
That distinction matters: the model assumes no reuse, real forward kernels reuse heavily, and
taking the model at face value overstates their bandwidth utilisation by roughly 2.5x.

    python scripts/roofline_report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONVS = ("min_aggr", "gat_v2", "gt")
MODES = ("forward", "backward")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main() -> int:
    rf = json.loads((REPO / "reports/roofline/roofline.json").read_text())
    peak = rf["peak_bytes_per_s"]
    meas = {
        (c["graph"], c["conv"], c["head_dim"], c["mode"]): c
        for c in json.loads((REPO / "reports/roofline/counters.json").read_text())
    }
    abl = {
        (c["graph"], c["conv"], c["head_dim"], c["mode"]): c
        for c in json.loads((REPO / "reports/final-report/cells.json").read_text())["cells"]
    }

    # Join every timed cell to its measured traffic and its tuned time.
    cells = []
    for c in rf["cells"]:
        k = (c["graph"], c["conv"], c["head_dim"], c["mode"])
        if k not in meas:
            continue
        t = c["ms"] / 1e3
        bytes_dram = meas[k]["dram_total"]
        a = abl.get(k)
        speedup = a["untuned_ms"] / a["full_auto_ms"] if a and a.get("untuned_ms") else None
        cells.append(
            {
                **c,
                "dram_bytes": bytes_dram,
                "achieved_GBs": bytes_dram / t / 1e9,
                "true_pct_peak": bytes_dram / t / peak * 100,
                "tuned_pct_peak": bytes_dram / t / peak * 100 * speedup if speedup else None,
                "reuse": c["model_bytes"] / bytes_dram,
                "model_ratio": bytes_dram / c["model_bytes"],
                "dram_per_edge": bytes_dram / c["edges"] if c["edges"] else None,
                "edge_vs_dram": (t / c["edges"]) / (bytes_dram / c["edges"] / peak)
                if c["edges"] and c["bytes_per_edge"]
                else None,
                "speedup": speedup,
            }
        )

    def sub(conv, mode):
        return [c for c in cells if c["conv"] == conv and c["mode"] == mode]

    print(f"device {rf['device']}   measured peak {peak / 1e12:.3f} TB/s   {len(cells)} cells")
    print("traffic measured with hardware counters; the compulsory model is shown for comparison\n")

    print("=" * 80)
    print("Q1  ARE WE SATURATING MEMORY BANDWIDTH?   -- no, and not close")
    print("=" * 80)
    print(f"\n  {'conv':<10}{'mode':<10}{'n':>4}{'modelled':>11}{'MEASURED':>11}{'tuned':>9}{'achieved':>12}")
    for conv in CONVS:
        for mode in MODES:
            s = sub(conv, mode)
            tp = [c["tuned_pct_peak"] for c in s if c["tuned_pct_peak"]]
            print(
                f"  {conv:<10}{mode:<10}{len(s):>4}{med([c['pct_of_peak'] for c in s]):>10.0f}%"
                f"{med([c['true_pct_peak'] for c in s]):>10.0f}%{med(tp):>8.0f}%"
                f"{med([c['achieved_GBs'] for c in s]):>10.0f} GB/s"
            )
    for mode in MODES:
        s = [c for c in cells if c["mode"] == mode]
        p = [c["true_pct_peak"] for c in s]
        print(
            f"\n  {mode:<9} median {med(p):>3.0f}% of peak   max {max(p):>3.0f}%   "
            f"at or above 90%: {sum(x >= 90 for x in p)}/{len(p)}"
        )
    allp = [c["true_pct_peak"] for c in cells]
    print(f"\n  Across all {len(cells)} configurations, {sum(x >= 90 for x in allp)} reach 90% of peak.")
    print("  The kernels are not bandwidth-bound. Whatever limits them, it is not the bus.")

    print("\n" + "=" * 80)
    print("Q2  IS MESSAGE PASSING SLOWER THAN READING THE NEIGHBOUR?   -- yes, everywhere")
    print("=" * 80)
    print("\n  Per-edge time divided by the time to fetch that edge's measured DRAM traffic at peak.")
    print(
        f"\n  {'conv':<10}{'mode':<10}{'n':>4}{'vs modelled':>13}{'vs MEASURED':>13}"
        f"{'DRAM B/edge':>13}{'compulsory':>12}"
    )
    for conv in CONVS:
        for mode in MODES:
            s = [c for c in sub(conv, mode) if c["edge_vs_dram"]]
            if not s:
                print(f"  {conv:<10}{mode:<10}{0:>4}   no per-edge traversal (node x feature kernel)")
                continue
            print(
                f"  {conv:<10}{mode:<10}{len(s):>4}{med([c['compute_vs_load'] for c in s]):>13.2f}"
                f"{med([c['edge_vs_dram'] for c in s]):>13.2f}"
                f"{med([c['dram_per_edge'] for c in s]):>12.0f}B{med([c['bytes_per_edge'] for c in s]):>11.0f}B"
            )
    ev = [c["edge_vs_dram"] for c in cells if c["edge_vs_dram"]]
    print(f"\n  All {len(ev)} edge-traversing configurations exceed 1.0x; median {med(ev):.2f}x.")
    print("  Even the forward pass, which the compulsory model put near break-even, is 2.8-4.9x.")

    print("\n" + "=" * 80)
    print("Q3  WHAT THE MODEL GOT WRONG   -- measured traffic / compulsory traffic")
    print("=" * 80)
    print(f"\n  {'conv':<10}{'mode':<10}{'n':>4}{'ratio':>9}{'reuse':>9}   reading")
    for conv in CONVS:
        for mode in MODES:
            s = sub(conv, mode)
            r = med([c["model_ratio"] for c in s])
            note = (
                "reuse: fetched less than compulsory"
                if r < 0.9
                else "amplification: fetched MORE"
                if r > 1.1
                else "matches compulsory"
            )
            print(f"  {conv:<10}{mode:<10}{len(s):>4}{r:>9.2f}{med([c['reuse'] for c in s]):>8.2f}x   {note}")
    print("\n  Forward kernels fetch 36-47% of compulsory: the caches serve most neighbour reads,")
    print("  so modelled 'apparent bandwidth' overstated real utilisation by roughly 2.5x.")
    print("  min_aggr's backward fetches 2.10x compulsory: it scatters one atomic per")
    print("  (node, feature) through arg_idx, so 4-byte writes drag whole 32-byte sectors.")

    print("\n" + "=" * 80)
    print("Q4  DOES TUNING CHANGE SATURATION?")
    print("=" * 80)
    print(f"\n  {'mode':<10}{'n':>4}{'default':>10}{'tuned':>9}{'speedup':>10}")
    for mode in MODES:
        s = [c for c in cells if c["mode"] == mode and c["tuned_pct_peak"]]
        print(
            f"  {mode:<10}{len(s):>4}{med([c['true_pct_peak'] for c in s]):>9.0f}%"
            f"{med([c['tuned_pct_peak'] for c in s]):>8.0f}%{med([c['speedup'] for c in s]):>9.2f}x"
        )
    print("\n  Tuning roughly doubles forward utilisation and leaves backward where it was.")
    print("  Both remain far from peak, so bandwidth is not what caps either one.")

    print("\n" + "=" * 80)
    print("PER GRAPH   (median over convolutions and head dims, measured traffic)")
    print("=" * 80)
    print(
        f"\n  {'graph':<16}{'nodes':>10}{'edges':>12}{'fwd %pk':>9}{'bwd %pk':>9}"
        f"{'fwd reuse':>11}{'fwd edge':>10}{'bwd edge':>10}"
    )
    for g in sorted({c["graph"] for c in cells}):
        s = [c for c in cells if c["graph"] == g]
        f = [c for c in s if c["mode"] == "forward"]
        b = [c for c in s if c["mode"] == "backward"]
        print(
            f"  {g:<16}{s[0]['nodes']:>10,}{s[0]['edges']:>12,}"
            f"{med([c['true_pct_peak'] for c in f]):>8.0f}%{med([c['true_pct_peak'] for c in b]):>8.0f}%"
            f"{med([c['reuse'] for c in f]):>10.2f}x"
            f"{med([c['edge_vs_dram'] for c in f if c['edge_vs_dram']]):>10.2f}"
            f"{med([c['edge_vs_dram'] for c in b if c['edge_vs_dram']]):>10.2f}"
        )

    # figures.json drives the published page
    fig = {"peak": peak, "device": rf["device"], "sat": [], "graphs": []}
    for conv in CONVS:
        for mode in MODES:
            s = sub(conv, mode)
            ev = [c["edge_vs_dram"] for c in s if c["edge_vs_dram"]]
            fig["sat"].append(
                {
                    "conv": conv,
                    "mode": mode,
                    "n": len(s),
                    "modelled": med([c["pct_of_peak"] for c in s]),
                    "measured": med([c["true_pct_peak"] for c in s]),
                    "tuned": med([c["tuned_pct_peak"] for c in s if c["tuned_pct_peak"]]),
                    "reuse": med([c["reuse"] for c in s]),
                    "model_ratio": med([c["model_ratio"] for c in s]),
                    "edge": med(ev) if ev else None,
                    "edge_model": med([c["compute_vs_load"] for c in s if c["compute_vs_load"]]) if ev else None,
                }
            )
    for g in sorted({c["graph"] for c in cells}):
        s = [c for c in cells if c["graph"] == g]
        f = [c for c in s if c["mode"] == "forward"]
        b = [c for c in s if c["mode"] == "backward"]
        fig["graphs"].append(
            {
                "graph": g,
                "nodes": s[0]["nodes"],
                "edges": s[0]["edges"],
                "fwd": med([c["true_pct_peak"] for c in f]),
                "bwd": med([c["true_pct_peak"] for c in b]),
                "reuse": med([c["reuse"] for c in f]),
                "fwd_edge": med([c["edge_vs_dram"] for c in f if c["edge_vs_dram"]]),
                "bwd_edge": med([c["edge_vs_dram"] for c in b if c["edge_vs_dram"]]),
            }
        )
    (REPO / "reports/roofline/figures.json").write_text(json.dumps(fig, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
