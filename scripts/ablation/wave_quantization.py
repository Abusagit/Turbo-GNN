import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from load_imbalance import (  # noqa: E402
    REAL_SOURCE, _signed_indptr, bin_sums_from_offsets,
    load_real, make_powerlaw, strided_bin_sums,
)
from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


def simulate(bins, SM):
    K = len(bins)
    num_waves = math.ceil(K / SM)
    wave_times = [bins[w * SM:(w + 1) * SM].max() for w in range(num_waves)]
    sim = float(sum(wave_times))
    ideal = float(bins.sum()) / SM
    waste = (sim - ideal) / sim if sim > 0 else 0.0
    last_wave_size = K - (num_waves - 1) * SM
    return {
        "K": K,
        "num_waves": num_waves,
        "last_wave_size": last_wave_size,
        "sim_time": sim,
        "ideal_time": ideal,
        "waste_pct": 100.0 * waste,
    }


def per_block_work(g, schedule, K):
    light = g.forward_light_nodes
    indptr = g.forward_indptr
    if schedule == "legacy":
        return _signed_indptr(indptr).diff()[light.long()].cpu().numpy()
    if schedule == "gsl":
        return strided_bin_sums(light, indptr, K)
    if schedule == "balanced":
        sorted_nodes, offsets = edge_balanced_partition(light, indptr, K)
        return bin_sums_from_offsets(sorted_nodes, indptr, offsets)
    raise ValueError(schedule)


def analyze(tag, edge_index, N, K_list, quantile, SM):
    g = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32,
    )
    print(f"\n### {tag}")
    print(f"N={N}  light={int(g.forward_light_nodes.numel())}  "
          f"heavy={int(g.forward_heavy_nodes.numel())}")

    legacy_bins = per_block_work(g, "legacy", 0)
    legacy_sim = simulate(legacy_bins, SM)
    print(f"\nlegacy: K={legacy_sim['K']}  waves={legacy_sim['num_waves']}  "
          f"last_wave={legacy_sim['last_wave_size']}/{SM}  "
          f"waste={legacy_sim['waste_pct']:.1f}%  "
          f"sim_time={legacy_sim['sim_time']:.3e} (edge-units)")

    header = (f"| {'schedule':<10} | {'K':>5} | {'waves':>5} | "
              f"{'last':>4} | {'sim_time':>12} | {'waste %':>7} | "
              f"{'speedup vs legacy':>17} |")
    print("\n" + header)
    print("|" + "-" * (len(header) - 2) + "|")
    for K in K_list:
        for sched in ("gsl", "balanced"):
            bins = per_block_work(g, sched, K)
            r = simulate(bins, SM)
            speedup = legacy_sim["sim_time"] / r["sim_time"] if r["sim_time"] > 0 else float("nan")
            print(f"| {sched:<10} | {r['K']:>5} | {r['num_waves']:>5} | "
                  f"{r['last_wave_size']:>4} | {r['sim_time']:>12.3e} | "
                  f"{r['waste_pct']:>7.1f} | {speedup:>17.2f}x |")


def _load_empirical(md_path):
    text = md_path.read_text()
    block_re = re.compile(
        r"\*\*(?P<dataset>[^*/]+)\s*/\s*(?P<op>gatv2|minaggr)\s*/\s*dim=(?P<dim>\d+)\*\*"
        r"\s*—\s*legacy\s*=\s*\*\*(?P<leg>[\d.]+)\s*ms\*\*",
    )
    row_re = re.compile(
        r"\|\s*(\d+)\s*\((\d+)x\)\s*\|"
        r"\s*([\d.]+)\s*\(([\d.]+)x\)\s*\|"
        r"\s*([\d.]+)\s*\(([\d.]+)x\)\s*\|"
        r"\s*([\d.]+)\s*\(([\d.]+)x\)\s*\|"
    )
    rows = {}
    matches = list(block_re.finditer(text))
    for i, m in enumerate(matches):
        ds = m.group("dataset").strip()
        op = m.group("op")
        dim = int(m.group("dim"))
        legacy_ms = float(m.group("leg"))
        rows.setdefault((ds, op, dim), {"legacy": legacy_ms})
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text.find("\n## ", start, end)
        if section != -1:
            end = section
        for r in row_re.finditer(text[start:end]):
            K = int(r.group(1))
            rows[(ds, op, dim)][("gsl", K)] = float(r.group(3))
            rows[(ds, op, dim)][("balanced", K)] = float(r.group(5))
    return rows


def _short_ds(ds):
    m = re.search(r"synth-powerlaw-N(\d+)-exp", ds)
    return f"synth-N{int(m.group(1))//1000}k" if m else ds


def compare(md_path, K_list, quantile, SM, data_root, avg_degree, exponent):
    empirical = _load_empirical(Path(md_path))
    tags_in_md = sorted({t for t, _, _ in empirical.keys()})

    print(f"# Simulation vs empirical\nSource: {md_path}\n")
    print(f"Only balanced schedule is shown. All numbers are ratios vs legacy of that "
          f"(dataset, op, dim).\n")
    header = (f"| {'dataset':<20} | {'op':>8} | {'dim':>3} | {'K':>5} | "
              f"{'sim speedup':>11} | {'emp speedup (gatv2 fp32)':>24} | {'delta':>6} |")
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    graphs = {}
    for tag in tags_in_md:
        short = _short_ds(tag)
        if short.startswith("synth-N"):
            N = int(short.split("N")[1].rstrip("k")) * 1000
            ei = make_powerlaw(N, avg_degree, exponent=exponent)
        else:
            try:
                ei, N = load_real(short, data_root)
            except Exception as e:
                print(f"skip {short}: {e}")
                continue
        g = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
            ei, N, quantile=quantile, index_dtype=torch.int32,
        )
        graphs[tag] = g

    for tag in tags_in_md:
        g = graphs.get(tag)
        if g is None:
            continue
        legacy_sim = simulate(per_block_work(g, "legacy", 0), SM)
        for (ds, op, dim), row in sorted(empirical.items()):
            if ds != tag:
                continue
            leg_ms = row["legacy"]
            for K in K_list:
                if ("balanced", K) not in row:
                    continue
                bal_sim = simulate(per_block_work(g, "balanced", K), SM)
                sim_speedup = legacy_sim["sim_time"] / bal_sim["sim_time"]
                emp_ms = row[("balanced", K)]
                emp_speedup = leg_ms / emp_ms
                delta = sim_speedup - emp_speedup
                print(f"| {_short_ds(ds):<20} | {op:>8} | {dim:>3} | {K:>5} | "
                      f"{sim_speedup:>10.2f}x | {emp_speedup:>23.2f}x | "
                      f"{delta:>+6.2f} |")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets", nargs="+",
                   default=["synth-N65k", "synth-N262k", "city-roads-L",
                            "web-traffic", "hm-categories"])
    p.add_argument("--K", type=int, nargs="+", default=[132, 264, 528, 1056, 2112])
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--exponent", type=float, default=2.3)
    p.add_argument("--avg-degree", type=int, default=8)
    p.add_argument("--sm", type=int, default=132)
    p.add_argument("--compare-ms", type=str, default=None,
                   help="path to a benchmark_final_sweep .md to compare against")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA required (for graph loading).")
        return 1
    torch.set_default_device("cuda")

    if args.compare_ms:
        compare(args.compare_ms, args.K, args.quantile, args.sm,
                args.data_root, args.avg_degree, args.exponent)
        return 0

    for name in args.datasets:
        if name == "synth-N65k":
            ei = make_powerlaw(65536, args.avg_degree, exponent=args.exponent)
            analyze(name, ei, 65536, args.K, args.quantile, args.sm)
        elif name == "synth-N262k":
            ei = make_powerlaw(262144, args.avg_degree, exponent=args.exponent)
            analyze(name, ei, 262144, args.K, args.quantile, args.sm)
        elif name in REAL_SOURCE:
            try:
                ei, n = load_real(name, args.data_root)
                analyze(name, ei, n, args.K, args.quantile, args.sm)
            except Exception as e:
                print(f"\n### {name}\nskip: {e}")
        else:
            print(f"\n### {name}\nunknown dataset")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
