import argparse
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

from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets  # noqa: E402
from turbo_gnn.scheduling import edge_balanced_partition  # noqa: E402


REAL_SOURCE = {
    "ogbn-arxiv":    "ogbn",
    "ogbn-products": "ogbn",
    "web-traffic":   "pyg",
    "hm-categories": "pyg",
    "city-roads-L":  "pyg",
}


def _finalize(src, dst, N, device):
    src_all = torch.cat([src, dst, torch.arange(N, device=device)])
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)])
    flat = src_all.long() * N + dst_all.long()
    flat = torch.unique(flat)
    return torch.stack([flat // N, flat % N])


def make_powerlaw(N, avg_degree, seed=42, exponent=2.3, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    E = N * avg_degree
    ranks = torch.arange(1, N + 1, device=device, dtype=torch.float)
    weights = ranks.pow(-1.0 / (exponent - 1.0))
    weights = weights[torch.randperm(N, device=device, generator=gen)]
    probs = weights / weights.sum()
    src = torch.multinomial(probs, E, replacement=True, generator=gen)
    dst = torch.multinomial(probs, E, replacement=True, generator=gen)
    return _finalize(src, dst, N, device)


def load_real(name, root):
    from src.data.datasets import DatasetConfig, load_single_graph
    cfg = DatasetConfig(source=REAL_SOURCE[name], name=name, root=root, conv_backend="cuda")
    graph = load_single_graph(cfg)
    ei = graph.edge_index
    if not isinstance(ei, torch.Tensor):
        ei = torch.as_tensor(ei)
    return ei.to("cuda"), int(graph.num_nodes)


def _signed_indptr(indptr):
    if indptr.dtype == torch.uint32:
        return indptr.view(torch.int32)
    if indptr.dtype == torch.uint64:
        return indptr.view(torch.int64)
    return indptr


def bin_sums_from_offsets(node_indices, indptr, offsets):
    signed = _signed_indptr(indptr)
    degrees = signed.diff()[node_indices.long()].cpu().numpy()
    offs = offsets.cpu().numpy()
    return np.array([degrees[offs[b]:offs[b + 1]].sum() for b in range(len(offs) - 1)])


def strided_bin_sums(node_indices, indptr, K):
    signed = _signed_indptr(indptr)
    degrees = signed.diff()[node_indices.long()].cpu().numpy()
    return np.array([degrees[b::K].sum() for b in range(K)])


def stats(bins, label):
    bins = np.asarray(bins, dtype=np.int64)
    total = bins.sum()
    mean = total / max(len(bins), 1)
    return {
        "label": label,
        "K": len(bins),
        "mean": mean,
        "max": int(bins.max()) if len(bins) else 0,
        "max_over_mean": float(bins.max()) / mean if mean > 0 else float("nan"),
        "std_over_mean": float(bins.std()) / mean if mean > 0 else float("nan"),
        "p99_over_mean": float(np.percentile(bins, 99)) / mean if mean > 0 else float("nan"),
    }


def analyze(tag, edge_index, N, K_list, quantile):
    g = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index, N, quantile=quantile, index_dtype=torch.int32,
    )
    light = g.forward_light_nodes
    indptr = g.forward_indptr
    signed = _signed_indptr(indptr)
    all_deg = signed.diff().cpu().numpy()
    light_deg = signed.diff()[light.long()].cpu().numpy()

    print(f"\n### {tag}")
    print(f"N={N}  light={len(light_deg)}  heavy={int(g.forward_heavy_nodes.numel())}")
    print(f"  full   deg: mean={all_deg.mean():.1f}  max={all_deg.max()}  "
          f"max/mean={all_deg.max()/all_deg.mean():.0f}")
    print(f"  light  deg: mean={light_deg.mean():.1f}  max={light_deg.max()}  "
          f"max/mean={light_deg.max()/light_deg.mean():.0f}")

    rows = [stats(light_deg, "legacy (per-vertex)")]
    for K in K_list:
        rows.append(stats(strided_bin_sums(light, indptr, K), f"gsl (strided, K={K})"))
        sorted_nodes, offsets = edge_balanced_partition(light, indptr, K)
        rows.append(stats(bin_sums_from_offsets(sorted_nodes, indptr, offsets),
                          f"balanced (LPT, K={K})"))

    print()
    header = f"| {'schedule':<28} | {'K':>7} | {'mean':>10} | {'max':>10} | "\
             f"{'max/mean':>8} | {'std/mean':>8} | {'p99/mean':>8} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for r in rows:
        print(f"| {r['label']:<28} | {r['K']:>7} | {r['mean']:>10.1f} | "
              f"{r['max']:>10} | {r['max_over_mean']:>8.2f} | "
              f"{r['std_over_mean']:>8.2f} | {r['p99_over_mean']:>8.2f} |")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets", nargs="+",
                   default=["synth-N65k", "synth-N262k", "city-roads-L",
                            "web-traffic", "hm-categories"])
    p.add_argument("--K", type=int, nargs="+", default=[132, 264, 528, 1056])
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--exponent", type=float, default=2.3)
    p.add_argument("--avg-degree", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    torch.set_default_device("cuda")

    for name in args.datasets:
        if name == "synth-N65k":
            ei = make_powerlaw(65536, args.avg_degree, exponent=args.exponent)
            analyze(name, ei, 65536, args.K, args.quantile)
        elif name == "synth-N262k":
            ei = make_powerlaw(262144, args.avg_degree, exponent=args.exponent)
            analyze(name, ei, 262144, args.K, args.quantile)
        elif name in REAL_SOURCE:
            try:
                ei, n = load_real(name, args.data_root)
                analyze(name, ei, n, args.K, args.quantile)
            except Exception as e:
                print(f"\n### {name}\nskip: {e}")
        else:
            print(f"\n### {name}\nunknown dataset")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
