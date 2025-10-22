import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

sys.path.append("./")

from src.backends.registry import BackendRegistry
from src.benchmarking.autotuner import TuningResult, grid_autotune
from src.data.converters import to_dgl_graph, to_pyg_data

doc = """
Autotuning launcher.

Grid-search over a parameter space for a given backend convolution (kernel),
measuring runtime and picking the best-performing config.
"""

# TODO add autotune for datasets
# TODO add other backends here


def _read_yaml(path: str) -> dict[str, Any]:
    """Read YAML to dict.

    Args:
        path (str): Path.

    Returns:
        Dict[str, Any]: Parsed.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_graph(num_nodes: int, avg_degree: int, device: torch.device):
    """Create a simple random graph as (edge_index, None).

    Args:
        num_nodes (int): Nodes.
        avg_degree (int): Avg out-degree.
        device (torch.device): Device.

    Returns:
        Any: Graph container (PyG Data/DGL Graph or tuple) depending on backend.
    """
    E = max(1, num_nodes * max(1, avg_degree))
    src = torch.randint(0, num_nodes, (E,), device=device)
    dst = torch.randint(0, num_nodes, (E,), device=device)
    return torch.stack([src.long(), dst.long()], dim=0), None


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: Args.
    """
    p = argparse.ArgumentParser(description="Autotune a backend convolution.")
    p.add_argument("--layer", type=str, required=True, choices=["gcn", "gat_v2", "sage", "gin", "mean_aggr"])
    p.add_argument("--backend", type=str, required=True)
    p.add_argument(
        "--param-space", type=str, required=True, help="YAML dict of lists, e.g., {'tile': [64,128], 'unroll':[1,2]}"
    )
    p.add_argument("--num-nodes", type=int, default=20000)
    p.add_argument("--avg-degree", type=int, default=10)
    p.add_argument("--in-ch", type=int, default=128)
    p.add_argument("--out-ch", type=int, default=128)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    """Entry: autotune a conv kernel.

    Returns:
        int: Exit code.
    """
    args = parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # inputs
    edge_index, edge_weight = _make_graph(args.num_nodes, args.avg_degree, device=device)
    backend = BackendRegistry.get_backend(args.backend)
    graph = (edge_index, edge_weight)
    if args.backend == "pyg":
        graph = to_pyg_data(edge_index, args.num_nodes, edge_weight)
    elif args.backend == "dgl":
        graph = to_dgl_graph(edge_index, args.num_nodes, edge_weight)
    else:
        raise NotImplementedError(f"Backend {args.backend} is not supported")
    x = torch.randn(args.num_nodes, args.in_ch, device=device)

    # conv to tune
    conv = backend.create_conv(
        args.layer, args.in_ch, args.out_ch, heads=args.heads if args.layer == "gat_v2" else 1
    ).to(device)

    # measure function
    def _measure() -> None:
        out = conv(x, graph)
        loss = (out**2).sum() * 1e-6
        loss.backward()

    # param space
    param_space = _read_yaml(args.param_space)

    # autotune
    result: TuningResult = grid_autotune(
        target=conv,
        measure=_measure,
        param_space=param_space,
        warmup=args.warmup,
        iters=args.iters,
    )

    payload = {
        "best_config": result.best_config,
        "best_ms_per_iter": result.best_result.ms_per_iter,
        "trials": [{"config": cfg, "ms_per_iter": r.ms_per_iter} for cfg, r in result.trials],
    }
    print(json.dumps(payload, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
