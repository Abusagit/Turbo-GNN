import argparse
import json
from typing import Optional, Tuple
from pathlib import Path
import torch
from src.benchmarking.microbench import time_callable, MicrobenchResult
from src.backends.registry import BackendRegistry
from src.data.converters import to_pyg_data, to_dgl_graph
from src.data.datasets import GraphSample, MODEL_BACKEND_TO_GRAPH_REPR

doc = """
Layer microbenchmark launcher.

Creates a random graph and features, instantiates a backend convolution, and
times forward/backward kernel using CUDA events (or wall-clock on CPU).
"""


def _make_random_graph(num_nodes: int, avg_degree: int, *, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generate an Erdos-Renyi-like random edge_index with approx avg_degree.

    Args:
        num_nodes (int): Number of nodes.
        avg_degree (int): Approximate average out-degree.
        device (torch.device): Torch device.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: (edge_index [2,E], edge_weight or None)
    """
    E = max(1, num_nodes * max(1, avg_degree))
    src = torch.randint(0, num_nodes, (E,), device=device, dtype=torch.long)
    dst = torch.randint(0, num_nodes, (E,), device=device, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, None


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: Parsed args.
    """
    p = argparse.ArgumentParser(description="Microbenchmark graph conv layers.")
    p.add_argument("--layer", type=str, required=True, choices=["gcn", "gat", "sage", "gin"])
    p.add_argument("--backend", type=str, required=True, help="Backend name (pyg|dgl|...).")
    p.add_argument("--num-nodes", type=int, default=20000)
    p.add_argument("--avg-degree", type=int, default=10)
    p.add_argument("--in-ch", type=int, default=128)
    p.add_argument("--out-ch", type=int, default=128)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--mode", type=str, default="forward", choices=["forward", "train"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--amp", type=str, default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON result.")
    return p.parse_args()


def main() -> int:
    """Entry: run the microbenchmark.

    Returns:
        int: Exit code.
    """
    args = parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # graph + features
    edge_index, edge_weight = _make_random_graph(args.num_nodes, args.avg_degree, device=device)

    x = torch.randn(args.num_nodes, args.in_ch, device=device)
    graph = GraphSample(backend=MODEL_BACKEND_TO_GRAPH_REPR[args.backend], x=x, y=torch.zeros(len(x)),edge_index=edge_index, edge_weight=edge_weight).graph_repr
    # conv
    backend = BackendRegistry.get_backend(args.backend)
    if args.layer != "gat":
        conv = backend.create_conv(args.layer, args.in_ch, args.out_ch)
    else:
        conv = backend.create_conv(args.layer, args.in_ch, args.out_ch, heads=args.heads)

    conv = conv.to(device)

    # measure function
    amp_dtype = None
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16

    def _fn_forward() -> None:
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ = conv(x, graph)
        else:
            _ = conv(x, graph)

    def _fn_train() -> None:
        opt = torch.optim.SGD(conv.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = conv(x, graph)
                loss = (out ** 2).sum() * 1e-6
            loss.backward()
        else:
            out = conv(x, graph)
            loss = (out ** 2).sum() * 1e-6
            loss.backward()
        opt.step()

    fn = _fn_forward if args.mode == "forward" else _fn_train
    res: MicrobenchResult = time_callable(fn, warmup=args.warmup, iters=args.iters)
    print(json.dumps({"iters": res.iters, "ms_per_iter": res.ms_per_iter, "device": res.device}, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"iters": res.iters, "ms_per_iter": res.ms_per_iter, "device": res.device}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
