from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys

import torch
import dgl


ALL_OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]

def dgl_op_name(op: str, reduce: str) -> str:
    return f"{op}_{reduce}" if op.startswith("copy") else f"u_{op}_e_{reduce}"


def time_ms(fn, iters: int, repeats: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def build_edge_index(args, device: str) -> tuple[torch.Tensor, int]:
    """-> edge_index [2, E] with a self-loop on every node, and N.

    Self-loops keep every in-degree non-zero, which removes the documented
    isolated-node divergence (DGL yields +-inf for min/max where turbo_gnn
    yields 0) from the reference check.
    """
    if args.graph in ("random", "skewed"):
        n = args.nodes
        g = torch.Generator(device=device).manual_seed(args.seed)
        m = n * args.avg_degree
        src = torch.randint(0, n, (m,), device=device, generator=g)
        if args.graph == "skewed":
            dst = (torch.rand(m, device=device, generator=g) ** 4 * n).long().clamp_(0, n - 1)
        else:
            dst = torch.randint(0, n, (m,), device=device, generator=g)
    else:
        from ogb.nodeproppred import NodePropPredDataset

        graph, _ = NodePropPredDataset(name=args.graph, root=args.ogb_root)[0]
        n = int(graph["num_nodes"])
        ei = torch.from_numpy(graph["edge_index"]).long().to(device)
        src, dst = ei[0], ei[1]

    loops = torch.arange(n, device=device)
    return torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])]), n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--graph", default="random", help="random | skewed | an OGB name (ogbn-arxiv, ogbn-products, ...)")
    p.add_argument("--nodes", type=int, default=200_000, help="synthetic graphs only")
    p.add_argument("--avg-degree", type=int, default=16, help="synthetic graphs only")
    p.add_argument("--feat-dims", default="32,64,128")
    p.add_argument("--ops", default=",".join(ALL_OPS),
                   help="restrict the op table; use copy_u alone for graphs whose [E, d] edge operand does not fit")
    p.add_argument("--quantile", type=float, default=0.95, help="degree quantile for the light/heavy split")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ogb-root", default=os.environ.get("OGB_ROOT", "data/ogb"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    device = "cuda"
    dims = [int(s) for s in args.feat_dims.split(",")]
    ops = args.ops.split(",")

    edge_index, num_nodes = build_edge_index(args, device)
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    num_edges = g.num_edges()

    indptr, indices, eids = g.adj_tensors("csc")
    indptr, indices = indptr.int().contiguous(), indices.int().contiguous()
    eids = eids.long()

    deg = (indptr[1:] - indptr[:-1]).float()
    thr = torch.quantile(deg, args.quantile)
    heavy = (deg > thr).nonzero(as_tuple=True)[0].int().contiguous()
    light = (deg <= thr).nonzero(as_tuple=True)[0].int().contiguous()

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(name: str, t: torch.Tensor) -> None:
        t.cpu().numpy().tofile(os.path.join(args.out_dir, name))

    dump("ptr.i32", indptr)
    dump("idx.i32", indices)
    dump("light.i32", light)
    dump("heavy.i32", heavy)

    print(f"{args.graph}: N={num_nodes} E={num_edges} avg_deg={num_edges / num_nodes:.1f} "
          f"max_deg={int(deg.max())} light={light.numel()} heavy={heavy.numel()}", flush=True)
    print(f"device {torch.cuda.get_device_name(0)} | dgl {dgl.__version__} | torch {torch.__version__}")
    print(f"timing {args.iters} iters x {args.repeats} repeats, median\n")

    torch.manual_seed(args.seed)
    meta = {
        "graph": args.graph, "N": num_nodes, "E": int(num_edges), "dims": dims, "ops": ops,
        "reducers": REDUCERS, "quantile": args.quantile, "num_light": int(light.numel()),
        "num_heavy": int(heavy.numel()), "max_degree": int(deg.max()),
        "iters": args.iters, "repeats": args.repeats,
        "device": torch.cuda.get_device_name(0), "dgl": dgl.__version__, "torch": torch.__version__,
        "timings": {},
    }

    needs_edge_operand = any(o != "copy_u" for o in ops)
    for d in dims:
        x = torch.randn(num_nodes, d, device=device)
        dump(f"x_{d}.f32", x)
        e = eb = None
        if needs_edge_operand:
            e = torch.rand(num_edges, d, device=device) + 0.5
            eb = torch.rand(num_edges, device=device) + 0.5
            dump(f"e_{d}.f32", e[eids])    # CSR order: the contract turbo_gnn expects
            dump(f"eb_{d}.f32", eb[eids])

        for op in ops:
            for reduce in REDUCERS:
                fn = getattr(dgl.ops, dgl_op_name(op, reduce))
                operands = [a for a in (None if op == "copy_e" else x, None if op == "copy_u" else e) if a is not None]
                meta["timings"][f"{op}|{reduce}|{d}"] = time_ms(
                    lambda: fn(g, *operands), args.iters, args.repeats
                )
                if d == dims[0]:  # reference output, for turbo_side.py to check against
                    with torch.no_grad():
                        dump(f"ref_{op}_{reduce}_{d}.f32", fn(g, *operands).reshape(num_nodes, d).contiguous())
            print(f"  dgl {op:8} d={d:4} " + "  ".join(
                f"{r}={meta['timings'][f'{op}|{r}|{d}']:9.3f}" for r in REDUCERS) + " ms", flush=True)

        del x, e, eb
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    for d in dims:
        vals = [meta["timings"][f"{op}|{r}|{d}"] for op in ops for r in REDUCERS]
        with open(os.path.join(args.out_dir, f"dgl_{d}.f64"), "wb") as fh:
            fh.write(struct.pack(f"{len(vals)}d", *vals))

    print(f"\nexported to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
