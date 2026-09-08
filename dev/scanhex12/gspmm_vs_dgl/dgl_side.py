"""Time dgl.ops and export the graph + operands for turbo_side.py.

    python dgl_side.py <out_dir> --graph ogbn-arxiv
    python dgl_side.py <out_dir> --graph ogbn-arxiv --dtype float16
    python dgl_side.py <out_dir> --graph skewed --backward

The forward CSR turbo_gnn wants has destination nodes as rows, which in DGL is
adj_tensors("csc"); its "csr" is out-edges per source, i.e. the transpose, which
is exactly the backward CSR that turbo_gnn's sum backward walks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys

import dgl
import torch

ALL_OPS = ["copy_u", "copy_e", "add", "sub", "mul", "div"]
REDUCERS = ["sum", "min", "max"]
EXT = {torch.float32: "f32", torch.float16: "f16"}


def dgl_op_name(op: str, reduce: str) -> str:
    return f"{op}_{reduce}" if op.startswith("copy") else f"u_{op}_e_{reduce}"


def warm_up_device(seconds: float = 1.5) -> None:
    """Burn the GPU clock ramp before anything is timed.

    Measured on a T4: the *first* timed cell of a process came out 11-18%
    slower than the next one and varied from run to run, while the second and
    third repeated to 0.5% -- the 10 warmup iterations time_ms does per
    measurement are not enough to cover the ramp.  Whichever cell happens to
    sit first in the loop would otherwise carry that penalty, and since both
    sides of the comparison measure in separate processes it does not cancel.
    """
    from time import perf_counter

    scratch = torch.randn(4096, 4096, device="cuda")
    deadline = perf_counter() + seconds
    while perf_counter() < deadline:
        for _ in range(5):
            scratch = scratch * 1.0001 + 0.0001
        torch.cuda.synchronize()
    del scratch
    torch.cuda.empty_cache()


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
        # Any named graph comes from the cache prepare_graph.py writes, so both
        # sides of the comparison walk the same edge list and neither needs
        # torch_geometric or ogb at measurement time.
        import numpy as np

        cache = os.path.join(args.graph_cache, f"{args.graph}.npz")
        if not os.path.exists(cache):
            sys.exit(f"{args.graph}: no {cache}. Run prepare_graph.py {args.graph} first.")
        blob = np.load(cache)
        n = int(blob["num_nodes"])
        ei = torch.from_numpy(blob["edge_index"]).long().to(device)
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
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--backward", action="store_true", help="also time forward+backward and export reference gradients")
    p.add_argument("--quantile", type=float, default=0.95, help="degree quantile for the light/heavy split")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ogb-root", default=os.environ.get("OGB_ROOT", "data/ogb"))
    p.add_argument("--graph-cache", default=os.environ.get("GRAPH_CACHE", "data/graph_cache"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    device = "cuda"
    dims = [int(s) for s in args.feat_dims.split(",")]
    ops = args.ops.split(",")
    dtype = getattr(torch, args.dtype)
    ext = EXT[dtype]

    edge_index, num_nodes = build_edge_index(args, device)
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    num_edges = g.num_edges()

    indptr, indices, eids = g.adj_tensors("csc")
    bwd_indptr, bwd_indices, _ = g.adj_tensors("csr")

    # DGL does not sort columns within a row, but graph.backward_edge_map does a
    # single argsort over (row, col) and assumes both CSRs are already in that
    # order -- feeding it unsorted arrays silently corrupts the mul/div sum node
    # gradient (the only path that uses the map). Sort here, and carry eids along
    # so the exported edge operand keeps matching the CSR it is indexed by.
    def sort_rows(p_, i_, payload=None):
        rows = torch.repeat_interleave(
            torch.arange(p_.numel() - 1, device=p_.device), (p_[1:] - p_[:-1]).long()
        )
        order = torch.argsort(rows * (p_.numel() - 1) + i_.long(), stable=True)
        return i_[order], (None if payload is None else payload[order])

    indices, eids = sort_rows(indptr, indices, eids)
    bwd_indices, _ = sort_rows(bwd_indptr, bwd_indices)

    indptr, indices = indptr.int().contiguous(), indices.int().contiguous()
    bwd_indptr, bwd_indices = bwd_indptr.int().contiguous(), bwd_indices.int().contiguous()
    eids = eids.long()

    deg = (indptr[1:] - indptr[:-1]).float()
    thr = torch.quantile(deg, args.quantile)
    heavy = (deg > thr).nonzero(as_tuple=True)[0].int().contiguous()
    light = (deg <= thr).nonzero(as_tuple=True)[0].int().contiguous()

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(name: str, t: torch.Tensor) -> None:
        t.detach().cpu().numpy().tofile(os.path.join(args.out_dir, name))

    dump("ptr.i32", indptr)
    dump("idx.i32", indices)
    dump("bwd_ptr.i32", bwd_indptr)
    dump("bwd_idx.i32", bwd_indices)
    dump("light.i32", light)
    dump("heavy.i32", heavy)

    print(f"{args.graph}: N={num_nodes} E={num_edges} avg_deg={num_edges / num_nodes:.1f} "
          f"max_deg={int(deg.max())} light={light.numel()} heavy={heavy.numel()}", flush=True)
    print(f"device {torch.cuda.get_device_name(0)} | dgl {dgl.__version__} | torch {torch.__version__}")
    print(f"dtype {args.dtype} | backward {args.backward} | {args.iters} iters x {args.repeats} repeats, median\n")

    torch.manual_seed(args.seed)
    meta = {
        "graph": args.graph, "N": num_nodes, "E": int(num_edges), "dims": dims, "ops": ops,
        "reducers": REDUCERS, "dtype": args.dtype, "backward": args.backward,
        "quantile": args.quantile, "num_light": int(light.numel()), "num_heavy": int(heavy.numel()),
        "max_degree": int(deg.max()), "iters": args.iters, "repeats": args.repeats,
        "device": torch.cuda.get_device_name(0), "dgl": dgl.__version__, "torch": torch.__version__,
        "timings": {},
    }

    # References are compared on a fixed random sample, not element-wise: a full
    # [E, d] reference gradient is 435 MB per cell on a 3.4M-edge graph, i.e.
    # ~7 GB for the 18-cell table, which overflows a tmpfs scratch dir. A sample
    # this size still catches any systematic error (the CSR-ordering bug this
    # harness had showed 39% relative error across the tensor); it will miss a
    # handful of differing elements, which is what argmin/argmax tie-breaking
    # produces and what we do not want to flag anyway.
    n_samp = 65536
    samp_gen = torch.Generator(device=device).manual_seed(1234)

    def sample_idx(numel):
        return torch.randint(0, numel, (min(n_samp, numel),), device=device, generator=samp_gen)

    needs_edge_operand = any(o != "copy_u" for o in ops)
    for d in dims:
        x_base = torch.randn(num_nodes, d, device=device, dtype=dtype)
        dump(f"x_{d}.{ext}", x_base)
        e_base = eb_base = None
        if needs_edge_operand:
            e_base = torch.rand(num_edges, d, device=device, dtype=dtype) + 0.5
            eb_base = torch.rand(num_edges, device=device, dtype=dtype) + 0.5
            dump(f"e_{d}.{ext}", e_base[eids])
            dump(f"eb_{d}.{ext}", eb_base[eids])
        gseed = torch.randn(num_nodes, d, device=device, dtype=dtype)
        dump(f"gseed_{d}.{ext}", gseed)

        if d == dims[0]:
            si_node = sample_idx(num_nodes * d)
            si_edge = sample_idx(num_edges * d)
            # per-dim names: the indices address an [N, d] / [E, d] tensor, so a
            # dim-independent name makes a later run read out-of-range indices
            dump(f"samp_node_{d}.i64", si_node)
            dump(f"samp_edge_{d}.i64", si_edge)

        warm_up_device()

        for op in ops:
            for reduce in REDUCERS:
                fn = getattr(dgl.ops, dgl_op_name(op, reduce))
                operands = [a for a in (None if op == "copy_e" else x_base,
                                        None if op == "copy_u" else e_base) if a is not None]
                meta["timings"][f"fwd|{op}|{reduce}|{d}"] = time_ms(
                    lambda: fn(g, *operands), args.iters, args.repeats
                )
                if d == dims[0]:
                    with torch.no_grad():
                        out = fn(g, *operands).reshape(num_nodes, d).contiguous()
                    dump(f"ref_{op}_{reduce}_{d}.{ext}", out.view(-1)[si_node])
                    del out

                if not args.backward:
                    continue

                # requires_grad on the base tensors instead of clones: a clone of
                # the [E, d] edge operand is 1.7 GB on a 3.4M-edge graph at
                # d=128, and holding base + clone + grad OOMs a 15 GB card
                x = x_base.requires_grad_(True) if op != "copy_e" else None
                e = e_base.requires_grad_(True) if op != "copy_u" else None
                leaves = [t for t in (x, e) if t is not None]
                grad_operands = [a for a in (x, e) if a is not None]

                def step():
                    for t in leaves:
                        t.grad = None
                    fn(g, *grad_operands).reshape(num_nodes, d).backward(gseed)

                meta["timings"][f"fb|{op}|{reduce}|{d}"] = time_ms(step, args.iters, args.repeats)

                # backward alone: build the graph once, replay the backward pass
                # with retain_graph. Zeroing via `grad = None` is O(1), so what
                # is timed is the backward kernels and nothing else.
                kept = fn(g, *grad_operands).reshape(num_nodes, d)

                def step_bwd():
                    for t in leaves:
                        t.grad = None
                    kept.backward(gseed, retain_graph=True)

                meta["timings"][f"bwd|{op}|{reduce}|{d}"] = time_ms(step_bwd, args.iters, args.repeats)
                del kept

                if d == dims[0]:
                    for t in leaves:
                        t.grad = None
                    fn(g, *grad_operands).reshape(num_nodes, d).backward(gseed)
                    if x is not None:
                        dump(f"refgx_{op}_{reduce}_{d}.{ext}", x.grad.view(-1)[si_node])
                    if e is not None:
                        # sample straight out of the DGL-ordered gradient: a flat
                        # index i of the CSR-ordered [E, d] tensor is edge
                        # eids[i // d], feature i % d. Permuting the whole thing
                        # first would allocate another full copy of it.
                        dump(f"refge_{op}_{reduce}_{d}.{ext}",
                             e.grad[eids[si_edge // d], si_edge % d])

                # each cell holds the operand clones plus their grads plus the
                # retained autograd graph; at d=128 on a 3.4M-edge graph that is
                # three copies of a 1.7 GB edge tensor, which OOMs a 15 GB card
                for t in leaves:
                    t.grad = None
                    t.requires_grad_(False)
                del x, e, leaves, grad_operands, step, step_bwd
                torch.cuda.empty_cache()

            kinds = ["fwd"] + (["bwd", "fb"] if args.backward else [])
            for kind in kinds:
                print(f"  dgl {kind:3} {op:8} d={d:4} " + "  ".join(
                    f"{r}={meta['timings'][f'{kind}|{op}|{r}|{d}']:9.3f}" for r in REDUCERS) + " ms", flush=True)

        del x_base, e_base, eb_base, gseed
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    for d in dims:
        vals = [meta["timings"][f"fwd|{op}|{r}|{d}"] for op in ops for r in REDUCERS]
        with open(os.path.join(args.out_dir, f"dgl_{d}.f64"), "wb") as fh:
            fh.write(struct.pack(f"{len(vals)}d", *vals))

    print(f"\nexported to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
