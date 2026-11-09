"""Mini-batch neighbor-sampling training: DGL vs our kernels (ICLR 2027 rebuttal).

One script, both benchmark environments (see REPRODUCING.md):

    # ours: our GPU sampler + our GATv2 kernels
    $OURS_PY scripts/train_sampled.py --dataset configs/datasets/main/ogbn_products.yaml \
        --backend cuda --sampler ours --fanouts 15,10,5
    # DGL, stock: dgl.dataloading.NeighborSampler + dgl.nn.GATv2Conv
    TURBO_GNN_ALLOW_MISSING_EXTENSION=1 $BASELINE_PY scripts/train_sampled.py ... --backend dgl --sampler dgl
    # control: DGL convs fed by our sampler (isolates the conv-layer difference)
    TURBO_GNN_ALLOW_MISSING_EXTENSION=1 $BASELINE_PY scripts/train_sampled.py ... --backend dgl --sampler ours

Both sides train on the same graph: the dataset loader's edge_index with self-loops added
once (loaded through the "pyg" representation, which builds nothing backend-specific). Node
features and labels live on the GPU in the compute dtype; the per-step input gather is the
only per-batch feature copy.

Timing (per run JSON under --out): epoch wall time (synchronised), and per step the exposed
data-loading wait, forward, backward and optimizer GPU time; with our sampler also the
batch-preparation host/GPU time, which prefetching hides. Warm-up epochs are excluded.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
# ogb starts a non-daemon "is there a newer version?" thread on import (OUTDATED_IGNORE only
# silences its warning); without network it hangs and the interpreter cannot exit after
# training. ogb imports `outdated` under `except ImportError`, so marking it unavailable
# skips the thread.
os.environ.setdefault("OUTDATED_IGNORE", "1")
sys.modules.setdefault("outdated", None)

from src.data.datasets import DatasetConfig, load_single_graph  # noqa: E402
from src.models.config import classifier_spec_from_config  # noqa: E402
from src.models.layers.blocks import ResidualBlock  # noqa: E402
from src.models.layers.prefix_grad import PREFIX_GRAD_MODES  # noqa: E402
from src.models.registry import build as build_registered  # noqa: E402
from src.sampling.loader import PrefetchLoader, resolve_gpu_ms  # noqa: E402
from src.sampling.async_sampler import AsyncNeighborSampler, GraphedNeighborSampler  # noqa: E402
from src.sampling.neighbor_sampler import GPUNeighborSampler, seed_batches  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
INDEX_DTYPES = {"int32": torch.int32, "int64": torch.int64}


# ------------------------------------------------------------------------- data
TASKS = ("multiclass", "binary", "multilabel", "regression")


def dataset_task(ds: dict) -> str:
    """The dataset's own downstream task, from its metadata (GraphLand's task map, OGB's master.csv)."""
    name = ds["name"]
    from src.data.graphland_datasets import GraphLandDataset

    if name in GraphLandDataset.GRAPHLAND_DATASETS:
        return {"multiclass_classification": "multiclass", "binary_classification": "binary", "regression": "regression"}[
            GraphLandDataset.GRAPHLAND_DATASETS[name]
        ]
    if name.lower().startswith("ogbn-"):
        import csv
        import importlib.util

        spec = importlib.util.find_spec("ogb")  # locate without importing (ogb's import starts a version-check thread)
        master = Path(spec.origin).parent / "nodeproppred" / "master.csv"
        rows = {r[0]: r for r in csv.reader(master.open())}
        col = rows[""].index(name)
        kind, num_tasks = rows["task type"][col], int(rows["num tasks"][col])
        if kind == "binary classification":
            return "multilabel" if num_tasks > 1 else "binary"
        return "multiclass"
    return "multiclass"  # Planetoid, Reddit, Amazon, ... are all single-label multiclass


def load_data(dataset_yaml: str, device: torch.device, dtype: torch.dtype, task: str = "auto", targets: str = "native"):
    """Graph, features, labels and splits; ``task`` "auto" takes the dataset's own task.

    ``targets="native"`` keeps GraphLand regression targets as standardized floats (and builds
    ogbn-proteins' node features from its edge features); "binned" reproduces the earlier runs,
    where regression datasets were binned into 5 classes.
    """
    with open(dataset_yaml, encoding="utf-8") as f:
        ds = yaml.safe_load(f)["dataset"]
    native = targets == "native"
    if task == "auto":
        task = dataset_task(ds)
        if task == "regression" and not native:
            task = "multiclass"
    # "pyg" representation: just (edge_index, edge_weight) with self-loops -- shared by both sides.
    sample = load_single_graph(
        DatasetConfig(
            source=ds["source"],
            name=ds["name"],
            # TURBO_GNN_DATA_ROOT: a data directory elsewhere (e.g. a mounted volume, or a writable
            # overlay when data/ is read-only and a new processed cache must be built).
            root=os.environ.get("TURBO_GNN_DATA_ROOT") or ds.get("root", "data"),
            conv_backend="pyg",
            allow_random_split=ds.get("allow_random_split", False),
            native_targets=native,
        )
    )
    edge_index = sample.edge_index.to(device)
    x = sample.x.to(device=device, dtype=dtype)
    splits = {k: getattr(sample, f"{k}_mask").to(device).nonzero().view(-1) for k in ("train", "val", "test")}
    return ds["name"], edge_index, x, Task(task, sample.y.to(device)), splits


class Task:
    """Head width, label tensor, loss and metrics of one downstream task -- shared by both backends.

    multiclass: CE, accuracy.  binary: 1 logit, BCE, average precision (GraphLand) + ROC-AUC.
    multilabel: one logit per task, BCE, mean per-task ROC-AUC (OGB, ogbn-proteins).
    regression: 1 output, MSE on the standardized target, R^2 (GraphLand) + RMSE (standardized units).
    """

    METRICS = {"multiclass": ("acc",), "binary": ("ap", "rocauc"), "multilabel": ("rocauc",), "regression": ("r2", "rmse")}

    def __init__(self, kind: str, y: torch.Tensor) -> None:
        assert kind in TASKS, kind
        self.kind = kind
        if kind == "multiclass":
            self.labels = y.view(-1).long()
            self.out_dim = int(self.labels.max().item()) + 1
        elif kind == "multilabel":
            assert y.ndim == 2, f"multilabel needs [N, T] labels, got {tuple(y.shape)}"
            self.labels = y.float()
            self.out_dim = y.shape[1]
        else:
            self.labels = y.view(-1).float()
            self.out_dim = 1
        self.metric = self.METRICS[kind][0]

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.kind == "multiclass":
            return F.cross_entropy(logits, y)
        if self.kind == "multilabel":
            return F.binary_cross_entropy_with_logits(logits, y)
        if self.kind == "binary":
            return F.binary_cross_entropy_with_logits(logits.view(-1), y)
        return F.mse_loss(logits.view(-1), y)

    def metrics(self, logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        logits = logits.float()
        if self.kind == "multiclass":
            return {"acc": (logits.argmax(-1) == y).float().mean().item()}
        if self.kind == "regression":
            pred = logits.view(-1)
            sse = ((pred - y) ** 2).sum()
            sst = ((y - y.mean()) ** 2).sum()
            return {"r2": (1 - sse / sst).item(), "rmse": (sse / y.numel()).sqrt().item()}
        from sklearn.metrics import average_precision_score, roc_auc_score

        if self.kind == "binary":
            p, t = torch.sigmoid(logits.view(-1)).cpu().numpy(), y.cpu().numpy()
            return {"ap": float(average_precision_score(t, p)), "rocauc": float(roc_auc_score(t, p))}
        p, t = logits.cpu().numpy(), y.cpu().numpy()  # multilabel: OGB evaluator -- tasks with both classes
        aucs = [roc_auc_score(t[:, j], p[:, j]) for j in range(t.shape[1]) if 0 < t[:, j].sum() < t.shape[0]]
        return {"rocauc": float(sum(aucs) / len(aucs))}


def in_csr(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """In-neighbor CSR: row v lists the sources u of edges u -> v."""
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=edge_index.device)
    indptr[1:] = torch.cumsum(counts, 0)
    return indptr, src[order].contiguous()


# ------------------------------------------------------------------------ model
def build_model(model_yaml: str, num_layers: int, backend: str, in_dim: int, num_classes: int, kernel_kwargs: dict):
    with open(model_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    template = cfg["encoder"]["layers"][0]
    layers = []
    for _ in range(num_layers):
        layer = copy.deepcopy(template)
        if backend == "cuda" and kernel_kwargs:
            layer["conv_kwargs"] = {**(layer.get("conv_kwargs") or {}), **kernel_kwargs}
        layers.append(layer)
    cfg["encoder"]["layers"] = layers
    spec = classifier_spec_from_config(
        cfg, backend_to_override=backend, conv_type_to_override="gat_v2", input_dim=in_dim, override_num_classes=num_classes
    )
    return build_registered(cfg.get("architecture", "node_classifier"), spec=spec)


# ---------------------------------------------------------------------- batches
class OursBatchMaker:
    """Exact sampler -> (x, y, per-layer graphs) for the cuda backend or (control) DGL blocks.

    Has host syncs (data-dependent sizes); used for full-neighbor blocks and evaluation.
    Training uses `AsyncBatchMaker`.
    """

    def __init__(self, sampler: GPUNeighborSampler, x, y, backend: str, args) -> None:
        self.sampler, self.x, self.y, self.backend, self.args = sampler, x, y, backend, args

    def graph_for(self, b):
        """One sampled block -> the graph object the backend's conv consumes."""
        if self.backend == "cuda":
            from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

            return G.from_block_csr(
                b.indptr,
                b.indices,
                b.num_src,
                max_in_degree=b.fanout if b.fanout > 0 else None,
                forward_heavy_threshold=self.args.fwd_heavy_threshold,
                backward_heavy_threshold=self.args.bwd_heavy_threshold,
                quantile=self.args.quantile,
            )
        import dgl

        return dgl.create_block(
            ("csc", (b.indptr, b.indices, torch.arange(b.num_edges, dtype=b.indptr.dtype, device=b.indptr.device))),
            num_src_nodes=b.num_src,
            num_dst_nodes=b.num_dst,
            idtype=b.indptr.dtype,
            device=b.indptr.device,
        )

    def __call__(self, seeds: torch.Tensor) -> dict[str, Any]:
        batch = self.sampler.sample(seeds)
        x = self.x[batch.input_nodes]
        y = self.y[batch.seeds]
        graphs = [self.graph_for(b) for b in batch.blocks]
        stats = [(b.num_dst, b.num_src, b.num_edges) for b in batch.blocks]
        # Keep the raw CSR tensors in the batch: for DGL blocks they are what record_stream must see.
        return {"x": x, "y": y, "graphs": graphs, "stats": stats, "_csr": [(b.indptr, b.indices) for b in batch.blocks]}


class AsyncBatchMaker:
    """Sync-free sampler -> the same batch dict, as a two-stage maker (launch, finalize).

    `PrefetchLoader` pipelines the stages; neither issues a blocking CUDA sync. The cuda
    graph object is assembled directly from the sampler's views -- passing the degree bounds
    (fanout, num_dst) and is_directed=True, so its __post_init__ reads nothing from the GPU.
    """

    def __init__(self, sampler: AsyncNeighborSampler, x, y, backend: str) -> None:
        self.sampler, self.x, self.y, self.backend = sampler, x, y, backend

    def launch(self, seeds):
        return self.sampler.launch(seeds)

    def graph_for(self, b):
        if self.backend == "cuda":
            from turbo_gnn.graph import AdjacencyForwardBackwardWithNodeBuckets as G

            return G(
                forward_indptr=b.indptr,
                forward_indices=b.indices,
                backward_indptr=b.bwd_indptr,
                backward_indices=b.bwd_indices,
                forward_light_nodes=b.fwd_light,
                forward_heavy_nodes=b.fwd_heavy,
                backward_light_nodes=b.bwd_light,
                backward_heavy_nodes=b.bwd_heavy,
                max_degree=b.fanout,
                backward_max_degree=b.num_dst,
                is_directed=True,
            )
        import dgl

        return dgl.create_block(
            ("csc", (b.indptr, b.indices, torch.arange(b.num_edges, dtype=b.indptr.dtype, device=b.indptr.device))),
            num_src_nodes=b.num_src,
            num_dst_nodes=b.num_dst,
            idtype=b.indptr.dtype,
            device=b.indptr.device,
        )

    def finalize(self, pending) -> dict[str, Any]:
        fb = self.sampler.finalize(pending)
        x = self.x[fb.input_nodes]
        y = self.y[fb.seeds]
        graphs = [self.graph_for(b) for b in fb.blocks]
        stats = [(b.num_dst, b.num_src, b.num_edges) for b in fb.blocks]
        return {"x": x, "y": y, "graphs": graphs, "stats": stats, "_parts": fb.blocks}


def ours_epoch(maker, idx, args, device, shuffle=True, prefetch=None):
    seeds = seed_batches(idx, args.batch_size, shuffle=shuffle)
    loader = PrefetchLoader(
        maker, seeds, prefetch=args.prefetch == "on" if prefetch is None else prefetch, depth=args.prefetch_depth, device=device
    )
    yield from loader


def dgl_native_loader(graph, idx, args, device, shuffle=True):
    import dgl

    sampler = dgl.dataloading.NeighborSampler(args.fanout_list, prefetch_node_feats=["feat"], prefetch_labels=["label"])
    kw: dict[str, Any] = dict(batch_size=args.batch_size, shuffle=shuffle, drop_last=False, device=device)
    if args.dgl_sampling == "cpu":
        kw.update(num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    else:
        kw.update(num_workers=0, use_uva=args.dgl_sampling == "uva")
    return dgl.dataloading.DataLoader(graph, idx, sampler, **kw)


def dgl_native_epoch(loader):
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            input_nodes, output_nodes, blocks = next(it)
        except StopIteration:
            return
        wait_ms = (time.perf_counter() - t0) * 1e3
        batch = {
            "x": blocks[0].srcdata["feat"],
            "y": blocks[-1].dstdata["label"],
            "graphs": blocks,
            "stats": [(b.num_dst_nodes(), b.num_src_nodes(), b.num_edges()) for b in blocks],
        }
        yield batch, {"wait_host_ms": wait_ms}


# ------------------------------------------------------------------------- loop
def run_epoch(model, optimizer, scaler, batches, args, dtype, task: Task, train=True):
    compute = torch.cuda.current_stream()
    autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
    steps = []
    outs, ys = [], []
    loss_sum = 0.0
    for batch, timing in batches:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        t_host = time.perf_counter()
        ev[0].record(compute)
        with torch.set_grad_enabled(train), autocast:
            logits = model(batch["x"], batch["graphs"])
            loss = task.loss(logits, batch["y"])
        ev[1].record(compute)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                ev[2].record(compute)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                ev[2].record(compute)
                optimizer.step()
        else:
            ev[2].record(compute)
            outs.append(logits.float())
            ys.append(batch["y"])
        ev[3].record(compute)
        loss_sum += loss.detach()
        steps.append((timing, ev, (time.perf_counter() - t_host) * 1e3, batch["stats"]))
    torch.cuda.synchronize()
    rows = []
    for timing, ev, step_host_ms, stats in steps:
        row = resolve_gpu_ms(timing)
        row.update(
            fwd_ms=ev[0].elapsed_time(ev[1]),
            bwd_ms=ev[1].elapsed_time(ev[2]),
            opt_ms=ev[2].elapsed_time(ev[3]),
            step_host_ms=step_host_ms,
            num_src0=stats[0][1],
            edges=sum(s[2] for s in stats),
        )
        rows.append(row)
    metrics = task.metrics(torch.cat(outs), torch.cat(ys)) if not train and outs else None
    return rows, float(loss_sum) / max(len(steps), 1), metrics


@torch.no_grad()
def layerwise_inference(model, x, task: Task, splits, args, dtype, make_full_blocks) -> dict[str, Any]:
    """Full-neighbor, layer-by-layer inference over all nodes (the standard DGL recipe).

    Each layer runs once over every node, in chunks of ``--infer-batch-size`` destination
    rows with their *complete* in-neighborhoods -- so the kernels see the original graph's
    degree skew again, which is where degree-aware partitioning applies. ``make_full_blocks``
    maps a chunk of node ids to (input_nodes, graph) for one full-neighbor layer.
    """
    num_nodes = x.shape[0]
    autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
    chunks = list(torch.arange(num_nodes, device=x.device).split(args.infer_batch_size))
    h = x
    layer_ms = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for block_module in model.encoder.blocks:
        t_layer = time.perf_counter()
        out = None
        for chunk in chunks:
            input_nodes, graph = make_full_blocks(chunk)
            with autocast:
                hb = block_module(h[input_nodes], graph)
            if out is None:
                out = torch.empty(num_nodes, hb.shape[1], device=x.device, dtype=hb.dtype)
            out[chunk] = hb
        h = out
        torch.cuda.synchronize()
        layer_ms.append((time.perf_counter() - t_layer) * 1e3)
    with autocast:
        logits = model.head(model.dropout(h))
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    out: dict[str, Any] = {"wall_s": total_s, "layer_ms": layer_ms, "chunks": len(chunks), "metric": task.metric}
    for k, idx in splits.items():
        if k == "train":
            continue
        m = task.metrics(logits[idx], task.labels[idx])
        out.update({f"{k}_{name}": v for name, v in m.items()})
        out[f"{k}_metric"] = m[task.metric]
    return out


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if vals:
            out[f"{k}_sum"] = float(sum(vals))
            out[f"{k}_median"] = float(statistics.median(vals))
    return out


# ------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset YAML, e.g. configs/datasets/main/ogbn_products.yaml")
    p.add_argument("--model", default="configs/models/sampling_gatv2_128x4.yaml")
    p.add_argument("--backend", choices=["cuda", "dgl"], required=True)
    p.add_argument("--sampler", choices=["ours", "dgl"], required=True)
    p.add_argument("--fanouts", default="15,10,5", help="outermost layer first; -1 = all neighbors")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--index-dtype", choices=list(INDEX_DTYPES), default="int32")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", choices=("auto",) + TASKS, default="auto", help="downstream task (auto: the dataset's own)")
    p.add_argument(
        "--targets",
        choices=["native", "binned"],
        default="native",
        help="native: GraphLand regression targets stay regression; binned: 5-class binning (earlier runs)",
    )
    # ours
    p.add_argument("--prefix-grad", choices=PREFIX_GRAD_MODES, default="autograd")
    p.add_argument(
        "--sampler-impl",
        choices=["async", "exact"],
        default="async",
        help="async: no host syncs while training (padded static shapes); exact: data-dependent sizes, syncs",
    )
    p.add_argument("--sampler-graphs", choices=["on", "off"], default="on", help="capture the async sampler in CUDA graphs")
    p.add_argument("--prefetch", choices=["on", "off"], default="on")
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--fwd-heavy-threshold", type=int, default=None, help="in-degree >= T is heavy (default: all light)")
    p.add_argument("--bwd-heavy-threshold", type=int, default=None, help="out-degree >= T is heavy (default: all light)")
    p.add_argument("--quantile", type=float, default=None, help="full-graph bucketing rule instead of thresholds")
    p.add_argument("--kernel-kwargs", default="{}", help="JSON forwarded to GATv2AggrKernel (warps, bucket launch, ...)")
    # dgl native
    p.add_argument("--dgl-sampling", choices=["gpu", "uva", "cpu"], default="gpu")
    p.add_argument("--num-workers", type=int, default=8, help="CPU sampling workers (--dgl-sampling cpu)")
    p.add_argument(
        "--eval",
        choices=["none", "val", "end"],
        default="end",
        help="after training: end = val + test metric; val = val only (speed runs: a full test pass on "
        "web-scale test splits costs more than the timed epochs)",
    )
    p.add_argument("--inference", choices=["none", "layerwise"], default="none", help="E5: timed full-neighbor inference")
    p.add_argument("--infer-batch-size", type=int, default=65536, help="destination rows per inference chunk")
    p.add_argument("--infer-repeats", type=int, default=2, help="first repeat is warm-up")
    p.add_argument("--out", default="results/rebuttal_sampling")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    args.fanout_list = [int(f) for f in args.fanouts.split(",")]
    return args


def main() -> None:
    args = parse_args()
    # An explicit index: torch 2.4's set_device (used again by the prefetch thread) rejects bare "cuda".
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    dtype = DTYPES[args.dtype]

    name, edge_index, x, task, splits = load_data(args.dataset, device, dtype, args.task, args.targets)
    y = task.labels
    num_nodes = x.shape[0]
    print(f"[{name}] task {task.kind} (metric {task.metric}), output dim {task.out_dim}", flush=True)

    model = build_model(
        args.model, len(args.fanout_list), args.backend, x.shape[1], task.out_dim, json.loads(args.kernel_kwargs)
    ).to(device)
    for m in model.modules():
        if isinstance(m, ResidualBlock):
            m.prefix_grad = args.prefix_grad
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if dtype == torch.float16 else None

    if args.sampler == "ours":
        indptr, indices = in_csr(edge_index, num_nodes)
        sampler = GPUNeighborSampler(indptr, indices, args.fanout_list, index_dtype=INDEX_DTYPES[args.index_dtype])
        maker = OursBatchMaker(sampler, x, y, args.backend, args)
        if args.sampler_impl == "async":
            assert args.quantile is None, "--quantile needs --sampler-impl exact (a quantile is a host-side value)"
            async_sampler = AsyncNeighborSampler(
                indptr,
                indices,
                args.fanout_list,
                index_dtype=INDEX_DTYPES[args.index_dtype],
                fwd_heavy_threshold=args.fwd_heavy_threshold,
                bwd_heavy_threshold=args.bwd_heavy_threshold,
            )
            if args.sampler_graphs == "on":
                # stage 1 replayed from CUDA graphs: ~0.7 ms of CPU per batch instead of ~9 ms
                async_sampler = GraphedNeighborSampler(async_sampler, args.batch_size)
            train_maker = AsyncBatchMaker(async_sampler, x, y, args.backend)
        else:
            train_maker = maker
        train_batches = lambda: ours_epoch(train_maker, splits["train"], args, device)  # noqa: E731
        eval_batches = lambda split: ours_epoch(train_maker, splits[split], args, device, shuffle=False, prefetch=False)  # noqa: E731
    else:
        assert args.backend == "dgl", "--sampler dgl needs --backend dgl"
        import dgl

        g = dgl.graph(
            (edge_index[0].contiguous(), edge_index[1].contiguous()),
            num_nodes=num_nodes,
            idtype=INDEX_DTYPES[args.index_dtype],
        )
        if args.dgl_sampling in ("uva", "cpu"):
            g = g.to("cpu")
            # contiguous: DGL pins in place (UVA) and shares with workers only contiguous storage
            g.ndata["feat"], g.ndata["label"] = x.cpu().contiguous(), y.cpu().contiguous()
            idx = {k: v.cpu() for k, v in splits.items()} if args.dgl_sampling == "cpu" else splits
            if args.dgl_sampling == "uva":
                g.pin_memory_()  # features pinned too; gathered by the GPU over UVA
        else:
            g.ndata["feat"], g.ndata["label"] = x, y
            idx = splits
        train_loader = dgl_native_loader(g, idx["train"].to(g.idtype), args, device)
        train_batches = lambda: dgl_native_epoch(train_loader)  # noqa: E731
        eval_batches = lambda split: dgl_native_epoch(  # noqa: E731
            dgl_native_loader(g, idx[split].to(g.idtype), args, device, shuffle=False)
        )

    epochs = []
    for epoch in range(args.warmup_epochs + args.epochs):
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows, loss, _ = run_epoch(model, optimizer, scaler, train_batches(), args, dtype, task, train=True)
        wall = time.perf_counter() - t0
        rec = {"epoch": epoch, "warmup": epoch < args.warmup_epochs, "wall_s": wall, "steps": len(rows), "loss": loss}
        rec.update(summarize(rows))
        epochs.append(rec)
        print(
            f"[{name} {args.backend}/{args.sampler}] epoch {epoch} wall {wall:.3f}s loss {loss:.4f} "
            f"wait {rec.get('wait_host_ms_sum', 0) / 1e3:.3f}s fwd {rec['fwd_ms_sum'] / 1e3:.3f}s "
            f"bwd {rec['bwd_ms_sum'] / 1e3:.3f}s",
            flush=True,
        )

    result: dict[str, Any] = {
        "dataset": name,
        "task": task.kind,
        "metric": task.metric,
        "args": {k: v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "epochs": epochs,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    timed = [e["wall_s"] for e in epochs if not e["warmup"]]
    result["epoch_wall_s_median"] = statistics.median(timed) if timed else None
    if args.eval != "none":
        model.eval()
        for split in ("val", "test") if args.eval == "end" else ("val",):
            _, _, m = run_epoch(model, None, None, eval_batches(split), args, dtype, task, train=False)
            result.update({f"{split}_{k}": v for k, v in m.items()})  # multiclass keeps val_acc / test_acc
            result[f"{split}_metric"] = m[task.metric]
        print(
            f"val {task.metric} {result['val_metric']:.4f}"
            + (f" test {task.metric} {result['test_metric']:.4f}" if "test_metric" in result else "")
        )

    if args.inference == "layerwise":
        model.eval()
        if args.sampler == "ours":
            full = GPUNeighborSampler(indptr, indices, [-1], index_dtype=INDEX_DTYPES[args.index_dtype])
            full_maker = OursBatchMaker(full, x, y, args.backend, args)

            # The layer input is the previous layer's output, not x, so only the block is built here.
            def make_full_blocks(chunk):
                batch = full.sample(chunk)
                blk = batch.blocks[0]
                return batch.input_nodes, full_maker.graph_for(blk)
        else:
            import dgl

            full_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            g_dev = g.to(device) if g.device != device else g

            def make_full_blocks(chunk):
                input_nodes, _, blocks = full_sampler.sample(g_dev, chunk.to(g_dev.idtype))
                return input_nodes.long(), blocks[0]

        runs = [layerwise_inference(model, x, task, splits, args, dtype, make_full_blocks) for _ in range(args.infer_repeats)]
        result["inference"] = runs[-1]
        result["inference_wall_s_all"] = [r["wall_s"] for r in runs]
        print(f"layerwise inference {runs[-1]['wall_s']:.3f}s layers {[round(t) for t in runs[-1]['layer_ms']]} ms "
              f"test {task.metric} {runs[-1].get('test_metric', float('nan')):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = f"{name}_{args.backend}_{args.sampler}_f{args.fanouts.replace(',', '-')}_b{args.batch_size}{tag}.json"
    (out_dir / fname).write_text(json.dumps(result, indent=2))
    print(f"wrote {out_dir / fname}")


if __name__ == "__main__":
    main()
