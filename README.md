# skewgnn

CUDA kernels for graph neural network aggregation on graphs with skewed degree
distributions. This repository accompanies an anonymous submission and contains the
kernels, their tests, and the scripts that produce every measurement in the paper.

The kernels cover:

- **Attention layers** -- GATv2 (`gatv2_aggr`) and graph transformer attention
  (`graph_transformer_aggr`), each a single fused pass over the neighborhood with online
  softmax.
- **Generalized SpMM** (`gspmm`) -- `{copy_u, copy_e, add, sub, mul, div} x {sum, min, max}`,
  the `dgl.ops` g-SpMM family, also exposed under its 18 names (`copy_u_sum`, `u_mul_e_max`, ...).
- **Generalized SDDMM** (`gsddmm`) -- per-edge `{add, sub, mul, div, dot, copy}` over
  source, destination and edge operands, with a node-partitioned and an edge-partitioned
  kernel (`variant="node"` / `"edge"`, or `"auto"` to time both once per graph).

All of them schedule work by degree: destination nodes are split into a light and a heavy
bucket at a degree quantile, each bucket gets its own warp count, buckets can be walked in
descending degree and launched concurrently, and the edge lists of heavy nodes can be cut
into fixed-size slices whose partial results are merged exactly. Launch parameters are
chosen per graph by a built-in autotuner.

## Requirements

- An NVIDIA GPU of compute capability 8.0 or newer (the paper's measurements use an A100).
- CUDA toolkit 12.x or newer with `nvcc`, and a C++20 compiler.
- Python >= 3.10 and PyTorch >= 2.0 built for the same CUDA major version.

## Build

The extension is compiled from source:

```bash
pip install --no-build-isolation .
```

or, for development, in place:

```bash
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=32 python setup.py build_ext --inplace
```

Set `TORCH_CUDA_ARCH_LIST` to your GPU's architecture. The attention and g-SDDMM kernels are
split into one translation unit per dtype (and per operator for g-SDDMM), so the build
parallelizes well; with 64 jobs it takes about 20 minutes.

Environment variables read by `setup.py`: `SKEWGNN_SKIP_CUDA_BUILD=TRUE` builds without the
extension, `SKEWGNN_FORCE_BUILD=TRUE` fails instead of falling back when CUDA is missing,
`SKEWGNN_NVCC_THREADS` sets nvcc's per-file parallelism (default 4).

## Usage

```python
import torch
from skewgnn import AdjacencyForwardBackwardWithNodeBuckets, gspmm, gsddmm

edge_index = torch.randint(0, 1000, (2, 20000), device="cuda")
graph = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
    edge_index, num_nodes=1000, quantile=0.99, index_dtype=torch.int32
).to("cuda")

x = torch.randn(1000, 128, device="cuda", dtype=torch.float16, requires_grad=True)

h = gspmm(graph, x, op="copy_u", reduce="sum")           # sum over in-neighbors
m = gspmm(graph, x, op="copy_u", reduce="max")           # max over in-neighbors
logits = gsddmm(graph, x, x, op="dot")                   # one score per edge
h.sum().backward()
```

Every op accepts `autotune=True` to search its launch parameters for the given graph once
and cache the result. The docstrings in `skewgnn/ops.py` list each op's parameters and
operand shapes.

## Tests

```bash
python -m pytest tests/correctness tests/unit
```

The correctness tests compare every kernel against a PyTorch reference in fp32, fp16 and
bf16, forward and backward.

## Reproducing the paper

Every table, the figure and every number quoted in the text come from the commands below.
All timings are fp16 on one NVIDIA A100-SXM4-80GB. Run each benchmark on an idle GPU:
a second process on the card changes the ratios. Commands are run from the repository root
unless stated otherwise, and write under `results/` or next to the script that produced them.

### Environments

Two Python environments are used, and the scripts take their interpreters from variables:

| variable | contents | used for |
| --- | --- | --- |
| `OURS_PY` | this package built (see Build), PyTorch, PyG 2.8 | our kernels; the PyG column of Table 1 |
| `BASELINE_PY` | Python 3.11, PyTorch 2.4.0+cu121, DGL 2.4.0+cu121, PyG 2.8 | every DGL column; the PyG columns of Tables 2 and 3 and Appendix I |

DGL publishes no build for newer PyTorch, hence the second environment. It cannot load an
extension compiled for a newer PyTorch, and the benchmark harness imports this package for
every backend, so the baseline runs set `SKEWGNN_ALLOW_MISSING_EXTENSION=1`: the package then
imports without its extension, and any attempt to launch one of our kernels raises. The
runners below set it where it is needed. `pandas` speeds up graph preparation and
`matplotlib` is needed for the figure and for the g-SpMM harness's charts.

### Data

The attention experiments load graphs through `configs/datasets/*.yaml` and download them
into `data/` on first use. The g-SpMM, g-SDDMM, PyG and dataset-statistics scripts read a
cache of edge lists instead, which `prepare_graph.py` builds once for all 27 graphs:

```bash
SP16="cora citeseer pubmed city-roads-M city-roads-L artnet-exp tolokers-2 city-reviews \
twitch-views hm-categories avazu-ctr pokec-regions web-fraud ogbn-arxiv ogbn-proteins ogbn-products"
SD25="cora citeseer pubmed CoraFull WikiCS amazon-photo amazon-computers coauthor-cs \
coauthor-physics Flickr city-roads-M city-roads-L artnet-exp artnet-views tolokers-2 ogbn-arxiv \
web-topics city-reviews reddit twitch-views hm-categories hm-prices avazu-ctr pokec-regions ogbn-products"

$BASELINE_PY benchmarks/gspmm_vs_dgl/prepare_graph.py $SP16 $SD25 \
    --out-dir data/graph_cache --graphland-root data/graphland --ogb-root data/ogb --pyg-root data
```

GraphLand graphs are read from the extracted Zenodo archive under `--graphland-root`, OGB
graphs from their raw download under `--ogb-root`, the Planetoid graphs through DGL and the
remaining benchmarks through PyG (which download them). `SP16` is the g-SpMM suite and `SD25`
the g-SDDMM suite; the attention suite is `SP16` as well (`benchmarks/paper/graphs.py`).

### Table 1 and Appendix J -- GATv2 ablation against DGL and PyG

```bash
for r in r1 r2 r3; do OURS_PY=$OURS_PY benchmarks/paper/run_attention_ablation.sh 0 $r; done
OURS_PY=$OURS_PY           benchmarks/paper/run_attention_baselines.sh pyg 0
BASELINE_PY=$BASELINE_PY   benchmarks/paper/run_attention_baselines.sh dgl 0

$OURS_PY benchmarks/paper/summarize_attention_ablation.py --heads 2 --width 128   # Table 1
$OURS_PY benchmarks/paper/summarize_attention_ablation.py --heads 4 --width 256   # one Appendix J block
```

Each (graph, heads, width) cell is one process that times a 160-point factorial of the
techniques on the same tensors; the summary reduces every point to its median over the three
repeats and fills each column with the fastest point that uses only the techniques enabled so
far (prior kernel, then bucketing, ordering, concurrent launch, slicing, asynchronous
staging), which is the rule the tables use. Both runners default to the full Appendix J grid,
heads {2, 4, 8} x width {64, 128, 256}; pass `2:128` to run only Table 1. `--width` is the
harness's `--feature-dim`, which for GATv2 is the width of each head.

### Section 4.1 -- hardware counters

```bash
sudo -E env PATH="$PATH" $OURS_PY benchmarks/paper/run_ncu_bandwidth.py \
    --gpu 0 --graphs ogbn-arxiv web-fraud --modes forward
```

Profiles the GATv2 kernels under Nsight Compute at both ends of the ladder (`baseline`: the
prior kernel; `best`: bucketing, ordering, concurrent launch and slicing) and writes one row
per launch to `results/ncu_bandwidth/metrics.json`. The share of active warps is
`warps_pct`; achieved bandwidth is `dram_bytes / ns` against the card's 2039 GB/s. Counter
access usually needs root.

### Table 2, Appendix K and the g-SpMM half of Appendix M

```bash
cd benchmarks/gspmm_vs_dgl
GRAPHS="$SP16" DIMS=64,128,256,512 DTYPES=float16 STAGES_LIST="0 1 2 4" REPEATS=3 \
PYTHON_DGL=$BASELINE_PY PYTHON_SKEWGNN=$OURS_PY ./run_matrix.sh
```

Writes `results/results_<graph>_float16_bwd.json` (pipeline depth 0) and
`..._float16_bwd_st{1,2,4}.json`, each holding every operator, reducer and width, forward,
backward and both, for DGL and for our kernels, 30 calls x 3 repeats. Table 2 is the
`copy_u_sum` and `copy_u_max` rows at width 128 of the depth-0 files, Appendix K all of them,
and Appendix M's g-SpMM rows compare depths 1, 2, 4 against 0. On graphs above 10M edges
only `copy_u` is measured, because an `[E, d]` operand does not fit.

### Tables 2 and 3, PyG columns, and Appendix I -- PyG and DGL in one process

```bash
cd benchmarks/paper
$BASELINE_PY pyg_vs_dgl.py --family gspmm  --graphs $SP16 --out gspmm_pyg_dgl.jsonl
$BASELINE_PY pyg_vs_dgl.py --family gsddmm --graphs $SD25 --out gsddmm_pyg_dgl.jsonl
```

Times PyG and DGL back to back in the same process at width 128 with the protocol of the
table they feed; PyG's speedups are taken against the DGL run of the same process.
`run_when_free.sh` runs both on the first GPU that stays idle.

### Table 3 and Appendix L -- g-SDDMM against DGL

```bash
OURS_PY=$OURS_PY BASELINE_PY=$BASELINE_PY benchmarks/paper/run_gsddmm_all.sh 0
```

Times all 32 operators at widths 32, 64, 128 and 256 on the 25 graphs, our node- and
edge-partitioned kernels and DGL, into `benchmarks/paper/gsddmm_all_{ours,dgl}.jsonl`
(5 warmup and 20 timed calls between CUDA events). Table 3 is `u_dot_v` and `u_add_v` at
width 128; Appendix L is every row. The script refuses to start on a busy GPU.

### Appendix M -- the g-SDDMM half

```bash
$OURS_PY benchmarks/paper/gsddmm_stages.py \
    --graphs cora citeseer pubmed city-roads-M city-roads-L artnet-exp tolokers-2 \
             ogbn-arxiv city-reviews twitch-views web-fraud pokec-regions \
    --dims 32 64 128 256 --warmup 5 --iters 20 --out results/gsddmm_stages.jsonl
```

Pipeline depths 0, 2, 3 against edges per warp 1, 4, 8, 16 in the edge-partitioned kernel;
edges per warp = 1 is the control, where there is nothing to prefetch across.

### Appendix A -- dataset statistics

```bash
$OURS_PY benchmarks/paper/graph_meta.py results/graph_meta.json
```

Degree statistics of every graph in the cache: nodes, edges as stored and as the harness
walks them, degree quantiles, skew and the edge share of the top 1% of nodes.

### Section 4.3, the figure and Appendix H -- the utilization simulator

The simulator is CPU-only except for step 1; `scripts/ablation/README.md` describes it in
full.

```bash
# 1. Time the real kernels: 14 graphs x {gt, gat_v2, min_aggr} x {fwd, bwd} x {128, 256} = 168 runs.
$OURS_PY scripts/ablation/measure_kernel_times.py \
    --graphs cora citeseer pubmed city-roads-M artnet-exp tolokers-2 city-roads-L ogbn-arxiv \
             city-reviews twitch-views hm-categories avazu-ctr pokec-regions web-fraud \
    --warmup 10 --iters 30 --out results/measurements.jsonl

# 2. Fit t(N, E) = c + aN + bE per cell (Appendix H's R^2 and median error).
$OURS_PY scripts/ablation/calibrate_cost_model.py --from-json results/measurements.jsonl --out cost_models.json

# 3. Appendix H's table: one run per graph; ogbn-arxiv also sweeps the assignment policy.
for g in tolokers-2 city-reviews city-roads-M web-fraud; do
  $OURS_PY scripts/ablation/simulate_load_imbalance.py --dataset $g --quantile 0.99 \
      --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
      --sms 108 --memory-bandwidth-gbps 2039 --seed 42 --assignments contiguous \
      --vertices-per-block 1 --heavy-slice-sizes 0 1024 \
      --launch-modes single sequential concurrent --out results/sim/$g
done
$OURS_PY scripts/ablation/simulate_load_imbalance.py --dataset ogbn-arxiv --quantile 0.99 \
    --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
    --sms 108 --memory-bandwidth-gbps 2039 --seed 42 \
    --assignments contiguous grid_strided lpt --vertices-per-block 1 4 \
    --heavy-slice-sizes 0 1024 --launch-modes single sequential concurrent --out results/sim/arxiv

# 4. Section 4.3 and the figure: bucketing alone against slicing + concurrent launch.
for g in tolokers-2 ogbn-arxiv; do
  $OURS_PY scripts/ablation/plot_heatmap_pair.py --dataset $g --cost-model cost_models.json \
      --conv gat_v2 --pass forward --head-dim 128 --quantile 0.99 \
      --sms 108 --memory-bandwidth-gbps 2039 --out results/tail-$g-gat_v2-forward-d128.pdf
done
```

Step 4 prints each panel's slot occupancy and makespan, which are the utilization and time
figures of Section 4.3; its `tolokers-2` PDF is the figure. Each step 3 run writes
`summary.csv` with the makespan, drain tail and mean slot utilization of every configuration.

## Layout

```
csrc/                   CUDA kernels: gatv2/, gt/ (attention), reduction/ + spmm/ (g-SpMM),
                        gsddmm/ (g-SDDMM), common/ (tiles, pipelines, scheduling)
skewgnn/                Python package: graph representation, autograd functions, ops,
                        autotuner, load-imbalance simulation and cost-model calibration
scripts/                benchmark_kernels.py (the kernel benchmark harness) and
                        ablation/ (the utilization simulator)
benchmarks/paper/       runners for the paper's attention, g-SDDMM, PyG and counter experiments
benchmarks/gspmm_vs_dgl g-SpMM against DGL, and prepare_graph.py for the graph cache
src/                    backends (ours, DGL, PyG, ...) and dataset loading used by the harness
configs/                dataset, model and benchmark configurations
tests/                  correctness, unit and performance tests
```

## License

See `LICENSE`.
