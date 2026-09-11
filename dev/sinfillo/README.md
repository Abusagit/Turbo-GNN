# Recalibrating the load-imbalance simulator on another GPU

The simulator charges a node `alpha + beta * degree` ticks. Both constants and the tick's
duration come from timing the real kernels, so they belong to the card they were measured on:
`alpha` is a ratio of two per-kernel costs limited by different resources, and the tick is a
duration. **Measure and simulate on the same GPU**, or the run mixes two machines.

Everything below assumes the repo is built (`turbo_gnn._C` importable) and the datasets are in
`data/`. If they are not, see "Datasets" at the end.

## One command

```bash
bash dev/sinfillo/recalibrate.sh
```

Defaults are an **A100-SXM4-80GB**: 108 SMs, 2039 GB/s. For an H100 SXM:

```bash
SMS=132 BANDWIDTH=3350 bash dev/sinfillo/recalibrate.sh
```

It runs four stages in order, each of which can be run alone with `STAGE=`:

| stage | what | cost |
|---|---|---|
| `measure` | times the kernels, writes `reports/measurements.jsonl` | ~1 min |
| `calibrate` | fits the cost model, writes `cost_models.json` | seconds |
| `sweep` | simulates the full grid into `reports/sweep.csv` | hours |
| `figures` | one figure per graph/kernel/quantile | hours |

The sweep and the figures are resumable: rows already in the CSV and figures already on disk are
skipped, so both survive being interrupted.

## Stage by stage

### 1. Measure

```bash
.venv/bin/python3 scripts/ablation/measure_kernel_times.py \
    --graphs cora citeseer pubmed tolokers-2 ogbn-arxiv ... \
    --warmup 10 --iters 30 --out reports/measurements.jsonl
```

Times `gt`, `gat_v2` and `min_aggr`, forward and backward, at head dims 128 and 256, on every
graph given. One JSON record per (graph, conv, pass, head dim) with the graph's node and edge
counts and the milliseconds per launch.

A graph that will not load is skipped with its reason and the run continues.

**Do not add `--synthetic`.** Generated graphs measure 2-6 ns per edge against 0.36-1.55 for
real graphs of any size, because their edge order carries none of the locality a real graph gets
from how it was collected. Mixing them put 60%+ median error into cells that fit to 15% on real
graphs alone.

### 2. Calibrate

```bash
.venv/bin/python3 scripts/ablation/calibrate_cost_model.py \
    --from-json reports/measurements.jsonl --out cost_models.json
```

Regresses `t(N, E) = c + a*N + b*E` per (conv, pass, head dim) and prints a table: `alpha`,
`ns/tick`, launch overhead, R², median error, and what share of the time edges explain.

Read the "do not predict wall clock from these cells" list. A cell is flagged when its `alpha`
hits the non-negativity constraint, when edges explain under 20% of its time (min-aggregation's
backward is O(N), not O(E), so no `alpha + beta * degree` model describes it), or when the
median error exceeds 35%. Flagged cells are skipped by the later stages.

`cost_models.json` is **not** committed: it is only valid for the card it was measured on.

### 3. Sweep

```bash
.venv/bin/python3 scripts/ablation/sweep_all.py --cost-model cost_models.json \
    --sms 108 --memory-bandwidth-gbps 2039 --out reports/sweep.csv
```

One row per (graph, cell, quantile, launch mode, split-K): makespan, the three lower bounds and
which one binds, the imbalance ratio, mean occupancy, predicted milliseconds.

Cost is dominated by the graphs with the largest hub -- a run without slicing lasts about as
long as the biggest node, and `web-fraud`'s is 228,991 ticks.

### 4. Figures

```bash
.venv/bin/python3 scripts/ablation/plot_all.py --cost-model cost_models.json \
    --sms 108 --memory-bandwidth-gbps 2039 --out reports/launch-modes
```

Writes `reports/launch-modes/<conv>-<pass>-d<dim>/<graph>-q<NN>.png`, five panels each: the
three launch modes plus the split-K variants of the two bucketed ones.

For one graph, or for the denser single-figure form:

```bash
.venv/bin/python3 scripts/ablation/plot_launch_modes.py --dataset ogbn-arxiv \
    --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
    --sms 108 --memory-bandwidth-gbps 2039 --out figure.png

.venv/bin/python3 scripts/ablation/plot_occupancy_profile.py --dataset ogbn-arxiv \
    --cost-model cost_models.json --conv gt --pass forward --head-dim 128 \
    --sms 108 --memory-bandwidth-gbps 2039 --out profile.png
```

## Building on the target machine

`turbo_gnn._C` is a CUDA extension and has to be compiled where it runs. `setup.py` already
defaults `TORCH_CUDA_ARCH_LIST` to `8.0 8.6 8.9 9.0`, and an A100 is **sm_80**, so no
architecture flag is needed -- but building only the one you have is much faster:

```bash
TORCH_CUDA_ARCH_LIST=8.0 make install-dev        # A100
TORCH_CUDA_ARCH_LIST=9.0 make install-dev        # H100
```

`install-dev` creates `.venv`, installs torch and the `[dev]` extras (PyG, ogb, pandas, ruff,
pytest), builds the extension in place and runs the test suite.

Two things that bite:

* `PYTHON_BIN` defaults to `$HOME/micromamba/envs/graph_ml/bin/python`. If that interpreter is
  not on the machine, point it at one that is -- 3.10 or 3.11:

  ```bash
  PYTHON_BIN=$(which python3.11) TORCH_CUDA_ARCH_LIST=8.0 make install-dev
  ```

* `CUDA_HOME` defaults to `/usr/local/cuda` and the torch wheel index is derived from
  `nvcc --version`, so nvcc must be on the machine and match the driver. Check with
  `nvcc --version` and `nvidia-smi`.

Reference build here: python 3.11, torch 2.11.0+cu129, CUDA 12.9.

### Verify

```bash
.venv/bin/python3 -c "import torch, turbo_gnn; from turbo_gnn.ops import graph_transformer_aggr; \
    print(torch.cuda.get_device_name(0), torch.__version__)"
```

The built object is per interpreter version -- `turbo_gnn/_C.cpython-311-*.so` -- so a different
python needs its own build.

### After editing CUDA

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 \
    .venv/bin/pip3 install -e . --no-build-isolation
```

Recompiles only what changed; the full `install-dev` reinstalls the dependencies too.

## Datasets

Normally they download on first use. Where egress is filtered they do not, and the failure
arrives as an SSL error under a DGL fallback. Fetch them from a machine with network:

```bash
python3 scripts/download_datasets.py --datasets all --archive graphs.tar.gz
# then on the GPU machine
tar xzf graphs.tar.gz -C /path/to/Turbo-GNN/
```

Standard library only, no torch import, so it runs anywhere. `--list` shows the names and sizes.

## What is not parameterised

Worth knowing before reading any number off a figure:

* **Everything is fp32.** The measurement hardcodes `torch.float32` and the calibration key has
  no dtype, so bf16 and fp16 need their own fits.
* **One head.** Measurements use `--heads 1`, and the bandwidth cap's row size ignores head
  count and the fact that GT reads Q, K and V per neighbour rather than one row.
* **Blocks per SM are hand-set** (`--max-blocks-light 8`, `--max-blocks-heavy 4`,
  `--max-blocks-merge 16`) and do not depend on head dim, though shared memory does.
* **Occupancy is 1.0** in the sweep and the figures; the 0.5/0.75/1.0 sweep is only in
  `simulate_load_imbalance.py`.
