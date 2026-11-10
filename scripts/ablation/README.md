# Load-imbalance simulation

The simulator assigns each node `alpha + beta * degree` ticks and schedules its
blocks onto SM slots. GPU measurements calibrate the cost model; the simulation
runs on CPU. Predicted times describe scheduling effects, not measured speedups
of new CUDA kernels. Recalibrate after changing the GPU or kernel implementation.

## Run on A100

Use the [project environment](../../README.md#installation), with CUDA toolkit
and datasets available. Run from the repository root. Rebuild the extension after
switching branches so measurements use the current kernels:

```bash
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 TURBO_GNN_FORCE_BUILD=TRUE \
  .venv/bin/python -m pip install -e . --no-build-isolation

# Measure kernels and fit cost_models.json.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  bash scripts/ablation/calibrate_device.sh a100 reports/simulation-a100

# Compare bucket counts and scheduling policies on CPU.
OMP_NUM_THREADS=4 .venv/bin/python scripts/ablation/sweep_degree_buckets.py \
  --hardware a100 --cost-model reports/simulation-a100/cost_models.json \
  --graphs tolokers-2 --conv gat_v2 --head-dim 128 \
  --buckets 2 4 8 --orderings natural --plot \
  --out reports/simulation-a100/sweep
```

Calibration uses eight real graphs, `gt/gat_v2/min_aggr` forward, one head,
head dimensions 128/256, fp32 and heavy quantile 0.99. Append
`--convs gat_v2 --head-dims 128` to the calibration command for a shorter run.
Set `PY` to use a different Python executable. Check the logs for skipped datasets
and rejected fits. These settings differ from the paper's fp16 benchmarks.

For H100, rebuild with `TORCH_CUDA_ARCH_LIST=9.0`, then use:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  bash scripts/ablation/calibrate_device.sh h100 reports/simulation-h100

OMP_NUM_THREADS=4 .venv/bin/python scripts/ablation/sweep_degree_buckets.py \
  --hardware h100 --cost-model reports/simulation-h100/cost_models.json \
  --graphs tolokers-2 --orderings natural --plot \
  --out reports/simulation-h100/sweep
```

The profiles assume SXM 80GB cards: A100 uses 108 SMs / 2039 GB/s;
H100 uses 132 SMs / 3350 GB/s. Override `--sms` and `--memory-bandwidth-gbps`
for other variants. GPU, dtype, head count and quantile metadata are checked.

## Bucket experiments

| Strategy | Partition |
|---|---|
| `quantile` | Split light nodes by quantile; keep the heavy bucket intact. |
| `log` | Split light nodes into logarithmic degree ranges; keep heavy intact. |
| `tail` | Split both light and heavy nodes with finer ranges in the tail. |

Defaults: 2/4/8 buckets, all three strategies, sequential/concurrent launches,
and seeds 42/43/44. `--orderings natural` preserves node order within each bucket;
a two-bucket descending-order control is always included. Use
`--orderings natural descending` to compare both orders for every bucket count.

Buckets launch from highest to lowest degree. Each node retains its resource
class: 8 light or 4 heavy blocks per SM by default. Empty buckets are omitted,
equal degrees stay together, and `tail` requires an even bucket count.
Edge slicing is disabled in this experiment.

`--extra-launch-us 0 2 5` adds a sensitivity surcharge per extra nonempty bucket.
It is an assumed launch cost, not a measured overhead. Preprocessing, cache
locality and per-bucket kernel tuning are not modelled.

## Datasets and results

Use `--graphs paper` for the 16 graphs in main-text Tables 1–3:

```text
artnet-exp avazu-ctr citeseer city-reviews city-roads-L city-roads-M
cora hm-categories ogbn-arxiv ogbn-products ogbn-proteins pokec-regions
pubmed tolokers-2 twitch-views web-fraud
```

Start with `tolokers-2 ogbn-arxiv web-fraud city-roads-L pubmed`. Additional options
include `web-traffic`, `amazon-computers`, `coauthor-physics` and `reddit`.
Use `synth-N16384` for scheduling stress tests, not cost-model calibration.

Calibration writes `measurements.jsonl`, `cost_models.json` and logs. The bucket
sweep writes `results.csv`, `manifest.json` and optional plots. CSV rows include
requested/effective bucket counts, makespan, predicted time, and speedups against
the two-bucket natural/ordered controls for the same launch mode and seed.
A speedup greater than 1 is an improvement.

The fit is `t = c + a*N + b*E`, with `alpha=a/b` and `beta=1`. One shared tick
duration is anchored to the two-bucket sequential baseline; `c` is added once.
Absolute times depend on fit quality. The new sweep supports forward only:
the existing measurement script's backward mode times forward and backward together.

## Other experiments and checks

`sweep_all.py` explores launch modes and edge slicing. `simulate_load_imbalance.py`
exposes individual scheduling parameters. `plot_heatmap_grid.py` and
`plot_heatmap_pair.py` produce occupancy figures. Each script supports `--help`.

```bash
.venv/bin/python -m pytest tests/unit/test_degree_bucket_simulation.py -q
```
