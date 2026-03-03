# GNN Benchmarking & Acceleration Framework

A framework for benchmarking and accelerating Graph Neural Network convolutions on GPUs.
Includes custom CUDA and Triton kernels for SpMM/attention, wrappers for PyG, DGL, cuGraph,
TCGNN, DFGNN, and FuseGNN, plus Optuna-based kernel autotuning. Models, datasets, and training
are all driven by YAML configs.

## Installation

Requires Python >= 3.10 and CUDA 12.x.

```bash
# Recommended: venv + full install (includes PyG, DGL, cuGraph, Triton, dev tools, pre-commit hooks)
python3.11 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip
make install-full
```

Custom CUDA backends (cuda, fusegnn, dfgnn) compile on first import via `torch.utils.cpp_extension`.

For other install targets (base only, dev only) see the `Makefile`. A conda alternative is available
via `environment.yml`, though it does not support `make test` or pre-commit hooks.

## Quick Start

```bash
# Train a GCN on Cora
python scripts/train.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn_dgl.yaml \
    --config configs/training/base.yaml \
    --config configs/comet/disabled.yaml \
    --conv_type gcn --backend pyg \
    --out runs/gcn_cora

# Benchmark a single conv layer
python scripts/benchmark.py --layer gcn --backend pyg --num-nodes 20000 --feature_dim 128

# Validate a trained checkpoint
python scripts/validate.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --checkpoint runs/gcn_cora/ckpts/best_model.pth \
    --conv_type gcn --backend pyg

# Profile training (outputs Perfetto/TensorBoard traces)
python scripts/run_profile.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --training configs/training/base.yaml \
    --profile configs/benchmarks/profile.yaml \
    --conv_type gcn --backend pyg \
    --out runs/profile

# Autotune kernel parameters with Optuna
python scripts/kernel_tune.py \
    --conv_type mean_aggr \
    --backend cusparse \
    --dataset configs/datasets/pyg_cora.yaml \
    --optuna-config configs/optuna/example_cusparse.yaml
```

## Scripts

### `train.py` — Train a GNN model

Merges one or more training YAMLs, builds dataset/model, attaches hooks (metrics, checkpoints, memory, optional profiler), and trains. Outputs checkpoints, logs, and history JSON.

```
--dataset           Dataset YAML path (required)
--model             Model YAML path (required)
--config            Training YAML(s), repeatable, later overrides earlier (required)
--conv_type         Convolution type, e.g. gcn, mean_aggr, gat_v2 (required)
--backend           Backend name, e.g. pyg, dgl, cuda (required)
--out               Output directory (default: runs/train)
--profile           Optional profiler YAML (configs/benchmarks/profile.yaml)
--record-snapshots  Flag to record CUDA memory snapshots
```

### `benchmark.py` — Microbenchmark a single conv layer

Creates a random graph (or loads one from a dataset YAML), instantiates a conv, and times forward or forward+backward using CUDA events.

```
--layer         Conv type: gcn, mean_aggr, gat_v2, gt, ... (required)
--backend       Backend name (required)
--dataset       Dataset YAML path (optional; if omitted, generates a random graph)
--num-nodes     Nodes in random graph (default: 20000)
--avg-degree    Average degree (default: 10)
--feature_dim   Feature dimension (default: 128)
--heads         Attention heads for gat_v2/gt (default: 1)
--mode          forward | train (default: forward)
--iters         Timing iterations (default: 100)
--warmup        Warmup iterations (default: 20)
--amp           none | bf16 | fp16 (default: none)
--json-out      Optional path to write JSON result
--device        CUDA device index (default: 0)
```

### `validate.py` — Validate a trained checkpoint

Loads dataset and model from YAMLs, restores weights from a `.pth` checkpoint, evaluates on validation and test splits.

```
--dataset       Dataset YAML path (required)
--model         Model YAML path (required)
--checkpoint    Path to .pth checkpoint (required)
--conv_type     Convolution type (required)
--backend       Backend name (required)
--batch-size    Loader batch size (default: 1)
--num-workers   DataLoader workers (default: 0)
--pin-memory    Enable pinned memory (flag)
```

### `run_profile.py` — Profile training

Runs a short training loop with `torch.profiler` attached. Outputs traces viewable in TensorBoard or [Perfetto UI](https://ui.perfetto.dev).

```
--dataset       Dataset YAML path (required)
--model         Model YAML path (required)
--training      Training YAML path (required)
--profile       Profiler YAML path (required)
--conv_type     Convolution type (required)
--backend       Backend name (required)
--out           Output directory (default: runs/profile)
```

### `kernel_tune.py` — Optuna-based kernel tuning

Loads a real graph dataset and uses Optuna (TPE sampler) to search over backend-specific kernel hyperparameters, minimizing forward-pass latency.

```
--conv_type       Convolution type (required)
--backend         Backend name (required)
--dataset         Dataset YAML path (required)
--optuna-config   YAML defining the parameter search space (required)
--in-ch           Feature dimension (default: 128)
--n-trials        Number of Optuna trials (default: 100)
--amp             none | bf16 | fp16 (default: none)
--json-out        Optional path to write best config JSON
```

The search space YAML (`--optuna-config`) defines parameters with Optuna suggest types. See `configs/optuna/example_cusparse.yaml` for the format.

### `autotune.py` — Grid-search autotuning

Exhaustive grid search over a parameter space for a backend conv on a random graph. Simpler than `kernel_tune.py` but does not use Optuna or real datasets.

```
--layer         Conv type: gcn, gat_v2, sage, gin, mean_aggr (required)
--backend       Backend name (required)
--param-space   YAML dict of parameter lists, e.g. {tile: [64,128]} (required)
--num-nodes     Nodes in random graph (default: 20000)
--avg-degree    Average degree (default: 10)
--in-ch         Input channels (default: 128)
--out-ch        Output channels (default: 128)
--heads         Attention heads (default: 1)
--iters         Timing iterations (default: 100)
--warmup        Warmup iterations (default: 20)
--json-out      Optional path to write JSON result
```

### `kernel_launch_comparison.py` — Multi-backend batch comparison

Runs a sweep of microbenchmarks across multiple backends, multiple datasets, and a grid of conv parameters (feature dims, heads, etc.). Measures both forward and backward pass. Outputs a pivot table comparing backends and optionally logs each measurement to Comet ML.

```
--conv_type                       Convolution type (required)
--backends                        Backend names, space-separated (required)
--target_backend                  Reference backend for comparison (required, default: cuda)
--conv_params_grid                YAML config defining parameter grid + datasets (required)
--device                          CUDA device index (default: 0)
--amp                             none | bf16 | fp16 (default: none)
--out                             Optional CSV output path
--use_comet                       Enable Comet ML logging (flag)
--comet_project_name              Comet project name (default: kernel_results)
--comet_workspace                 Comet workspace (default: accelerating-gnns-2)
--comet_experiment_name_prefix    Prefix for Comet experiment names
```

The `--conv_params_grid` YAML has three sections:

```yaml
# Example: configs/kernels_measurements/gcn.yaml
params_grid:                              # conv parameter grid
  all:                                    # shared across all backends
    feature_dim: [64, 128, 256, 512]
  cusparse:                               # backend-specific overrides (optional)
    feature_dim: [128]

kernel_related_kwargs:                    # graph-repr hyperparams (e.g. reordering)
  all:
    graph_reordering_partition_size: [-1]

datasets:                                 # which dataset configs to load
  base_path: configs/datasets
  dirs:
    main:
      all: true                           # load all .yaml files in the directory
    secondary:
      all: false
      choices: [cora, citeseer, pubmed]   # or pick specific ones
```

Requires: `pandas`, `comet_ml` (if `--use_comet`), and all backends listed in `--backends` to be importable. Existing config files live under `configs/kernels_measurements/`.

## Backends

| Backend | Type | Registered names | Supported conv types |
|---------|------|------------------|----------------------|
| PyG | Library wrapper | `pyg` | gcn, mean_aggr, sum_aggr, gat, gat_v2, gin, sage |
| DGL | Library wrapper | `dgl` | gcn, mean_aggr, sum_aggr, min_aggr, max_aggr, gat, gat_v2, gt |
| cuGraph | Library wrapper | `cugraph` | gcn, mean_aggr, sum_aggr, min_aggr, max_aggr, gat_v2, gt |
| cuSPARSE | Library wrapper | `cusparse`, `cusparse_precomputed_bwd` | gcn, sum_aggr, mean_aggr, random_walk |
| TCGNN | Library wrapper | `tcgnn` | gcn, agnn |
| CUDA | Custom CUDA | `cuda` | gcn, sum_aggr, mean_aggr, min_aggr, max_aggr, gat_v2, gt |
| CUDA Test | Custom CUDA | `cuda_test` | mean_aggr, dot_aggr |
| FuseGNN | Custom CUDA | `fusegnn` | gcn, gat |
| DFGNN | Custom CUDA | `dfgnn` | gt |
| Triton | Triton kernels | `triton_block_sparse` | gcn, mean_aggr, sum_aggr, gt |
| Torch Native | Pure PyTorch | `torch_native_gcn`, `torch_native_mean_aggr`, `torch_native_sum_aggr`, `torch_native_adj_mat` | gcn, mean_aggr, sum_aggr, min_aggr, max_aggr |

## Configuration

All configs live under `configs/` in four categories:

| Category | Path | Purpose |
|----------|------|---------|
| Datasets | `configs/datasets/` | Data source, name, root path (OGB, PyG, DGL) |
| Models | `configs/models/` | Architecture, layers, backends, conv kwargs |
| Training | `configs/training/` | Epochs, optimizer, scheduler, AMP, early stopping |
| Benchmarks | `configs/benchmarks/` | Profiler settings (wait, warmup, active, memory) |

Additional config dirs: `configs/optuna/` (kernel tuning search spaces), `configs/comet/` (experiment tracking).

See the YAML files in each directory for the full set of available options.

## Project Structure

```
.
├── configs/              # YAML configurations (datasets, models, training, benchmarks, optuna)
├── scripts/              # Entry-point scripts (train, validate, benchmark, profile, autotune)
├── src/
│   ├── backends/         # Backend implementations (one subdir per backend)
│   ├── benchmarking/     # Microbench, memory profiling, autotuner
│   ├── data/             # Dataset loading, graph format converters, data loaders
│   ├── models/           # Model specs, layer blocks (GCN, GATv2, SAGE, GIN), registry
│   ├── training/         # Trainer, hooks, metrics, optimizer/scheduler factories
│   └── utils/            # Logging, checkpointing
├── tests/
│   ├── correctness/      # Backend correctness & numerical checks
│   ├── unit/             # Unit tests
│   ├── integration/      # End-to-end pipeline tests
│   └── performance/      # Performance regression tests
├── Makefile              # Install, test, lint, format targets
└── pyproject.toml        # Package metadata & tool config
```

## Testing

```bash
make test                  # Run all tests (pytest)
pytest tests/              # Equivalent
pytest tests/correctness/  # Backend correctness only
pytest tests/unit/         # Unit tests only

bash tests/integration/launch_training_pipeline.sh  # Integration smoke test
```

## CLI Entry Points

After `pip install -e .`, the following console scripts are available:

| Command | Script | Description |
|---------|--------|-------------|
| `gnn-train` | `scripts/train.py` | Train a model |
| `gnn-validate` | `scripts/validate.py` | Validate a checkpoint |
| `gnn-benchmark` | `scripts/benchmark.py` | Microbenchmark a conv layer |
| `gnn-profile` | `scripts/run_profile.py` | Profile training |
| `gnn-autotune` | `scripts/autotune.py` | Optuna-based kernel autotuning |
