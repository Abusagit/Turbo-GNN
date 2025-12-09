# Graph Neural Network Benchmarking Framework

A comprehensive framework for benchmarking and accelerating Graph Neural Networks (GNNs) through efficient GPU utilization and backend optimization.

## Table of Contents

- [Graph Neural Network Benchmarking Framework](#graph-neural-network-benchmarking-framework)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Via Makefile (RECOMMENDED)](#via-makefile-recommended)
    - [Conda installation (Error-prone)](#conda-installation-error-prone)
  - [Quick Start](#quick-start)
    - [1. Train a GCN on Cora](#1-train-a-gcn-on-cora)
    - [2. Benchmark GCN Layer Across Backends](#2-benchmark-gcn-layer-across-backends)
    - [3. Profile Training with Memory Tracking](#3-profile-training-with-memory-tracking)
    - [4. Validate Trained Model](#4-validate-trained-model)
  - [Core Concepts](#core-concepts)
    - [Backend System](#backend-system)
    - [Graph Representations](#graph-representations)
    - [Model Architecture](#model-architecture)
    - [Training Pipeline](#training-pipeline)
  - [Configuration System](#configuration-system)
    - [Dataset Configuration](#dataset-configuration)
    - [Model Configuration](#model-configuration)
    - [Training Configuration](#training-configuration)
    - [Profiling Configuration](#profiling-configuration)
  - [Usage Guide](#usage-guide)
    - [Training Models](#training-models)
    - [Benchmarking](#benchmarking)
    - [Profiling](#profiling)
    - [Validation](#validation)
    - [Autotuning](#autotuning)
  - [Extending the Framework](#extending-the-framework)
    - [Adding Custom Backends](#adding-custom-backends)
    - [Adding Custom Models](#adding-custom-models)
    - [Adding Custom Datasets](#adding-custom-datasets)
    - [Adding Custom Hooks](#adding-custom-hooks)
  - [Testing](#testing)
    - [Correctness Tests](#correctness-tests)
    - [Integration Tests](#integration-tests)
    - [Writing Tests](#writing-tests)
  - [Contributing](#contributing)
    - [Adding Features](#adding-features)
    - [Code Style](#code-style)
    - [Reporting Issues](#reporting-issues)

---

## Overview

This framework addresses the critical challenge of accelerating Graph Neural Networks by optimizing graph convolution operations on modern GPUs. Graph convolutions primarily rely on Sparse Matrix-Dense Matrix Multiplication (SpMM), which suffers from poor GPU scalability in existing frameworks like Deep Graph Library (DGL).

### Key Features

- **Multi-Backend Support**: PyTorch Geometric, DGL, native PyTorch, and custom CUDA/cuSPARSE/cuBLAS/Triton backends
- **Unified Training Pipeline**: Backend-agnostic training with automatic mixed precision (AMP)
- **Comprehensive Benchmarking**: Microbenchmarking, profiling, and memory tracking
- **Flexible Architecture**: YAML-based configuration for models, datasets, and training
- **Multiple Graph Representations**: Seamless support for edge lists, adjacency matrices, CSR/COO formats, DGL graphs, and PyG Data objects
- **Hook System**: Extensible training pipeline with hooks for profiling, checkpointing, metrics and more
- **Autotuning**: Grid-search optimization for custom backend parameters

---

## Project Structure

```
.
├── configs/                    # YAML configurations
│   ├── benchmarks/            # Profiling and microbenchmark configs
│   ├── datasets/              # Dataset configurations (PyG, DGL, OGB)
│   ├── models/                # Model architecture definitions
│   └── training/              # Training hyperparameters
├── scripts/                   # Executable scripts
│   ├── train.py              # Main training script
│   ├── validate.py           # Model validation
│   ├── benchmark.py          # Layer microbenchmarking
│   ├── run_profile.py        # Profiling launcher
│   └── autotune.py           # Backend parameter autotuning
├── src/                      # Source code
│   ├── backends/             # Backend implementations
│   │   ├── base.py          # Abstract base classes
│   │   ├── registry.py      # Backend registration system
│   │   ├── pyg_backend/     # PyTorch Geometric backend
│   │   ├── dgl_backend/     # Deep Graph Library backend
│   │   ├── torch_native_backend/  # Native PyTorch backend
│   │   ├── cuda_backend/    # Custom CUDA kernels (placeholder)
│   │   ├── cusparse_backend/  # cuSPARSE backend (placeholder)
│   │   ├── cublas_backend/  # cuBLAS backend (placeholder)
│   │   └── triton_backend/  # Triton backend (placeholder)
│   ├── benchmarking/        # Benchmarking utilities
│   │   ├── microbench.py   # Layer timing utilities
│   │   ├── memory.py       # Memory profiling
│   │   ├── profiler.py     # PyTorch profiler wrappers
│   │   └── autotuner.py    # Grid-search autotuning
│   ├── data/               # Data loading and conversion
│   │   ├── datasets.py     # Dataset loaders (OGB, PyG, DGL)
│   │   ├── converters.py   # Graph format converters
│   │   └── loaders.py      # DataLoader builders
│   ├── models/             # Model components
│   │   ├── architecture/   # High-level architectures
│   │   ├── layers/         # Layer blocks (GCN, GATv2, SAGE, GIN)
│   │   ├── base.py        # Model specifications
│   │   ├── config.py      # YAML-based model building
│   │   └── registry.py    # Model registration
│   ├── training/          # Training pipeline
│   │   ├── trainer.py     # Main trainer class
│   │   ├── hooks.py       # Training hooks
│   │   ├── metrics.py     # Evaluation metrics
│   │   ├── optimizer.py   # Optimizer factory
│   │   └── scheduler.py   # LR scheduler factory
│   └── utils/             # Utilities
│       ├── logger.py      # Logging configuration
│       └── checkpointing.py  # Checkpoint management
└── tests/                 # Test suite
    ├── correctness/       # Correctness tests
    ├── integration/       # Integration tests
    └── unit/             # Unit tests
```

---

## Installation

### Via Makefile (RECOMMENDED)

You need python>=3.11 to install everything. Run these commands - they install everything you need to `.venv` local folder & install pre-commit hooks:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip && make install-full # full installation
```

For other types of installation, see `Makefile` & `pyproject.toml`

### Conda installation (Error-prone)

**NOTE!!!** - this version doesn't support `make tests` options and pre-commits via VS-Code (@mightyneighbor is working on it)

```bash
# Create environment
conda env create -f environment.yml
conda activate gnn_bench


### TODO
# Optional: Install custom backends
python setup.py develop
```

---

## Quick Start

### 1. Train a GCN on Cora

```bash
python scripts/train.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn_dgl.yaml \
    --config configs/training/base.yaml \
    --config configs/comet/disabled.yaml \
    --out runs/gcn_cora
```

For experiments use `configs/comet/exp_run.yaml` (probably with another `project_name`) to enable comet logging.

### 2. Benchmark GCN Layer Across Backends

```bash
# PyTorch Geometric
python scripts/benchmark.py --layer gcn --backend pyg --num-nodes 20000 --in-ch 128 --out-ch 128

# DGL
python scripts/benchmark.py --layer gcn --backend dgl --num-nodes 20000 --in-ch 128 --out-ch 128

# Native PyTorch
python scripts/benchmark.py --layer gcn --backend torch_native_gcn --num-nodes 20000 --in-ch 128 --out-ch 128
```

### 3. Profile Training with Memory Tracking

```bash
python scripts/run_profile.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --training configs/training/base.yaml \
    --profile configs/benchmarks/profile.yaml \
    --out runs/profile
```

### 4. Validate Trained Model

```bash
python scripts/validate.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --checkpoint runs/gcn_cora/ckpts/best_model.pth
```

### 5. Kernel tune

```bash
python scripts/kernel_tune.py \
    --conv_type mean_aggr \
    --backend cusparse \
    --dataset configs/datasets/pyg_cora.yaml \
    --optuna-config configs/optuna/example_cusparse.yaml
```

---

## Core Concepts

### Backend System

The framework uses a **registry pattern** to manage different backend implementations. Each backend provides convolution layers (GCN, GATv2, GraphSAGE, GIN) with consistent interfaces.

**Available Backends:**
- `pyg`: PyTorch Geometric
- `dgl`: Deep Graph Library
- `torch_native_gcn`: Native PyTorch (normalized adjacency)
- `cuda`: Custom CUDA kernels (placeholder)
- `cusparse`: cuSPARSE-based (placeholder)
- `cublas`: cuBLAS-based (placeholder)
- `triton`: Triton kernels (placeholder)

**Backend Interface:**
```python
from src.backends.base import BaseBackend, BaseConvolution

class MyBackend(BaseBackend):
    def create_conv(self, conv_type: str, in_channels: int,
                    out_channels: int, **kwargs):
        # Return convolution instance
        pass
```

### Graph Representations

Different backends expect different graph formats. The framework handles conversions automatically:

| Backend | Graph Format | Description |
|---------|-------------|-------------|
| `pyg` | `(edge_index, edge_weight)` | Tuple of COO edge list and optional weights |
| `dgl` | `DGLGraph` | DGL graph object with edge data |
| `torch_native_gcn` | `sparse_coo_tensor` | Normalized adjacency matrix |
| `edge_list` | `(edge_list, edge_weight)` | Edge list format |
| `csr`/`coo`/`csc` | Sparse matrix formats | (Future support) |

The `GraphSample` class in `src/data/datasets.py` automatically converts graphs to the appropriate format based on the backend.

### Model Architecture

Models are defined via YAML configurations with a flexible encoder-decoder structure:

**Key Components:**
- **LayerSpec**: Configuration for a single GNN layer (conv type, backend, dimensions, activation, normalization, dropout, residual)
- **EncoderSpec**: Stack of LayerSpec definitions
- **ClassifierSpec**: Encoder + classification head

**Example Model (configs/models/gcn.yaml):**
```yaml
architecture: node_classifier
num_classes: 7
dropout: 0.5

encoder:
  layers:
    - conv_type: gcn
      backend: pyg
      in_channels: auto  # Inferred from dataset
      out_channels: 128
      norm: batch
      activation: relu
      dropout: 0.5
      residual: false
      conv_kwargs:
        cached: true

    - conv_type: gcn
      backend: pyg
      in_channels: auto  # Inferred from previous layer
      out_channels: 128
      norm: batch
      activation: relu
      dropout: 0.5
      residual: true
```

### Training Pipeline

The `GNNTrainer` class provides a complete training loop with:

- **Automatic Mixed Precision (AMP)**: bf16/fp16 support
- **Gradient Accumulation**: For large batch simulation
- **Hook System**: Extensible event-driven architecture
- **Early Stopping**: Based on validation metrics
- **Checkpointing**: Automatic model saving
- **Memory Profiling**: CUDA memory tracking

**Training Flow:**
```
on_training_start
  ↓
for epoch in epochs:
    on_epoch_start
      ↓
    for batch in train_loader:
        on_batch_start
          ↓
        forward pass
          ↓
        on_forward_end
          ↓
        backward pass
          ↓
        on_batch_end
      ↓
    validation
      ↓
    on_epoch_end (includes scheduler step, metric logging)
      ↓
    on_best_model (if new best)
  ↓
on_training_end
```

---

## Configuration System

### Dataset Configuration

**Example (configs/datasets/ogbn_arxiv.yaml):**
```yaml
dataset:
  source: ogbn       # 'ogbn', 'pyg', 'dgl', or 'auto'
  name: ogbn-arxiv
  root: data
```

**Supported Datasets:**
- **OGB**: `ogbn-arxiv`, `ogbn-products`, etc.
- **PyG**: `Cora`, `CiteSeer`, `PubMed`, `Reddit`
- **DGL**: `cora`, `citeseer`, `pubmed`, `reddit`

### Model Configuration

Models support **auto-inference** of `in_channels` from the dataset and previous layers.

**Key Fields:**
- `conv_type`: `gcn`, `gat_v2`, `sage`, `gin`
- `backend`: Backend name (must be registered)
- `in_channels`: Input features (use `auto` for inference)
- `out_channels`: Output features
- `heads`: Number of attention heads (GAT only)
- `norm`: `batch`, `layer`, or `none`
- `activation`: `relu`, `gelu`, `prelu`, `elu`, `tanh`, `sigmoid`, `none`
- `dropout`: Dropout probability
- `residual`: Add skip connection (requires matching dimensions)
- `conv_kwargs`: Backend-specific parameters

### Training Configuration

**Example (configs/training/base.yaml):**
```yaml
training:
  epochs: 200
  learning_rate: 0.01
  weight_decay: 0.0005
  batch_size: 1              # Full-batch for single-graph datasets
  accumulation_steps: 1
  use_amp: false
  clip_grad_norm: null
  patience: 100
  device: cuda
  num_workers: 0
  pin_memory: false
  log_interval: 10
  checkpoint_interval: 10
  profile: false

optimizer:
  name: adamw               # 'adamw', 'adam', 'sgd', 'rmsprop', 'adagrad'
  lr: 0.01
  weight_decay: 0.0005
  betas: [0.9, 0.999]
  no_decay_norm_bias: true  # Exclude bias/norm from weight decay

scheduler:
  name: cosine_warmup       # 'none', 'step', 'cosine', 'cosine_warmup', etc.
  warmup_epochs: 5
  T_max: 200
  eta_min: 0.0
```

### Profiling Configuration

**Example (configs/benchmarks/profile.yaml):**
```yaml
profiler:
  output_dir: runs/profiler
  wait: 1
  warmup: 1
  active: 3
  repeat: 1
  with_stack: true
  profile_memory: true
```

---

## Usage Guide

### Training Models

**Basic Training:**
```bash
python scripts/train.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --config configs/training/base.yaml \
    --out runs/experiment
```


**Training Output:**
```
runs/experiment/
├── ckpts/
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── logs/
│   └── metrics.json
│
├── profiler/
│    # traces of a profiler if profiling is enabled
│
└── history.json
```

### Benchmarking

**Layer Microbenchmark:**
```bash
python scripts/benchmark.py \
    --layer gcn \
    --backend pyg \
    --num-nodes 100000 \
    --avg-degree 20 \
    --in-ch 512 \
    --out-ch 512 \
    --mode train \
    --iters 100 \
    --warmup 20 \
    --amp bf16 \
    --json-out results.json
```

**Output:**
```json
{
  "iters": 100,
  "ms_per_iter": XY.ZW,
  "device": "cuda"
}
```

**Batch Benchmark Script:**
```bash
for backend in pyg dgl torch_native_gcn; do
    python scripts/benchmark.py \
        --layer gcn --backend $backend \
        --num-nodes 50000 --in-ch 256 --out-ch 256 \
        --json-out results_${backend}.json
done
```

### Profiling

**Profile Training:**
```bash
python scripts/run_profile.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --training configs/training/base.yaml \
    --profile configs/benchmarks/profile.yaml \
    --out runs/profile
```

**View Results:**
```bash
tensorboard --logdir runs/profile/profiler
```

Or open it in [Perfetto UI](https://ui.perfetto.dev)

**Available Hooks:**
- `ProfilerHook`: PyTorch profiler integration
- `MetricHook`: Track and log metrics
- `CheckpointHook`: Save model checkpoints
- `MemoryHook`: CUDA memory profiling
- `LRSchedulerStepHook`: Learning rate scheduling

### Validation

**Validate Checkpoint:**
```bash
python scripts/validate.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --checkpoint runs/experiment/ckpts/best_model.pth
```

### Autotuning

**Autotune Backend Parameters:**

If you have custom backend named `my_custom_backend`, you can launch its autotuning:

```bash
python scripts/autotune.py \
    --layer gcn \
    --backend my_custom_backend \
    --param-space params.yaml \
    --num-nodes 50000 \
    --in-ch 256 \
    --out-ch 256 \
    --json-out best_params.json
```

**Parameter Space (params.yaml):**
```yaml
tile_size: [64, 128, 256]
unroll_factor: [1, 2, 4]
use_shared_mem: [true, false]
```

---

## Extending the Framework

### Adding Custom Backends

**Step 1: Implement Backend Class**

Create `src/backends/my_backend/conv.py`:

```python
from typing import Any, Optional
import torch
import torch.nn as nn
from ..base import BaseBackend, BaseConvolution
from ..registry import BackendRegistry

class _MyGCNConv(BaseConvolution):
    """Custom GCN implementation."""

    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True, **kwargs):
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs) -> torch.Tensor:
        # Your custom forward implementation
        # graph is in the format specified by MODEL_BACKEND_TO_GRAPH_REPR
        out = torch.matmul(x, self.weight)
        # ... apply graph convolution
        if self.bias is not None:
            out = out + self.bias
        return out

@BackendRegistry.register_backend("my_backend")
class MyBackend(BaseBackend):
    """Custom backend."""

    def create_conv(self, conv_type: str, in_channels: int,
                    out_channels: int, **kwargs):
        if conv_type.lower() == "gcn":
            return _MyGCNConv(in_channels, out_channels, **kwargs)
        raise KeyError(f"Unsupported conv_type: {conv_type}")
```

**Step 2: Register Graph Representation**

Update `src/data/datasets.py`:

```python
MODEL_BACKEND_TO_GRAPH_REPR: Mapping[str, GraphBackendOption] = {
    "pyg": "pyg",
    "dgl": "dgl",
    "torch_native_gcn": "normalized_adj_mat_gcn",
    "my_backend": "csr",  # Your preferred format
}
```

**Step 3: Handle Graph Conversion**

Update `GraphSample.__post_init__()` in `src/data/datasets.py`:

```python
elif self.backend == "my_custom_format":
    # Convert edge_index to your format
    graph = my_conversion_function(self.edge_index, self.num_nodes)
    graph = self._to_default_device(graph)
```

**Step 4: Import Backend**

Add to `src/backends/__init__.py`:

```python
from . import my_backend
```

**Step 5: Use Your Backend**

Create model config:
```yaml
encoder:
  layers:
    - conv_type: gcn
      backend: my_backend  # Your backend name
      in_channels: 128
      out_channels: 64
```

### Adding Custom Models

**Step 1: Define Architecture**

Create `src/models/architecture/my_model.py`:

```python
import torch.nn as nn
from ..base import ClassifierSpec
from ..registry import register

class MyCustomModel(nn.Module):
    def __init__(self, spec: ClassifierSpec):
        super().__init__()
        # Build your architecture
        self.encoder = ...
        self.head = ...

    def forward(self, batch_or_x, graph=None):
        # Your forward pass
        pass

@register("my_model")
def build_my_model(spec: ClassifierSpec) -> nn.Module:
    return MyCustomModel(spec)
```

**Step 2: Import Architecture**

Add to `src/models/architecture/__init__.py`:

```python
from . import my_model
```

**Step 3: Create Config**

Create `configs/models/my_model.yaml`:

```yaml
architecture: my_model  # Registered name
num_classes: 7
# Your custom parameters
```

### Adding Custom Datasets

**Step 1: Implement Loader**

Add to `src/data/datasets.py`:

```python
def load_my_dataset(name: str, graph_backend: GraphBackendOption,
                    root: str = "data") -> GraphSample:
    # Load your dataset
    x = ...  # Node features
    y = ...  # Labels
    edge_index = ...  # Graph edges

    return GraphSample(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        backend=graph_backend,
    )
```

**Step 2: Register Source**

Update `load_single_graph()` in `src/data/datasets.py`:

```python
def load_single_graph(cfg: DatasetConfig) -> GraphSample:
    s = cfg.source.lower()
    if s == "my_source":
        return load_my_dataset(cfg.name, cfg.root, cfg.graph_backend)
    # ... existing sources
```

**Step 3: Create Config**

Create `configs/datasets/my_dataset.yaml`:

```yaml
dataset:
  source: my_source
  name: my_dataset_name
  root: data
```

### Adding Custom Hooks

**Step 1: Implement Hook**

Create hook class:

```python
from src.training.hooks import Hook

class MyCustomHook(Hook):
    def on_training_start(self, model, config):
        # Initialization logic
        pass

    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        # Custom logging or processing
        pass

    # Override other hook methods as needed
```

**Step 2: Register Hook**

Add to trainer in `scripts/train.py`:

```python
from my_hooks import MyCustomHook

trainer = build_trainer(model, merged_cfg)
trainer.add_hook(MyCustomHook(
    # Your parameters
))
```

---

## Testing

### Correctness Tests

**Run All Tests:**
```bash
python tests/correctness/test_verify_backends.py
```

**Specific Tests:**
```bash
# Test backend registration
pytest tests/correctness/test_verify_backends.py::test_backend_registration

# Test dataset loading
pytest tests/correctness/test_verify_backends.py::test_dataset_loading

# Test convolution layers
pytest tests/correctness/test_verify_backends.py::test_backend_convolutions
```

### Integration Tests

```bash
bash tests/integration/launch_training_pipeline.sh
```

### Writing Tests

**Backend Correctness Test Template:**
```python
def test_my_backend():
    backend = BackendRegistry.get_backend("my_backend")
    conv = backend.create_conv("gcn", in_channels=16, out_channels=32)

    # Forward pass
    x = torch.randn(100, 16)
    edge_index = torch.randint(0, 100, (2, 500))
    graph = prepare_graph(edge_index, backend="my_backend")
    out = conv(x, graph)

    # Check output shape
    assert out.shape == (100, 32)
    assert not torch.isnan(out).any()

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check gradients
    assert conv.weight.grad is not None
    assert not torch.isnan(conv.weight.grad).any()
```
---

## Contributing

### Adding Features

1. **Fork and Branch**: Create a feature branch from `main`
2. **Implement**: Add your backend/model/dataset following the patterns above
3. **Test**: Add correctness tests and integration tests
4. **Document**: Update this README and add docstrings
5. **Submit PR**: Include benchmark results if applicable

### Code Style

- **Type Hints**: All functions should have type annotations
- **Docstrings**: Use Google-style docstrings
- **Naming**: Follow PEP 8 conventions
- **Imports**: Group by standard library, third-party, local

### Reporting Issues

Include:
- Python/PyTorch/CUDA versions
- GPU model
- Dataset and model configuration
- Error traceback
- Minimal reproducible example
