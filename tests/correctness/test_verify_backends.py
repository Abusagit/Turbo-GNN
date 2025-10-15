#!/usr/bin/env python3
"""
Test script to verify all backends work correctly with different datasets.
Run from repository root: python test_verify_backends.py
"""

import sys
import traceback
from pathlib import Path

import pytest
import torch
import yaml

from src.backends.registry import BackendRegistry
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR, DatasetConfig, GraphSample, load_single_graph


def test_backend_registration():
    """Test that backends are properly registered."""
    print("\n" + "=" * 60)
    print("Testing Backend Registration")
    print("=" * 60)

    # Import backend modules to trigger registration
    try:
        import src.backends.dgl_backend  # noqa: F401
        import src.backends.pyg_backend  # noqa: F401
        import src.backends.torch_native_backend  # noqa: F401

        print("✓ Backend modules imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import backends: {e}")

    # Check registered backends
    backends = BackendRegistry.list_backends()
    print(f"Registered backends: {backends}")

    expected = {"pyg", "dgl", "torch_native_gcn"}
    missing = expected - set(backends)
    if missing:
        pytest.fail(f"Missing backends: {missing}")


def test_dataset_loading():
    """Test dataset loading from different sources."""
    print("\n" + "=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    test_configs = [
        DatasetConfig(source="pyg", name="cora", root="data", graph_backend="pyg"),
        DatasetConfig(source="dgl", name="cora", root="data", graph_backend="pyg"),
        DatasetConfig(source="ogbn", name="ogbn-arxiv", root="data", graph_backend="pyg"),  # Large dataset
    ]

    for cfg in test_configs:
        try:
            print(f"\nLoading {cfg.source}/{cfg.name}...")
            sample = load_single_graph(cfg)
            print(f"  ✓ Loaded: {sample.num_nodes} nodes, {sample.num_features} features")
            print(f"    Classes: {sample.num_classes}")
            print(f"    Edges: {sample.edge_index.shape}")
            print(f"    Train mask: {sample.train_mask.sum().item()} nodes")
        except Exception as e:
            pytest.fail(f"Failed: {e}")


def test_backend_convolutions():
    """Test that each backend can create and run convolutions."""
    print("\n" + "=" * 60)
    print("Testing Backend Convolutions")
    print("=" * 60)

    # Create small test graph
    num_nodes = 100
    num_edges = 500
    in_channels = out_channels = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    torch.set_default_device(device)
    # Generate random graph
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    x = torch.randn(num_nodes, in_channels, device=device, requires_grad=True)

    backends_to_test = ["pyg", "dgl", "torch_native_gcn"]
    conv_types = ["gcn"]

    results = {}

    for backend_name in backends_to_test:
        print(f"\n{backend_name.upper()} Backend:")
        try:
            backend = BackendRegistry.get_backend(backend_name)

            for conv_type in conv_types:
                try:
                    # Create convolution
                    conv = backend.create_conv(conv_type, in_channels, out_channels, bias=True).to(device)

                    graph = GraphSample(
                        backend=MODEL_BACKEND_TO_GRAPH_REPR[backend_name],
                        x=x,
                        y=torch.zeros_like(x),
                        edge_index=edge_index,
                    ).graph_repr

                    # Forward pass
                    out = conv(x, graph)

                    # Check output
                    assert out.shape == (num_nodes, out_channels), f"Wrong output shape: {out.shape}"
                    assert not torch.isnan(out).any(), "NaN in output"

                    # Backward pass
                    loss = out.sum()
                    loss.backward()

                    # Check gradients
                    for name, param in conv.named_parameters():
                        if param.requires_grad:
                            assert param.grad is not None, f"No gradient for {name}"
                            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"

                    print(f"  ✓ {conv_type.upper()}: forward/backward pass successful")
                    results[f"{backend_name}_{conv_type}"] = "PASSED"

                except KeyError:
                    print(f"  ⚠ {conv_type.upper()}: not implemented")
                    results[f"{backend_name}_{conv_type}"] = "NOT_IMPLEMENTED"
                except Exception as e:
                    print(f"  ✗ {conv_type.upper()}: {e}")
                    results[f"{backend_name}_{conv_type}"] = "FAILED"

        except Exception as e:
            print(f"  ✗ Backend initialization failed: {e}")
            for conv_type in conv_types:
                results[f"{backend_name}_{conv_type}"] = "BACKEND_FAILED"

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for key, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗" if "FAILED" in status else "⚠"

    ok = all(v in ("PASSED", "NOT_IMPLEMENTED") for v in results.values())
    if not ok:
        failed = {k: v for k, v in results.items() if v not in ("PASSED", "NOT_IMPLEMENTED")}
        pytest.fail(f"Backend convolution failures: {failed}")


def test_microbenchmarking():
    """Test microbenchmarking functionality."""
    print("\n" + "=" * 60)
    print("Testing Microbenchmarking")
    print("=" * 60)

    from src.benchmarking.microbench import time_callable

    def test_fn():
        x = torch.randn(1000, 1000)
        y = torch.matmul(x, x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        result = time_callable(test_fn, warmup=5, iters=10)
    except Exception as e:
        pytest.fail(f"Microbench failed: {e}")


def test_memory_profiling():
    """Test memory profiling utilities."""
    print("\n" + "=" * 60)
    print("Testing Memory Profiling")
    print("=" * 60)

    from src.benchmarking.memory import capture_cuda_snapshot, human_bytes, measure_peak_cuda_memory_during

    def memory_test():
        x = torch.randn(1000, 1000, device="cuda" if torch.cuda.is_available() else "cpu")
        y = x @ x.T
        return y

    try:
        # Test snapshot
        snapshot = capture_cuda_snapshot()
        print(
            f"Current memory - Allocated: {human_bytes(snapshot.allocated_bytes, binary=True)}, "
            f"Reserved: {human_bytes(snapshot.reserved_bytes, binary=True)}"
        )
        result = measure_peak_cuda_memory_during(memory_test)
    except Exception as e:
        pytest.fail(f"Memory profiling failed: {e}")


def test_model_building():
    """Test YAML-based model building."""
    print("\n" + "=" * 60)
    print("Testing Model Building from YAML")
    print("=" * 60)

    import src.models.architecture  # Import to register architectures
    from src.models.config import classifier_spec_from_config
    from src.models.registry import ModelRegistry

    # Test config
    config = {
        "architecture": "node_classifier",
        "num_classes": 7,
        "dropout": 0.5,
        "encoder": {
            "layers": [
                {
                    "layer_type": "residual_block",
                    "conv_type": "gcn",
                    "backend": "pyg",
                    "in_channels": 128,
                    "out_channels": 64,
                    "norm": "batch",
                    "activation": "relu",
                    "dropout": 0.5,
                    "residual": False,
                    "conv_kwargs": {"cached": True},
                },
                {
                    "layer_type": "residual_block",
                    "conv_type": "gcn",
                    "backend": "pyg",
                    "in_channels": 64,
                    "out_channels": 32,
                    "norm": "layer",
                    "activation": "relu",
                    "dropout": 0.5,
                    "residual": True,
                    "conv_kwargs": {"cached": True},
                },
            ]
        },
    }

    try:
        # Build spec
        spec = classifier_spec_from_config(config, input_dim=128)
        print(f"✓ Model spec created with {len(spec.encoder.layers)} layers")

        # Build model
        model = ModelRegistry.build("node_classifier", spec=spec)
        print(f"✓ Model built: {type(model).__name__}")

        # Test forward pass
        x = torch.randn(100, 128)
        edge_index = torch.randint(0, 100, (2, 500))
        torch.set_default_device(x.device)

        graph = GraphSample(
            backend=MODEL_BACKEND_TO_GRAPH_REPR["pyg"], x=x, y=torch.zeros(len(x)), edge_index=edge_index
        ).graph_repr
        logits = model(x, graph)
        assert logits.shape == (100, 7), f"Wrong output shape: {logits.shape}"
        print(f"✓ Forward pass successful: output shape {logits.shape}")

        logits.sum().backward()

    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Model building failed: {e}")
