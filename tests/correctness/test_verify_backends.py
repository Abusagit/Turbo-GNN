#!/usr/bin/env python3
"""
Test script to verify all backends work correctly with different datasets.
Run from repository root: python test_verify_backends.py
"""

import sys
import torch
import traceback
from pathlib import Path

import yaml

# pytest: keep imports identical; we only add assertions so failures fail
# (no deletions or large refactors)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "./")

from src.backends.registry import BackendRegistry
from src.data.datasets import load_single_graph, DatasetConfig, GraphSample, MODEL_BACKEND_TO_GRAPH_REPR


def test_backend_registration():
    """Test that backends are properly registered."""
    print("\n" + "="*60)
    print("Testing Backend Registration")
    print("="*60)

    # Import backend modules to trigger registration
    try:
        import src.backends.pyg_backend  # noqa: F401
        import src.backends.dgl_backend  # noqa: F401
        import src.backends.torch_native_backend  # noqa: F401
        print("✓ Backend modules imported successfully")
    except Exception as e:
        msg = f"✗ Failed to import backends: {e}"
        print(msg)
        assert False, msg  # <-- pytest: fail test
        return False

    # Check registered backends
    backends = BackendRegistry.list_backends()
    print(f"Registered backends: {backends}")

    expected = {"pyg", "dgl", "torch_native_gcn"}
    missing = expected - set(backends)
    if missing:
        msg = f"✗ Missing backends: {missing}"
        print(msg)
        assert False, msg  # <-- pytest: fail test
        return False

    print("✓ All expected backends registered")
    assert True
    return True


def test_dataset_loading():
    """Test dataset loading from different sources."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)

    test_configs = [
        DatasetConfig(source="pyg", name="Cora", root="data", graph_backend="pyg"),
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
            msg = f"  ✗ Failed: {e}"
            print(msg)
            assert False, msg  # <-- pytest: fail test
            return False

    assert True
    return True


def test_backend_convolutions():
    """Test that each backend can create and run convolutions."""
    print("\n" + "="*60)
    print("Testing Backend Convolutions")
    print("="*60)

    # Create small test graph
    num_nodes = 100
    num_edges = 500
    in_channels = 16
    out_channels = 32

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
                    conv = backend.create_conv(
                        conv_type,
                        in_channels,
                        out_channels,
                        bias=True
                    ).to(device)

                    graph = GraphSample(backend=MODEL_BACKEND_TO_GRAPH_REPR[backend_name], x=x, y=torch.zeros_like(x), edge_index=edge_index).graph_repr

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
    print("\n" + "="*60)
    print("Summary:")
    for key, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗" if "FAILED" in status else "⚠"
        print(f"  {symbol} {key}: {status}")

    ok = all(v in ("PASSED", "NOT_IMPLEMENTED") for v in results.values())
    if not ok:
        failed = {k: v for k, v in results.items() if v not in ("PASSED", "NOT_IMPLEMENTED")}
        assert False, f"Backend convolution failures: {failed}"  # <-- pytest: fail test

    return ok


def test_microbenchmarking():
    """Test microbenchmarking functionality."""
    print("\n" + "="*60)
    print("Testing Microbenchmarking")
    print("="*60)

    from src.benchmarking.microbench import time_callable

    def test_fn():
        x = torch.randn(1000, 1000)
        y = torch.matmul(x, x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        result = time_callable(test_fn, warmup=5, iters=10)
        print(f"✓ Microbench completed: {result.ms_per_iter:.3f} ms/iter on {result.device}")
        assert True
        return True
    except Exception as e:
        msg = f"✗ Microbench failed: {e}"
        print(msg)
        assert False, msg  # <-- pytest: fail test
        return False


def test_memory_profiling():
    """Test memory profiling utilities."""
    print("\n" + "="*60)
    print("Testing Memory Profiling")
    print("="*60)

    from src.benchmarking.memory import (
        capture_cuda_snapshot,
        human_bytes,
        measure_peak_cuda_memory_during
    )

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

        # Test peak measurement
        result = measure_peak_cuda_memory_during(memory_test)
        print(f"✓ Peak memory during operation: {human_bytes(result.peak_allocated, binary=True)}")
        assert True
        return True
    except Exception as e:
        print(f"✗ Memory profiling failed: {e}")
        traceback.print_exc()
        assert False, f"Memory profiling failed: {e}"  # <-- pytest: fail test
        return False


def test_model_building():
    """Test YAML-based model building."""
    print("\n" + "="*60)
    print("Testing Model Building from YAML")
    print("="*60)

    from src.models.config import classifier_spec_from_config
    from src.models.registry import ModelRegistry
    import src.models.architecture  # Import to register architectures

    # Test config
    config = {
        "architecture": "node_classifier",
        "num_classes": 7,
        "dropout": 0.5,
        "encoder": {
            "layers": [
                {
                    "conv_type": "gcn",
                    "backend": "pyg",
                    "in_channels": 128,
                    "out_channels": 64,
                    "norm": "batch",
                    "activation": "relu",
                    "dropout": 0.5,
                    "residual": False,
                    "conv_kwargs": {"cached": True}
                },
                {
                    "conv_type": "gcn",
                    "backend": "pyg",
                    "in_channels": 64,
                    "out_channels": 32,
                    "norm": "layer",
                    "activation": "relu",
                    "dropout": 0.5,
                    "residual": True,
                    "conv_kwargs": {"cached": True}
                }
            ]
        }
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

        graph = GraphSample(backend=MODEL_BACKEND_TO_GRAPH_REPR["pyg"], x=x, y=torch.zeros(len(x)), edge_index=edge_index).graph_repr
        logits = model(x, graph)
        assert logits.shape == (100, 7), f"Wrong output shape: {logits.shape}"
        print(f"✓ Forward pass successful: output shape {logits.shape}")

        logits.sum().backward()

        assert True
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        traceback.print_exc()
        assert False, f"Model building failed: {e}"  # <-- pytest: fail test
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GNN BENCHMARKING REPOSITORY VERIFICATION")
    print("="*60)

    tests = [
        ("Backend Registration", test_backend_registration),
        ("Dataset Loading", test_dataset_loading),
        ("Backend Convolutions", test_backend_convolutions),
        ("Microbenchmarking", test_microbenchmarking),
        ("Memory Profiling", test_memory_profiling),
        ("Model Building", test_model_building),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {sum(not p for _, p in results)} tests failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
