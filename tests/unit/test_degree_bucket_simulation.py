from argparse import Namespace

import numpy as np
import pytest

from scripts.ablation.calibrate_cost_model import measurement_metadata
from scripts.ablation.sweep_degree_buckets import validate_model_metadata
from turbo_gnn.bucket_simulation import build_bucket_workloads, partition_degree_buckets
from turbo_gnn.simulation import CostModel, KernelConfig, SimulationConfig, build_blocks, simulate


@pytest.mark.parametrize("strategy", ["quantile", "log", "tail"])
@pytest.mark.parametrize("count", [2, 4, 8])
def test_partition_preserves_nodes_edges_and_natural_order(strategy, count):
    degrees = np.array([0, 5, 1, 1000, 2, 5, 80, 3, 30, 5, 200, 1])
    buckets = partition_degree_buckets(degrees, count, strategy, heavy_quantile=0.8)
    assert sorted(np.concatenate([b.node_ids for b in buckets])) == list(range(len(degrees)))
    assert all(np.all(np.diff(b.node_ids) > 0) for b in buckets)
    reference = partition_degree_buckets(degrees, 2, heavy_quantile=0.8)
    assert np.array_equal(np.sort(np.concatenate([b.node_ids for b in buckets if b.heavy])), reference[0].node_ids)
    workloads, _ = build_bucket_workloads(degrees, buckets, CostModel(), num_heads=2)
    blocks = [block for group in workloads.values() for block in group]
    assert len(blocks) == 2 * len(degrees)
    assert sum(sum(b.degrees) for b in blocks) == 2 * degrees.sum()
    assert sorted(node for block in blocks for node in block.node_ids) == sorted(list(range(len(degrees))) * 2)
    for degree in np.unique(degrees):
        assert sum(np.any(degrees[b.node_ids] == degree) for b in buckets) == 1


def test_ordering_is_explicit_and_does_not_change_partition():
    degrees = np.array([2, 10, 1, 9, 8, 7, 100, 3])
    natural = partition_degree_buckets(degrees, 2)
    ordered = partition_degree_buckets(degrees, 2, degree_ordering=True)
    assert natural[1].node_ids.tolist() != ordered[1].node_ids.tolist()
    for a, b in zip(natural, ordered):
        assert sorted(a.node_ids) == sorted(b.node_ids)
        assert np.all(np.diff(degrees[b.node_ids]) <= 0)


def test_tied_degrees_and_empty_inputs_do_not_create_empty_launches():
    assert partition_degree_buckets([], 8) == []
    buckets = partition_degree_buckets([5] * 10, 8)
    assert len(buckets) == 1
    assert buckets[0].node_ids.tolist() == list(range(10))


def test_arbitrary_bucket_names_are_serialized_on_one_stream():
    workloads = {f"bucket_{i}": build_blocks([degree], f"bucket_{i}") for i, degree in enumerate([8, 4, 1])}
    kernels = {name: KernelConfig(name, 4) for name in workloads}
    sequential = simulate(workloads, kernels, SimulationConfig(num_sms=1, launch_mode="sequential"))
    concurrent = simulate(workloads, kernels, SimulationConfig(num_sms=1, launch_mode="concurrent"))
    assert sequential.makespan == 19
    assert concurrent.makespan == 10
    assert sequential.active_blocks["bucket_1"][:10].sum() == 0
    assert sequential.active_blocks["bucket_2"][:16].sum() == 0
    assert sequential.imbalance_ratio >= 1
    assert concurrent.imbalance_ratio >= 1


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_two_buckets_reproduce_legacy_schedule(mode):
    degrees = np.array([5, 10, 0, 3, 1, 7, 50, 2, 4, 7])
    buckets = partition_degree_buckets(degrees, 2, heavy_quantile=0.8)
    workloads, kernels = build_bucket_workloads(degrees, buckets, CostModel())
    legacy = {
        "heavy": build_blocks(degrees[buckets[0].node_ids], "heavy"),
        "light": build_blocks(degrees[buckets[1].node_ids], "light"),
    }
    legacy_kernels = {"heavy": KernelConfig("heavy", 4), "light": KernelConfig("light", 8)}
    config = SimulationConfig(num_sms=2, launch_mode=mode, light_launch_latency=0)
    new, old = simulate(workloads, kernels, config), simulate(legacy, legacy_kernels, config)
    assert new.makespan == old.makespan
    assert new.total_work == old.total_work
    np.testing.assert_array_equal(new.sm_utilisation, old.sm_utilisation)


def test_calibration_refuses_mixed_gpus(tmp_path):
    path = tmp_path / "timings.jsonl"
    path.write_text('{"gpu": "NVIDIA H100"}\n{"gpu": "NVIDIA A100"}\n')
    with pytest.raises(ValueError, match="mixed gpu"):
        measurement_metadata([path])


def test_simulation_refuses_cost_model_from_other_gpu():
    with pytest.raises(ValueError, match="does not identify"):
        validate_model_metadata({"measurement_metadata": {"gpu": "NVIDIA A100"}}, Namespace(hardware="h100"))
