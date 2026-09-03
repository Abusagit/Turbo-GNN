import numpy as np
import pytest

from turbo_gnn.simulation import (
    KernelConfig,
    SimulationConfig,
    bandwidth_cap_from_hardware,
    build_blocks,
    build_heavy_slices,
    simulate,
)


def test_single_block_cost_is_degree_plus_two():
    blocks = build_blocks([3], "light")
    result = simulate(
        {"light": blocks}, {"light": KernelConfig("light", 1)}, SimulationConfig(num_sms=1)
    )
    assert blocks[0].cost == 5
    assert result.makespan == 5
    assert result.imbalance_ratio == pytest.approx(1.0)


def test_occupancy_allows_multiple_resident_blocks():
    blocks = build_blocks([2, 2], "light")
    full = simulate(
        {"light": blocks}, {"light": KernelConfig("light", 2, 1.0)}, SimulationConfig(num_sms=1)
    )
    half = simulate(
        {"light": blocks}, {"light": KernelConfig("light", 2, 0.5)}, SimulationConfig(num_sms=1)
    )
    assert full.makespan == 4
    assert half.makespan == 8


def test_bandwidth_cap_stalls_resident_blocks():
    blocks = build_blocks([0, 0], "light")
    result = simulate(
        {"light": blocks},
        {"light": KernelConfig("light", 2)},
        SimulationConfig(num_sms=1, bandwidth_cap=1),
    )
    assert result.makespan == 4
    assert np.all(result.bandwidth_utilisation == 1)


def test_bandwidth_cap_must_be_positive():
    with pytest.raises(ValueError, match="bandwidth_cap"):
        simulate(
            {"light": build_blocks([1], "light")},
            {"light": KernelConfig("light", 1)},
            SimulationConfig(num_sms=1, bandwidth_cap=0),
        )


def test_work_retires_only_when_a_block_finishes():
    result = simulate(
        {"light": build_blocks([3], "light")},
        {"light": KernelConfig("light", 1)},
        SimulationConfig(num_sms=1),
    )
    assert result.retired_work.tolist() == [0, 0, 0, 0, 5]
    assert result.retired_95_time == 5


def test_assignments_cover_every_vertex_once():
    degrees = [9, 1, 7, 2, 4, 3, 8]
    for policy in ("contiguous", "grid_strided", "lpt"):
        blocks = build_blocks(degrees, "light", policy, vertices_per_block=2, num_blocks=3)
        nodes = sorted(node for block in blocks for node in block.node_ids)
        assert nodes == list(range(len(degrees)))


def test_num_heads_creates_one_task_per_node_and_head():
    blocks = build_blocks([2, 5], "light", num_heads=3)
    assert [block.node_ids for block in blocks] == [(0,), (1,), (0,), (1,), (0,), (1,)]
    assert [block.degrees for block in blocks] == [(2,), (5,), (2,), (5,), (2,), (5,)]


def test_num_blocks_is_per_head():
    blocks = build_blocks([1, 2, 3, 4], "light", "grid_strided", num_blocks=2, num_heads=3)
    assert len(blocks) == 6
    assert [block.node_ids for block in blocks] == [(0, 2), (1, 3)] * 3


def test_lpt_reduces_peak_block_cost_for_skewed_order():
    degrees = [10, 9, 8, 1, 1, 1]
    contiguous = build_blocks(degrees, "light", "contiguous", vertices_per_block=2, num_blocks=3)
    lpt = build_blocks(degrees, "light", "lpt", vertices_per_block=2, num_blocks=3)
    assert max(b.cost for b in lpt) < max(b.cost for b in contiguous)


def test_heavy_slices_do_not_cross_nodes():
    blocks = build_heavy_slices([0, 5], slice_size=2)
    assert [b.degrees for b in blocks] == [(0,), (2,), (2,), (1,)]
    assert [b.node_ids for b in blocks] == [(0,), (1,), (1,), (1,)]


def test_heavy_slices_are_created_for_each_head():
    blocks = build_heavy_slices([3], slice_size=2, num_heads=2)
    assert [block.degrees for block in blocks] == [(2,), (1,), (2,), (1,)]


def test_concurrent_streams_start_light_after_latency():
    workloads = {
        "heavy": build_blocks([8], "heavy"),
        "light": build_blocks([1], "light"),
    }
    kernels = {"heavy": KernelConfig("heavy", 2), "light": KernelConfig("light", 2)}
    concurrent = simulate(
        workloads, kernels, SimulationConfig(num_sms=1, launch_mode="concurrent", light_launch_latency=2)
    )
    sequential = simulate(workloads, kernels, SimulationConfig(num_sms=1, launch_mode="sequential"))
    assert concurrent.active_blocks["light"][:2].sum() == 0
    assert concurrent.makespan < sequential.makespan


def test_hardware_bandwidth_conversion():
    # 100 bytes/ns for 1 ns, 4-byte x 10-element row => two full rows.
    assert bandwidth_cap_from_hardware(100, 10, 4, 1) == 2
