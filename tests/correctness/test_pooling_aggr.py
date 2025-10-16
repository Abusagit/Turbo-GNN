import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__)))

from fixtures import (
    connectivity_component_and_isolated_vertice_data,
    create_conv_layer,
    create_graph_sample,
    device,
    dgl_available,
    fully_connected_on_3_vertices_data,
    karate_like_club_graph,
    set_default_device,
    small_graph_data,
)

from src.backends.registry import BackendRegistry
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR


class TestBackendRegistration:
    """Test backend registration and basic setup."""

    def test_backend_is_registered(self):
        """Verify torch_native_mean backend is registered."""
        backends = BackendRegistry.list_backends()
        assert "dgl" in backends, f"dgl not in registered backends: {backends}"

    def test_graph_representation_mapping(self):
        """Verify backend has correct graph representation mapping."""
        assert "dgl" in MODEL_BACKEND_TO_GRAPH_REPR
        assert MODEL_BACKEND_TO_GRAPH_REPR["dgl"] == "dgl"

    def test_backend_instantiation(self):
        """Verify backend can be instantiated."""
        backend = BackendRegistry.get_backend("dgl")
        assert backend is not None
        assert hasattr(backend, "create_conv")

    @pytest.mark.parametrize("aggr_type", ["min_aggr", "max_aggr"])
    def test_pooling_creation(self, aggr_type):
        """Verify backend can create pooling layers declared."""
        backend = BackendRegistry.get_backend("dgl")
        conv = backend.create_conv(aggr_type, in_channels=16, out_channels=16, bias=True)

        assert conv is not None
        assert hasattr(conv, "forward")


class TestAggregationCorrectness:
    """Test mean aggregation mathematical correctness against DGL."""

    @pytest.mark.parametrize(
        "gt_key, aggr_type",
        [
            ("expected_min", "min_aggr"),
            ("expected_max", "max_aggr"),
        ],
    )
    def test_star_graph(self, small_graph_data, create_conv_layer, gt_key, aggr_type):
        """
        Test that our mean aggregation matches DGL's copy_u_mean exactly.

        This is the core correctness test: we compare our implementation with
        DGL's reference implementation on the Karate Club graph.
        """
        try:
            import dgl
            import dgl.ops as dgl_ops
        except ImportError:
            pytest.skip("DGL not installed - cannot verify correctness")

        data = small_graph_data
        features = data["features"]
        out_channels = features.shape[1]

        # ===== gt =====

        gt = data[gt_key]

        # =====  Our Implementation =====

        our_graph = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data["num_nodes"]).to(
            data["device"]
        )

        conv = create_conv_layer("dgl", aggr_type, data["in_channels"], out_channels, bias=False)

        our_output = conv(features, our_graph)

        # ===== Compare Outputs =====
        max_abs_diff = (gt - our_output).abs().max().item()
        mean_abs_diff = (gt - our_output).abs().mean().item()

        # Relative error (avoid division by zero)
        relative_error = ((gt - our_output).abs() / (gt.abs() + 1e-8)).mean().item()

        print("\nComparison with ground truth:")
        print(f"  Max absolute difference:  {max_abs_diff:.8e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8e}")
        print(f"  Mean relative error:      {relative_error:.8e}")

        # Assert numerical equivalence
        assert torch.allclose(
            gt, our_output, atol=1e-6, rtol=1e-5
        ), f"Output doesn't match ground truth: max_diff={max_abs_diff:.8e}, mean_diff={mean_abs_diff:.8e}"

    @pytest.mark.parametrize(
        "gt_key, aggr_type",
        [
            ("expected_min", "min_aggr"),
            ("expected_max", "max_aggr"),
        ],
    )
    def test_fully_connected_graph(self, fully_connected_on_3_vertices_data, create_conv_layer, gt_key, aggr_type):
        """
        Test that our mean aggregation matches DGL's copy_u_mean exactly.

        This is the core correctness test: we compare our implementation with
        DGL's reference implementation on the Karate Club graph.
        """
        try:
            import dgl
            import dgl.ops as dgl_ops
        except ImportError:
            pytest.skip("DGL not installed - cannot verify correctness")

        data = fully_connected_on_3_vertices_data
        features = data["features"]
        out_channels = features.shape[1]

        # ===== gt =====

        gt = data[gt_key]

        # =====  Our Implementation =====

        our_graph = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data["num_nodes"]).to(
            data["device"]
        )

        conv = create_conv_layer("dgl", aggr_type, data["in_channels"], out_channels, bias=False)

        our_output = conv(features, our_graph)

        # ===== Compare Outputs =====
        max_abs_diff = (gt - our_output).abs().max().item()
        mean_abs_diff = (gt - our_output).abs().mean().item()

        # Relative error (avoid division by zero*)
        relative_error = ((gt - our_output).abs() / (gt.abs() + 1e-8)).mean().item()

        print("\nComparison with ground truth:")
        print(f"  Max absolute difference:  {max_abs_diff:.8e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8e}")
        print(f"  Mean relative error:      {relative_error:.8e}")

        # Assert numerical equivalence
        assert torch.allclose(
            gt, our_output, atol=1e-6, rtol=1e-5
        ), f"Output doesn't match ground truth: max_diff={max_abs_diff:.8e}, mean_diff={mean_abs_diff:.8e}"

    @pytest.mark.parametrize(
        "gt_key, aggr_type",
        [
            ("expected_min", "min_aggr"),
            ("expected_max", "max_aggr"),
        ],
    )
    def test_connectivity_component_and_isolated_vertice(
        self, connectivity_component_and_isolated_vertice_data, create_conv_layer, gt_key, aggr_type
    ):
        """
        Test that our mean aggregation matches DGL's copy_u_mean exactly.

        This is the core correctness test: we compare our implementation with
        DGL's reference implementation on the Karate Club graph.
        """
        try:
            import dgl
            import dgl.ops as dgl_ops
        except ImportError:
            pytest.skip("DGL not installed - cannot verify correctness")

        data = connectivity_component_and_isolated_vertice_data
        features = data["features"]
        out_channels = features.shape[1]

        # ===== gt =====

        gt = data[gt_key]

        # =====  Our Implementation =====

        our_graph = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data["num_nodes"]).to(
            data["device"]
        )

        conv = create_conv_layer("dgl", aggr_type, data["in_channels"], out_channels, bias=False)

        our_output = conv(features, our_graph)

        # ===== Compare Outputs =====
        max_abs_diff = (gt - our_output).abs().max().item()
        mean_abs_diff = (gt - our_output).abs().mean().item()

        # Relative error (avoid division by zero*)
        relative_error = ((gt - our_output).abs() / (gt.abs() + 1e-8)).mean().item()

        print("\nComparison with ground truth:")
        print(f"  Max absolute difference:  {max_abs_diff:.8e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8e}")
        print(f"  Mean relative error:      {relative_error:.8e}")

        # Assert numerical equivalence
        assert torch.allclose(
            gt, our_output, atol=1e-6, rtol=1e-5
        ), f"Output doesn't match ground truth: max_diff={max_abs_diff:.8e}, mean_diff={mean_abs_diff:.8e}"


# TODO test_empty_graph

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
