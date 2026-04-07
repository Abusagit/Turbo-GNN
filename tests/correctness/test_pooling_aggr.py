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
    fully_connected_on_3_vertices_data,
    karate_like_club_graph,
    random_graph_data,
    set_default_device,
    small_graph_data,
)

from src.backends.registry import BackendRegistry
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR


class TestBackendRegistration:
    """Test backend registration and basic setup."""

    @pytest.mark.parametrize(
        "backend_type",
        ["torch_native", "torch_native_adj_mat"],
    )
    def test_backend_is_registered(self, backend_type):
        """Verify the backends necessary are registered."""
        backends = BackendRegistry.list_backends()
        assert backend_type in backends, f"{backend_type} not in registered backends: {backends}"

    @pytest.mark.parametrize(
        "backend_type, graph_repr",
        [
            ("torch_native", "coo"),
            ("torch_native_adj_mat", "adj_mat"),
        ],
    )
    def test_graph_representation_mapping(self, backend_type, graph_repr):
        """Verify the backends necessary have correct graph representation mapping."""
        assert backend_type in MODEL_BACKEND_TO_GRAPH_REPR
        assert MODEL_BACKEND_TO_GRAPH_REPR[backend_type] == graph_repr

    @pytest.mark.parametrize(
        "backend_type",
        ["torch_native", "torch_native_adj_mat"],
    )
    def test_backend_instantiation(self, backend_type):
        """Verify the backends necessary can be instantiated."""
        backend = BackendRegistry.get_backend(backend_type)
        assert backend is not None
        assert hasattr(backend, "create_conv")

    @pytest.mark.parametrize(
        "backend_type, aggr_type",
        [
            ("torch_native", "min_aggr"),
            ("torch_native", "max_aggr"),
            ("torch_native_adj_mat", "min_aggr"),
            ("torch_native_adj_mat", "max_aggr"),
        ],
    )
    def test_pooling_creation(self, backend_type, aggr_type):
        """Verify the backends necessary can create pooling layers declared."""
        backend = BackendRegistry.get_backend(backend_type)
        conv = backend.create_conv(aggr_type, feature_dim=16, bias=True)

        assert conv is not None
        assert hasattr(conv, "forward")


class TestAggregationCorrectness:
    """Test pooling aggregation mathematical correctness against hand-computed ground truth on minimal corner cases."""

    @pytest.mark.parametrize(
        "backend_type, gt_key, aggr_type",
        [
            ("torch_native", "expected_min", "min_aggr"),
            ("torch_native", "expected_max", "max_aggr"),
            ("torch_native_adj_mat", "expected_min", "min_aggr"),
            ("torch_native_adj_mat", "expected_max", "max_aggr"),
        ],
    )
    def test_star_graph(
        self, small_graph_data, create_conv_layer, create_graph_sample, backend_type, gt_key, aggr_type
    ):
        """
        star graph - all nodes except center pass messages to the center; no other messages are passed
        """
        data = small_graph_data
        features = data["features"]

        # ===== gt =====
        gt = data[gt_key]

        # =====  Our Implementation =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend=backend_type,
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )

        conv = create_conv_layer(backend_type, aggr_type, feature_dim=data["in_channels"], bias=False)

        our_output = conv(features, graph_sample.graph_repr)

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
        "backend_type, gt_key, aggr_type",
        [
            ("torch_native", "expected_min", "min_aggr"),
            ("torch_native", "expected_max", "max_aggr"),
            ("torch_native_adj_mat", "expected_min", "min_aggr"),
            ("torch_native_adj_mat", "expected_max", "max_aggr"),
        ],
    )
    def test_fully_connected_graph(
        self,
        fully_connected_on_3_vertices_data,
        create_conv_layer,
        create_graph_sample,
        backend_type,
        gt_key,
        aggr_type,
    ):
        """
        fully connected graph - all nodes pass messages to all the other nodes
        """
        data = fully_connected_on_3_vertices_data
        features = data["features"]

        # ===== gt =====
        gt = data[gt_key]

        # =====  Our Implementation =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend=backend_type,
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )

        conv = create_conv_layer(backend_type, aggr_type, feature_dim=data["in_channels"], bias=False)

        our_output = conv(features, graph_sample.graph_repr)

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
        "backend_type, gt_key, aggr_type",
        [
            ("torch_native", "expected_min", "min_aggr"),
            ("torch_native", "expected_max", "max_aggr"),
            ("torch_native_adj_mat", "expected_min", "min_aggr"),
            ("torch_native_adj_mat", "expected_max", "max_aggr"),
        ],
    )
    def test_connectivity_component_and_isolated_vertice(
        self,
        connectivity_component_and_isolated_vertice_data,
        create_conv_layer,
        create_graph_sample,
        backend_type,
        gt_key,
        aggr_type,
    ):
        """
        graph with 3 vertices and edges 0->1 1->0 to test isolation layer result
        """
        data = connectivity_component_and_isolated_vertice_data
        features = data["features"]

        # ===== gt =====
        gt = data[gt_key]

        # =====  Our Implementation =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend=backend_type,
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )

        conv = create_conv_layer(backend_type, aggr_type, feature_dim=data["in_channels"], bias=False)

        our_output = conv(features, graph_sample.graph_repr)

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
# star graph also tests for directions


class TestAggregationEquivalence:
    """
    Test equivalence of implementation outputs on hard graph samples.
    Compares scatter-based torch_native vs sparse-COO torch_native_adj_mat.
    """

    @pytest.mark.parametrize(
        "gt_key, aggr_type",
        [("expected_min", "min_aggr"), ("expected_max", "max_aggr")],
    )
    def test_karate_equivalence(
        self, karate_like_club_graph, create_conv_layer, create_graph_sample, gt_key, aggr_type
    ):
        """
        Test equivalence of output of our implementations on Karate Club graph.
        """
        data = karate_like_club_graph
        features = data["features"]

        # ===== scatter-based output =====
        scatter_graph = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native",
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )
        conv_scatter = create_conv_layer("torch_native", aggr_type, feature_dim=data["in_channels"], bias=False)
        scatter_output = conv_scatter(features, scatter_graph.graph_repr)

        # ===== sparse COO output =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_adj_mat",
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )
        conv = create_conv_layer("torch_native_adj_mat", aggr_type, feature_dim=data["in_channels"], bias=False)
        native_output = conv(features, graph_sample.graph_repr)

        # ===== Compare Outputs =====
        max_abs_diff = (scatter_output - native_output).abs().max().item()
        mean_abs_diff = (scatter_output - native_output).abs().mean().item()

        # Relative error (avoid division by zero*)
        relative_error = ((scatter_output - native_output).abs() / (scatter_output.abs() + 1e-8)).mean().item()

        print("\nComparison scatter vs sparse COO:")
        print(f"  Max absolute difference:  {max_abs_diff:.8e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8e}")
        print(f"  Mean relative error:      {relative_error:.8e}")

        # Assert numerical equivalence
        assert torch.allclose(
            scatter_output, native_output, atol=1e-6, rtol=1e-5
        ), f"Output doesn't match: max_diff={max_abs_diff:.8e}, mean_diff={mean_abs_diff:.8e}"

    @pytest.mark.parametrize(
        "gt_key, aggr_type",
        [("expected_min", "min_aggr"), ("expected_max", "max_aggr")],
    )
    def test_random_graph_equivalence(
        self, random_graph_data, create_conv_layer, create_graph_sample, gt_key, aggr_type
    ):
        """
        Test equivalence of output of our implementations on random graph fixture.
        """
        data = random_graph_data
        features = data["features"]

        # ===== scatter-based output =====
        scatter_graph = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native",
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )
        conv_scatter = create_conv_layer("torch_native", aggr_type, feature_dim=data["in_channels"], bias=False)
        scatter_output = conv_scatter(features, scatter_graph.graph_repr)

        # ===== sparse COO output =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_adj_mat",
            num_nodes=data["num_nodes"],
            add_self_loops=False,
        )
        conv = create_conv_layer("torch_native_adj_mat", aggr_type, feature_dim=data["in_channels"], bias=False)
        native_output = conv(features, graph_sample.graph_repr)

        # ===== Compare Outputs =====
        max_abs_diff = (scatter_output - native_output).abs().max().item()
        mean_abs_diff = (scatter_output - native_output).abs().mean().item()

        # Relative error (avoid division by zero*)
        relative_error = ((scatter_output - native_output).abs() / (scatter_output.abs() + 1e-8)).mean().item()

        print("\nComparison scatter vs sparse COO:")
        print(f"  Max absolute difference:  {max_abs_diff:.8e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8e}")
        print(f"  Mean relative error:      {relative_error:.8e}")

        # Assert numerical equivalence
        assert torch.allclose(
            scatter_output, native_output, atol=1e-6, rtol=1e-5
        ), f"Output doesn't match: max_diff={max_abs_diff:.8e}, mean_diff={mean_abs_diff:.8e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
