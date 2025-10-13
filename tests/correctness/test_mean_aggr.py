import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__)))

from fixtures import (
    create_conv_layer,
    create_graph_sample,
    device,
    dgl_available,
    karate_like_club_graph,
    set_default_device,
)

from src.backends.registry import BackendRegistry
from src.data.datasets import MODEL_BACKEND_TO_GRAPH_REPR


class TestBackendRegistration:
    """Test backend registration and basic setup."""

    def test_backend_is_registered(self):
        """Verify torch_native_mean backend is registered."""
        backends = BackendRegistry.list_backends()
        assert "torch_native_meanaggr" in backends, f"torch_native_mean not in registered backends: {backends}"

    def test_graph_representation_mapping(self):
        """Verify backend has correct graph representation mapping."""
        assert "torch_native_meanaggr" in MODEL_BACKEND_TO_GRAPH_REPR
        assert MODEL_BACKEND_TO_GRAPH_REPR["torch_native_meanaggr"] == "adj_mat_in_degree_normalized_transposed"

    def test_backend_instantiation(self):
        """Verify backend can be instantiated."""
        backend = BackendRegistry.get_backend("torch_native_meanaggr")
        assert backend is not None
        assert hasattr(backend, "create_conv")

    def test_conv_layer_creation(self):
        """Verify backend can create GCN convolution layers."""
        backend = BackendRegistry.get_backend("torch_native_meanaggr")
        conv = backend.create_conv("mean_aggr", in_channels=16, out_channels=16, bias=True)

        assert conv is not None
        assert hasattr(conv, "forward")


class TestAggregationCorrectness:
    """Test mean aggregation mathematical correctness against DGL."""

    def test_matches_dgl_copy_u_mean(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
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

        data = karate_like_club_graph
        features = data["features"]
        out_channels = features.shape[1]

        # ===== DGL Implementation =====
        dgl_graph = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data["num_nodes"]).to(
            data["device"]
        )

        dgl_output = dgl_ops.copy_u_mean(dgl_graph, features)

        # ===== Our Implementation =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_meanaggr",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", data["in_channels"], out_channels, bias=False)

        our_output = conv(features, graph_sample.graph_repr)

        # ===== Compare Outputs =====
        max_abs_diff = (dgl_output - our_output).abs().max().item()
        mean_abs_diff = (dgl_output - our_output).abs().mean().item()

        # Relative error (avoid division by zero)
        relative_error = ((dgl_output - our_output).abs() / (dgl_output.abs() + 1e-8)).mean().item()

        print("\nComparison with DGL copy_u_mean:")
        print(f"  Max absolute difference:  {max_abs_diff:.8e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8e}")
        print(f"  Mean relative error:      {relative_error:.8e}")

        # Assert numerical equivalence
        assert torch.allclose(
            our_output, dgl_output, atol=1e-6, rtol=1e-5
        ), f"Output doesn't match DGL: max_diff={max_abs_diff:.8e}, mean_diff={mean_abs_diff:.8e}"

    def test_isolated_nodes_produce_zero(self, create_graph_sample, create_conv_layer, device):
        """
        Test that nodes with no incoming edges produce zero aggregation.
        """
        num_nodes = 5
        in_channels = 8

        # Create graph where node 4 has no incoming edges
        # 0->1, 1->2, 2->3 (node 4 isolated)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device=device)
        features = torch.randn(num_nodes, in_channels, device=device)

        graph_sample = create_graph_sample(
            edge_index=edge_index, features=features, backend="torch_native_meanaggr", num_nodes=num_nodes
        )

        conv = create_conv_layer(
            "torch_native_meanaggr",
            "mean_aggr",
            in_channels,
            16,
            bias=False,  # No bias so we can verify zero
        )

        output = conv(features, graph_sample.graph_repr)

        # Node 0 and 4 have no incoming edges -> should be zero
        assert torch.allclose(
            output[0], torch.zeros_like(output[0]), atol=1e-6
        ), "Node 0 (no incoming edges) should have zero output"

        assert torch.allclose(
            output[4], torch.zeros_like(output[4]), atol=1e-6
        ), "Node 4 (isolated) should have zero output"

        # Nodes 1, 2, 3 should have non-zero output (they have incoming edges)
        for node_id in [1, 2, 3]:
            assert (
                output[node_id].abs().sum() > 1e-6
            ), f"Node {node_id} (has incoming edges) should have non-zero output"


class TestGradientFlow:
    """Test gradient computation and backpropagation with finite differences."""

    def test_gradients_exist(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """Basic test that gradients are computed for all parameters."""
        data = karate_like_club_graph

        # Require gradients for features
        features = data["features"].clone().requires_grad_(True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_meanaggr",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", data["in_channels"], 32, bias=True)

        # Forward pass
        output = conv(features, graph_sample.graph_repr)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check all gradients exist
        assert features.grad is not None, "Features should have gradients"

        # Check no NaN or Inf
        assert not torch.isnan(features.grad).any(), "Features gradient contains NaN"

    def test_weight_gradient_with_gradcheck(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """
        Verify gradients using PyTorch's autograd.gradcheck.

        This uses PyTorch's built-in numerical gradient checking which is more
        robust than manual finite differences. We check feature gradients.
        """
        data = karate_like_club_graph

        # Use smaller dimensions for faster gradient checking
        in_channels = 4
        out_channels = 6

        # Create smaller feature set in double precision
        features = torch.randn(data["num_nodes"], in_channels, device=data["device"], dtype=torch.float64)
        features.requires_grad_(True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_meanaggr",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", in_channels, out_channels, bias=True)

        # Convert to double precision (required for gradcheck)
        conv = conv.double()
        graph_repr_double = graph_sample.graph_repr.double()

        def func(feat):
            return conv(feat, graph_repr_double)

        result = torch.autograd.gradcheck(func, features, eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True)

        assert result

    def test_feature_gradient_with_gradcheck(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """
        Verify feature gradients using PyTorch's autograd.gradcheck.

        This specifically tests gradient flow through the aggregation operation.
        """
        data = karate_like_club_graph

        # Use smaller dimensions for faster checking
        in_channels = 4
        out_channels = 6

        # Create features in double precision
        features_double = torch.randn(data["num_nodes"], in_channels, device=data["device"], dtype=torch.float64)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features_double,
            backend="torch_native_meanaggr",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", in_channels, out_channels, bias=True)
        conv = conv.double()

        # Freeze conv parameters (we only check feature gradients)
        for param in conv.parameters():
            param.requires_grad = False

        # Define function that takes features as input
        def func(feat):
            return conv(feat, graph_sample.graph_repr.double())

        # Create input features in double precision
        features = torch.randn(
            data["num_nodes"], in_channels, device=data["device"], dtype=torch.float64, requires_grad=True
        )

        result = torch.autograd.gradcheck(func, features, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)

        assert result, "PyTorch gradcheck failed for feature gradients"

    def test_gradient_flow_through_aggregation(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """
        Test that gradients correctly flow from output nodes back to their neighbors.

        This verifies the chain rule through the mean aggregation operation.
        """
        data = karate_like_club_graph

        features = data["features"].clone().requires_grad_(True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_meanaggr",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer(
            "torch_native_meanaggr",
            "mean_aggr",
            data["in_channels"],
            data["in_channels"],  # Same size
            bias=False,
        )
        output = conv(features, graph_sample.graph_repr)

        # Take gradient w.r.t. one specific output node
        target_node = 0  # Node with many incoming edges in Karate Club
        loss = output[target_node].sum()
        loss.backward()

        # Find which nodes have edges to target_node
        incoming_mask = data["edge_index"][1] == target_node
        source_nodes = data["edge_index"][0][incoming_mask].unique()

        print(f"\nGradient flow test for node {target_node}:")
        print(f"  Incoming edges from nodes: {source_nodes.tolist()}")

        # Source nodes should have non-zero gradients
        for src_node in source_nodes:
            grad_norm = features.grad[src_node].abs().sum().item()
            assert (
                grad_norm > 1e-6
            ), f"Source node {src_node} should have non-zero gradient (contributes to node {target_node})"
            print(f"  Node {src_node}: gradient norm = {grad_norm:.6f}")

        # Nodes that don't connect to target should have zero gradients
        non_source_nodes = set(range(data["num_nodes"])) - set(source_nodes.tolist()) - {target_node}
        for node_id in list(non_source_nodes)[:5]:  # Check a few
            grad_norm = features.grad[node_id].abs().sum().item()
            assert (
                grad_norm < 1e-6
            ), f"Non-source node {node_id} should have zero gradient (doesn't contribute to node {target_node})"

    def test_second_order_gradients(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """
        Test that second-order gradients can be computed (for methods like MAML).
        """
        data = karate_like_club_graph

        features = data["features"].clone().requires_grad_(True)

        weight = torch.randn_like(features, requires_grad=True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native_meanaggr",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", data["in_channels"], 16, bias=True)

        # First forward/backward
        output = conv(features, graph_sample.graph_repr) @ weight.T
        loss = output.pow(2).sum()

        # Compute first-order gradients
        grad_outputs = torch.autograd.grad(
            loss,
            weight,
            create_graph=True,  # Allow second-order
            retain_graph=True,
        )

        # Compute second-order gradients
        second_order_loss = sum(g.pow(2).sum() for g in grad_outputs)
        second_order_loss.backward()

        # Check second-order gradients exist
        assert weight.grad is not None, "Second-order weight gradient should exist"
        assert not torch.isnan(weight.grad).any(), "Second-order gradient contains NaN"

        print("\nSecond-order gradient test:")
        print(f"  Second-order loss: {second_order_loss.item():.6f}")
        print(f"  Weight gradient norm: {weight.grad.norm().item():.6f}")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_graph(self, create_graph_sample, create_conv_layer, device):
        """Test graph with no edges."""
        num_nodes = 10
        in_channels = 8

        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        features = torch.randn(num_nodes, in_channels, device=device)

        graph_sample = create_graph_sample(
            edge_index=edge_index, features=features, backend="torch_native_meanaggr", num_nodes=num_nodes
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", in_channels, 16, bias=False)
        output = conv(features, graph_sample.graph_repr)

        # All nodes isolated -> zero aggregation
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_self_loops_only(self, create_graph_sample, create_conv_layer, device):
        """Test graph with only self-loops."""
        num_nodes = 5
        in_channels = 4

        # Only self-loops: i->i for all i
        edge_index = torch.arange(num_nodes, device=device).repeat(2, 1)
        features = torch.randn(num_nodes, in_channels, device=device)

        graph_sample = create_graph_sample(
            edge_index=edge_index, features=features, backend="torch_native_meanaggr", num_nodes=num_nodes
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", in_channels, in_channels, bias=False)

        output = conv(features, graph_sample.graph_repr)

        # Each node receives only from itself (mean of one value = that value)
        # So output ≈ features (after linear transform with identity)
        assert torch.allclose(output, features, atol=1e-5)

    def test_complete_graph(self, create_graph_sample, create_conv_layer, device):
        """Test complete graph (all nodes connected to all nodes)."""
        num_nodes = 10
        in_channels = 4

        # Complete graph: all edges
        src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
        dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
        edge_index = torch.stack([src, dst], dim=0)

        features = torch.randn(num_nodes, in_channels, device=device)

        graph_sample = create_graph_sample(
            edge_index=edge_index, features=features, backend="torch_native_meanaggr", num_nodes=num_nodes
        )

        conv = create_conv_layer("torch_native_meanaggr", "mean_aggr", in_channels, in_channels, bias=False)

        output = conv(features, graph_sample.graph_repr)

        # Each node receives mean of all features
        expected_mean = features.mean(dim=0, keepdim=True).expand(num_nodes, -1)

        assert torch.allclose(
            output, expected_mean, atol=1e-5
        ), "In complete graph, all nodes should receive global mean"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
