import sys
from pathlib import Path

import pytest
import torch
from fixtures import (
    create_conv_layer,
    create_graph_sample,
    device,
    karate_like_club_graph,
    set_default_device,
    small_graph_data,
)

from src.backends.registry import BackendRegistry

try:
    from pylibcugraphops.pytorch import CSC, operators

    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False

pytestmark = pytest.mark.skipif(not HAS_CUGRAPH, reason="cugraph not installed")


class TestCugraphBasicAggregation:
    """Test basic aggregation operations (sum/mean/min/max)."""

    @pytest.mark.parametrize("aggr_type", ["sum", "mean", "min", "max"])
    def test_aggregation_matches_torch_native(
        self, aggr_type, karate_like_club_graph, create_graph_sample, create_conv_layer
    ):
        """Test that cugraph aggregation matches torch_native scatter reference."""
        data = karate_like_club_graph
        features = data["features"]

        # ===== torch_native scatter reference =====
        ref_graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="torch_native",
            num_nodes=data["num_nodes"],
        )
        ref_conv = create_conv_layer("torch_native", f"{aggr_type}_aggr", feature_dim=data["in_channels"], bias=False)
        ref_output = ref_conv(features, ref_graph_sample.graph_repr)

        # ===== CuGraph =====
        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )
        conv = create_conv_layer("cugraph", f"{aggr_type}_aggr", feature_dim=data["in_channels"], bias=False)
        cugraph_output = conv(features, graph_sample.graph_repr)

        assert torch.allclose(
            cugraph_output, ref_output, atol=1e-6, rtol=1e-5
        ), f"CuGraph {aggr_type} aggregation doesn't match torch_native reference"


class TestCugraphGATv2:
    """Test GATv2 convolution with cugraph backend."""

    @pytest.mark.parametrize("heads", [1, 4])
    def test_gatv2_forward_backward(self, heads, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """Test GATv2 forward and backward passes."""
        data = karate_like_club_graph
        features = data["features"].clone().requires_grad_(True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("cugraph", "gat_v2", feature_dim=data["in_channels"], heads=heads, bias=False)

        output = conv(features, graph_sample.graph_repr)
        assert output.shape == (data["num_nodes"], 16)
        assert not torch.isnan(output).any()

        loss = output.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

        for param in conv.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


@pytest.mark.skip("mha_simple_n2n is broken and doesn't work with correct inputs")
class TestCugraphMultiHeadAttention:
    """Test multi-head self-attention (mha_simple_n2n wrapper)."""

    @pytest.mark.parametrize("heads", [1, 4])
    def test_mha_simple_basic(self, heads, small_graph_data, create_graph_sample):
        """Test basic multi-head attention forward pass."""
        from pylibcugraphops.pytorch import operators

        data = small_graph_data
        features = data["features"]
        head_dim = data["in_channels"]

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )

        csc_graph, _ = graph_sample.graph_repr

        qkv = features.unsqueeze(1).repeat(1, heads, 1)
        qkv = qkv.reshape(data["num_nodes"], heads * head_dim)

        output = operators.mha_simple_n2n(
            key_emb=qkv,
            query_emb=qkv,
            value_emb=qkv,
            graph=csc_graph,
            num_heads=heads,
            concat_heads=True,
        )

        assert output.shape == (data["num_nodes"], heads * head_dim)
        assert not torch.isnan(output).any()
        assert output.is_cuda

    @pytest.mark.parametrize("heads", [1, 4])
    def test_mha_simple_gradients(self, heads, small_graph_data, create_graph_sample):
        """Test gradients flow through mha_simple_n2n."""

        data = small_graph_data
        features = data["features"].clone().requires_grad_(True)
        head_dim = data["in_channels"]

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )

        csc_graph, _ = graph_sample.graph_repr

        qkv = features.unsqueeze(1).repeat(1, heads, 1).reshape(data["num_nodes"], heads * head_dim)

        output = operators.mha_simple_n2n(
            key_emb=qkv,
            query_emb=qkv,
            value_emb=qkv,
            graph=csc_graph,
            num_heads=heads,
            concat_heads=True,
        )

        loss = output.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


class TestCugraphGCN:
    """Test GCN (sum aggregation with edge weights)."""

    def test_gcn_basic(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """Test GCN forward and backward."""
        data = karate_like_club_graph
        features = data["features"].clone().requires_grad_(True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("cugraph", "gcn", feature_dim=data["in_channels"], bias=False)

        output = conv(features, graph_sample.graph_repr)
        assert output.shape == (data["num_nodes"], data["in_channels"])
        assert not torch.isnan(output).any()

        loss = output.sum()
        loss.backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
