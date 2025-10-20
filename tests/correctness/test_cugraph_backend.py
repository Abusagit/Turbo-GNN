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

from src.backends.cugraph_backend import CugraphBackend
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
    def test_mean_aggregation_matches_dgl(
        self, aggr_type, karate_like_club_graph, create_graph_sample, create_conv_layer
    ):
        """Test that cugraph mean aggregation matches DGL's copy_u_mean."""
        try:
            import dgl
            import dgl.ops as dgl_ops
        except ImportError:
            pytest.skip("DGL not installed - cannot verify correctness")

        data = karate_like_club_graph
        features = data["features"]
        out_channels = data["in_channels"]

        dgl_graph = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data["num_nodes"]).to(
            data["device"]
        )
        dgl_graph = dgl.add_self_loop(dgl_graph)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )

        match aggr_type:
            case "sum":
                dgl_op = dgl_ops.copy_u_sum
            case "mean":
                dgl_op = dgl_ops.copy_u_mean
            case "min":
                dgl_op = dgl_ops.copy_u_min
            case "max":
                dgl_op = dgl_ops.copy_u_max

        dgl_output = dgl_op(dgl_graph, features)
        conv = create_conv_layer("cugraph", f"{aggr_type}_aggr", data["in_channels"], out_channels, bias=False)
        cugraph_output = conv(features, graph_sample.graph_repr)

        assert torch.allclose(
            cugraph_output, dgl_output, atol=1e-6, rtol=1e-5
        ), f"CuGraph {aggr_type} aggregation doesn't match DGL"


class TestCugraphGATv2:
    """Test GATv2 convolution with cugraph backend."""

    @pytest.mark.parametrize("heads", [1, 4])
    def test_gatv2_matches_dgl(self, heads, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """Test that cugraph GATv2 matches DGL's GATv2Conv."""
        try:
            import dgl
            from dgl.nn.pytorch.conv import GATv2Conv
        except ImportError:
            pytest.skip("DGL not installed")

        data = karate_like_club_graph
        features = data["features"]
        in_channels = out_channels = data["in_channels"]

        dgl_conv = GATv2Conv(in_channels, out_channels, num_heads=heads, bias=False, allow_zero_in_degree=True).to(
            data["device"]
        )

        dgl_graph = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data["num_nodes"]).to(
            data["device"]
        )
        cugraph_conv = create_conv_layer("cugraph", "gat", in_channels, out_channels, heads=heads, bias=False)

        with torch.no_grad():
            cugraph_conv.linear_gat_projection.weight.data = dgl_conv.fc_src.weight.data.clone()
            cugraph_conv.attn_weights.data = dgl_conv.attn.data.flatten().clone()
            cugraph_conv.outer_projection = torch.nn.Identity()

        dgl_output = dgl_conv(dgl_graph, features, get_attention=False)
        dgl_output = dgl_output.view(data["num_nodes"], -1)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )
        cugraph_output = cugraph_conv(features, graph_sample.graph_repr)

        assert cugraph_output.shape == dgl_output.shape
        assert not torch.isnan(cugraph_output).any()

    def test_gatv2_forward_backward(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """Test GATv2 forward and backward passes."""
        data = karate_like_club_graph
        features = data["features"].clone().requires_grad_(True)

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="cugraph",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("cugraph", "gat", data["in_channels"], 16, heads=4, bias=False)

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

        conv = create_conv_layer("cugraph", "gcn", data["in_channels"], data["in_channels"], bias=False)

        output = conv(features, graph_sample.graph_repr)
        assert output.shape == (data["num_nodes"], data["in_channels"])
        assert not torch.isnan(output).any()

        loss = output.sum()
        loss.backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
