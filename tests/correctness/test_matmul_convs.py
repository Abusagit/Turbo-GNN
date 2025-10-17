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
    random_graph_data,
    set_default_device,
)

from src.backends.registry import BackendRegistry


class TestMatMulConvCorrectness:
    """Test basic aggregation operations (gcn, mean, sum)."""

    @pytest.mark.parametrize("aggr_type", ["gcn", "mean_aggr", "sum_aggr"])
    @pytest.mark.parametrize("backend", ["pyg", "torch_native"])
    def test_matmul_conv_matches_dgl_on_undirected_graph(
        self, aggr_type, backend, karate_like_club_graph, create_graph_sample, create_conv_layer
    ):
        self._test_matmul_conv_matches_dgl(
            aggr_type, backend, karate_like_club_graph, create_graph_sample, create_conv_layer
        )

    # works on ("gcn", "torch_native"), but not on ("gcn", "pyg")
    @pytest.mark.parametrize("aggr_type", ["mean_aggr", "sum_aggr"])
    @pytest.mark.parametrize("backend", ["pyg", "torch_native"])
    def test_matmul_conv_matches_dgl_on_directed_graph(
        self, aggr_type, backend, random_graph_data, create_graph_sample, create_conv_layer
    ):
        self._test_matmul_conv_matches_dgl(
            aggr_type, backend, random_graph_data, create_graph_sample, create_conv_layer
        )

    @pytest.mark.parametrize("aggr_type,backend", [("gcn", "torch_native")])
    def test_matmul_conv_matches_dgl_on_directed_graph2(
        self, aggr_type, backend, random_graph_data, create_graph_sample, create_conv_layer
    ):
        self._test_matmul_conv_matches_dgl(
            aggr_type, backend, random_graph_data, create_graph_sample, create_conv_layer
        )

    def _test_matmul_conv_matches_dgl(self, aggr_type, backend, cur_data, create_graph_sample, create_conv_layer):
        """Test that particular convolution matches DGL's one."""
        try:
            import dgl
            import dgl.ops as dgl_ops
        except ImportError:
            pytest.skip("DGL not installed - cannot verify correctness")

        backend = f"{backend}_{aggr_type}" if backend == "torch_native" else backend

        data = cur_data
        features = data["features"]
        features.requires_grad = True
        out_channels = data["in_channels"]

        def apply_conv(backend):
            graph = create_graph_sample(
                edge_index=data["edge_index"],
                features=features,
                backend=backend,
                num_nodes=data["num_nodes"],
            )

            conv = create_conv_layer(backend, aggr_type, data["in_channels"], out_channels, bias=False)

            output = conv(features, graph.graph_repr)
            output.sum().backward()
            grad = features.grad.clone()
            features.grad = None
            return output, grad

        (output_dgl, grad_dgl), (output_test, grad_test) = apply_conv("dgl"), apply_conv(backend)

        assert torch.allclose(
            output_dgl, output_test, atol=1e-6, rtol=1e-5
        ), f"MatMul conv ({backend=}, {aggr_type=}) doesn't match DGL's one"

        assert torch.allclose(
            grad_dgl, grad_test, atol=1e-6, rtol=1e-5
        ), f"MatMul conv grad ({backend=}, {aggr_type=}) doesn't match DGL's one"
