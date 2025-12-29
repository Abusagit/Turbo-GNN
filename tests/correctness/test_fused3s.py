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

    def test_mean_aggregation_matches_dgl(self, karate_like_club_graph, create_graph_sample, create_conv_layer):
        """Test that cugraph mean aggregation matches DGL's copy_u_mean."""
        try:
            import dgl
            import dgl.ops as dgl_ops
        except ImportError:
            pytest.skip("DGL not installed - cannot verify correctness")

        data = karate_like_club_graph
        features = data["features"]

        graph_sample = create_graph_sample(
            edge_index=data["edge_index"],
            features=features,
            backend="f3s",
            num_nodes=data["num_nodes"],
        )

        conv = create_conv_layer("f3s", "graph_transformer", feature_dim=data["in_channels"], bias=False)
        cugraph_output = conv(features, graph_sample.graph_repr)
