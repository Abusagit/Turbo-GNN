import pytest
import torch
import torch.nn as nn
from fixtures import (
    create_conv_layer,
    create_graph_sample,
    device,
    karate_like_club_graph,
    random_graph_data,
    set_default_device,
    simple_graph_data,
)

from src.backends.registry import BackendRegistry


class TestGraphTransformer:
    "Test Graph Transformer correctness against manual computation"

    def test_correctness_torch_native(self):
        backend = BackendRegistry.get_backend("torch_native")

        hidden_dim = 16
        num_heads = 4

        # initialize conv
        conv = backend.create_conv("gt", feature_dim=hidden_dim, heads=num_heads)

        # create test graph
        num_nodes = 6
        node_features = torch.randn(num_nodes, hidden_dim)
        edges = torch.tensor([(1, 0), (2, 0), (3, 0), (1, 4), (2, 5), (1, 5)])
        num_edges = len(edges)

        # Build COO graph: (edge_index [2, E], edge_weight, num_nodes)
        edge_index = edges.T.long()  # [2, E]
        graph = (edge_index, None, num_nodes)

        out_ref = conv(node_features, graph)

        # dummy loss to trigger backward pass
        dummy_loss = (out_ref**2).sum()
        dummy_loss.backward()

        # save gradients for correctness checking
        ref_grads_qkv = conv.qkv_proj.weight.grad.clone()
        ref_grads_qkv_bias = conv.qkv_proj.bias.grad.clone()

        # zero current gradients
        conv.qkv_proj.weight.grad.zero_()
        conv.qkv_proj.bias.grad.zero_()

        assert out_ref.shape == (num_nodes, hidden_dim)

        # calculate output manually
        qkv = conv.qkv_proj(nn.functional.layer_norm(node_features, (node_features.shape[-1],)))
        q, k, v = qkv.split(hidden_dim, -1)

        q = q.view(num_nodes, num_heads, -1)
        k = k.view(num_nodes, num_heads, -1)
        v = v.view(num_nodes, num_heads, -1)

        assert q.shape == (num_nodes, num_heads, hidden_dim // num_heads)
        assert k.shape == (num_nodes, num_heads, hidden_dim // num_heads)
        assert v.shape == (num_nodes, num_heads, hidden_dim // num_heads)

        multiplier = conv.attn_scores_multiplier

        attn_scores = torch.zeros(num_edges, num_heads)

        for i in range(num_edges):
            src, dst = edges[i]
            attn_scores[i] = torch.einsum("hd,hd->h", q[src], k[dst]) * multiplier

        out = torch.zeros(num_nodes, hidden_dim)

        # calculate softmax on edges -- find incoming edges per node via edge_index
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        for i in range(num_nodes):
            in_edge_mask = dst_nodes == i
            in_edge_indices = torch.nonzero(in_edge_mask, as_tuple=True)[0]
            if len(in_edge_indices) == 0:
                continue
            exp_scores = torch.exp(attn_scores[in_edge_indices])
            exp_scores = exp_scores / exp_scores.sum(dim=0)

            source_node_values = v[edges[in_edge_indices, 0]]
            out[i] += torch.einsum("ehd,eh->hd", source_node_values, exp_scores).reshape(-1)

        assert torch.allclose(out, out_ref, atol=1e-6), "Output mismatch"

        dummy_loss = (out**2).sum()
        dummy_loss.backward()

        # check gradient correctness
        assert torch.allclose(
            conv.qkv_proj.weight.grad, ref_grads_qkv, atol=1e-6
        ), "Gradient mismatch for qkv_proj.weight"
        assert torch.allclose(
            conv.qkv_proj.bias.grad, ref_grads_qkv_bias, atol=1e-6
        ), "Gradient mismatch for qkv_proj.bias"
