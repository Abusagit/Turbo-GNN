import pytest
import torch
import torch.nn as nn
from dgl import DGLGraph

from src.backends.registry import BackendRegistry


class TestGraphTransformer:
    "Test Graph Transformer correctness"

    def test_correctness_dgl(self):
        backend = BackendRegistry.get_backend("dgl")

        hidden_dim = 16
        num_heads = 4

        # initialize conv
        conv = backend.create_conv("gt", feature_dim=hidden_dim, heads=num_heads)
        conv.v_proj.bias.data = torch.randn(hidden_dim)

        # create test graph
        num_nodes = 6
        node_features = torch.randn(num_nodes, hidden_dim)
        edges = [(1, 0), (2, 0), (3, 0), (1, 4), (2, 5), (1, 5)]
        edges = torch.tensor(edges)
        num_edges = len(edges)

        graph = DGLGraph()
        graph.add_nodes(num_nodes)

        for src, dst in edges:
            graph.add_edges(src, dst)

        out_dgl = conv(node_features, graph)

        # dummy loss to trigger backward pass
        dummy_loss = (out_dgl**2).sum()
        dummy_loss.backward()

        # save gradients for correctness checking

        dgl_grads_q = conv.q_proj.weight.grad.clone()
        dgl_grads_k = conv.k_proj.weight.grad.clone()
        dgl_grads_v = conv.v_proj.weight.grad.clone()
        dgl_grads_q_bias = conv.q_proj.bias.grad.clone()
        dgl_grads_k_bias = conv.k_proj.bias.grad.clone()
        dgl_grads_v_bias = conv.v_proj.bias.grad.clone()

        # zero current gradients

        conv.q_proj.weight.grad.zero_()
        conv.k_proj.weight.grad.zero_()
        conv.v_proj.weight.grad.zero_()
        conv.q_proj.bias.grad.zero_()
        conv.k_proj.bias.grad.zero_()
        conv.v_proj.bias.grad.zero_()

        assert out_dgl.shape == (num_nodes, hidden_dim)

        # calculate output manually

        q = conv.q_proj(node_features).view(num_nodes, num_heads, -1)
        k = conv.k_proj(node_features).view(num_nodes, num_heads, -1)
        v = conv.v_proj(node_features).view(num_nodes, num_heads, -1)

        assert q.shape == (num_nodes, num_heads, hidden_dim // num_heads)
        assert k.shape == (num_nodes, num_heads, hidden_dim // num_heads)
        assert v.shape == (num_nodes, num_heads, hidden_dim // num_heads)

        multiplier = conv.attn_scores_multiplier

        attn_scores = torch.zeros(num_edges, num_heads)

        for i in range(num_edges):
            src, dst = edges[i]
            attn_scores[i] = torch.einsum("hd,hd->h", q[src], k[dst]) * multiplier

        out = torch.zeros(num_nodes, hidden_dim)

        # calculate softmax on edges

        for i in range(num_nodes):
            in_edges_indexes = graph.in_edges(i, form="eid")
            if len(in_edges_indexes) == 0:
                continue
            exp_scores = torch.exp(attn_scores[in_edges_indexes])
            exp_scores = exp_scores / exp_scores.sum(dim=0)

            source_node_values = v[edges[in_edges_indexes, 0]]
            out[i] += torch.einsum("ehd,eh->hd", source_node_values, exp_scores).reshape(-1)

        assert torch.allclose(out, out_dgl, atol=1e-6), "Output mismatch"

        dummy_loss = (out**2).sum()
        dummy_loss.backward()

        # check gradient correctess
        assert torch.allclose(conv.q_proj.weight.grad, dgl_grads_q, atol=1e-6), "Gradient mismatch for q_proj.weight"
        assert torch.allclose(conv.k_proj.weight.grad, dgl_grads_k, atol=1e-6), "Gradient mismatch for k_proj.weight"
        assert torch.allclose(conv.v_proj.weight.grad, dgl_grads_v, atol=1e-6), "Gradient mismatch for v_proj.weight"
        assert torch.allclose(conv.q_proj.bias.grad, dgl_grads_q_bias, atol=1e-6), "Gradient mismatch for q_proj.bias"
        assert torch.allclose(conv.k_proj.bias.grad, dgl_grads_k_bias, atol=1e-6), "Gradient mismatch for k_proj.bias"
        assert torch.allclose(conv.v_proj.bias.grad, dgl_grads_v_bias, atol=1e-6), "Gradient mismatch for v_proj.bias"
