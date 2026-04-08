"""
Reduction aggregation (min/max) correctness: CUDA kernel vs torch_native scatter reference.
"""

import torch

from src.backends.cuda_backend.reduction_aggr.utils import reduction_aggr, reduction_aggr_forward_partitioned
from src.backends.registry import BackendRegistry
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets
from src.data.datasets import load_pyg_single_graph


def partition_nodes(indptr, threshold=100):
    deg = indptr[1:] - indptr[:-1]
    index_dtype = indptr.dtype
    light = torch.nonzero(deg <= threshold, as_tuple=True)[0].to(index_dtype).to(indptr.device)
    heavy = torch.nonzero(deg > threshold, as_tuple=True)[0].to(index_dtype).to(indptr.device)
    return light, heavy


def make_graph_repr(indptr, indices, light, heavy):
    """Wrap raw tensors into AdjacencyForwardBackwardWithNodeBuckets."""
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=indptr,
        forward_indices=indices,
        backward_indptr=indptr,
        backward_indices=indices,
        forward_light_nodes=light,
        forward_heavy_nodes=heavy,
        backward_light_nodes=light,
        backward_heavy_nodes=heavy,
    )


def check_dataset(name: str, device, reduce="min"):
    print(f"\n========== DATASET {name} (reduce={reduce}) ==========")
    torch.manual_seed(0)
    torch.set_default_device(device)

    sample = load_pyg_single_graph(name=name, graph_backend="csr", root="data", allow_random_split=True)

    F = sample.num_features
    N = sample.num_nodes
    x = torch.randn(N, F, device=device)
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    grad_output = torch.ones_like(x)
    indptr, indices, _ = sample.graph_repr
    indptr = indptr.to(device).to(torch.int32)
    indices = indices.to(device).to(torch.int32)
    light, heavy = partition_nodes(indptr)

    # ===== CUDA kernel forward =====
    out_cuda, arg_idx = reduction_aggr_forward_partitioned(indptr, indices, x, light, heavy, 8, 128, reduce=reduce)
    out_cuda[out_cuda.isinf()] = 0

    # ===== torch_native scatter reference =====
    ref_backend = BackendRegistry.get_backend("torch_native")
    aggr_type = "min_aggr" if reduce == "min" else "max_aggr"
    ref_conv = ref_backend.create_conv(aggr_type, feature_dim=F).to(device)
    # Build COO graph from edge_index
    edge_index = sample.edge_index.to(device)
    coo_graph = (edge_index, None, N)
    out_ref = ref_conv(x1, coo_graph)

    max_diff_fwd = (out_cuda - out_ref).abs().max().item()
    print(f"[{name}] forward: max |diff| = {max_diff_fwd:.3e}")
    mean_diff_fwd = (out_cuda - out_ref).abs().mean().item()
    print(f"[{name}] forward: mean |diff| = {mean_diff_fwd:.3e}")

    # ===== backward =====
    graph = make_graph_repr(indptr, indices, light, heavy)
    out_cuda2 = reduction_aggr(graph, x1, reduce=reduce)
    out_cuda2[out_cuda2.isinf()] = 0
    out_cuda2.backward(grad_output)
    grad_x_cuda = x1.grad.detach().clone()

    ref_conv2 = ref_backend.create_conv(aggr_type, feature_dim=F).to(device)
    out_ref2 = ref_conv2(x2, coo_graph)
    out_ref2.backward(grad_output)
    grad_x_ref = x2.grad.detach().clone()

    max_diff_bwd = (grad_x_cuda - grad_x_ref).abs().max().item()
    print(f"[{name}] backward: max |diff| = {max_diff_bwd:.3e}")
    mean_diff_bwd = (grad_x_cuda - grad_x_ref).abs().mean().item()
    print(f"[{name}] backward: mean |diff| = {mean_diff_bwd:.3e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ["cora", "citeseer", "pubmed"]:
        for reduce in ["min", "max"]:
            check_dataset(name, device, reduce=reduce)


if __name__ == "__main__":
    main()
