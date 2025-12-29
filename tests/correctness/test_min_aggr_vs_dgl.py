import dgl
import dgl.function as fn
import torch

from src.backends.cuda_backend.min_aggr.utils import min_aggr, min_aggr_forward
from src.data.datasets import load_pyg_single_graph


def check_dataset(name: str, device):
    print(f"\n========== DATASET {name} ==========")
    torch.manual_seed(0)
    torch.set_default_device(device)

    sample = load_pyg_single_graph(name=name, graph_backend="csr", root="data", allow_random_split=True)

    # x = sample.x
    F = sample.num_features
    N = sample.num_nodes
    x = torch.randn(N, F, device=device)
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    grad_output = torch.ones_like(x)
    indptr, indices, _ = sample.graph_repr
    src = sample.edge_index[0]
    dst = sample.edge_index[1]
    g = dgl.graph((src, dst), num_nodes=N).to(device)
    out_cuda, argmin = min_aggr_forward(indptr, indices, x)

    x_aggregated = dgl.ops.copy_u_min(g, x1)
    x_aggregated[x_aggregated.isinf()] = 0
    out_dgl = x_aggregated

    max_diff_fwd = (out_cuda - out_dgl).abs().max().item()
    print(f"[{name}] forward: max |diff| = {max_diff_fwd:.3e}")
    mean_diff_fwd = (out_cuda - out_dgl).abs().mean().item()
    print(f"[{name}] forward: meann |diff| = {mean_diff_fwd:.3e}")

    out_cuda2 = min_aggr(indptr, indices, x1)
    out_cuda2.backward(grad_output)
    # loss_cuda = out_cuda2.sum()
    # loss_cuda.backward()
    grad_x_cuda = x1.grad.detach().clone()

    x_aggregated2 = dgl.ops.copy_u_min(g, x2)
    x_aggregated2[x_aggregated2.isinf()] = 0
    x_aggregated2.backward(grad_output)
    # loss_dgl = out_dgl2.sum()
    # loss_dgl.backward()
    grad_x_dgl = x2.grad.detach().clone()

    max_diff_bwd = (grad_x_cuda - grad_x_dgl).abs().max().item()
    print(f"[{name}] backward: max |diff| = {max_diff_bwd:.3e}")
    mean_diff_bwd = (grad_x_cuda - grad_x_dgl).abs().mean().item()
    print(f"[{name}] backward: mean |diff| = {mean_diff_bwd:.3e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ["cora", "citeseer", "pubmed"]:
        check_dataset(name, device)


if __name__ == "__main__":
    main()
