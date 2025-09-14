# test_edge_norm.py
import torch
from torch.utils.cpp_extension import load
from scipy import sparse as sp

edge_norm = load(
    name="edge_norm",
    sources=["edge_norm_kernels.cu", "edge_norm_bindings.cpp"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

def edges_to_transposed_csr(num_nodes: int, src: torch.Tensor, dst: torch.Tensor):
    """
    Build CSR for A^T:
      - rows = dst
      - indices = src (neighbors listed per dst)
    Returns (indptr[int32], indices[int32]) and a permutation that maps E to CSR order.
    """
    num_edges = torch.ones(len(src)).numpy()
    destination_nodes_np = dst.cpu().numpy()
    source_nodes_np = src.cpu().numpy()

    adj_transposed = sp.csr_matrix(
        (num_edges, (destination_nodes_np, source_nodes_np)),
        shape=(num_nodes, num_nodes)
    )

    indptr = torch.from_numpy(adj_transposed.indptr).int().cuda()
    indices = torch.from_numpy(adj_transposed.indices).int().cuda()
    return indptr, indices


def cpu_degrees(num_nodes: int, src: torch.Tensor, dst: torch.Tensor):
    in_deg  = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
    out_deg = torch.bincount(src, minlength=num_nodes).to(torch.float32)
    return in_deg, out_deg

def cpu_norm_weights(norm, w, src, dst, in_deg, out_deg):

    if w.numel() == 0:
        w = torch.ones_like(src, dtype=torch.float32)

    w = w.to(torch.float32)
    weight = w

    if norm == "NONE":
        return w
    if norm == "RIGHT":
        return w / torch.clamp(in_deg[dst], min=1.0)
    if norm == "LEFT":
        return w / torch.clamp(out_deg[src], min=1.0)
    if norm == "BOTH":
        return w / torch.sqrt(torch.clamp(out_deg[src], min=1.0) * torch.clamp(in_deg[dst], min=1.0))
    raise ValueError(norm)


def run_once(num_nodes, src, dst, w=None, atol=1e-6, rtol=1e-6):
    device = torch.device("cuda")
    src = src.to(torch.int64).to(device)
    dst = dst.to(torch.int64).to(device)

    # Build TRANSPOSED CSR
    indptr, indices = edges_to_transposed_csr(num_nodes, src, dst)
    E = indices.numel()

    # Degrees: CPU reference
    in_deg_cpu, out_deg_cpu = cpu_degrees(num_nodes, src, dst)

    # GPU degrees
    in_deg_gpu  = torch.empty(num_nodes, dtype=torch.float32, device=device)
    out_deg_gpu = torch.empty(num_nodes, dtype=torch.float32, device=device)
    edge_norm.compute_degrees(indptr, indices, in_deg_gpu, out_deg_gpu)


    torch.testing.assert_close(in_deg_gpu,  in_deg_cpu.to(device),  atol=atol, rtol=rtol)
    torch.testing.assert_close(out_deg_gpu, out_deg_cpu.to(device), atol=atol, rtol=rtol)


    # Weights
    if w is None:
        w = torch.tensor([], dtype=torch.float32, device=device)  # signals "no weights"
    else:
        # Reorder weights to CSR order (dst,src) to match kernel write order
        w = w.to(device).to(torch.float32)

    for name, enumv in [("NONE", edge_norm.NormType.NONE),
                        ("RIGHT", edge_norm.NormType.RIGHT),
                        ("LEFT",  edge_norm.NormType.LEFT),
                        ("BOTH",  edge_norm.NormType.BOTH)]:
        normed_gpu = torch.empty(E, dtype=torch.float32, device=device)
        edge_norm.compute_normalized_weights(indptr, indices, w, normed_gpu,
                                             in_deg_gpu, out_deg_gpu, enumv)

        # CPU baseline in original edge order, then reorder to CSR order
        w_cpu = cpu_norm_weights(name,
                                 w if w.numel() > 0 else torch.tensor([], device=device),
                                 src, dst, in_deg_cpu.to(device), out_deg_cpu.to(device))
        
    

        print(f"NORM={name}, {normed_gpu=}, {w_cpu=}")
        assert torch.allclose(normed_gpu, w_cpu, atol=atol, rtol=rtol), f"Mismatch for {name}"

def test_small_cases():
    # 5 nodes, mixed structure incl. self-loop and isolated node
    # num_nodes = 5
    # src = torch.tensor([0,0,1,1,3,4,4], dtype=torch.int64)  # edges: 0->1,0->2,1->2,1->1,3->1,4->0,4->4
    # dst = torch.tensor([1,2,2,1,1,0,4], dtype=torch.int64)


    num_nodes = 3

    src = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    dst = torch.tensor([1, 0, 2, 2], dtype=torch.int64)

    # Unweighted
    run_once(num_nodes, src, dst)

    # Weighted
    w = torch.tensor([1.5, 2.0, 0.5, 3.0], dtype=torch.float32)

    run_once(num_nodes, src, dst, w=w)

def test_random():
    torch.manual_seed(0)
    num_nodes = 10
    density = 0.04
    device = torch.device("cuda")
    # Sample E
    E = int(num_nodes * num_nodes * density)
    src = torch.randint(0, num_nodes, (E,), dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (E,), dtype=torch.int64)

    # Add some self-loops and duplicates
    w = torch.rand(E, dtype=torch.float32)

    # sort by row indices first then by col indices:
    order = list(range(E))
    order.sort(key=lambda i: (dst[i], src[i]))  # we need to rearrange them so that for A^T receievers would be adjacent in memory

    src = src[order]
    dst = dst[order]
    w = w[order]


    run_once(num_nodes, src, dst)
    run_once(num_nodes, src, dst, w=w)


def test_random_with_dgl():
    torch.manual_seed(0)
    num_nodes = 100
    density = 0.04
    device = torch.device("cuda")
    # Sample E
    E = int(num_nodes * num_nodes * density)
    src = torch.randint(0, num_nodes, (E,), dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (E,), dtype=torch.int64)

    # Add some self-loops and duplicates
    w = torch.rand(E, dtype=torch.float32)

    # sort by row indices first then by col indices:
    order = list(range(E))
    order.sort(key=lambda i: (dst[i], src[i]))  # we need to rearrange them so that for A^T receievers would be adjacent in memory

    src = src[order]
    dst = dst[order]
    w = w[order]

    run_once(num_nodes, src, dst)
    run_once(num_nodes, src, dst, w=w)

if __name__ == "__main__":
    test_small_cases()
    print("Small case Passed ✅")


    test_random()
    print("All tests passed ✅")
