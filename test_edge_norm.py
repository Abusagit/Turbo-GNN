# test_edge_norm.py
import torch
from torch.utils.cpp_extension import load

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
    E = src.numel()
    device = src.device
    # Sort by (dst, src) to make CSR canonical
    order = torch.lexsort([src, dst]) if hasattr(torch, "lexsort") else torch.argsort(dst * (num_nodes + 1) + src)
    dst_s = dst[order]
    src_s = src[order]

    counts = torch.bincount(dst_s, minlength=num_nodes)
    indptr = torch.empty(num_nodes + 1, dtype=torch.int32, device=device)
    indptr[0] = 0
    indptr[1:] = counts.cumsum(0).to(torch.int32)
    indices = src_s.to(torch.int32)
    return indptr.contiguous(), indices.contiguous(), order


def cpu_degrees(num_nodes: int, src: torch.Tensor, dst: torch.Tensor):
    in_deg  = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
    out_deg = torch.bincount(src, minlength=num_nodes).to(torch.float32)
    return in_deg, out_deg

def cpu_norm_weights(norm, w, src, dst, in_deg, out_deg):
    if w.numel() == 0:
        w = torch.ones_like(src, dtype=torch.float32)
    w = w.to(torch.float32)
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
    indptr, indices, order = edges_to_transposed_csr(num_nodes, src, dst)
    E = indices.numel()

    # Degrees: CPU reference
    in_deg_cpu, out_deg_cpu = cpu_degrees(num_nodes, src, dst)

    # GPU degrees
    in_deg_gpu  = torch.empty(num_nodes, dtype=torch.float32, device=device)
    out_deg_gpu = torch.empty(num_nodes, dtype=torch.float32, device=device)
    edge_norm.compute_degrees(indptr, indices, in_deg_gpu, out_deg_gpu)

    assert torch.allclose(in_deg_gpu,  in_deg_cpu.to(device),  atol=atol, rtol=rtol)
    assert torch.allclose(out_deg_gpu, out_deg_cpu.to(device), atol=atol, rtol=rtol)

    # Weights
    if w is None:
        w = torch.tensor([], dtype=torch.float32, device=device)  # signals "no weights"
    else:
        # Reorder weights to CSR order (dst,src) to match kernel write order
        w = w.to(device).to(torch.float32)
        w = w[order]

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
        if w_cpu.numel() == 0:
            w_cpu = torch.ones_like(indices, dtype=torch.float32, device=device)
            if name == "RIGHT":
                w_cpu = w_cpu / torch.clamp(in_deg_gpu[dst], min=1.0)
            elif name == "LEFT":
                w_cpu = w_cpu / torch.clamp(out_deg_gpu[src], min=1.0)
            elif name == "BOTH":
                w_cpu = w_cpu / torch.sqrt(torch.clamp(out_deg_gpu[src], min=1.0) *
                                           torch.clamp(in_deg_gpu[dst], min=1.0))
        # reorder baseline to CSR order
        w_cpu_csr = w_cpu[order]

        assert torch.allclose(normed_gpu, w_cpu_csr, atol=atol, rtol=rtol), f"Mismatch for {name}"

def test_small_cases():
    # 5 nodes, mixed structure incl. self-loop and isolated node
    num_nodes = 5
    src = torch.tensor([0,0,1,1,3,4,4], dtype=torch.int64)  # edges: 0->1,0->2,1->2,1->1,3->1,4->0,4->4
    dst = torch.tensor([1,2,2,1,1,0,4], dtype=torch.int64)

    # Unweighted
    run_once(num_nodes, src, dst)

    # Weighted
    w = torch.tensor([1.5, 2.0, 0.5, 3.0, 0.7, 4.2, 1.1], dtype=torch.float32)
    run_once(num_nodes, src, dst, w=w)

def test_random():
    torch.manual_seed(0)
    num_nodes = 100
    density = 0.04
    device = torch.device("cuda")
    # Sample E
    E = int(num_nodes * num_nodes * density)
    src = torch.randint(0, num_nodes, (E,), dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (E,), dtype=torch.int64)
    # Add some self-loops and duplicates
    src[:10] = torch.arange(10)
    dst[:10] = torch.arange(10)
    src[10:20] = 5
    dst[10:20] = 5
    w = torch.rand(E, dtype=torch.float32)

    run_once(num_nodes, src, dst)
    run_once(num_nodes, src, dst, w=w)

if __name__ == "__main__":
    test_small_cases()
    test_random()
    print("All tests passed ✅")
