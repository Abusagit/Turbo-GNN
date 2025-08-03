import torch
import cusparse_spmm
from data import generate_random_graph
from cusparse_spmm import csr_SPMM


def benchmark_cusparse_spmm(indptr, indices, feats, num_iters=100):


    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(10):
        out = csr_SPMM(indptr, indices, feats)

        print('warmup')

    for _ in range(num_iters):
        starter.record()
        out = csr_SPMM(indptr, indices, feats)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

    avg_time_ms = sum(timings) / len(timings)
    return avg_time_ms, out

# Example usage:

if __name__ == "__main__":
    
    print(torch.version.cuda)       # CUDA version PyTorch was built with
    print(torch.cuda.is_available()) # If CUDA is usable in PyTorch
    print(torch.cuda.get_device_name(0)) # Your GPU name

    num_nodes, avg_degree, feature_dim = 10000, 20, 64
    device = 'cuda'

    g, indptr, indices, feats = generate_random_graph(num_nodes, avg_degree, feature_dim, device)


    avg_time, out_feats = benchmark_cusparse_spmm(indptr, indices, feats)
    print(f"cuSPARSE SpMM average time: {avg_time:.3f} ms")
