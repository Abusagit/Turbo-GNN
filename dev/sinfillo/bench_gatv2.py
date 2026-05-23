import argparse
import torch
import numpy as np

import turbo_gnn._C as _C


def make_random_csr_graph(num_nodes: int, avg_degree: int, seed: int = 42):
    rng = np.random.RandomState(seed)

    row_counts = rng.poisson(avg_degree, size=num_nodes).astype(np.int64)
    row_counts = np.clip(row_counts, 1, num_nodes - 1)

    col_lists = []
    for i in range(num_nodes):
        deg = int(row_counts[i])
        neighbors = rng.choice(num_nodes, size=deg, replace=False)
        col_lists.append(np.sort(neighbors))

    row_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        row_ptr[i + 1] = row_ptr[i] + len(col_lists[i])

    col_idx = np.concatenate(col_lists).astype(np.int32)

    row_ptr_t = torch.from_numpy(row_ptr).cuda()
    col_idx_t = torch.from_numpy(col_idx).cuda()

    return row_ptr_t, col_idx_t


def partition_nodes(row_ptr: torch.Tensor, threshold: int = 32):
    degrees = row_ptr[1:] - row_ptr[:-1]
    N = degrees.shape[0]
    light_mask = degrees <= threshold
    light_nodes = torch.arange(N, device=row_ptr.device, dtype=torch.int32)[light_mask]
    heavy_nodes = torch.arange(N, device=row_ptr.device, dtype=torch.int32)[~light_mask]
    return light_nodes, heavy_nodes


def run_forward(l, r, row_ptr, col_idx, attn_vec, negative_slope,
                light_nodes, heavy_nodes,
                light_wpb, heavy_wpb, use_pipeline, num_iters=1):
    for _ in range(num_iters):
        _C.gatv2_forward(
            l, r, row_ptr, col_idx, attn_vec,
            negative_slope,
            light_nodes, heavy_nodes,
            light_wpb, heavy_wpb,
            use_pipeline,
        )


def bench_timing(l, r, row_ptr, col_idx, attn_vec, negative_slope,
                 light_nodes, heavy_nodes,
                 light_wpb, heavy_wpb, use_pipeline,
                 warmup=5, repeats=20):
    run_forward(l, r, row_ptr, col_idx, attn_vec, negative_slope,
                light_nodes, heavy_nodes, light_wpb, heavy_wpb,
                use_pipeline, num_iters=warmup)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _C.gatv2_forward(l, r, row_ptr, col_idx, attn_vec,
                         negative_slope, light_nodes, heavy_nodes,
                         light_wpb, heavy_wpb, use_pipeline)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def main():
    parser = argparse.ArgumentParser(description="GATv2 Forward Kernel Benchmark")
    parser.add_argument("--use-pipeline", action="store_true", help="Enable async pipeline")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable async pipeline (baseline)")
    parser.add_argument("--compare", action="store_true", help="Run both and print timing comparison")

    parser.add_argument("--num-nodes", type=int, default=100_000)
    parser.add_argument("--avg-degree", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64, choices=[32, 64, 128, 256])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--negative-slope", type=float, default=0.2)
    parser.add_argument("--light-wpb", type=int, default=2)
    parser.add_argument("--heavy-wpb", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    N, H, D = args.num_nodes, args.heads, args.head_dim

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"N={N}, avg_degree={args.avg_degree}, H={H}, D={D}, dtype={args.dtype}")
    print(f"light_wpb={args.light_wpb}, heavy_wpb={args.heavy_wpb}")
    print()

    row_ptr, col_idx = make_random_csr_graph(N, args.avg_degree, seed=args.seed)
    light_nodes, heavy_nodes = partition_nodes(row_ptr, threshold=32)
    print(f"Graph: {N} nodes, {col_idx.shape[0]} edges, "
          f"light={light_nodes.shape[0]}, heavy={heavy_nodes.shape[0]}")

    torch.manual_seed(args.seed)
    l = torch.randn(N, H, D, device="cuda", dtype=dtype) * 0.1
    r = torch.randn(N, H, D, device="cuda", dtype=dtype) * 0.1
    attn_vec = torch.randn(H, D, device="cuda", dtype=dtype) * 0.1

    common_args = (l, r, row_ptr, col_idx, attn_vec, args.negative_slope,
               light_nodes, heavy_nodes, args.light_wpb, args.heavy_wpb)

    if args.compare:
        times_base = bench_timing(*common_args, use_pipeline=False,
                                  warmup=args.warmup, repeats=args.repeats)
        times_pipe = bench_timing(*common_args, use_pipeline=True,
                                  warmup=args.warmup, repeats=args.repeats)

        h_base, _ = _C.gatv2_forward(*common_args, False)
        h_pipe, _ = _C.gatv2_forward(*common_args, True)
        max_diff = (h_base.float() - h_pipe.float()).abs().max().item()

        import statistics
        mean_b, std_b = statistics.mean(times_base), statistics.stdev(times_base)
        mean_p, std_p = statistics.mean(times_pipe), statistics.stdev(times_pipe)

        print(f"Baseline:  {mean_b:.4f} ± {std_b:.4f} ms  (min={min(times_base):.4f}, max={max(times_base):.4f})")
        print(f"Pipeline:  {mean_p:.4f} ± {std_p:.4f} ms  (min={min(times_pipe):.4f}, max={max(times_pipe):.4f})")
        print(f"Speedup:   {mean_b / mean_p:.3f}x")
        print(f"Max diff:  {max_diff:.2e}")

    elif args.use_pipeline:
        print("Running with pipeline=True ...")
        run_forward(*common_args, use_pipeline=True, num_iters=args.warmup + args.repeats)
        torch.cuda.synchronize()
        print("Done.")

    elif args.no_pipeline:
        print("Running with pipeline=False ...")
        run_forward(*common_args, use_pipeline=False, num_iters=args.warmup + args.repeats)
        torch.cuda.synchronize()
        print("Done.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
