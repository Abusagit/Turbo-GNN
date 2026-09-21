# Occupancy audit

Static analysis of `turbo_gnn/_C*.so` at `3f761b0` on 2026-08-23. Analysis only — no changes made.

## Method, and why it is static

`ncu` cannot read performance counters on this machine (`ERR_NVGPUCTRPERM`), and a timing
run needs an idle GPU, which is currently scarce. Register allocation is fixed at compile
time though, so `cuobjdump -res-usage` answers "how many warps can be resident" from the
binary alone. Dynamic shared memory is not in the binary — these kernels size it at the
launch site — so that half is computed from the launch-site formulas in `csrc/`.

Reproduce with `python scripts/occupancy_audit.py` (full output in `audit.txt`).

A100 (sm_80) limits used: 65,536 registers/SM allocated per warp at 256-register
granularity, 64 warps/SM, 32 blocks/SM, 164 KiB shared/SM, 48 KiB per block without an
explicit opt-in.

## Headline

Across the **6,780** instantiations whose block size is recoverable from their template
arguments (of 8,152 total):

- median occupancy **50%**, mean 60%
- **314** instantiations below 50%; 647 reach 100%
- the binding resource is **registers in 96% of cases** (6,512 of 6,780); the rest hit the
  warps/blocks-per-SM ceiling. Shared memory binds essentially nowhere.

## Per kernel

`occ` is at the block size the instantiation was compiled for. `spill` counts
instantiations with a non-zero stack frame.

| kernel | instantiations | median REG | max REG | spilling | median occ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GATv2Backward_AL` | 1,152 | 64 | 80 | 185 | 50% |
| `GATv2Forward_Kernel` | 1,152 | 56 | 80 | 149 | 50% |
| `GraphAttentionForward_CSR_MH_v2_D` | 1,152 | 40 | 96 | 111 | 50% |
| `graph_attn_backward_fwd_csr_undirected_kernel_D` | 192 | 64 | 90 | 5 | 50% |
| `GATv2Backward_ALR_Undirected` | 192 | 57 | 79 | 0 | 50% |
| `GATv2Backward_G_Kernel` | 192 | 42 | 64 | 0 | 50% |
| `compute_D_mh_kernel_D` | 48 | 48 | 64 | 0 | 50% |
| `reduction_aggr_forward_light_kernel_1d` | 672 | 48 | 72 | 128 | 56% |
| `graph_attn_backward_csrT_kernel_D` | 1,152 | 48 | 80 | 209 | 62% |
| `GATv2Backward_R` | 1,152 | 48 | 72 | 151 | 62% |
| `reduction_aggr_backward_typed` | 336 | 24 | 28 | 0 | 100% |
| `reduction_aggr_forward_heavy_kernel` | 588 | 42 | 57 | 168 | — |
| `ReduceGradAKernel` | 21 | 20 | 20 | 0 | — |
| `reduction_aggr_forward_heavy_kernel_2d` | 96 | 56 | 70 | 0 | — |
| `unpack_results_kernel` | 42 | 16 | 18 | 0 | — |
| `compute_edge_weights_kernel` | 4 | 20 | 21 | 0 | — |
| `compute_degrees_kernel` | 4 | 16 | 16 | 0 | — |

## Finding 1 — nearly everything is register-bound at 50%

Median occupancy is 50%, and registers are the constraint in 96% of instantiations. The
attention backward kernels are the worst: `GATv2Backward_AL` and
`graph_attn_backward_fwd_csr_undirected_kernel_D` sit at a median of 64 registers and reach
80–90 at their largest, which caps them at half the SM.

## Finding 2 — 1,106 instantiations spill to local memory

| kernel | spilling | median stack | max stack |
| --- | ---: | ---: | ---: |
| `graph_attn_backward_csrT_kernel_D` | 209 | 8 B | 32 B |
| `GATv2Backward_AL` | 185 | 16 B | 48 B |
| `reduction_aggr_forward_heavy_kernel` | 168 | 12 B | 48 B |
| `GATv2Backward_R` | 151 | 16 B | 32 B |
| `GATv2Forward_Kernel` | 149 | 16 B | 24 B |
| `reduction_aggr_forward_light_kernel_1d` | 128 | 28 B | 152 B |
| `GraphAttentionForward_CSR_MH_v2_D` | 111 | 8 B | 56 B |
| `graph_attn_backward_fwd_csr_undirected_kernel_D` | 5 | 8 B | 8 B |

**1,106 of 8,152** instantiations carry a stack frame, i.e. registers that did not
fit and were pushed to local memory. Local memory is DRAM-backed and cached, so a spill in
an inner loop is far more expensive than the register it saved.

Some of this is self-inflicted and recent: the `__launch_bounds__
minBlocksPerMultiprocessor` added to GT forward to fix its occupancy cliff bought registers
back at the cost of spills in the `dynamic` instantiations (16 B at head dim 128, 56 B at
256). That trade was measured as net positive for GT, but it shows the two are coupled and
that pushing registers down further is not free.

## Finding 3 — a 1-warp block can never exceed 50% occupancy

This is structural, not a tuning matter. An A100 SM holds at most 32 blocks. A block of one
warp therefore tops out at 32 resident warps against a maximum of 64 — 50% — no matter how
few registers it uses. Two warps per block is the smallest size that can reach 100%.

It matters because several defaults are 1-warp blocks: GATv2's `forward_light_warps=1` and
`backward_light_warps=1`, and the graph transformer's `backward_light_warps=1`. Those light
buckets are the ones handling ~99% of nodes on most graphs.

The autotuner does search `[1, 2, 4]` for these, so it can escape the ceiling — but only if
the extra warps do not cost more than the occupancy gains, which the earlier benchmarks
suggest is graph-dependent.

## Finding 4 — two configurations cannot launch at all

Computing the launch-site shared-memory formulas over every (warps, head dim) combination,
shared memory is nowhere the binding constraint — except two combinations that exceed the
48 KiB per-block default outright:

| kernel | warps | head dim | shared memory |
| --- | ---: | ---: | ---: |
| `GATv2Backward_AL` | 32 | 256 | 66.1 KiB |
| `graph_attn_backward_csrT_kernel_D` | 32 | 256 | 66.0 KiB |

There is no `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
anywhere in `csrc/`, so the cap is never raised and both fail at launch with `invalid
argument`. Both are reachable: head dim comes from `MakeIntVariant<32, 64, 128, 256>` and
the heavy warp count from `MakeIntVariant<8, 16, 32>`. This extends finding (b) in
`KERNEL_ISSUES.md`, which recorded only the GATv2 case — the graph transformer has the same
problem. An A100 allows up to 163 KiB per block with the opt-in, so both would fit.

## What this suggests, untested

Listed for the record; **none of it has been implemented or measured**.

1. **Per-instantiation `__launch_bounds__` tuned to just below the spill threshold.** The
   1,106 spilling instantiations are the clearest waste, and the GT experience shows the
   knob works — but also that it can create the spills it is meant to avoid.
2. **Raise the shared-memory cap** with `cudaFuncSetAttribute`. Fixes the two launch
   failures outright, and would let larger tiles trade shared memory for registers.
3. **Avoid 1-warp blocks in the light buckets**, which cannot exceed 50% by construction.
4. **Shrink live state in the hot loops**, the only route that lowers registers without
   trading them for spills. The scheduler objects hold 2 pointers and 3 ints across the
   whole loop body in the `dynamic` policy, which is where its register premium comes from.

Occupancy is not automatically throughput: these kernels are frequently memory-bound, and
`scripts/scheduler_headroom.py` showed ogbn-proteins already exceeding what HBM could
deliver. Raising occupancy on a kernel that is waiting on memory buys nothing. Any of the
above needs measuring, not assuming.
