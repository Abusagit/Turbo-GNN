# Kernel issues found while auditing `csrc/` for the persistent-scheduler work

Found while mapping which kernels launch one thread block per output node, in preparation
for the templated scheduler. All pre-date that work; this file records them so the scheduler
work does not get blamed for pre-existing behaviour, and so they can be triaged on their own
merits.

**(a), (d) and (f) have since been fixed** — (a) stopped being latent the moment the persistent loop
landed and became a hard deadlock; (d) became load-bearing when the light and heavy buckets
started launching on separate streams; (f) was a one-line fence with no performance cost.
**(b), (c) and (e) are documented only, not fixed.**

Branch: `scheduler`, at `5a883d2`. Verified on an A100-SXM4-80GB, CUDA 13.2, PyTorch 2.13.

Each finding states how strong the evidence is.

---

## (a) `__syncthreads()` inside a `threadIdx.x`-dependent loop — REAL, and now FIXED

**Where:** `csrc/reduction/reduction_aggr.cu`, in `reduction_aggr_forward_heavy_kernel_2d`.

The feature loop was written as

```cpp
for (size_t fv = fid; fv < d_vec; fv += F_BLOCK) {   // fid = threadIdx.x, F_BLOCK = blockDim.x
    ...
    __syncthreads();          // x3 per iteration
}
```

The trip count is `ceil((d_vec - fid) / F_BLOCK)`, which differs across threads whenever
`d_vec` is not a multiple of `F_BLOCK`, and is `0` for threads with `fid >= d_vec`. Threads
therefore execute *different numbers of `__syncthreads()`* — undefined behaviour.

**This is the one finding that changed status during the scheduler work, in both directions.**

*First assessment (wrong): "not reproducible."* Two configurations built specifically to force
divergence — whole-warp (`d_vec=32, F_BLOCK=64`) and intra-warp (`d_vec=48, F_BLOCK=32`) —
both ran clean under `compute-sanitizer --tool synccheck`. nvcc was evidently normalising the
loop's trip count around the barrier, so the emitted SASS had every thread reach every one.

*What actually happened:* wrapping the body in the scheduler's outer per-node loop defeated
that normalisation, and the kernel **deadlocked**. It presented as the whole test suite
hanging at 63% with the GPU stuck in `cudaStreamSynchronize`; bisecting reached
`tests/correctness/test_reduction_aggr_precision.py`, where the `use_2d_kernel` subset hung
indefinitely while the non-2D subset passed in 6 s. It also silently hung
`test_autotune_integration.py`, because `forward_use_2d_kernel` is one of that file's swept
tunables.

**Fixed** (this was a prerequisite for the scheduler work, not optional): the loop now runs a
block-uniform `n_feature_iters = ceil_div(d_vec, F_BLOCK)` times with the body predicated on
`active = fv < d_vec`. Inactive lanes still publish `(identity, INVALID)` to shared memory so
the cross-tile tree reduction never reads uninitialised memory, and the final output write is
guarded so they never write past `d_vec`.

After the fix: 200 previously-hanging tests pass in 31 s, all four scheduler policies are
bit-exact on that kernel, and `compute-sanitizer` reports 0 sync errors and 0 race hazards
across `d_vec % F_BLOCK != 0` shapes.

**The lesson worth keeping:** a clean `synccheck` run did not mean the code was correct. It
meant the compiler happened to be hiding a programming-model violation, and a later,
unrelated change removed the accident that was hiding it.

---

## (b) `GATv2Backward_AL` cannot launch at `D_CONST=256, W=32` — confirmed, reproducible

**Where:** shared-memory size at `csrc/gatv2/gatv2_kernel.cu:359`.

```cpp
size_t sh_al = 2 * DC * sizeof(cuda_t) + W * 2 * DC * sizeof(float) + (W + 1) * sizeof(float);
```

At `DC=256, W=32` that is **67 716 B (66.1 KiB)** in fp32 and 66 692 B in fp16, against the
48 KiB default dynamic shared-memory limit. There is no `cudaFuncSetAttribute(...,
cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` anywhere in `csrc/`, so the cap is never
raised.

Exactly one combination in the instantiated space overflows:

| `DC` | `W=8` | `W=16` | `W=32` |
| --- | --- | --- | --- |
| 128 | 9.0 KiB | 17.1 KiB | 33.1 KiB |
| 256 | 18.0 KiB | 34.1 KiB | **66.1 KiB — over** |

It is reachable: head dim comes from `MakeIntVariant<32, 64, 128, 256>(D)` and the backward
heavy path from `MakeIntVariant<8, 16, 32>(heavy_warps_per_block)`.

**Reproduced:**

```python
out = gatv2_aggr(g, x_left, x_right, attn, 0.2, backward_heavy_warps=32)  # N=4000, H=1, D=256
out.backward(grad)
# backward_heavy_warps=16 -> OK
# backward_heavy_warps=32 -> AcceleratorError: CUDA error: invalid argument
```

`invalid argument` at launch is the standard symptom of requesting more dynamic shared
memory than the device permits. Fix is either to raise the cap with `cudaFuncSetAttribute`
(A100 allows up to 164 KiB/SM) or to stop instantiating that combination.

---

## (c) Hardcoded `max_degree = 131070` over-provisions `gridDim.y` — confirmed, test-path only

**Where:** `src/backends/cuda_backend/reduction_aggr/utils.py:30`, passed as the `max_degree`
argument of the shim wrapper.

The packed-heavy kernel sizes its grid from it (`csrc/reduction/reduction_aggr.cu:558`):

```cpp
dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);
```

With `max_degree = 131070` and the default `EDGES_PER_BLOCK = 128`, `gridDim.y = 1024`
regardless of the actual graph. On a graph whose true maximum degree is, say, 27, one
`blockIdx.y` does the work and the other 1023 immediately hit the early-out at
`reduction_aggr.cu:157`:

```cpp
if (chunk_start >= row_end) [[unlikely]] { return; }
```

This is the **default** heavy path — `use_2d_kernel` defaults to `false`
(`csrc/kernels.cuh:21`) — so the waste is on unless the caller opts out.

**Blast radius is narrower than it looks.** The production path is unaffected:
`turbo_gnn/ops.py:70` passes the real `graph.max_degree`. Only the `src/backends` shim
hardcodes it, and its callers are all tests and benchmarks —
`tests/correctness/test_min_aggr_2d_kernel.py`, `test_reduction_aggr_precision.py`,
`test_reduction_aggr_vs_dgl.py`, and `tests/performance/min_aggr_benchmarks.py`. The
practical consequence is that `min_aggr_benchmarks.py` reports the packed path as slower
than it is.

---

## (d) Two GATv2 entry points ignore the current CUDA stream — REAL, and now FIXED

**Where:** `csrc/gatv2/gatv2_kernel.cu:189` and `:331`.

```cpp
cudaStream_t stream = 0;     // default stream, not the caller's
```

and both `ReduceGradAKernel` launches omit the stream argument entirely
(`gatv2_kernel.cu:48` and `:426`):

```cpp
ReduceGradAKernel<CHUNK, cuda_t><<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(...)
```

The GT path shows the intended pattern (`csrc/gt/graph_transformer.cu:22`):

```cpp
at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(Q.device().index());
```

Benign while everything runs on the default stream; incorrect under a non-default stream,
and a hazard for CUDA-graph capture or multi-stream overlap.

**Fixed**, because the concurrent light/heavy bucket work made it load-bearing rather than
theoretical: launching on stream 0 serialises against every other stream, so overlapping the
two bucket kernels would have been a silent no-op for GATv2 while appearing to work. Both entry
points now take `at::cuda::getCurrentCUDAStream(l.device().index())`.

---

## (e) Dead `GATv2Backward_CSR_Impl_UNUSED` has stale template arguments — confirmed by inspection

**Where:** `csrc/gatv2/gatv2_kernel.cu:61-122`.

It calls `GATv2Backward_AL<D_CONST, cuda_t, index_t>` and `GATv2Backward_R<...>` with the
old three-argument template list. The current signatures take
`<WARPS_PER_BLOCK, D_CONST, cuda_t, index_t, accum_t>`
(`csrc/gatv2/gatv2_backward.cu:10` and `:234`). It compiles only because the function
template is never instantiated.

Harmless as-is, but it is misleading as a reference when refactoring these kernels — anyone
copying its call shape will write code that does not compile. Either delete it or update it.

---

## (f) Warp-synchronous shared-memory hazard in GT forward — REAL, and now FIXED

**Where:** `csrc/gt/gt_forward.cu`, the cross-warp reduction at `:204-246`.

Inside `if (block_neighbor_id == 0)` (warp 0 only), lane 0 rewrites the shared `neighbor_sum`
array in place:

```cpp
if (lane_id == 0) {
    for (size_t w = 0; w < neighbor_block_size; ++w) {
        neighbor_sum[w] = AccumOps::exp(neighbor_max[w] - global_max);   // :217
    }
    ...
}
inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);
...
combined[ep] = AccumOps::fma(neighbor_sum[w], neighbor_out[...], combined[ep]);   // :240, all lanes
```

All 32 lanes then read `neighbor_sum` at `:240`. The only thing between the write and the reads
is the `__shfl_sync`, so the code is relying on a shuffle for *memory* ordering rather than on
`__syncwarp()`. Under Volta's independent thread scheduling that is the classic
warp-synchronous-programming assumption the CUDA memory model no longer guarantees.

`compute-sanitizer --tool racecheck` reports it: **7 errors, 39 warnings, 169,312 hazards** on a
25-iteration GT forward workload, pointing at `gt_forward.cu:240`.

**It is pre-existing, and unrelated to the scheduler or the stream work.** The reported
instantiation is `ScheduleKind = 0` (`one_per_block`, the historical launch); `git diff
d3033b4..HEAD -- csrc/gt/gt_forward.cu` shows none of the barrier or `neighbor_sum` logic was
touched by either change; and the hazard is inside a single block's shared memory, which
concurrent bucket streams cannot affect since shared memory is per-block.

It has presumably been benign in practice — the results are correct on every test — because the
write and the reads are all in warp 0 and nvcc has been keeping the lanes converged. That is the
same kind of accident that hid finding (a) until an unrelated change removed it. The fix is one
line: `__syncwarp()` after the `lane_id == 0` block, before the reads at `:240`.

**Fixed.** `__syncwarp()` after the `lane_id == 0` block, before the reads. racecheck goes from
169,312 hazards to **0**, register allocation is byte-identical so occupancy is unaffected, and
768 matched GT configurations re-measured before and after show a **median of 1.0003x** — no
performance change either way. Details in `reports/gt-syncwarp/REPORT.md`.

---

## (g) One unreproduced illegal memory access in the GT backward path — open

**Where:** unknown. Seen once, on `ogbn-arxiv` / `gt` / head dim 128 / backward, during a
24-configuration sweep (4 schedules x 2 bucket launches x 3 node orders) inside a single
process:

```
CUDA warning: an illegal memory access was encountered (function ~CUDAEvent)
```

**It has not reproduced.** Re-running the identical command completed cleanly; all eight
(schedule, bucket_launch) combinations pass individually with the same replayed kernel
parameters; `compute-sanitizer --tool memcheck` over the whole 24-point sweep reports **0
errors**; and 13 further cells of the same matrix have run without recurrence.

Recorded rather than dismissed because an illegal access is never noise. Two candidate
explanations, neither confirmed:

* something stateful across sweep points -- each point builds a new reordered/repartitioned
  graph while the previous point's retained autograd graph may still reference the old one;
* a rare race in the backward path, of the same family as (f) but not yet located. (f) was
  itself invisible to normal runs and only showed up under `racecheck`, so a second one is
  plausible.

Worth a targeted hunt if it recurs: run the sweep repeatedly under `memcheck` with
`CUDA_LAUNCH_BLOCKING=1`, which turns an asynchronous report into one attributable to a
specific launch.

---

## (h) Every convolution serialises two DRAM round-trips per edge — measured, unfixed

Counter-measured profiling (`reports/roofline/`, hardware counters via `sudo ncu`) puts these
kernels at **24% of peak HBM bandwidth forward and 29% backward**; 4 of 192 configurations reach
90%. They are not bandwidth-bound.

The cause is visible in the source. A neighbour's feature row cannot be addressed until its
index has arrived:

```cpp
const index_t src = edge_idx[eid];                            // DRAM round-trip 1
const auto val    = Tile::read(&X[static_cast<size_t>(src) * d], fv);  // round-trip 2, depends on src
```

`reduction_aggr.cu:56`, and the same shape at `gt_forward.cu:143-147`, where GT and GATv2
additionally interpose a `warp_reduce_sum` shuffle chain and an online-softmax update *between*
the two loads. Only one neighbour is ever in flight per warp.

Three things this is **not**: load width is already optimal (`SelectTW`, `tile.cuh:66`, picks
128-bit loads at both D=128 and D=256); feature rows are contiguous so coalescing is fine; and
capacity is adequate — Little's law puts the requirement at ~10 KB in flight per SM to sustain
1.75 TB/s at ~600 ns, which ~32 resident warps × 512 B already clears. The shortfall is **duty
cycle**: warps spend most cycles with no request outstanding, blocked behind the index load, the
shuffle, and barriers. That predicts ~25% of peak, and 24% is measured.

A batched prefetch was written and reverted unmeasured (no free GPU at the time). It fetches a
run of contiguous indices, then issues that batch's feature loads back-to-back so several
overlap; comparison order is preserved within and across batches, so argmin tie-breaking stays
bit-identical. It compiles clean. Static register cost from `nvcc --resource-usage` over the 672
recoverable `reduction_aggr_forward_light_kernel_1d` instantiations:

| prefetch depth | median REG | mean REG | mean occupancy |
| ---: | ---: | ---: | ---: |
| 1 (today) | 48 | 47.8 | 69.4% |
| 2 | 48 | 49.2 | 67.2% |
| **4** | **50** | **50.1** | **66.0%** |
| 8 | 52 | 51.7 | 63.9% |

Depth 4 buys 4x memory-level parallelism for 3.4 points of mean occupancy; depth 8 starts
costing real occupancy. **Never benchmarked** — the trade needs measuring, not assuming.

The larger follow-up is `cp.async` / `__pipeline_memcpy_async` double-buffering. `sm_80` is
already the build target (`setup.py:189`) and async copy appears **nowhere** in `csrc/`. It
supplies the same memory-level parallelism without consuming registers, which is the constraint
that otherwise caps the prefetch approach — registers bind occupancy in 96% of instantiations
(`reports/occupancy/REPORT.md`).

---

## (i) One-warp blocks cap occupancy at 50% structurally — confirmed by inspection

The light-bucket kernels launch 32-thread blocks: `ncu` reports `GATv2Forward_Kernel` at block
`(32,1,1)` across 167,628 blocks on ogbn-arxiv. On A100 an SM hosts at most 32 blocks, so
32 blocks x 1 warp = **32 of 64 warps, a hard 50% ceiling** that no register tuning can lift.
This is consistent with the occupancy audit's median of 50%, and it compounds (h): fewer
resident warps means fewer loads in flight.

Raising it requires more warps per block doing useful work. For small `d` the feature dimension
does not supply enough parallelism for a second warp (at D=128 with TW=4 there are only 32
vector elements), so the warps would have to come from processing **several nodes per block** —
which the persistent scheduler already makes expressible.

## Note on evidence

(a) is reproduced, understood and fixed; the write-up above keeps the wrong first assessment
on the record because the way it was wrong is the useful part. (b) is reproduced from Python
and fails deterministically. (c), (d) and (e) are plain readings of the source with the call
sites enumerated. (h) rests on counter-measured DRAM traffic plus static register counts;
its proposed fix is explicitly unmeasured and labelled as such. (i) is read off an `ncu`
launch record and the A100 blocks-per-SM limit.
