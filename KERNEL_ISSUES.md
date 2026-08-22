# Kernel issues found while auditing `csrc/` for the persistent-scheduler work

Found while mapping which kernels launch one thread block per output node, in preparation
for the templated scheduler. All pre-date that work; this file records them so the scheduler
work does not get blamed for pre-existing behaviour, and so they can be triaged on their own
merits.

**(a) has since been fixed** — it stopped being latent the moment the persistent loop landed
and became a hard deadlock, so it had to be. **(b) through (e) are documented only, not
fixed.**

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

## (d) Two GATv2 entry points ignore the current CUDA stream — confirmed by inspection

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

## Note on evidence

(a) is reproduced, understood and fixed; the write-up above keeps the wrong first assessment
on the record because the way it was wrong is the useful part. (b) is reproduced from Python
and fails deterministically. (c), (d) and (e) are plain readings of the source with the call
sites enumerated.
