# Making the persistent scheduler actually faster

Notes from tuning `csrc/common/scheduler.cuh` against the one-block-per-node launch it
replaces. Recorded because the first two hypotheses were wrong in an instructive way, and
because the numbers say something non-obvious about what these kernels are bound by.

Measured on an A100-SXM4-80GB, CUDA 13.2, PyTorch 2.13, on GPUs verified idle at launch
(`scripts/free_gpus.py` — see the note at the end).

---

## The baseline is much stronger than it looks

`one_per_block` launches `grid.x = num_nodes`. That looks wasteful, and the whole premise of
the persistent rewrite was that it is. It is not, for a reason worth stating plainly:

**The hardware block scheduler is already a dynamic work queue, and it is a locality-optimal
one.** Blocks are dispatched roughly in `blockIdx.x` order as slots free, so the blocks
resident at any instant cover a *contiguous window* of node ids. Contiguous node ids mean
contiguous CSR rows, which means `indptr`/`indices` stream sequentially and neighbouring
nodes' feature reads hit in L2. When a block retires, the next block dispatched into that
slot is the next node — the window slides. It costs nothing and no software queue can improve
on the access pattern.

So a persistent kernel does not win by "balancing better". It wins, if at all, by launching
~N fewer blocks — and it can easily *lose* by disturbing that access pattern.

## Hypothesis 1 (wrong): the per-node atomic is the bottleneck

`ncu` on ogbn-arxiv showed 698,160 atomic sectors against 670,512 work items — one global
atomic per node — with registers unchanged at 64, achieved occupancy *higher* than the
baseline (48.4% vs 37.5%), and barrier stalls at 0.11. One atomic per node looked damning, so
`DynamicQueue` gained chunked claiming: `chunk` consecutive items per atomic.

It did not help where it mattered. The sweep (speedup vs `one_per_block`, forward, d=128):

    ogbn-proteins / min_aggr        c1     c2     c4     c8    c16    c32
      bps 32                      0.56   0.55   0.54   0.51   0.47   0.33
      bps 256                     0.57   0.55   0.53   0.49   0.43   0.34

Every extra item per atomic made it *worse*, monotonically, all the way to 0.34x. If atomic
traffic were the cost, dividing it by 32 could not cost 40%.

## Hypothesis 2 (wrong): the grid is sized badly

The same table read the other way says speedup rises with the grid and saturates:
0.70 → 0.77 → 0.77 → 0.77 for GT on proteins as `blocks_per_sm` goes 32 → 256. So a fixed
~23% penalty survives any grid size. Sizing the grid to
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` was implemented and then deleted: above about
128 blocks/SM the grid stops mattering, and the residual penalty is not a grid-size problem.

But the *shape* of both trends is the tell, and it is the same tell twice:

> **the closer the launch came to one-block-per-node, the faster it ran.** More blocks is
> closer. Smaller chunks is closer.

That is not an overhead signature. That is the access pattern.

## What was actually wrong: `+ gridDim.x` assumes a single wave

The original `DynamicQueue` followed FlashAttention's `tile_scheduler.hpp`: statically assign
item `rank` to block `rank`, then draw the tail with `atomicAdd(counter, 1) + gridDim.x`. The
`+ gridDim.x` is what lets the counter start at zero instead of being pre-seeded.

That form is locality-preserving **only when the grid is exactly one resident wave**, which is
the regime FlashAttention launches in. Ours is not. On proteins' light bucket the grid is
6912 blocks against roughly 864 resident. So block rank 5, having finished node 5, jumped
straight to node 6912. Within a few iterations the set of nodes in flight had fragmented from
one contiguous window into scattered ones, and the CSR streaming the baseline gets for free
was gone. Chunking made it strictly worse because it widens every jump by a further factor of
`chunk` — which is exactly the monotone degradation in the table.

**Fix: one monotone cursor.** Every block draws its first chunk from the same counter as every
later one, so items go out strictly in demand order 0, 1, 2, … and the in-flight set is always
a contiguous window whose width is the number of *resident* blocks — the hardware's own
pattern, reproduced. It also drops the separate rank claim entirely: the cursor is already
ascending in start order, which is all the rank was for.

`tests/csrc/test_scheduler.py::test_dynamic_claims_are_a_dense_prefix` pins this invariant so
the `+ gridDim.x` form cannot come back unnoticed.

## Also fixed: `precomputed` was splitting on the wrong axis

`default_block_offsets` split the node list into equal-*count* slices. Cost is proportional to
degree, so that balances nothing on a skewed graph — and combined with a descending-degree
(LPT) order it is actively pathological: block 0 receives every hub in the graph (+1174% at
`bps=8`). `degree_balanced_block_offsets` splits on the degree prefix sum instead, giving each
block equal *edges* while keeping its nodes contiguous. Two call sites keep the even split and
say why in a comment: the reduction backward has no CSR in scope, and the undirected GATv2
impl receives the CSR as raw pointers.

## How much was ever on the table (`scripts/scheduler_headroom.py`)

A scheduler can remove block-launch overhead and load imbalance. It cannot remove a byte of
neighbour-feature traffic. Comparing measured time against the time those reads alone would
take at achievable HBM bandwidth (1.55 TB/s on this A100) bounds what any scheduling change
could do, and it splits these graphs into two regimes that behave completely differently:

    graph            d     edges       measured    traffic floor    floor/measured
    ogbn-arxiv     128   1,166,243     0.652 ms       0.445 ms            68%
    ogbn-arxiv     256   1,166,243     1.176 ms       0.886 ms            75%
    tolokers-2     128   1,038,000     0.240 ms       0.349 ms           146%
    ogbn-proteins  128  79,122,504    10.018 ms      26.384 ms           263%
    ogbn-proteins  256  79,122,504    18.056 ms      52.564 ms           291%

**Below 100% — DRAM-bound.** On ogbn-arxiv two thirds to three quarters of the runtime is
feature traffic that no scheduler can touch. Everything a scheduler *can* address — launch
overhead, imbalance, compute — is competing for the remaining ~30%. Measured wins of
1.05-1.10x are a fair share of that budget, and there is no 2x hiding in it.

**Above 100% — L2-reuse-bound.** ogbn-proteins could not have moved its own traffic across HBM
in the time it took, so the caches absorbed most of it: adjacent nodes share neighbours, and
the same feature rows get re-read from L2. Reuse depends entirely on the order nodes are
visited in, which is exactly what a scheduler changes.

That single distinction explains every result above. On ogbn-proteins the policies span 0.57x
to 1.10x — a factor of two — while on ogbn-arxiv they barely move, because on a cache-bound
graph the scheduler's first job is *not to destroy locality* and only its second job is to
balance. It is also why the two wrong hypotheses were wrong in the same direction: both
chunking and the `+ gridDim.x` claim widen the window of nodes in flight, and on a cache-bound
graph a wider window is a lower hit rate.

It also says where a decisive win would have to come from, and it is not scheduling: on a
cache-bound graph, reordering nodes so that ones sharing neighbours are visited together (BFS
or RCM, as `src/data/converters.py::reorder_graph` already does) raises the hit rate directly.
That permutes the output and is a data-layout change rather than a scheduler change, so it is
out of scope here -- but the `nodes` indirection array is exactly the hook it would use.

## What this means for the defaults

- The per-node work decides everything. On a dense graph (proteins, p50 degree 402) the
  baseline's access pattern is the whole ball game and the scheduler's job is to not break it.
  On a sparse graph (arxiv, p50 degree 1) per-node work is tiny, block-launch overhead is
  visible, and chunking pays.
- `sched_chunk` is therefore a genuine tunable, not a constant to be tuned once. It is
  deliberately *not* a `TunableParam`: adding it and `schedule` to the autotuner would take
  the search space from 1,344 to 24,192 combinations.
- Neither `schedule` nor `blocks_per_sm` changes any result — every policy is bit-exact
  against `one_per_block` on the forward path, and matches to float-atomic tolerance on the
  backward. `tests/correctness/test_scheduler_policies.py` covers that.

## The GT case: an occupancy cliff, and the fix

GT was the one convolution that lost consistently on the forward pass (0.81-0.98x). `ncu`
counters are unavailable on this machine (`ERR_NVGPUCTRPERM`), but register usage is static and
`cuobjdump -res-usage` needs no permission at all.

The mean over all 288 instantiations pointed the right way (40.4 for `one_per_block` versus
42.0 / 42.6 / 48.6 for the persistent policies), but the *specific* instantiation the light
bucket actually launches -- 4 warps, `D_CONST=128`, fp32, which handles ~99% of nodes -- is
where it bites:

    policy           REG   warps/SM   occupancy
    one_per_block     32         64        100%
    grid_stride       38         52         81%
    precomputed       39         52         81%
    dynamic           48         40         62%
    dynamic (D=256)   56         36         56%

32 registers is *exactly* the budget for 2048 resident threads on an A100. The baseline sits
precisely on the full-occupancy threshold, and lengthening live ranges by wrapping the body in
the scheduler's loop pushes it over. Measured 0.81-0.98x against occupancy ratios of 0.62-0.81:
the same numbers.

The cliff is specific to this kernel, which is why only GT regressed. `reduction_aggr_forward_light_kernel_1d`
goes 47.6 -> 48.2 across policies and `GATv2Forward_Kernel` goes 55.9 -> 54.5/58.2; neither is
near a threshold, and neither conv loses.

**The fix.** Every kernel in `csrc/` passed `__launch_bounds__` a thread count and nothing else,
which constrains *nothing* about registers. Supplying the second argument
(`minBlocksPerMultiprocessor`) does: asking for B blocks of T threads caps the compiler at
65536/(B*T) registers per thread. `kGtFwdMinBlocksPerSM` targets 2048 resident threads, but only
where that is reachable -- the 8-warp heavy instantiation needs 47 registers on its own, so
demanding 32 there would trade an occupancy loss for a local-memory spill, and it is left
unconstrained.

Afterwards, on that same instantiation: `grid_stride` and `precomputed` drop to **30 registers
with zero spill** -- below the baseline, so full occupancy -- and `dynamic` reaches 32 but
spills 16 bytes, which is why it is no longer the policy that wins GT cells. The measured
effect, best policy per cell:

    cell                        before   after
    ogbn-proteins d=128 fwd       0.81    0.92
    city-roads-M  d=256 fwd       0.82    0.96
    city-roads-M  d=128 fwd       0.86    0.97
    ogbn-arxiv    d=128 fwd       0.98    1.03

Baselines moved by under 2% across these, so the gain is on the persistent path rather than a
shifted reference. It removes most of GT's regression without reaching a decisive win, and it
is the clearest remaining evidence that what limits these kernels is occupancy and memory
behaviour rather than scheduling policy.

## Results: 108 cells, forward and backward

Every graph in `configs/datasets/main/` plus ogbn-proteins; head dims 128 and 256; forward and
backward timed separately; `min_aggr`, `gat_v2`, `gt`.

`results/` is gitignored, so the raw rows are not in the repo. Regenerate them with:

    CUDA_VISIBLE_DEVICES=$(python scripts/free_gpus.py --count 3) \
      python scripts/benchmark_scheduler_suite.py --conv gt --head-dims 128 256 \
        --blocks-per-sm 256 1024 --sched-chunk 1 4 --json results/full_gt.json
    python scripts/summarize_scheduler_bench.py results/full_*.json

Choosing the best policy per cell -- an oracle, since no runtime can know which to pick --
gives **geomean 1.02x with 66/108 cells at or above baseline**, and every graph is at or above 1.01x. As a single blanket default,
no persistent policy breaks even, which is why `DEFAULT_SCHEDULE` is now `one_per_block`.

Where the persistent path genuinely pays:

    min_aggr backward, small sparse graphs   tolokers-2 1.35x, avazu-ctr 1.29x, city-roads-M 1.24x
    min_aggr forward, cache-bound            ogbn-proteins 1.10x (precomputed, degree-balanced)
    gat_v2, head dim 256                     ogbn-proteins 1.08x, twitch-views 1.08x (dynamic)

Where it does not:

    gt forward                               0.87-1.03x after the occupancy fix, still the weakest
    ogbn-products                            1.00x -- big enough that launch overhead vanishes

The shape is consistent with the headroom table: wins concentrate where per-node work is small
(so block-launch overhead is a real fraction) or where a better *contiguous* assignment raises
cache reuse. They vanish where the kernel is already DRAM-bound.

## Where the win actually was: visit order

Having exhausted the scheduling policies, the headroom table said the remaining lever had to be
cache reuse -- and reuse is a function of the order nodes are visited in. That is not a
data-layout change and it does not permute anything: the scheduler's `nodes` array decides
*which node a block visits*, never where the result is written, which is why
`sorted_by_degree()` has always been bit-exact. The same hook takes any order.

First, does order matter at all? On ogbn-proteins with `min_aggr` at d=128:

    identity       9.411 ms   x1.00
    rcm           10.485 ms   x0.90
    degree_desc   11.842 ms   x0.79
    random        13.411 ms   x0.70

A random shuffle costs 30%, so order is worth real money -- but ogbn-proteins is already stored
in a good order and nothing beats it. That is exactly why the first probe was discouraging, and
why it would have been wrong to stop there: other graphs are *not* stored in a good order.

Sweeping all three convs over three orders x four policies, 108 cells, best per cell against
the untouched baseline. **Forward at the head dims that matter:**

    d=128 forward   geomean 1.125x   25/27 cells at or above baseline
    d=256 forward   geomean 1.170x   26/27
    d=128 backward  geomean 1.046x   24/27
    d=256 backward  geomean 1.026x   21/27

Overall the oracle goes from 1.02x to **1.09x**, and cells at or above baseline from 66/108 to
**96/108**. For the first time a single blanket configuration beats the baseline on average:
`one_per_block` on an RCM-ordered bucket, geomean 1.05 with a best of 1.78x.

The largest cells:

    ogbn-products  gt        256 fwd   573.4 -> 321.8 ms   x1.78   one_per_block + rcm
    ogbn-products  gt        128 fwd   288.1 -> 173.4 ms   x1.66   one_per_block + rcm
    ogbn-products  min_aggr  256 fwd    80.3 ->  51.2 ms   x1.57   one_per_block + rcm
    ogbn-products  gat_v2    256 fwd   307.0 -> 199.2 ms   x1.54   one_per_block + rcm
    tolokers-2     min_aggr  128 bwd     0.13 ->  0.09 ms  x1.45   grid_stride   + degree
    city-reviews   gt        256 fwd     8.9 ->   6.7 ms   x1.33   one_per_block + rcm
    twitch-views   gt        256 fwd    47.2 ->  38.4 ms   x1.23   one_per_block + degree

And by graph (geomean of the best config per cell):

    ogbn-products  1.26x      tolokers-2   1.16x      twitch-views  1.09x
    avazu-ctr      1.08x      city-reviews 1.08x      ogbn-arxiv    1.06x
    hm-categories  1.06x      city-roads-M 1.03x      ogbn-proteins 1.02x

The backward pass gains much less than the forward one (1.03-1.05x against 1.13-1.17x). Its
kernels scatter through float `atomicAdd` rather than reading a neighbourhood per output row,
so there is far less reuse for a visit order to improve. ogbn-proteins is the weakest graph
throughout, and GT on it is the one place still below baseline (0.88-0.92x); both facts have
the same cause, that ogbn-proteins was already stored in a good order.

ogbn-products is the largest graph here (2.4M nodes, 123M edges) and gains the most, which fits
the mechanism: the bigger the working set relative to L2, the more a good visit order is worth.
ogbn-proteins gains the least, for the same reason in reverse -- it was already ordered well.

Two orders, because neither dominates:

* `sorted_by_degree()` balances. Cost tracks degree, so descending order is
  longest-processing-time-first. Wins on tolokers-2 (1.45x) and twitch-views.
* `sorted_by_locality()` clusters, via reverse Cuthill-McKee. Wins on ogbn-products (1.57x)
  and ogbn-arxiv. Needs scipy (an optional dependency) and costs 0.1 s on ogbn-arxiv, 6.4 s on
  ogbn-products -- one-off and cacheable on the graph, so it pays back over a training run but
  not over a single inference call.

Both are opt-in for the same reason the policies are: the right choice is per graph, and
picking wrong costs more than picking right gains (ogbn-proteins loses 21% under
descending-degree order).

## A landmine found on the way: `precomputed` at 0.03x

Combining a reordered bucket with `precomputed` ran **33x slower than baseline**. The
degree-balanced split was weighing nodes by edge count alone, so a zero-degree node had cost
zero and `searchsorted` piled every one of them into a single slice -- and a zero-degree node
is cheap per edge but still costs its block a loop iteration. ogbn-arxiv has 62,006 of them in
a light bucket of 167,628, so one block received 62,014 nodes against a mean of 1.52. Under a
descending-degree order, where the zeros are contiguous at the tail, it was deterministic.

The cost model is now `1 + degree` rather than `degree`: a fixed per-node part plus a per-edge
part. Worst slice on ogbn-arxiv drops from 62,014 nodes to 9, and from 95 to 8 even in the
natural order, where the bug was silently costing ~20%.

## Benchmarking hygiene

Every number above was taken on a GPU with no other compute process on it. The first run of
this sweep was not: it reported GT on ogbn-arxiv at 2.07x, which a clean re-run put at 1.01x.
`scripts/free_gpus.py` queries `nvidia-smi` for compute processes (not just memory, since an
idle context still preempts) and prints the free indices:

    CUDA_VISIBLE_DEVICES=$(python scripts/free_gpus.py --count 3) python scripts/...
