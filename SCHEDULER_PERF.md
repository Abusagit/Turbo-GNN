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

## The GT case: the persistent loop costs registers

GT is the one convolution that loses consistently on the forward pass (0.81-0.98x everywhere).
`ncu` counters are unavailable on this machine (`ERR_NVGPUCTRPERM`), but register usage is
static and `cuobjdump -res-usage` needs no permission. Averaged over all 288 instantiations of
`GraphAttentionForward_CSR_MH_v2_D` per policy:

    policy           instantiations   mean REG   max
    one_per_block              288       40.4     62
    grid_stride                288       42.0     64
    precomputed                288       42.6     64
    dynamic                    288       48.6     64

Wrapping the body in a loop lengthens live ranges, and GT has register headroom at baseline
(40 of a 64 cap) for that to consume. At 256 threads/block that is 6.3 blocks/SM at 40.4
registers versus 5.2 at 48.6 -- roughly 15% less occupancy, which is the right size to explain
the loss.

The contrast confirms it: `GATv2Forward_Kernel` already sits at the 64-register cap for
*every* policy (64 / 62-64 / 64 / 64), so the loop cannot cost it any occupancy -- and GATv2
is the conv that holds up best, 1.02-1.08x on the same graphs.

The obvious lever is the second argument of `__launch_bounds__`
(`minBlocksPerMultiprocessor`), which is absent everywhere in `csrc/` -- every kernel passes
only the thread count, which constrains nothing about registers. Capping GT's registers back
to ~40 would plausibly recover the loss. It is **not** done here: it can only reach parity for
GT, and forcing the cap risks spilling to local memory in the GATv2 and reduction kernels that
are currently healthy. It is the first thing to try if GT forward matters.

## Results: 108 cells, forward and backward

Every graph in `configs/datasets/main/` plus ogbn-proteins; head dims 128 and 256; forward and
backward timed separately; `min_aggr`, `gat_v2`, `gt`.

`results/` is gitignored, so the raw rows are not in the repo. Regenerate them with:

    CUDA_VISIBLE_DEVICES=$(python scripts/free_gpus.py --count 3) \
      python scripts/benchmark_scheduler_suite.py --conv gt --head-dims 128 256 \
        --blocks-per-sm 256 1024 --sched-chunk 1 4 --json results/full_gt.json
    python scripts/summarize_scheduler_bench.py results/full_*.json

Choosing the best policy per cell -- an oracle, since no runtime can know which to pick --
gives **geomean 1.02x with 64/108 cells at or above baseline**. As a single blanket default,
no persistent policy breaks even, which is why `DEFAULT_SCHEDULE` is now `one_per_block`.

Where the persistent path genuinely pays:

    min_aggr backward, small sparse graphs   tolokers-2 1.35x, avazu-ctr 1.29x, city-roads-M 1.24x
    min_aggr forward, cache-bound            ogbn-proteins 1.10x (precomputed, degree-balanced)
    gat_v2, head dim 256                     ogbn-proteins 1.08x, twitch-views 1.08x (dynamic)

Where it does not:

    gt forward                               0.81-0.98x everywhere (registers, above)
    ogbn-products                            1.00x -- big enough that launch overhead vanishes

The shape is consistent with the headroom table: wins concentrate where per-node work is small
(so block-launch overhead is a real fraction) or where a better *contiguous* assignment raises
cache reuse. They vanish where the kernel is already DRAM-bound.

## Benchmarking hygiene

Every number above was taken on a GPU with no other compute process on it. The first run of
this sweep was not: it reported GT on ogbn-arxiv at 2.07x, which a clean re-run put at 1.01x.
`scripts/free_gpus.py` queries `nvidia-smi` for compute processes (not just memory, since an
idle context still preempts) and prints the free indices:

    CUDA_VISIBLE_DEVICES=$(python scripts/free_gpus.py --count 3) python scripts/...
