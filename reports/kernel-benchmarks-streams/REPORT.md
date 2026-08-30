# Concurrent light/heavy bucket launches

Generated from `reports/kernel-benchmarks-streams/` at `3f761b0` on 2026-08-22.

Hardware: NVIDIA A100-SXM4-80GB. Every run on a GPU verified idle at launch.

## The change

These convolutions split nodes into a *light* and a *heavy* bucket by degree quantile and
launch a kernel for each. The two touch disjoint output rows and have no data dependence, but
historically went out back to back on one stream, so the heavy launch could not begin until
the light one had fully drained. Three strategies are compared:

| `bucket_launch` | behaviour |
| --- | --- |
| `sequential` | light then heavy, one stream — the historical behaviour, and the reference |
| `heavy_first` | heavy then light, one stream — ordering change only, no overlap |
| `concurrent` | heavy and light on separate streams, heavy issued first |

`concurrent` orders with events rather than synchronisation: a fork event on the caller's
stream gates the side stream at entry, a join event gates the caller at exit (also from the
destructor, so an early return cannot leave the caller running ahead). Every tensor allocated
on the caller's stream but touched on the side one is passed to `recordStream`, or the caching
allocator could hand its memory out while the side stream is still reading.

**A prerequisite fix:** GATv2 was launching on `cudaStream_t stream = 0`, the legacy default
stream, which serialises against every other stream. Overlapping its buckets would have been a
silent no-op. That was finding (d) in `KERNEL_ISSUES.md`, now fixed.

All three strategies are bit-exact against `sequential` on the forward pass and on every
backward except GT's directed `dK`, which accumulates through float `atomicAdd` and so depends
on summation order — the same documented exception the scheduler policies have.

## What was measured

16 graphs x 3 convs x head dims 128/256 x forward and backward x 3 strategies x
2 schedules x 3 node orders = **1,152 runs, 0 failures, 192 cells**.

Each strategy is given **its own best** schedule and node order per cell, so no strategy is
judged on a configuration that suits another. Schedules are restricted to `one_per_block` and
`grid_stride`, the only two that were competitive in `reports/kernel-benchmarks/`; that makes
the `sequential` column here slightly different from that report's best-of, so the two are
internally consistent but not cross-comparable.

```bash
python scripts/run_kernel_benchmark_matrix.py --out reports/kernel-benchmarks-streams \
  --schedules one_per_block grid_stride --bucket-launch sequential heavy_first concurrent
python scripts/compare_bucket_launch.py reports/kernel-benchmarks-streams
```

## Result

| strategy | geomean | worst | best | cells won |
| --- | ---: | ---: | ---: | ---: |
| `sequential` *(today)* | 1.0000 | 1.000 | 1.000 | 40/192 |
| `heavy_first` | 0.9642 | 0.320 | 2.182 | 39/192 |
| `concurrent` | 1.0365 | 0.336 | 2.008 | 113/192 |

**The gain is the overlap, not the ordering.** `heavy_first` lands at 0.9642 — worse than
doing nothing. That is the expected result and it is worth stating: reordering two launches on
one stream cannot help, because the second still waits for the first to drain. It is reported
here only because it isolates the mechanism.

## The finding: it splits by pass

| | `heavy_first` | `concurrent` |
| --- | ---: | ---: |
| head dim 128, forward | 0.9501 | **1.1402** |
| head dim 128, backward | 0.9602 | **0.9212** |
| head dim 256, forward | 0.9673 | **1.1126** |
| head dim 256, backward | 0.9793 | **0.9877** |

Forward gains 11-14%; backward *loses* 1-8%. This held at all six intermediate checkpoints
while the overall geomean drifted between 1.02 and 1.05 with graph mix, so the split is the
more reliable result of the two.

Backward has more serialised phases around its buckets — GATv2 runs its AL and R kernels in
sequence, GT has a D-kernel prepass — so there is less to overlap, while the fork/join and
`recordStream` bookkeeping is a fixed per-call cost that does not shrink with it.

### Which gives the obvious policy

| policy | geomean | worst cell | cells at or above 1.0 |
| --- | ---: | ---: | ---: |
| always `sequential` (today) | 1.0000 | 1.000 | 192/192 |
| always `heavy_first` | 0.9642 | 0.320 | 133/192 |
| always `concurrent` | 1.0365 | 0.336 | 147/192 |
| **`concurrent` on forward, `sequential` on backward** | **1.0613** | 0.493 | 177/192 |
| oracle: best strategy per cell (not achievable at runtime) | 1.0934 | 1.000 | 192/192 |

Choosing per pass captures 66% of what a perfect oracle would get, and lifts
cells-at-or-above from 147/192 to 177/192. On the forward pass alone, always-concurrent is
**1.1263** with 81/96 cells at or above baseline.

## Per graph

| graph | nodes | avg deg | `heavy_first` | `concurrent` | `concurrent` fwd only | best cell |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| city-reviews | 148,801 | 16.7 | 0.884 | 1.225 | **1.304** | 2.01x gat_v2/128/for |
| avazu-ctr | 76,269 | 289.0 | 1.002 | 1.176 | **1.376** | 1.71x gat_v2/256/for |
| hm-categories | 46,563 | 461.9 | 1.004 | 1.117 | **1.257** | 1.47x gat_v2/128/for |
| twitch-views | 168,114 | 81.9 | 0.999 | 1.112 | **1.233** | 1.56x gat_v2/128/for |
| tolokers-2 | 11,758 | 89.3 | 0.974 | 1.106 | **1.248** | 1.41x gt/128/for |
| cora | 2,708 | 4.9 | 0.916 | 1.066 | **1.210** | 1.39x gat_v2/256/for |
| ogbn-arxiv | 169,343 | 7.9 | 1.002 | 1.059 | **1.099** | 1.19x gt/256/for |
| web-fraud | 2,890,331 | 5.5 | 1.008 | 1.051 | **1.093** | 1.15x gt/128/for |
| artnet-exp | 50,405 | 12.1 | 1.005 | 1.038 | **1.082** | 1.12x min_aggr/128/for |
| pubmed | 19,717 | 5.5 | 0.987 | 1.031 | **1.143** | 1.24x gat_v2/256/for |
| ogbn-proteins | 132,534 | 598.0 | 1.004 | 1.022 | **1.040** | 1.10x min_aggr/128/for |
| city-roads-L | 142,257 | 3.0 | 0.972 | 1.001 | **1.132** | 1.79x gat_v2/256/for |
| citeseer | 3,327 | 3.7 | 0.767 | 0.931 | **1.088** | 1.40x gat_v2/256/for |
| pokec-regions | 1,632,803 | 19.8 | 1.020 | 0.918 | **0.963** | 1.12x min_aggr/256/bac |
| city-roads-M | 57,073 | 3.3 | 0.975 | 0.909 | **0.964** | 1.02x gt/256/for |
| ogbn-products | 2,449,029 | 51.5 | 0.944 | 0.887 | **0.905** | 1.05x gt/128/for |

Wins concentrate on dense or heavily skewed graphs — city-reviews, avazu-ctr (p99 degree
3,914), hm-categories (p99 6,315), twitch-views, tolokers-2 — which is where the heavy bucket
is a genuine long pole worth overlapping.

## Where it loses, and why

| cell | sequential | concurrent | | |
| --- | ---: | ---: | ---: | --- |
| citeseer min_aggr d=128 backward | 0.020 ms | 0.061 ms | **0.336x** | N=3,327 |
| pokec-regions gat_v2 d=128 backward | 49.761 ms | 133.212 ms | **0.374x** | N=1,632,803 |
| city-roads-M gat_v2 d=128 backward | 0.517 ms | 1.350 ms | **0.383x** | N=57,073 |
| ogbn-products gt d=256 backward | 391.774 ms | 901.449 ms | **0.435x** | N=2,449,029 |
| city-roads-L gat_v2 d=128 backward | 1.062 ms | 2.430 ms | **0.437x** | N=142,257 |
| ogbn-products gt d=256 forward | 143.747 ms | 291.296 ms | **0.493x** | N=2,449,029 |
| citeseer min_aggr d=128 forward | 0.048 ms | 0.080 ms | **0.607x** | N=3,327 |
| pubmed min_aggr d=128 backward | 0.062 ms | 0.086 ms | **0.728x** | N=19,717 |

Two distinct mechanisms, at opposite ends of the size range:

* **Very large graphs** (ogbn-products 2.4M nodes, pokec-regions 1.6M). A single bucket kernel
  already saturates the device, so a second concurrent kernel adds no parallelism and both
  halve their effective L2. ogbn-products GT at head dim 256 goes 143.7 -> 291.3 ms, almost
  exactly 2x, which is what pure cache-capacity contention would predict.
* **Very small graphs** (citeseer 3.3k nodes, city-roads-M). At 0.05-0.5 ms per call there is
  nothing to overlap, and the fork/join events plus per-tensor `recordStream` become a visible
  fraction of runtime. That part is implementation cost rather than anything fundamental, and
  is probably reducible — `recordStream` is currently called unconditionally on tensors that
  in practice outlive the call.

## Recommendation

Do not switch the default globally. Gate `concurrent` on the forward pass, which is worth
1.126x on its own and 1.0613x overall when backward stays sequential, and consider
additionally gating it off for graphs large enough to saturate the device on one bucket alone.
Drop `heavy_first`; it isolates the mechanism but has no standalone value.

## Files

- `summary.txt` — full text output of `scripts/compare_bucket_launch.py`.
- `grid.json` — every cell with all three strategies, their best configurations and speedups.
- `*.json` — one file per run. Gitignored; regenerate with the commands above.
