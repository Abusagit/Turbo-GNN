# Counter metrics: edge-sliced heavy bucket vs node-per-block

Nsight Compute on an A100-SXM4-80GB, measured peak **1.747 TB/s** (a large read+write stream,
not the 2.0 TB/s datasheet figure). One steady-state iteration per configuration, warmup outside
the profiled region via `cudaProfilerStart`.

Both arms use the **same argmax parameters**, taken from that cell's completed autotune record;
only `heavy_edge_slice` differs. Measuring the old path in its default configuration would have
flattered the comparison.

Light and heavy launches of the attention kernels share a kernel name and differ only in their
warps template argument (light dispatched over {1,2,4}, heavy over {8,16,32}), so attribution
splits on that; without it the heavy bucket's cost hides inside the total.

## Forward — 23 cells

| metric | old | new |
| --- | ---: | ---: |
| heavy-bucket time | — | **median 3.57x faster** (max 34.1x, min 0.80x) |
| heavy % of peak bandwidth | 7.8% | **58.3%** |
| heavy achieved occupancy | 42.3% | **57.1%** |
| end-to-end (whole conv) | — | median 1.65x, max 6.09x |
| merge kernel overhead | — | median 0.49% of new total |
| light bucket (untouched) | — | 1.0001x |

The light bucket coming out at 1.0001x is the control: this change touches only the
heavy path, and the measurement confirms nothing else moved.

### Largest heavy-bucket gains

| graph | conv | dim | old | new | speedup | % peak |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| web-fraud | min_aggr | 128 | 44.1 ms | 1.29 ms | **34.1x** | 1.7% → **58.3%** |
| web-fraud | min_aggr | 256 | 51.9 ms | 2.17 ms | **23.9x** | 3.2% → **77.7%** |
| web-fraud | gt | 128 | 41.8 ms | 2.31 ms | **18.1x** | 3.8% → **69.3%** |
| web-fraud | gat_v2 | 128 | 25.2 ms | 1.57 ms | **16.1x** | 3.0% → **48.5%** |
| web-fraud | gat_v2 | 256 | 34.1 ms | 2.49 ms | **13.7x** | 4.7% → **64.7%** |
| web-fraud | gt | 256 | 45.0 ms | 3.93 ms | **11.4x** | 7.8% → **90.2%** |
| ogbn-arxiv | gt | 128 | 2.1 ms | 0.23 ms | **9.1x** | 6.5% → **62.6%** |
| ogbn-arxiv | gat_v2 | 128 | 1.1 ms | 0.15 ms | **7.4x** | 3.6% → **29.0%** |

## Backward — flat, and the counters say why

The autotuner selected `slice=0` for nearly every backward cell. Forcing the split on ogbn-arxiv
(a *directed* graph, so its backward genuinely has a heavy bucket) shows what it declined:

| cell | heavy old → new | end-to-end | heavy share of pass |
| --- | --- | ---: | ---: |
| ogbn-arxiv gt d128 | 172 → 161 us (1.07x) | 1.00x | 6% |
| ogbn-arxiv gt d256 | 363 → 364 us (1.00x) | 1.00x | 6% |
| ogbn-arxiv gat_v2 d128 | 1124 → 1120 us (1.00x) | 1.00x | 36% |
| ogbn-arxiv gat_v2 d256 | 2275 → 2330 us (0.98x) | 0.99x | 40% |

Three separate reasons, all visible in the counters:

1. **GT's directed backward splits, but its heavy bucket is only ~6% of the pass.** At d=128 the
   heavy kernel is 172 us against 2,141 us of light work. A 1.07x on 6% cannot move the total,
   and Amdahl caps it there regardless of how good the split is.
2. **GATv2's backward is not split** — deliberately, and it shows: 1.00x. Its heavy bucket *is*
   36% of the pass, so this is the one remaining backward opportunity worth taking.
3. **Undirected backward has no heavy bucket at all.** On ogbn-proteins the backward profile
   contains only a light launch, because that path is unbucketed. There is nothing to slice
   until bucketing is introduced there.

One caution for anyone extending the backward split: at d=128 the sliced backward kernel's
occupancy *drops* (71.8% → 46.4%) from higher register pressure, which is part of why the gain
is only 1.07x even on the bucket it targets.

## Reading

The heavy bucket ran at a **median 7.8% of peak bandwidth at 42% occupancy** while the light
bucket beside it reached 48-91% at 87-99%. It was never short of work — it was short of
*parallelism*: a grid sized by node count, with per-block runtimes spanning the whole degree
distribution, so every launch waited on its longest row. Slicing by edges raises it to a median
**58% of peak**, and on web-fraud to **90%**.

That is the concrete mechanism behind the earlier roofline result that these kernels sit near
24% of peak overall: a bucket holding 13-35% of the edges was contributing almost no bandwidth.

The two-stage structure costs essentially nothing — the merge kernel is a **median 0.49%** of the
new total.

Reproduce: `python scripts/run_edge_slice_ncu.py --gpu N` (needs sudo for counters); add
`--force-slice 256` to profile cells the autotuner rejected.
