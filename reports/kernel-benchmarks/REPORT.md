# Kernel benchmark report: node scheduling and visit order

Generated from `reports/kernel-benchmarks/` at `559d339` on 2026-08-22.

Hardware: NVIDIA A100-SXM4-80GB. Every run on a GPU verified idle at launch (`scripts/free_gpus.py`).

## What was measured

`scripts/benchmark_kernels.py` times the projection-free aggregation alone — no linear/QKV
projections, no bias, no framework dispatch — through `--backend cuda`, which reaches the
turbo_gnn kernels. Autotuning is off, so each run measures exactly the configuration asked for.

- **16 graphs**: all of `configs/datasets/main/`, all of `configs/datasets/graphland_remaining/`,
  `cora` / `citeseer` / `pubmed`, and `ogbn-proteins`.
- **3 convolutions**: `min_aggr`, `gat_v2`, `gt`.
- **2 head dims**: 128 and 256, with `--heads 1` so head dim is unambiguous and identical
  across conv families (they differ in how `--feature-dim` maps to per-head width).
- **Both passes**: `--mode forward` and `--mode backward`. Backward reuses one forward graph,
  so only the gradient kernels are inside the timed region.
- **12 configurations per cell**: 4 schedules x 3 node orders.

That is **192 cells** and **2304 measured configurations**, from 768 runs, 0 failures.

```bash
python scripts/run_kernel_benchmark_matrix.py --out reports/kernel-benchmarks
python scripts/summarize_kernel_benchmarks.py reports/kernel-benchmarks
```

## Read this before the numbers

**The baseline is one of the 12 candidates.** `best` is the fastest of the grid, and the grid
contains `one_per_block+natural`, so a best-of speedup can never fall below 1.00x. Counting
"cells at or above baseline" would report 100% and mean nothing. The honest counts below are
**cells improved** — where something other than the baseline won by more than 0.5%.

**These numbers are not comparable to `SCHEDULER_PERF.md`.** That report used a different
harness: no self-loops and 4 attention heads. This one goes through the repo's own loader,
which adds one self-loop per node, and uses 1 head. Same kernels, different workload.

**Sub-millisecond cells carry the most relative noise.** `cora` and `citeseer` run in 0.05–0.25 ms,
where launch overhead dominates — which is exactly where persistent kernels should win, and
they do, but treat a 4.35x on a 0.047 ms kernel as directional rather than precise.

## Headline

| | geomean | cells improved |
| --- | --- | --- |
| head dim 128, forward | **1.172x** | 44/48 |
| head dim 128, backward | **1.126x** | 37/48 |
| head dim 256, forward | **1.155x** | 42/48 |
| head dim 256, backward | **1.085x** | 34/48 |
| **overall** | **1.134x** | 157/192 |

## If one configuration had to be the default

Picked per cell, the grid is worth 1.134x. Picked once for everything, no configuration is
safe — the two that look best on average also lose 45% in their worst cell.

| configuration | geomean | worst | best | cells at or above 1.0 |
| --- | --- | --- | --- | --- |
| `grid_stride+degree` | 1.053 | 0.55 | 3.32 | 105/192 |
| `grid_stride+locality` | 1.050 | 0.43 | 3.29 | 119/192 |
| `one_per_block+locality` | 1.046 | 0.17 | 2.53 | 141/192 |
| `one_per_block+degree` | 1.037 | 0.15 | 2.53 | 126/192 |
| `grid_stride+natural` | 1.004 | 0.49 | 2.42 | 66/192 |
| `one_per_block+natural` *(baseline)* | 1.000 | 1.00 | 1.00 | 192/192 |
| `dynamic+natural` | 0.781 | 0.18 | 3.17 | 17/192 |
| `dynamic+locality` | 0.779 | 0.25 | 4.35 | 34/192 |
| `dynamic+degree` | 0.726 | 0.21 | 4.29 | 27/192 |
| `precomputed+locality` | 0.714 | 0.04 | 1.85 | 41/192 |
| `precomputed+natural` | 0.703 | 0.07 | 2.01 | 27/192 |
| `precomputed+degree` | 0.699 | 0.07 | 1.82 | 48/192 |

`precomputed` and `dynamic` hold the highest peaks in the whole matrix (up to 4.35x) and the
worst averages (0.70–0.78). They are worth reaching for per graph and wrong as a default.

## Per-graph summary

| graph | nodes | edges | avg deg | geomean | improved | best cell |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cora | 2,708 | 13,264 | 4.9 | **1.387x** | 12/12 | 4.35x `dynamic+locality` |
| ogbn-products | 2,449,029 | 126,167,309 | 51.5 | **1.236x** | 8/12 | 1.69x `one_per_block+locality` |
| citeseer | 3,327 | 12,431 | 3.7 | **1.232x** | 12/12 | 3.32x `grid_stride+degree` |
| city-reviews | 148,801 | 2,479,631 | 16.7 | **1.210x** | 12/12 | 1.82x `grid_stride+degree` |
| artnet-exp | 50,405 | 611,101 | 12.1 | **1.189x** | 10/12 | 2.68x `grid_stride+degree` |
| pubmed | 19,717 | 108,365 | 5.5 | **1.182x** | 12/12 | 1.79x `one_per_block+locality` |
| tolokers-2 | 11,758 | 1,049,758 | 89.3 | **1.172x** | 11/12 | 1.44x `grid_stride+degree` |
| twitch-views | 168,114 | 13,763,228 | 81.9 | **1.125x** | 12/12 | 1.29x `grid_stride+degree` |
| hm-categories | 46,563 | 21,508,553 | 461.9 | **1.122x** | 12/12 | 1.24x `grid_stride+locality` |
| avazu-ctr | 76,269 | 22,044,423 | 289.0 | **1.095x** | 10/12 | 1.31x `one_per_block+degree` |
| ogbn-arxiv | 169,343 | 1,335,586 | 7.9 | **1.090x** | 11/12 | 1.22x `one_per_block+degree` |
| web-fraud | 2,890,331 | 15,785,700 | 5.5 | **1.073x** | 7/12 | 1.52x `grid_stride+locality` |
| city-roads-L | 142,257 | 421,318 | 3.0 | **1.033x** | 10/12 | 1.10x `one_per_block+locality` |
| city-roads-M | 57,073 | 189,643 | 3.3 | **1.031x** | 10/12 | 1.10x `one_per_block+locality` |
| ogbn-proteins | 132,534 | 79,255,038 | 598.0 | **1.020x** | 5/12 | 1.11x `one_per_block+locality` |
| pokec-regions | 1,632,803 | 32,255,367 | 19.8 | **1.008x** | 3/12 | 1.06x `one_per_block+degree` |

## Every graph, every cell

`baseline` is `one_per_block` in natural node order; `best` is the fastest of the 12
configurations. Times are milliseconds per iteration.

### cora

`N=2,708` &middot; `E=13,264` &middot; `avg_deg=4.9` &middot; `max_deg=169` &middot; `heavy_fwd=29` &middot; geomean **1.387x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 0.7315 | 0.7230 | **1.01x** | `grid_stride+locality` |
| gat_v2 | 128 | forward | 0.0460 | 0.0346 | **1.33x** | `grid_stride+locality` |
| gat_v2 | 256 | backward | 0.8795 | 0.8059 | **1.09x** | `grid_stride+natural` |
| gat_v2 | 256 | forward | 0.0478 | 0.0439 | **1.09x** | `grid_stride+locality` |
| gt | 128 | backward | 0.1982 | 0.1847 | **1.07x** | `one_per_block+locality` |
| gt | 128 | forward | 0.0834 | 0.0435 | **1.92x** | `one_per_block+degree` |
| gt | 256 | backward | 0.3207 | 0.2999 | **1.07x** | `grid_stride+locality` |
| gt | 256 | forward | 0.0491 | 0.0441 | **1.11x** | `one_per_block+locality` |
| min_aggr | 128 | backward | 0.2051 | 0.0471 | **4.35x** | `dynamic+locality` |
| min_aggr | 128 | forward | 0.0505 | 0.0437 | **1.15x** | `grid_stride+degree` |
| min_aggr | 256 | backward | 0.1106 | 0.0860 | **1.29x** | `grid_stride+natural` |
| min_aggr | 256 | forward | 0.0997 | 0.0499 | **2.00x** | `grid_stride+locality` |

### ogbn-products

`N=2,449,029` &middot; `E=126,167,309` &middot; `avg_deg=51.5` &middot; `max_deg=17,482` &middot; `heavy_fwd=24,593` &middot; geomean **1.236x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 183.0328 | 182.2321 | **1.00x** | `one_per_block+degree` *(baseline already best)* |
| gat_v2 | 128 | forward | 45.3832 | 35.5564 | **1.28x** | `one_per_block+locality` |
| gat_v2 | 256 | backward | 360.0466 | 356.4667 | **1.01x** | `dynamic+degree` |
| gat_v2 | 256 | forward | 78.5132 | 52.7759 | **1.49x** | `one_per_block+locality` |
| gt | 128 | backward | 208.4321 | 194.9716 | **1.07x** | `dynamic+locality` |
| gt | 128 | forward | 73.7690 | 46.4092 | **1.59x** | `one_per_block+locality` |
| gt | 256 | backward | 385.5329 | 385.4715 | **1.00x** | `one_per_block+degree` *(baseline already best)* |
| gt | 256 | forward | 146.2374 | 86.3324 | **1.69x** | `one_per_block+locality` |
| min_aggr | 128 | backward | 30.6203 | 30.6101 | **1.00x** | `one_per_block+degree` *(baseline already best)* |
| min_aggr | 128 | forward | 39.9724 | 26.9554 | **1.48x** | `one_per_block+locality` |
| min_aggr | 256 | backward | 61.2987 | 61.2526 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| min_aggr | 256 | forward | 81.7848 | 52.8353 | **1.55x** | `one_per_block+locality` |

### citeseer

`N=3,327` &middot; `E=12,431` &middot; `avg_deg=3.7` &middot; `max_deg=100` &middot; `heavy_fwd=38` &middot; geomean **1.232x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 0.7205 | 0.6333 | **1.14x** | `one_per_block+locality` |
| gat_v2 | 128 | forward | 0.0318 | 0.0299 | **1.06x** | `grid_stride+degree` |
| gat_v2 | 256 | backward | 0.7806 | 0.6869 | **1.14x** | `dynamic+degree` |
| gat_v2 | 256 | forward | 0.0380 | 0.0352 | **1.08x** | `one_per_block+locality` |
| gt | 128 | backward | 0.1648 | 0.1333 | **1.24x** | `grid_stride+locality` |
| gt | 128 | forward | 0.0359 | 0.0333 | **1.08x** | `one_per_block+locality` |
| gt | 256 | backward | 0.2177 | 0.2097 | **1.04x** | `one_per_block+locality` |
| gt | 256 | forward | 0.0409 | 0.0387 | **1.06x** | `grid_stride+locality` |
| min_aggr | 128 | backward | 0.0993 | 0.0763 | **1.30x** | `dynamic+degree` |
| min_aggr | 128 | forward | 0.0446 | 0.0390 | **1.14x** | `one_per_block+degree` |
| min_aggr | 256 | backward | 0.2279 | 0.0687 | **3.32x** | `grid_stride+degree` |
| min_aggr | 256 | forward | 0.0513 | 0.0447 | **1.15x** | `one_per_block+degree` |

### city-reviews

`N=148,801` &middot; `E=2,479,631` &middot; `avg_deg=16.7` &middot; `max_deg=7,438` &middot; `heavy_fwd=1,500` &middot; geomean **1.210x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 19.6219 | 10.7725 | **1.82x** | `grid_stride+degree` |
| gat_v2 | 128 | forward | 1.5219 | 1.3009 | **1.17x** | `grid_stride+degree` |
| gat_v2 | 256 | backward | 19.5413 | 19.0876 | **1.02x** | `dynamic+degree` |
| gat_v2 | 256 | forward | 2.2528 | 1.8142 | **1.24x** | `grid_stride+degree` |
| gt | 128 | backward | 11.9835 | 11.0436 | **1.09x** | `one_per_block+locality` |
| gt | 128 | forward | 2.1505 | 1.9468 | **1.10x** | `one_per_block+degree` |
| gt | 256 | backward | 22.0587 | 21.5107 | **1.03x** | `one_per_block+locality` |
| gt | 256 | forward | 3.2692 | 2.6311 | **1.24x** | `one_per_block+locality` |
| min_aggr | 128 | backward | 1.0002 | 0.5953 | **1.68x** | `one_per_block+locality` |
| min_aggr | 128 | forward | 0.7707 | 0.6707 | **1.15x** | `grid_stride+locality` |
| min_aggr | 256 | backward | 1.3380 | 1.3286 | **1.01x** | `one_per_block+locality` |
| min_aggr | 256 | forward | 1.4721 | 1.2076 | **1.22x** | `grid_stride+locality` |

### artnet-exp

`N=50,405` &middot; `E=611,101` &middot; `avg_deg=12.1` &middot; `max_deg=199` &middot; `heavy_fwd=507` &middot; geomean **1.189x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 1.3598 | 1.3185 | **1.03x** | `one_per_block+degree` |
| gat_v2 | 128 | forward | 0.2779 | 0.2104 | **1.32x** | `grid_stride+degree` |
| gat_v2 | 256 | backward | 2.0265 | 2.0265 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gat_v2 | 256 | forward | 0.3891 | 0.3041 | **1.28x** | `grid_stride+degree` |
| gt | 128 | backward | 1.0910 | 1.0283 | **1.06x** | `grid_stride+locality` |
| gt | 128 | forward | 0.3111 | 0.3016 | **1.03x** | `one_per_block+degree` |
| gt | 256 | backward | 2.0356 | 1.9237 | **1.06x** | `one_per_block+locality` |
| gt | 256 | forward | 0.4472 | 0.4472 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 128 | backward | 0.2112 | 0.1506 | **1.40x** | `grid_stride+locality` |
| min_aggr | 128 | forward | 0.5622 | 0.2101 | **2.68x** | `grid_stride+degree` |
| min_aggr | 256 | backward | 0.3249 | 0.3184 | **1.02x** | `one_per_block+locality` |
| min_aggr | 256 | forward | 0.3766 | 0.3646 | **1.03x** | `grid_stride+degree` |

### pubmed

`N=19,717` &middot; `E=108,365` &middot; `avg_deg=5.5` &middot; `max_deg=172` &middot; `heavy_fwd=214` &middot; geomean **1.182x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 0.9251 | 0.7895 | **1.17x** | `one_per_block+locality` |
| gat_v2 | 128 | forward | 0.0887 | 0.0742 | **1.20x** | `one_per_block+degree` |
| gat_v2 | 256 | backward | 1.8957 | 1.0568 | **1.79x** | `one_per_block+locality` |
| gat_v2 | 256 | forward | 0.1184 | 0.1018 | **1.16x** | `one_per_block+degree` |
| gt | 128 | backward | 0.3856 | 0.3586 | **1.08x** | `one_per_block+locality` |
| gt | 128 | forward | 0.1111 | 0.1071 | **1.04x** | `one_per_block+locality` |
| gt | 256 | backward | 0.7011 | 0.6837 | **1.03x** | `one_per_block+locality` |
| gt | 256 | forward | 0.1431 | 0.1367 | **1.05x** | `one_per_block+locality` |
| min_aggr | 128 | backward | 0.1080 | 0.0683 | **1.58x** | `grid_stride+degree` |
| min_aggr | 128 | forward | 0.0956 | 0.0840 | **1.14x** | `one_per_block+locality` |
| min_aggr | 256 | backward | 0.1181 | 0.1116 | **1.06x** | `one_per_block+locality` |
| min_aggr | 256 | forward | 0.1491 | 0.1340 | **1.11x** | `one_per_block+locality` |

### tolokers-2

`N=11,758` &middot; `E=1,049,758` &middot; `avg_deg=89.3` &middot; `max_deg=2,139` &middot; `heavy_fwd=118` &middot; geomean **1.172x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 3.3788 | 3.0887 | **1.09x** | `grid_stride+locality` |
| gat_v2 | 128 | forward | 0.6800 | 0.5525 | **1.23x** | `one_per_block+degree` |
| gat_v2 | 256 | backward | 5.3441 | 4.7325 | **1.13x** | `dynamic+degree` |
| gat_v2 | 256 | forward | 0.9232 | 0.7108 | **1.30x** | `grid_stride+degree` |
| gt | 128 | backward | 2.8647 | 2.5766 | **1.11x** | `one_per_block+degree` |
| gt | 128 | forward | 0.6210 | 0.5030 | **1.23x** | `grid_stride+degree` |
| gt | 256 | backward | 5.2526 | 4.8298 | **1.09x** | `one_per_block+degree` |
| gt | 256 | forward | 0.8165 | 0.7303 | **1.12x** | `one_per_block+degree` |
| min_aggr | 128 | backward | 0.0739 | 0.0739 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 128 | forward | 0.3290 | 0.2596 | **1.27x** | `one_per_block+degree` |
| min_aggr | 256 | backward | 0.0910 | 0.0821 | **1.11x** | `one_per_block+locality` |
| min_aggr | 256 | forward | 0.4655 | 0.3224 | **1.44x** | `grid_stride+degree` |

### twitch-views

`N=168,114` &middot; `E=13,763,228` &middot; `avg_deg=81.9` &middot; `max_deg=35,280` &middot; `heavy_fwd=1,685` &middot; geomean **1.125x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 52.6582 | 51.3331 | **1.03x** | `dynamic+natural` |
| gat_v2 | 128 | forward | 6.9846 | 5.4015 | **1.29x** | `grid_stride+degree` |
| gat_v2 | 256 | backward | 92.6423 | 85.2480 | **1.09x** | `dynamic+degree` |
| gat_v2 | 256 | forward | 9.9442 | 7.6919 | **1.29x** | `one_per_block+degree` |
| gt | 128 | backward | 52.0858 | 51.2799 | **1.02x** | `dynamic+locality` |
| gt | 128 | forward | 9.8954 | 8.0306 | **1.23x** | `one_per_block+degree` |
| gt | 256 | backward | 95.8781 | 95.0979 | **1.01x** | `one_per_block+degree` |
| gt | 256 | forward | 15.3391 | 13.0390 | **1.18x** | `grid_stride+degree` |
| min_aggr | 128 | backward | 0.8772 | 0.8576 | **1.02x** | `one_per_block+locality` |
| min_aggr | 128 | forward | 2.9213 | 2.5259 | **1.16x** | `one_per_block+degree` |
| min_aggr | 256 | backward | 2.3182 | 2.2438 | **1.03x** | `one_per_block+degree` |
| min_aggr | 256 | forward | 5.9030 | 4.8492 | **1.22x** | `grid_stride+degree` |

### hm-categories

`N=46,563` &middot; `E=21,508,553` &middot; `avg_deg=461.9` &middot; `max_deg=31,459` &middot; `heavy_fwd=466` &middot; geomean **1.122x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 39.7824 | 37.2178 | **1.07x** | `precomputed+locality` |
| gat_v2 | 128 | forward | 6.6941 | 5.4795 | **1.22x** | `one_per_block+degree` |
| gat_v2 | 256 | backward | 66.7945 | 66.2487 | **1.01x** | `precomputed+locality` |
| gat_v2 | 256 | forward | 8.8604 | 7.4694 | **1.19x** | `one_per_block+degree` |
| gt | 128 | backward | 39.3078 | 37.5823 | **1.05x** | `one_per_block+locality` |
| gt | 128 | forward | 7.8049 | 6.7242 | **1.16x** | `one_per_block+locality` |
| gt | 256 | backward | 76.0300 | 72.5780 | **1.05x** | `one_per_block+degree` |
| gt | 256 | forward | 12.1094 | 10.6010 | **1.14x** | `grid_stride+degree` |
| min_aggr | 128 | backward | 0.2771 | 0.2559 | **1.08x** | `grid_stride+locality` |
| min_aggr | 128 | forward | 2.9652 | 2.4070 | **1.23x** | `one_per_block+degree` |
| min_aggr | 256 | backward | 0.4792 | 0.4539 | **1.06x** | `one_per_block+locality` |
| min_aggr | 256 | forward | 4.9935 | 4.0275 | **1.24x** | `grid_stride+locality` |

### avazu-ctr

`N=76,269` &middot; `E=22,044,423` &middot; `avg_deg=289.0` &middot; `max_deg=42,032` &middot; `heavy_fwd=842` &middot; geomean **1.095x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 64.1372 | 64.1014 | **1.00x** | `precomputed+locality` *(baseline already best)* |
| gat_v2 | 128 | forward | 8.9203 | 6.8085 | **1.31x** | `one_per_block+degree` |
| gat_v2 | 256 | backward | 106.8165 | 105.5406 | **1.01x** | `one_per_block+locality` |
| gat_v2 | 256 | forward | 11.7030 | 9.6641 | **1.21x** | `one_per_block+degree` |
| gt | 128 | backward | 56.7859 | 56.4808 | **1.01x** | `one_per_block+locality` |
| gt | 128 | forward | 11.4011 | 10.2937 | **1.11x** | `one_per_block+degree` |
| gt | 256 | backward | 107.8436 | 107.5620 | **1.00x** | `one_per_block+degree` *(baseline already best)* |
| gt | 256 | forward | 14.3942 | 12.9473 | **1.11x** | `one_per_block+degree` |
| min_aggr | 128 | backward | 0.4517 | 0.4366 | **1.03x** | `one_per_block+locality` |
| min_aggr | 128 | forward | 3.1076 | 2.5659 | **1.21x** | `one_per_block+degree` |
| min_aggr | 256 | backward | 0.8444 | 0.7987 | **1.06x** | `dynamic+locality` |
| min_aggr | 256 | forward | 5.0380 | 4.4579 | **1.13x** | `grid_stride+degree` |

### ogbn-arxiv

`N=169,343` &middot; `E=1,335,586` &middot; `avg_deg=7.9` &middot; `max_deg=13,156` &middot; `heavy_fwd=1,715` &middot; geomean **1.090x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 4.6123 | 4.1491 | **1.11x** | `one_per_block+degree` |
| gat_v2 | 128 | forward | 1.6425 | 1.4891 | **1.10x** | `one_per_block+locality` |
| gat_v2 | 256 | backward | 7.0952 | 6.6869 | **1.06x** | `grid_stride+degree` |
| gat_v2 | 256 | forward | 2.5925 | 2.1201 | **1.22x** | `one_per_block+degree` |
| gt | 128 | backward | 2.9910 | 2.7050 | **1.11x** | `grid_stride+locality` |
| gt | 128 | forward | 2.7720 | 2.5763 | **1.08x** | `one_per_block+degree` |
| gt | 256 | backward | 5.9909 | 5.7074 | **1.05x** | `one_per_block+locality` |
| gt | 256 | forward | 3.6656 | 3.3396 | **1.10x** | `one_per_block+locality` |
| min_aggr | 128 | backward | 0.6132 | 0.6098 | **1.01x** | `one_per_block+degree` |
| min_aggr | 128 | forward | 0.6939 | 0.6135 | **1.13x** | `one_per_block+locality` |
| min_aggr | 256 | backward | 1.2736 | 1.2732 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| min_aggr | 256 | forward | 1.2782 | 1.1246 | **1.14x** | `grid_stride+locality` |

### web-fraud

`N=2,890,331` &middot; `E=15,785,700` &middot; `avg_deg=5.5` &middot; `max_deg=228,991` &middot; `heavy_fwd=29,575` &middot; geomean **1.073x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 198.9110 | 198.9110 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gat_v2 | 128 | forward | 28.7870 | 28.3136 | **1.02x** | `precomputed+natural` |
| gat_v2 | 256 | backward | 255.1470 | 255.1470 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gat_v2 | 256 | forward | 38.5459 | 37.6463 | **1.02x** | `grid_stride+natural` |
| gt | 128 | backward | 200.2668 | 200.2668 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 128 | forward | 46.9673 | 45.9018 | **1.02x** | `precomputed+natural` |
| gt | 256 | backward | 270.7272 | 270.7272 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 256 | forward | 52.1615 | 52.1615 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 128 | backward | 17.6263 | 13.1830 | **1.34x** | `grid_stride+degree` |
| min_aggr | 128 | forward | 44.6280 | 43.5231 | **1.03x** | `precomputed+natural` |
| min_aggr | 256 | backward | 38.0180 | 24.9781 | **1.52x** | `grid_stride+locality` |
| min_aggr | 256 | forward | 57.0860 | 54.1757 | **1.05x** | `precomputed+natural` |

### city-roads-L

`N=142,257` &middot; `E=421,318` &middot; `avg_deg=3.0` &middot; `max_deg=10` &middot; `heavy_fwd=4,899` &middot; geomean **1.033x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 1.0800 | 1.0535 | **1.03x** | `grid_stride+degree` |
| gat_v2 | 128 | forward | 0.2813 | 0.2726 | **1.03x** | `one_per_block+locality` |
| gat_v2 | 256 | backward | 1.8050 | 1.7972 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| gat_v2 | 256 | forward | 0.3762 | 0.3723 | **1.01x** | `grid_stride+natural` |
| gt | 128 | backward | 1.2353 | 1.2062 | **1.02x** | `grid_stride+degree` |
| gt | 128 | forward | 0.4415 | 0.4235 | **1.04x** | `one_per_block+locality` |
| gt | 256 | backward | 2.4129 | 2.3657 | **1.02x** | `one_per_block+locality` |
| gt | 256 | forward | 0.5377 | 0.5139 | **1.05x** | `one_per_block+locality` |
| min_aggr | 128 | backward | 0.3753 | 0.3688 | **1.02x** | `one_per_block+locality` |
| min_aggr | 128 | forward | 0.2938 | 0.2700 | **1.09x** | `grid_stride+locality` |
| min_aggr | 256 | backward | 0.7240 | 0.7240 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 256 | forward | 0.5722 | 0.5225 | **1.10x** | `one_per_block+locality` |

### city-roads-M

`N=57,073` &middot; `E=189,643` &middot; `avg_deg=3.3` &middot; `max_deg=7` &middot; `heavy_fwd=4,705` &middot; geomean **1.031x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 0.5134 | 0.4980 | **1.03x** | `one_per_block+locality` |
| gat_v2 | 128 | forward | 0.1377 | 0.1367 | **1.01x** | `grid_stride+locality` |
| gat_v2 | 256 | backward | 0.8275 | 0.8094 | **1.02x** | `one_per_block+locality` |
| gat_v2 | 256 | forward | 0.1864 | 0.1795 | **1.04x** | `grid_stride+locality` |
| gt | 128 | backward | 0.5569 | 0.5473 | **1.02x** | `one_per_block+locality` |
| gt | 128 | forward | 0.2002 | 0.1877 | **1.07x** | `one_per_block+locality` |
| gt | 256 | backward | 1.0708 | 1.0421 | **1.03x** | `one_per_block+locality` |
| gt | 256 | forward | 0.2478 | 0.2472 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| min_aggr | 128 | backward | 0.1629 | 0.1629 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 128 | forward | 0.1431 | 0.1362 | **1.05x** | `one_per_block+locality` |
| min_aggr | 256 | backward | 0.3080 | 0.3055 | **1.01x** | `one_per_block+locality` |
| min_aggr | 256 | forward | 0.2630 | 0.2393 | **1.10x** | `one_per_block+locality` |

### ogbn-proteins

`N=132,534` &middot; `E=79,255,038` &middot; `avg_deg=598.0` &middot; `max_deg=7,751` &middot; `heavy_fwd=1,329` &middot; geomean **1.020x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 63.8239 | 63.7737 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| gat_v2 | 128 | forward | 17.3103 | 17.3103 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gat_v2 | 256 | backward | 139.7156 | 126.0093 | **1.11x** | `one_per_block+locality` |
| gat_v2 | 256 | forward | 26.9848 | 26.5236 | **1.02x** | `one_per_block+locality` |
| gt | 128 | backward | 83.0802 | 83.0802 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 128 | forward | 18.8817 | 18.8817 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 256 | backward | 161.0906 | 160.9134 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| gt | 256 | forward | 33.5012 | 33.5012 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 128 | backward | 0.5381 | 0.5328 | **1.01x** | `grid_stride+natural` |
| min_aggr | 128 | forward | 9.4022 | 9.0370 | **1.04x** | `precomputed+natural` |
| min_aggr | 256 | backward | 1.0445 | 1.0445 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 256 | forward | 18.1690 | 16.9570 | **1.07x** | `precomputed+natural` |

### pokec-regions

`N=1,632,803` &middot; `E=32,255,367` &middot; `avg_deg=19.8` &middot; `max_deg=13,734` &middot; `heavy_fwd=16,670` &middot; geomean **1.008x**

| conv | dim | pass | baseline | best | speedup | winning config |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| gat_v2 | 128 | backward | 49.7152 | 49.7152 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gat_v2 | 128 | forward | 13.2221 | 12.5006 | **1.06x** | `one_per_block+degree` |
| gat_v2 | 256 | backward | 90.3905 | 90.3905 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gat_v2 | 256 | forward | 19.9666 | 19.9666 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 128 | backward | 58.4929 | 58.4929 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 128 | forward | 19.0070 | 19.0070 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 256 | backward | 121.2273 | 121.2273 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| gt | 256 | forward | 36.3310 | 36.3310 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 128 | backward | 12.1766 | 11.8926 | **1.02x** | `one_per_block+locality` |
| min_aggr | 128 | forward | 11.1506 | 11.1506 | **1.00x** | `one_per_block+natural` *(baseline already best)* |
| min_aggr | 256 | backward | 23.5978 | 23.5205 | **1.00x** | `one_per_block+locality` *(baseline already best)* |
| min_aggr | 256 | forward | 21.6550 | 21.3576 | **1.01x** | `one_per_block+locality` |

## Where nothing helped

35 of 192 cells were already best on the untouched baseline. They cluster,
and the clustering is informative:

- **pokec-regions** — 9 of 12 cells
- **ogbn-proteins** — 7 of 12 cells
- **web-fraud** — 5 of 12 cells
- **ogbn-products** — 4 of 12 cells
- **artnet-exp** — 2 of 12 cells
- **avazu-ctr** — 2 of 12 cells
- **city-roads-L** — 2 of 12 cells
- **city-roads-M** — 2 of 12 cells
- **ogbn-arxiv** — 1 of 12 cells
- **tolokers-2** — 1 of 12 cells

`pokec-regions` and `ogbn-proteins` account for most of them. Both are large and already
stored in a locality-friendly order, so the largest lever is spent before we start; and
25 of the 35 are backward cells, whose kernels scatter through float
`atomicAdd` rather than reading a neighbourhood per output row, leaving far less reuse for a
visit order to improve.

## Files

- `summary.txt` — the summariser's full text output.
- `grid.json` — every cell as one JSON record (baseline, best, winning config, graph facts).
- `*.json` — one file per run, each carrying all three node orders in its `sweep` array.
  Gitignored; regenerate with the commands above.
