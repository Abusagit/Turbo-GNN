# Autotuned kernels vs the pre-scheduler baseline

Generated from `reports/autotune-comparison/` at `3f761b0` on 2026-08-23.

A100-SXM4-80GB. Every run on a GPU held **exclusively** for its duration — see *Measurement
hygiene* below, which is not boilerplate here.

## What is being compared

| | configuration |
| --- | --- |
| **baseline** | `schedule=one_per_block`, both bucket launches sequential, autotuning off — the launch these kernels used before any of this work |
| **autotuned** | `--autotune`: a grid search over the declared `TunableParam`s, which now include `forward_bucket_launch` and `backward_bucket_launch` |

16 graphs x 3 convs x head dims 128/256 x forward and backward = **192 cells, 384 runs,
0 failures**. The search cost is excluded from the timed region, so what is compared is the
chosen configuration's steady-state cost, not the cost of finding it.

## Headline

| | geomean | cells at or above baseline |
| --- | ---: | ---: |
| head dim 128, forward | **1.5712x** | 45/48 |
| head dim 128, backward | **1.0304x** | 26/48 |
| head dim 256, forward | **1.4921x** | 47/48 |
| head dim 256, backward | **1.0182x** | 28/48 |
| **overall** | **1.2523x** | 146/192 |

Range: worst 0.806x, best 9.09x. By convolution: gat_v2 1.286x, gt 1.216x, min_aggr 1.256x.

## The important caveat: what is actually responsible

The baseline is untuned in **every** dimension, not just the new ones, so this comparison
measures the autotuner as a whole rather than the scheduler or stream work. Ablating three
of the largest wins, one parameter at a time, shows how the gain is really distributed:

| cell | graph partition | warps | **bucket concurrency** | total |
| --- | ---: | ---: | ---: | ---: |
| web-fraud `min_aggr` d=128 fwd | +1.68x | +1.30x | **1.00x** | 9.10x |
| ogbn-arxiv `gt` d=128 fwd | 1.00x | 2.11x | **+1.20x** | 2.54x |
| tolokers-2 `gat_v2` d=128 fwd | 1.60x | +1.13x | **+1.22x** | 2.20x |

web-fraud's 9.09x — the largest number in the whole matrix — owes **nothing** to the stream
work. It is 4.18x from `edges_per_block_heavy_nodes` alone (128 -> 1024), then the bucketing
quantile, then warps; adding concurrency changed the time by 0.000 ms. That graph has a
maximum degree of 228,991 against an average of 5.5, and the packed heavy kernel sizes its
grid as `(num_heavy, max_degree / edges_per_block)`, so the default 128 launches
`gridDim.y = 1,789`. This is finding (c) in `KERNEL_ISSUES.md` showing up as a 4x.

On the attention convolutions concurrency does pay, contributing **+1.20x and +1.22x** on
top of tuned warps — consistent with, and slightly above, the 1.11-1.14x measured in
isolation in `reports/kernel-benchmarks-streams/`, where every other parameter was held
fixed. That isolated figure remains the honest estimate of what the stream work is worth.

## The forward/backward split survives autotuning

Forward gains 1.49-1.57x; backward gains 1.02-1.03x. The search's own choices show why it
is not simply that backward is harder to tune:

| pass | chose `concurrent` | geomean when it chose `concurrent` | when it chose `sequential` |
| --- | ---: | ---: | ---: |
| forward | 59/96 (61%) | 1.715x | 1.277x |
| backward | 57/96 (59%) | 0.999x | 1.063x |

On the forward pass, choosing concurrency goes with a better outcome. On the backward pass it
goes with a *worse* one — cells where the search picked `concurrent` average 0.999x while
those where it picked `sequential` average 1.063x. **This is association, not causation:** the
search picks per cell, so the comparison is confounded by which cells are easy. But it points
the same way as the controlled sweep, where backward measured 0.92-0.99 under concurrency,
and it suggests the thin autotuning budget is sometimes choosing concurrency on backward
where it does not help.

## Per graph

| graph | nodes | avg deg | overall | forward | backward | best cell |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| web-fraud | 2,890,331 | 5.5 | **2.124** | 4.465 | 1.011 | 9.09x min_aggr/128/for |
| ogbn-arxiv | 169,343 | 7.9 | **1.463** | 2.090 | 1.024 | 3.00x gt/128/for |
| tolokers-2 | 11,758 | 89.3 | **1.403** | 1.986 | 0.991 | 2.67x gat_v2/128/for |
| avazu-ctr | 76,269 | 289.0 | **1.337** | 1.768 | 1.010 | 2.50x gat_v2/128/for |
| pubmed | 19,717 | 5.5 | **1.293** | 1.363 | 1.227 | 3.16x gat_v2/128/bac |
| city-reviews | 148,801 | 16.7 | **1.291** | 1.630 | 1.022 | 2.00x gat_v2/128/for |
| twitch-views | 168,114 | 81.9 | **1.248** | 1.580 | 0.985 | 2.01x gat_v2/128/for |
| city-roads-M | 57,073 | 3.3 | **1.240** | 1.526 | 1.007 | 1.99x min_aggr/256/for |
| hm-categories | 46,563 | 461.9 | **1.219** | 1.491 | 0.997 | 1.93x gat_v2/128/for |
| city-roads-L | 142,257 | 3.0 | **1.186** | 1.415 | 0.995 | 1.81x min_aggr/256/for |
| artnet-exp | 50,405 | 12.1 | **1.163** | 1.325 | 1.020 | 1.69x min_aggr/128/for |
| citeseer | 3,327 | 3.7 | **1.155** | 1.248 | 1.070 | 1.60x min_aggr/128/for |
| cora | 2,708 | 4.9 | **1.089** | 1.067 | 1.112 | 1.37x min_aggr/128/for |
| ogbn-proteins | 132,534 | 598.0 | **1.072** | 1.155 | 0.995 | 1.39x gat_v2/128/for |
| pokec-regions | 1,632,803 | 19.8 | **1.051** | 1.143 | 0.967 | 1.31x gat_v2/128/for |
| ogbn-products | 2,449,029 | 51.5 | **1.020** | 1.057 | 0.984 | 1.17x gat_v2/128/for |

## Largest gains and losses

| cell | baseline | autotuned | | picked |
| --- | ---: | ---: | ---: | --- |
| web-fraud min_aggr d=128 forward | 44.6264 ms | 4.9085 ms | **9.09x** | bl=concurrent |
| web-fraud min_aggr d=256 forward | 57.0890 ms | 8.7405 ms | **6.53x** | bl=concurrent |
| web-fraud gt d=128 forward | 47.0415 ms | 13.1155 ms | **3.59x** | bl=concurrent |
| web-fraud gat_v2 d=256 forward | 38.5475 ms | 11.1759 ms | **3.45x** | bl=concurrent |
| web-fraud gat_v2 d=128 forward | 28.8324 ms | 8.6240 ms | **3.34x** | bl=concurrent |
| web-fraud gt d=256 forward | 52.1288 ms | 16.1618 ms | **3.23x** | bl=concurrent |
| pubmed gat_v2 d=128 backward | 2.6376 ms | 0.8354 ms | **3.16x** | bl=sequential |
| ogbn-arxiv gt d=128 forward | 2.7572 ms | 0.9193 ms | **3.00x** | bl=concurrent |

| worst cells | baseline | autotuned | | picked |
| --- | ---: | ---: | ---: | --- |
| hm-categories gat_v2 d=256 backward | 64.3031 ms | 72.1684 ms | **0.89x** | bl=concurrent |
| city-roads-M min_aggr d=128 backward | 0.1634 ms | 0.1863 ms | **0.88x** | bl=sequential |
| pokec-regions min_aggr d=256 backward | 23.6572 ms | 27.1773 ms | **0.87x** | bl=sequential |
| city-roads-L min_aggr d=128 backward | 0.3725 ms | 0.4377 ms | **0.85x** | bl=sequential |
| cora gat_v2 d=128 forward | 0.0387 ms | 0.0478 ms | **0.81x** | bl=sequential |
| tolokers-2 min_aggr d=128 backward | 0.0664 ms | 0.0824 ms | **0.81x** | bl=concurrent |

46 of 192 cells are below baseline, the worst at 0.806x. An
exhaustive search should not lose to a configuration inside its own space, so these are
selection error: with 1 warmup and 3 timed iterations per trial, argmin over thousands of
noisy candidates sometimes picks a configuration that measured fast by luck. All the losing
cells are sub-millisecond or backward, which is where that bites hardest.

## Measurement hygiene

The first attempt at this comparison was **discarded**. It ran on GPUs [0, 5, 6]; other
tenants arrived partway through, and — worse — investigation scripts were run on GPU 0 while
the matrix was still using it. Per-GPU geomeans diverged 1.34 / 1.17 / 1.14, and one cell
reported 0.13x that could not be reproduced afterwards. That data is kept in
`reports/autotune-comparison-CONTAMINATED/` as a record, and is not used anywhere.

`scripts/free_gpus.py` only checked at launch, which is not enough for a multi-hour run. It
now exposes `is_free()` / `wait_until_free()`, and the runner waits for an exclusive GPU
before each run, re-checks afterwards, and **discards and retries any run whose GPU was
shared for any part of it**. This run recorded 0 discards on a dedicated GPU.

## Two bugs found while building this

**Stale defaults in the autotuner's kernel classes.** `turbo_gnn/_kernels.py` hardcoded
`schedule="dynamic"` and `blocks_per_sm=8`, left behind when the library defaults changed to
`one_per_block` / 1024. By the earlier measurements that is close to the worst configuration
in the entire matrix, so every autotuned run was handicapped: the first comparison showed
backward at 0.62-0.96x purely because of it.

**Sixteen identical candidates per configuration.** `features_per_block` and `tiles_y` are
read only by the 2-D heavy kernel, but were searched unconditionally, so every non-2-D
configuration appeared 16 times. The search timed each separately and took the argmin, which
means noise decided the winner among behaviourally identical candidates. A
`canonicalise_config` hook now collapses them: grid 2,688 -> 1,428.

**`tune_backward` defaulted to `False`**, so backward parameters were never searched at all;
a `--mode backward --autotune` run reported the *default* backward configuration under an
"autotuned" label. It now follows `--mode`.

## Files

- `summary.txt` — full text output of `scripts/summarize_autotune_comparison.py`.
- `grid.json` — every cell with both configurations, the chosen parameters and the speedup.
- `*.json` — one file per run. Gitignored; regenerate with `scripts/run_autotune_comparison.py`.
- Related: `reports/kernel-benchmarks/` (schedule x node order), `reports/kernel-benchmarks-streams/` (bucket launch in isolation), `reports/occupancy/`.
