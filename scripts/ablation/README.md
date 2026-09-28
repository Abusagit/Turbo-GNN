# Load-imbalance simulator: from an empty machine to the figures

The simulator charges a vertex `alpha + beta * degree` ticks and schedules the resulting
blocks onto SM slots. `beta` is fixed at 1 by definition -- one tick is the cost of one
neighbour -- so the whole model rests on two numbers measured from the real kernels: `alpha`,
the per-vertex prologue and epilogue, and the duration of a tick.

Both belong to the card they were measured on. **Measure and simulate on the same GPU**, or
the run silently mixes two machines.

## The pipeline

| # | step | script | needs a GPU | cost |
|---|---|---|---|---|
| 1 | time the real kernels | `measure_kernel_times.py` | **yes** | ~1 min |
| 2 | fit the cost model | `calibrate_cost_model.py` | no | seconds |
| 3 | sweep the grid | `sweep_all.py` | no | hours |
| 4 | draw | `plot_heatmap_grid.py` and friends | no | minutes to hours |

Only step 1 touches the GPU. Everything after it is CPU-bound and can run while the card is
busy with something else.

### 1. Measure

```bash
python scripts/ablation/measure_kernel_times.py \
    --graphs cora citeseer pubmed tolokers-2 ogbn-arxiv ... \
    --warmup 10 --iters 30 --out reports/measurements.jsonl
```

Times `gt`, `gat_v2` and `min_aggr`, forward and backward, at head dimensions 128 and 256,
on every graph given, using CUDA events. One JSON record per (graph, conv, pass, head dim)
with the graph's node and edge counts, the milliseconds per launch and the device name. A
graph that will not load is skipped with its reason and the run continues.

### 2. Calibrate

```bash
python scripts/ablation/calibrate_cost_model.py \
    --from-json reports/measurements.jsonl --out cost_models.json
```

Regresses `t(N, E) = c + a*N + b*E` per (conv, pass, head dim), under `a, b >= 0`. Since
`sum(alpha + beta * degree) = alpha*N + beta*E`, normalising `beta` to 1 gives
`alpha = a/b` and a tick of `b` nanoseconds. Prints a table: `alpha`, ns per tick, launch
overhead, R2, median error, and what share of the time edges explain.

Read the "do not predict wall clock from these cells" list. A cell is flagged when `alpha`
hits the non-negativity constraint, when edges explain under 20% of its time, or when the
median error exceeds 35%. Flagged cells are skipped by every later step. One or two flags is
normal -- min-aggregation's backward pass is O(N), not O(E), so no `alpha + beta * degree`
model describes it.

`cost_models.json` is **not** committed: it is only valid for the card it was measured on.

### 3. Sweep

```bash
python scripts/ablation/sweep_all.py --cost-model cost_models.json \
    --sms 108 --memory-bandwidth-gbps 2039 --out reports/sweep.csv
```

One row per (graph, cell, quantile, launch mode, split-K): makespan `T`, the perfect-packing
time `T*`, which of the three lower bounds attains it, the ratio, the drain tail, mean slot
occupancy and predicted milliseconds. This is where the conclusions come from; the figures
only illustrate them.

Resumable -- rows already in the CSV are skipped -- and dominated by the graphs with the
largest hub, since a run without slicing lasts about as long as the biggest vertex.

### 4. Draw

```bash
# Five panels per figure, as PDF, for the paper.
python scripts/ablation/plot_heatmap_grid.py \
    --setups tolokers-2:gat_v2:forward:128:0.99 ogbn-arxiv:gt:forward:128:0.99 \
    --cost-model cost_models.json --sms 108 --memory-bandwidth-gbps 2039 \
    --out-dir reports/figures

# Two panels: the bucketed baseline against concurrent + edge slicing.
python scripts/ablation/plot_heatmap_pair.py --dataset tolokers-2 \
    --cost-model cost_models.json --conv gat_v2 --pass forward --head-dim 128 \
    --sms 108 --memory-bandwidth-gbps 2039 --out pair.png
```

`plot_heatmap_grid.py` prints the graph's shape and every panel's `T`, `T*` and `T/T*`, which
is everything a caption needs. Pass `--tex` if you also want that written as LaTeX.

The older figure scripts are still here: `plot_all.py` walks the whole grid into per-cell
directories of PNGs, `plot_launch_modes.py` draws one such figure, and
`plot_occupancy_profile.py` collapses the SM axis into a single occupancy curve.

## Exploring a single configuration

`simulate_load_imbalance.py` is the layer underneath all of the above, and runs one point of
the space with every knob exposed -- assignment order, vertices per block, blocks per SM,
occupancy, launch mode, slice size. Start here when changing the model rather than using it.
`dump_timeline.py` prints the resulting schedule block by block.

## Checking that the simulation is sound

Three things are worth re-deriving after any change to the cost model.

**The tick should track memory bandwidth.** It is a per-neighbour cost, and that cost is
memory-bound, so moving to a card with `k` times less bandwidth should multiply it by `k`.
Between an H100 and an A100 the measured ratio was 1.62 against a bandwidth ratio of 1.64.
Nothing in the calibration knows about bandwidth, so this is an independent check.

**`alpha` should be roughly card-independent.** It is a ratio of two fitted coefficients, so
a uniform slowdown cancels. Large moves in `alpha` between cards mean the fit is unstable,
not that the hardware changed.

**`T/T*` should never fall below 1.** It is a makespan over a lower bound on the same work;
a value under 1 means a bound is computed wrong.

Unit tests cover the fit and the launch barrier:

```bash
python -m pytest tests/unit/test_cost_model_calibration.py \
                 tests/unit/test_load_imbalance_simulation.py
```

## Runbooks

`dev/sinfillo/recalibrate.sh` runs steps 1--4 end to end on a new card.
`dev/sinfillo/paper_figures.sh` runs step 4 for the setups chosen for the paper.
