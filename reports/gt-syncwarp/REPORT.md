# Fixing the warp-synchronous hazard in GT forward

Generated 2026-08-23 at `3f761b0`. A100-SXM4-80GB, GPUs held exclusively.

## The bug

`csrc/gt/gt_forward.cu`, cross-warp reduction. Lane 0 rewrites the shared `neighbor_sum` array
in place, then **all 32 lanes** of warp 0 read it a few lines later. The only thing between the
write and the reads was `__shfl_sync`, which converges the warp but is **not a memory fence** —
the classic warp-synchronous assumption that Volta's independent thread scheduling stopped
honouring. This was finding (f) in `KERNEL_ISSUES.md`, and it pre-dated all of the scheduler and
stream work (`git diff d3033b4..HEAD` never touched that logic).

## The fix

One line: `__syncwarp()` after the `lane_id == 0` block, before the reads.

| | before | after |
| --- | --- | --- |
| `compute-sanitizer --tool racecheck` | 7 errors, 39 warnings, **169,312 hazards** | **0 hazards, 0 errors, 0 warnings** |
| registers (GT fwd light, 4 warps, D=128, fp32) | 30 / 30 / 30 / 32 | **identical** |
| correctness suites | 770 passing | 770 passing |

The fix costs nothing: register allocation is unchanged for every policy, so occupancy is
unaffected.

## Performance: no measurable change

256 runs, 768 matched configurations (16 graphs x head dims 128/256 x both passes x 2 schedules
x 2 bucket launches x 3 node orders), before vs after:

| statistic | all | forward | backward |
| --- | ---: | ---: | ---: |
| **median** | **1.0003x** | 1.0003x | 1.0004x |
| geomean | 1.0623x | 1.0926x | 1.0328x |

Faster in 235/768 configurations, slower in 180/768; p10 0.970, p90 1.083.

**The median is the number to read.** It sits at 1.0003x — no change. The geomean of
1.062 is pulled up by a handful of outliers in *both* directions,
from 3.88x to 0.18x, on the cells already known to be unreliable: cross-validating the identical
baseline configuration across two independent matrices showed 30 of 192 cells disagreeing by
more than 1.5x. A symmetric spread around a median of exactly 1.000 is what a
performance-neutral change looks like when measured through that noise.

That is the expected outcome. `__syncwarp()` on an already-converged warp is close to free; the
compiler was evidently keeping the lanes in step anyway, which is precisely why the missing
fence was benign in practice and why the results were correct before. What changes is that the
code no longer *depends* on that accident.

## Files

- `*.json` — one file per run. Gitignored; regenerate with
  `scripts/run_kernel_benchmark_matrix.py --conv gt --out reports/gt-syncwarp`.
