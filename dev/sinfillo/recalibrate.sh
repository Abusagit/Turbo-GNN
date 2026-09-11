#!/usr/bin/env bash
# Recalibrate the load-imbalance simulator on whichever GPU this machine has, then redraw
# everything from it.
#
# alpha is a ratio of two per-kernel constants and the tick is a duration, so neither travels
# between architectures: measure and simulate on the same card or the numbers mix two machines.
#
#   bash dev/sinfillo/recalibrate.sh              # measure, calibrate, sweep, draw
#   STAGE=measure bash dev/sinfillo/recalibrate.sh   # just one stage
#
# Stages are independent and skip work that is already done, so a run can be interrupted and
# restarted. Expect minutes for the measurement and hours for the sweep and the figures.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=${PY:-.venv/bin/python3}
GRAPHS=${GRAPHS:-"cora citeseer pubmed city-roads-M artnet-exp tolokers-2 city-roads-L ogbn-arxiv \
city-reviews twitch-views hm-categories avazu-ctr pokec-regions web-fraud"}
STAGE=${STAGE:-all}
OUT=${OUT:-reports}
MODEL=${MODEL:-cost_models.json}

# A100-SXM4-80GB. Override for another card: an H100 SXM is 132 and 3350.
SMS=${SMS:-108}
BANDWIDTH=${BANDWIDTH:-2039}

run() { echo; echo "=== $*"; "$@"; }

if [ "$STAGE" = all ] || [ "$STAGE" = measure ]; then
  run $PY scripts/ablation/measure_kernel_times.py \
      --graphs $GRAPHS --warmup 10 --iters 30 --out "$OUT/measurements.jsonl"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = calibrate ]; then
  run $PY scripts/ablation/calibrate_cost_model.py \
      --from-json "$OUT/measurements.jsonl" --out "$MODEL"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = sweep ]; then
  run $PY scripts/ablation/sweep_all.py --cost-model "$MODEL" \
      --graphs $GRAPHS --sms "$SMS" --memory-bandwidth-gbps "$BANDWIDTH" \
      --out "$OUT/sweep.csv"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = figures ]; then
  run $PY scripts/ablation/plot_all.py --cost-model "$MODEL" \
      --graphs $GRAPHS --sms "$SMS" --memory-bandwidth-gbps "$BANDWIDTH" \
      --out "$OUT/launch-modes"
fi

echo
echo "done. $MODEL, $OUT/sweep.csv, $OUT/launch-modes/"
