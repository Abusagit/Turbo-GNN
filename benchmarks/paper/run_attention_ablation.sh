#!/usr/bin/env bash
# Table 1 and Appendix J: the GATv2 forward ablation.
#
#   benchmarks/paper/run_attention_ablation.sh GPU REPEAT [HEADS:WIDTH ...]
#
#   Table 1      HEADS:WIDTH = 2:128
#   Appendix J   HEADS:WIDTH = 2:64 2:128 2:256 4:64 4:128 4:256 8:64 8:128 8:256 (the default)
#   REPEAT       a tag such as r1; the paper reports the median of r1, r2 and r3
#
# WIDTH is --feature-dim, which for GATv2 is the width of each head: inputs are shaped
# [nodes, HEADS, WIDTH].
#
# Each (graph, HEADS, WIDTH) cell is one scripts/benchmark_kernels.py process that times the
# full 160-point factorial below on the same tensors, so every rung of the ladder is compared
# in one process. The rungs are then read out of that sweep by summarize_attention_ablation.py.
#
#   split quantile   -1 (no bucketing, the prior kernel), 0.85, 0.9, 0.99, 0.995
#   node order       natural, degree
#   bucket launch    sequential, concurrent
#   heavy slicing    off (0), 1024 edges per slice
#   staging depth    0, 6 for the light and the heavy bucket separately
#
# Cells that run out of memory are recorded with their exit code and skipped.
set -u
cd "$(dirname "$0")/../.."
GPU=${1:?usage: $0 GPU REPEAT [HEADS:WIDTH ...]}
R=${2:?usage: $0 GPU REPEAT [HEADS:WIDTH ...]}
shift 2
COMBOS=("$@")
[ ${#COMBOS[@]} -eq 0 ] && COMBOS=(2:64 2:128 2:256 4:64 4:128 4:256 8:64 8:128 8:256)
PY=${OURS_PY:-python}
RESULTS=${RESULTS:-results/attention_ablation}

mapfile -t CFG < <("$PY" -c "
import sys; sys.path.insert(0, 'benchmarks/paper')
from graphs import ATTENTION_GRAPHS
print('\n'.join(c for _, c in ATTENTION_GRAPHS))")

for COMBO in "${COMBOS[@]}"; do
  H=${COMBO%%:*}; D=${COMBO##*:}
  OUT="$RESULTS/h${H}-d${D}-$R"; mkdir -p "$OUT"
  for DS in "${CFG[@]}"; do
    NAME=$(basename "$DS" .yaml)
    J="$OUT/${NAME}__forward.json"
    [ -s "$J" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU timeout 7200 "$PY" scripts/benchmark_kernels.py \
      --backend cuda --conv gat_v2 --dataset "$DS" --feature-dim "$D" --heads "$H" \
      --mode forward --dtype fp16 -K schedule=one_per_block \
      --sweep quantile=-1,0.85,0.9,0.99,0.995 --sweep node_order=natural,degree \
      --sweep-kernel forward_bucket_launch=sequential,concurrent \
      --sweep-kernel forward_heavy_edge_slice=0,1024 \
      --sweep-kernel pipeline_stages=0,6 \
      --sweep-kernel heavy_pipeline_stages=0,6 \
      --json-out "$J" > /dev/null 2>"$OUT/${NAME}.err"
    echo "  h$H d$D $R $NAME rc=$?"
  done
done
