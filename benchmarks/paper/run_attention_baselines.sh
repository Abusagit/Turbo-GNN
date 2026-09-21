#!/usr/bin/env bash
# The DGL and PyG columns of Table 1 and Appendix J: GATv2 forward, fp16, same graphs, same
# harness and timing loop as run_attention_ablation.sh.
#
#   benchmarks/paper/run_attention_baselines.sh dgl GPU [HEADS:WIDTH ...]
#   benchmarks/paper/run_attention_baselines.sh pyg GPU [HEADS:WIDTH ...]
#
# Default HEADS:WIDTH is the Appendix J grid; Table 1 needs only 2:128.
#
# The frameworks expose no kernel parameters, so each (graph, HEADS, WIDTH) is one measurement.
# DGL needs PyTorch 2.4 and therefore its own interpreter (BASELINE_PY); that environment
# cannot load our extension, so it runs with SKEWGNN_ALLOW_MISSING_EXTENSION=1 (see README).
set -u
cd "$(dirname "$0")/../.."
BACKEND=${1:?usage: $0 dgl|pyg GPU [HEADS:WIDTH ...]}
GPU=${2:?usage: $0 dgl|pyg GPU [HEADS:WIDTH ...]}
shift 2
COMBOS=("$@")
[ ${#COMBOS[@]} -eq 0 ] && COMBOS=(2:64 2:128 2:256 4:64 4:128 4:256 8:64 8:128 8:256)
case "$BACKEND" in
  dgl) PY=${BASELINE_PY:-python}; export DGLBACKEND=pytorch SKEWGNN_ALLOW_MISSING_EXTENSION=1 ;;
  pyg) PY=${OURS_PY:-python} ;;
  *) echo "backend must be dgl or pyg" >&2; exit 2 ;;
esac
OUT=${RESULTS:-results/attention_baselines}/$BACKEND; mkdir -p "$OUT"

mapfile -t CFG < <("$PY" -c "
import sys; sys.path.insert(0, 'benchmarks/paper')
from graphs import ATTENTION_GRAPHS
print('\n'.join(c for _, c in ATTENTION_GRAPHS))")

for COMBO in "${COMBOS[@]}"; do
  H=${COMBO%%:*}; D=${COMBO##*:}
  for DS in "${CFG[@]}"; do
    NAME=$(basename "$DS" .yaml)
    J="$OUT/gat_v2-h${H}-d${D}-forward__${NAME}.json"
    [ -s "$J" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU timeout 3600 "$PY" scripts/benchmark_kernels.py \
      --backend "$BACKEND" --conv gat_v2 --dataset "$DS" --feature-dim "$D" --heads "$H" \
      --mode forward --dtype fp16 --json-out "$J" \
      > /dev/null 2>"${J%.json}.err"
    echo "  $BACKEND h$H d$D $NAME rc=$?"
  done
done
