#!/usr/bin/env bash
# Five-panel occupancy figures for the paper, as PDF, with the LaTeX that wraps them.
#
#   bash dev/sinfillo/paper_figures.sh            # the six setups worth printing, ~1 hour
#   ALL=1 bash dev/sinfillo/paper_figures.sh      # every calibrated cell, ~19 hours
#
# Output lands in reports/figures: one PDF per setup, figures.tex with the mode legend, the
# definition of T/T* and a caption per figure, and preview.tex to compile it on its own.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=${PY:-.venv/bin/python3}
MODEL=${MODEL:-cost_models.json}
OUT=${OUT:-reports/figures}
SMS=${SMS:-108}
BANDWIDTH=${BANDWIDTH:-2039}
MIN_FREE_MB=${MIN_FREE_MB:-2048}

# The PDFs are tiny, but the disk on this box has been full twice and a write that fails
# halfway leaves a truncated file, so refuse to start without headroom.
free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
  echo "only ${free_mb} MB free, want at least ${MIN_FREE_MB}. Free some space or lower MIN_FREE_MB." >&2
  exit 1
fi
echo "${free_mb} MB free, output needs about 10"

# Picked from the sweep: the four largest gaps between the bucketed baseline and
# concurrent + slicing, plus tolokers-2, which is the one setup where the drain tail after
# slicing is wide enough to see.
SETUPS=${SETUPS:-"
web-fraud:gt:backward:128:0.99
web-fraud:min_aggr:backward:128:0.99
web-fraud:gat_v2:forward:256:0.99
web-fraud:gt:forward:128:0.99
ogbn-arxiv:gat_v2:forward:128:0.99
tolokers-2:gat_v2:forward:128:0.99
"}

if [ "${ALL:-0}" = "1" ]; then
  GRAPHS=${GRAPHS:-"cora citeseer pubmed city-roads-M artnet-exp tolokers-2 city-roads-L \
ogbn-arxiv city-reviews twitch-views hm-categories avazu-ctr pokec-regions web-fraud"}
  # Cells come from the calibration, minus the ones it flagged as not describable by
  # alpha + beta * degree -- plotting those would put meaningless ratios in a caption.
  SETUPS=$($PY - "$MODEL" "$GRAPHS" <<'PYEOF'
import json, sys
models = json.load(open(sys.argv[1]))["models"]
cells = sorted(c for c, v in models.items() if not v.get("note"))
for graph in sys.argv[2].split():
    for cell in cells:
        conv, pass_name, dim = cell.split("/")
        for q in ("0.99", "0.95"):
            print(f"{graph}:{conv}:{pass_name}:{dim}:{q}")
PYEOF
)
  echo "full grid: $(echo "$SETUPS" | grep -c .) figures, expect around 19 hours"
fi

$PY scripts/ablation/plot_heatmap_grid.py \
    --setups $SETUPS \
    --cost-model "$MODEL" --sms "$SMS" --memory-bandwidth-gbps "$BANDWIDTH" \
    --out-dir "$OUT" --standalone

echo
echo "done. $(ls "$OUT"/*.pdf 2>/dev/null | wc -l) PDFs, $(du -sh "$OUT" | cut -f1) total"
df -Pm . | awk 'NR==2 {print $4 " MB still free"}'
