#!/bin/bash
# Launch the PyG/DGL sweep on the first GPU that stays idle (<1 GB used, <=2% util) for 3 checks
# in a row. Shared box: a contended GPU corrupts the ratios, so never launch on a busy one.
set -u
cd "$(dirname "$0")"
PY="${BASELINE_PY:-python}"
SP="citeseer cora pubmed city-roads-M city-roads-L artnet-exp tolokers-2 ogbn-arxiv city-reviews twitch-views web-fraud hm-categories avazu-ctr pokec-regions ogbn-proteins ogbn-products"
SD="cora citeseer pubmed CoraFull WikiCS amazon-photo amazon-computers coauthor-cs coauthor-physics Flickr city-roads-M city-roads-L artnet-exp artnet-views tolokers-2 ogbn-arxiv web-topics city-reviews reddit twitch-views hm-categories hm-prices avazu-ctr pokec-regions ogbn-products"
declare -A streak
while true; do
  while IFS=', ' read -r idx mem util; do
    if [ "$mem" -lt 1000 ] && [ "$util" -le 2 ]; then streak[$idx]=$(( ${streak[$idx]:-0} + 1 )); else streak[$idx]=0; fi
    if [ "${streak[$idx]}" -ge 3 ]; then GPU=$idx; break 2; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  sleep 60
done
echo "LAUNCH on GPU $GPU at $(date -Is)"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
export CUDA_VISIBLE_DEVICES=$GPU
$PY pyg_vs_dgl.py --family gsddmm --graphs $SD --out gsddmm_pyg_dgl.jsonl 2>&1 | grep -v Warning
$PY pyg_vs_dgl.py --family gspmm  --graphs $SP --out gspmm_pyg_dgl.jsonl 2>&1 | grep -v Warning
echo "GPU state at end:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "SWEEP DONE rc=$? $(date -Is)"
