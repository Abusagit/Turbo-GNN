#!/bin/bash
# Lean g-SDDMM sweep, both backends, on one idle GPU. Refuses to start on a busy GPU.
set -u
cd "$(dirname "$0")"
GPU=$1
read mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $GPU | tr -d ',')
if [ "$mem" -ge 1000 ] || [ "$util" -gt 2 ]; then echo "GPU $GPU busy ($mem MiB, $util%), not launching"; exit 1; fi
G="cora citeseer pubmed CoraFull WikiCS amazon-photo amazon-computers coauthor-cs coauthor-physics Flickr city-roads-M city-roads-L artnet-exp artnet-views tolokers-2 ogbn-arxiv web-topics city-reviews reddit twitch-views hm-categories hm-prices avazu-ctr pokec-regions ogbn-products"
export CUDA_VISIBLE_DEVICES=$GPU
echo "LAUNCH GPU $GPU $(date -Is)"
"${OURS_PY:-python}" gsddmm_all.py --backend ours --graphs $G --out gsddmm_all_ours.jsonl 2>&1 | grep --line-buffered -v Warning
"${BASELINE_PY:-python}" gsddmm_all.py --backend dgl --graphs $G --out gsddmm_all_dgl.jsonl 2>&1 | grep --line-buffered -v Warning
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i $GPU
echo "ALL DONE $(date -Is)"
