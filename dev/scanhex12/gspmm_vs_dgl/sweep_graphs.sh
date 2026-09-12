#!/usr/bin/env bash
# Turbo vs DGL over every prepared graph, smallest first.
#
#   cd dev/scanhex12/gspmm_vs_dgl
#   nohup ./sweep_graphs.sh > sweep.log 2>&1 &
#
# One run.sh invocation per (graph, dtype), each measuring forward, backward and
# forward+backward for the whole 6x3 op table.  Resumable: a results file that
# already exists and is non-empty is skipped, so an interrupted sweep continues
# where it stopped.
#
# Past EDGE_LIMIT edges a graph measures only the ops that read no edge data
# (copy_u).  An [E, d] operand and its gradient come to 2*E*d*sizeof(dtype) and
# both libraries hold one at once, but the operands are not the whole cost -- the
# reference gradients and DGL's own temporaries scale with E too, and at fp16 a
# limit of 22M edges (half the fp32 one, as the arithmetic suggests) still ran a
# 13M-edge graph out of memory.  Ten million holds for both dtypes.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_DGL="${PYTHON_DGL:-/tmp/bench-env/venv/bin/python}"
export PYTHON_TURBO="${PYTHON_TURBO:-/tmp/bench-env/venv13/bin/python}"
export PYTHON_PLOT="${PYTHON_PLOT:-$PYTHON_DGL}"
export GRAPH_CACHE="${GRAPH_CACHE:-/home/ubuntu/work/graph_cache}"
export OGB_ROOT="${OGB_ROOT:-/home/ubuntu/work/ogb}"
export WORK="${WORK:-/home/ubuntu/work/gspmm_scratch}"
export DIMS="${DIMS:-64}"
export ITERS="${ITERS:-30}"
export REPEATS="${REPEATS:-2}"
export BACKWARD="${BACKWARD:-1}"
export STAGES="${STAGES:-0}"

GRAPHS="${GRAPHS:-citeseer cora pubmed city-roads-M city-roads-L artnet-exp tolokers-2 ogbn-arxiv city-reviews web-fraud twitch-views hm-categories avazu-ctr pokec-regions ogbn-proteins ogbn-products}"
DTYPES="${DTYPES:-float32 float16}"

started=$(date +%s)
for graph in $GRAPHS; do
    for dtype in $DTYPES; do
        tag="${dtype}"
        [ "$BACKWARD" = "1" ] && tag="${tag}_bwd"
        [ "$STAGES" != "0" ] && tag="${tag}_st${STAGES}"
        out="$HERE/results/results_${graph}_${tag}.json"
        if [ -s "$out" ]; then
            echo "== skip $graph $dtype (have $(basename "$out"))"
            continue
        fi
        limit=10000000
        echo "############ $graph $dtype  ($(date +%H:%M:%S), $(( ($(date +%s) - started) / 60 )) min in)"
        GRAPHS="$graph" DTYPE="$dtype" EDGE_LIMIT="${EDGE_LIMIT:-$limit}" "$HERE/run.sh" \
            || echo "!! failed: $graph $dtype"
    done
done
echo "sweep finished $(date -Is), $(( ($(date +%s) - started) / 60 )) min"
