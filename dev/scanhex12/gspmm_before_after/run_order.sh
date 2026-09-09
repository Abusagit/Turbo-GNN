#!/usr/bin/env bash
# Visit-order sweep: natural bucket order against descending degree, every
# prepared graph.  Same build on both sides -- only the bucket arrays differ.
#
#   nohup ./run_order.sh > order.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BENCH=/home/ubuntu/work/turbo-bench

export TORCH_CUDA_ARCH_LIST=7.5
export CUDA_HOME=/usr/local/cuda
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export PYTHONWARNINGS=ignore
export TURBO_REPO="${TURBO_REPO:-$REPO}"
# The partial extension is named after the checkout it is built from, and this
# one is the repo itself: "Turbo-GNN" has a hyphen in it, which is not a C
# identifier, so PYBIND11_MODULE would not compile.
export TURBO_EXT_NAME="${TURBO_EXT_NAME:-turbo_bench_order_C}"

PYTHON="${PYTHON:-$REPO/.venv/bin/python3}"
BENCH_CMD="${BENCH_CMD:-$PYTHON $BENCH/harness/bench_wrapper.py}"
GRAPH_CACHE="${GRAPH_CACHE:-/home/ubuntu/work/graph_cache}"
GRAPHS="${GRAPHS:-citeseer cora pubmed city-roads-M city-roads-L artnet-exp tolokers-2 ogbn-arxiv city-reviews web-fraud twitch-views hm-categories avazu-ctr pokec-regions ogbn-proteins ogbn-products}"
DTYPES="${DTYPES:-float16 float32}"
DIMS="${DIMS:-64}"
RESULTS="$HERE/results_order"
mkdir -p "$RESULTS"

for graph in $GRAPHS; do
    for dtype in $DTYPES; do
        out="$RESULTS/order_${graph}_${dtype}.json"
        if [ -s "$out" ]; then echo "== skip $graph $dtype"; continue; fi
        echo "############ $graph $dtype ($(date +%H:%M:%S))"
        # shellcheck disable=SC2086
        $BENCH_CMD "$HERE/bench_order.py" "$out" --graph "$graph" --dtype "$dtype" \
            --dims "$DIMS" --graph-cache "$GRAPH_CACHE" || { echo "!! failed: $graph $dtype"; rm -f "$out"; }
    done
done
echo "order sweep done $(date -Is)"
