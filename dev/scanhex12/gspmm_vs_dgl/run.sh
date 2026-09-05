#!/usr/bin/env bash
# ./run.sh   [GRAPHS="ogbn-arxiv"] [DIMS=32,64] [PYTHON_DGL=...] [PYTHON_TURBO=...] [PYTHON_PLOT=...] [WORK=...] [REPO=...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
WORK="${WORK:-/tmp/gspmm_vs_dgl}"
PYTHON_DGL="${PYTHON_DGL:-python}"
PYTHON_TURBO="${PYTHON_TURBO:-$PYTHON_DGL}"
PYTHON_PLOT="${PYTHON_PLOT:-$PYTHON_DGL}"
DIMS="${DIMS:-32,64,128}"
GRAPHS="${GRAPHS:-random skewed ogbn-arxiv ogbn-products}"
RESULTS="$HERE/results"

mkdir -p "$WORK" "$RESULTS"

for graph in $GRAPHS; do
    dir="$WORK/xchg_${graph}"
    ops="copy_u,copy_e,add,sub,mul,div"
    case "$graph" in ogbn-products) ops="copy_u";; esac

    echo
    echo "=================== $graph ==================="
    if [ ! -f "$dir/meta.json" ]; then
        "$PYTHON_DGL" "$HERE/dgl_side.py" "$dir" --graph "$graph" --feat-dims "$DIMS" --ops "$ops"
    else
        echo "reusing exported graph in $dir (delete it to re-measure DGL)"
    fi

    : > "$WORK/turbo_${graph}.tsv"
    first=1
    for d in ${DIMS//,/ }; do
        check=""; [ $first -eq 1 ] && check="--check"; first=0
        echo "--- $graph d=$d ---"
        PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_TURBO" "$HERE/turbo_side.py" "$dir" "$d" $check \
            2> >(grep '^RESULT' >> "$WORK/turbo_${graph}.tsv")
    done

    "$PYTHON_DGL" - "$dir/meta.json" "$WORK/turbo_${graph}.tsv" "$RESULTS/results_${graph}.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
cells = []
for line in open(sys.argv[2]):
    _, op, red, d, turbo_ms = line.rstrip("\n").split("\t")
    key = f"{op}|{red}|{d}"
    if key not in meta["timings"]:
        raise SystemExit(f"no DGL timing for {key}")
    cells.append({"op": op, "reduce": red, "d": int(d),
                  "dgl_ms": meta["timings"][key], "turbo_ms": float(turbo_ms)})
out = {k: v for k, v in meta.items() if k != "timings"}
json.dump({"meta": out, "cells": cells}, open(sys.argv[3], "w"), indent=1)
print(f"  -> {sys.argv[3]} ({len(cells)} cells)")
PY
done

echo
echo "== plots =="
"$PYTHON_PLOT" "$HERE/plot.py" "$RESULTS"/results_*.json --dim 64 -o "$RESULTS/speedup_d64.png"
"$PYTHON_PLOT" "$HERE/plot.py" "$RESULTS"/results_*.json -o "$RESULTS/speedup_all.png" --max-bars 40
