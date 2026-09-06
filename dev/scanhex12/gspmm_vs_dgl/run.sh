#!/usr/bin/env bash
# ./run.sh   [GRAPHS="ogbn-arxiv"] [DIMS=32,64] [DTYPE=float16] [BACKWARD=1] [STAGES=0]
#            [PYTHON_DGL=...] [PYTHON_TURBO=...] [PYTHON_PLOT=...] [WORK=...] [REPO=...]
#
# One feature width at a time: export -> measure -> delete that width's operands.
# Exporting all widths up front needs E*(32+64+128)*4 bytes of scratch (3 GB on a
# 3.4M-edge graph, times three configurations), which overflows a tmpfs WORK dir.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
WORK="${WORK:-/home/ubuntu/work/gspmm_scratch}"   # NOT /tmp: it is a RAM-backed tmpfs here
PYTHON_DGL="${PYTHON_DGL:-python}"
PYTHON_TURBO="${PYTHON_TURBO:-$PYTHON_DGL}"
PYTHON_PLOT="${PYTHON_PLOT:-$PYTHON_DGL}"
DIMS="${DIMS:-32,64,128}"
DTYPE="${DTYPE:-float32}"
BACKWARD="${BACKWARD:-0}"
STAGES="${STAGES:-0}"
ITERS="${ITERS:-30}"
REPEATS="${REPEATS:-7}"
GRAPHS="${GRAPHS:-random skewed ogbn-arxiv ogbn-products}"
RESULTS="$HERE/results"

bwd_flag=""
kinds="fwd"
if [ "$BACKWARD" = "1" ]; then
    bwd_flag="--backward"
    kinds="fwd bwd fb"
fi

# DIRTAG has no stage suffix: the export does not depend on pipeline_stages, so a
# stage sweep can REUSE=1 the operands exported by the stages=0 run
DIRTAG="${DTYPE}"
if [ "$BACKWARD" = "1" ]; then DIRTAG="${DIRTAG}_bwd"; fi
TAG="$DIRTAG"
if [ "$STAGES" != "0" ]; then TAG="${TAG}_st${STAGES}"; fi

mkdir -p "$WORK" "$RESULTS"

for graph in $GRAPHS; do
    dir="$WORK/xchg_${graph}_${DIRTAG}"
    ops="copy_u,copy_e,add,sub,mul,div"
    case "$graph" in ogbn-products) ops="copy_u";; esac

    echo
    echo "=================== $graph  ($TAG) ==================="
    if [ "${REUSE:-0}" != "1" ]; then rm -rf "$dir"; fi
    mkdir -p "$dir"
    tsv="$WORK/turbo_${graph}_${TAG}.tsv"
    : > "$tsv"
    mismatched=0

    for d in ${DIMS//,/ }; do
        echo "--- $graph d=$d: export ---"
        if [ "${REUSE:-0}" = "1" ] && [ -f "$dir/meta_$d.json" ]; then
            echo "reusing export for d=$d"
            cp "$dir/meta_$d.json" "$dir/meta.json"
        else
            "$PYTHON_DGL" "$HERE/dgl_side.py" "$dir" --graph "$graph" --feat-dims "$d" \
                --ops "$ops" --dtype "$DTYPE" --iters "$ITERS" --repeats "$REPEATS" $bwd_flag
            cp "$dir/meta.json" "$dir/meta_$d.json"
        fi

        echo "--- $graph d=$d: turbo ---"
        chk="--check"
        if [ "${CHECK:-1}" = "0" ]; then chk=""; fi
        PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_TURBO" "$HERE/turbo_side.py" \
            "$dir" "$d" $chk $bwd_flag --pipeline-stages "$STAGES" \
            --iters "$ITERS" --repeats "$REPEATS" \
            2> >(tee -a "$WORK/mismatches_${TAG}.log" | grep '^RESULT' >> "$tsv") || mismatched=1

        # the operands are the bulk of the scratch; the sampled references stay
        if [ "${KEEP_OPERANDS:-0}" != "1" ]; then
            rm -f "$dir"/x_"$d".* "$dir"/e_"$d".* "$dir"/eb_"$d".* "$dir"/gseed_"$d".*
        fi
    done

    if [ "$mismatched" = "1" ]; then
        echo "!! $graph ($TAG): were MISMATCHes, see $WORK/mismatches_${TAG}.log"
    fi

    "$PYTHON_DGL" - "$dir" "$tsv" "$RESULTS/results_${graph}_${TAG}.json" "$TAG" <<'PY'
import glob, json, os, sys
d, tsv, out, tag = sys.argv[1:5]
meta, timings = None, {}
for p in sorted(glob.glob(os.path.join(d, "meta_*.json"))):
    m = json.load(open(p))
    timings.update(m.pop("timings"))
    if meta is None:
        meta = m
meta["dims"] = sorted({int(k.split("|")[-1]) for k in timings})
meta["tag"] = tag
cells = []
for line in open(tsv):
    _, kind, op, red, dim, turbo_ms, stages = line.rstrip("\n").split("\t")
    key = f"{kind}|{op}|{red}|{dim}"
    if key not in timings:
        raise SystemExit(f"no DGL timing for {key}")
    cells.append({"kind": kind, "op": op, "reduce": red, "d": int(dim),
                  "dgl_ms": timings[key], "turbo_ms": float(turbo_ms),
                  "pipeline_stages": int(stages)})
json.dump({"meta": meta, "cells": cells}, open(out, "w"), indent=1)
print(f"  -> {out} ({len(cells)} cells)")
PY

    if [ "${KEEP_OPERANDS:-0}" != "1" ]; then rm -rf "$dir"; fi
done

echo
echo "== plots =="
for kind in $kinds; do
    case "$kind" in
        fwd) label="forward";;
        bwd) label="backward";;
        *)   label="forward+backward";;
    esac
    "$PYTHON_PLOT" "$HERE/plot.py" "$RESULTS"/results_*_"$TAG".json --dim 64 --kind "$kind" \
        --title "g-SpMM: turbo_gnn против DGL по конфигурациям (d=64, $DTYPE, $label)" \
        --xlabel "Ускорение: dgl / turbo_gnn ($label, медиана из $REPEATS прогонов по $ITERS запусков)" \
        -o "$RESULTS/speedup_d64_${TAG}_${kind}.png"
done
