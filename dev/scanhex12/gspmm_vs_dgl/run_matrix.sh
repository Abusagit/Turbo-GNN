#!/usr/bin/env bash
# Every configuration of the turbo-vs-DGL comparison, unattended and resumable.
#
#   cd dev/scanhex12/gspmm_vs_dgl
#   OGB_ROOT=... PYTHON_DGL=... PYTHON_TURBO=... nohup ./run_matrix.sh > matrix.log 2>&1 &
#
# Loops run.sh over {float32, float16} x stages {0,1,2,4} with --backward on,
# so each pass yields forward, backward and forward+backward charts over all
# four graphs.  That is the same 4x3 grid of graphs and feature widths the
# original charts used, times every configuration.
#
# Resumable: a pass whose results JSON already exists for all graphs is
# skipped.  The stages=0 pass exports the operands and keeps them
# (KEEP_OPERANDS=1); later passes reuse that export (REUSE=1), since DGL's
# timings and the operands do not depend on turbo's pipeline_stages -- which is
# what run.sh's stage-free DIRTAG is for.  The scratch is released at the end.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DTYPES="${DTYPES:-float32 float16}"
STAGES_LIST="${STAGES_LIST:-0 1 2 4}"
GRAPHS="${GRAPHS:-random skewed ogbn-arxiv ogbn-products}"
DIMS="${DIMS:-32,64,128}"
WORK="${WORK:-/home/ubuntu/work/gspmm_scratch}"
ITERS="${ITERS:-30}"
# Two, not run.sh's default of seven: inside one process these timings repeat
# to under 0.5%, so the extra samples buy almost nothing and cost 3.5x the wall
# clock over a matrix this size.  Two still leaves a consistency check; one
# would leave none.
REPEATS="${REPEATS:-2}"
RESULTS="$HERE/results"

export GRAPHS DIMS WORK ITERS REPEATS

started=$(date +%s)
echo "matrix started $(date -Is)"
echo "  dtypes=[$DTYPES] stages=[$STAGES_LIST] graphs=[$GRAPHS] dims=$DIMS"
echo "  $ITERS iters x $REPEATS repeats"
echo

done_count=0
skip_count=0
fail_count=0

# Stages outer, dtype inner, so both dtypes at stages=0 -- the configurations
# the headline charts come from -- land before any stage variant.
for stages in $STAGES_LIST; do
    for dtype in $DTYPES; do
        tag="${dtype}_bwd"
        [ "$stages" != "0" ] && tag="${tag}_st${stages}"

        # Every graph already reported for this tag -> nothing to do.
        missing=0
        for graph in $GRAPHS; do
            [ -s "$RESULTS/results_${graph}_${tag}.json" ] || missing=1
        done
        if [ "$missing" = "0" ]; then
            echo "== $tag: already complete, skipping"
            skip_count=$((skip_count + 1))
            continue
        fi

        # The stages=0 pass of a dtype produces the export the others reuse.
        reuse=1
        [ "$stages" = "0" ] && reuse=0

        echo "== $tag  ($(date +%H:%M:%S))"
        if DTYPE="$dtype" STAGES="$stages" BACKWARD=1 REUSE="$reuse" KEEP_OPERANDS=1 "$HERE/run.sh"; then
            done_count=$((done_count + 1))
        else
            echo "!! FAILED: $tag"
            fail_count=$((fail_count + 1))
        fi
        echo
    done
done

# The operands are the bulk of the scratch and nothing downstream reads them.
rm -rf "$WORK"/xchg_* 2>/dev/null

echo "matrix finished $(date -Is), $(( ($(date +%s) - started) / 60 )) min"
echo "passes run $done_count, skipped $skip_count, failed $fail_count"
echo "results and charts in $RESULTS"
exit 0
