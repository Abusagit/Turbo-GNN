#!/usr/bin/env bash
# Full before/after sweep of turbo_gnn.gspmm, unattended and resumable.
#
#   cd dev/scanhex12/gspmm_before_after
#   nohup ./run.sh > run.log 2>&1 &
#
# Measures two commits over the same four graphs the DGL charts used --
# random, skewed, ogbn-arxiv, ogbn-products -- times {float16, float32} x
# {stages 0,1,2,4} x {d 32,64,128}, forward and backward, then draws every
# chart from the results.  Several hours on a T4; safe to leave.
#
# ogbn-products is 126M edges: an [E, d] edge operand does not fit in 15 GB, so
# on that graph only the copy_u cells measure and the rest are recorded as
# skipped -- which is also why the original four-graph chart showed copy_u
# alone.  Its raw csv.gz has to be under OGB_ROOT; bench.py reads that directly
# rather than depending on the ogb package.
#
# Resumable: a results file that already exists and is non-empty is skipped, so
# an interrupted run continues where it stopped.  Delete the file (or the whole
# results/ directory) to force a re-measure.  Charts are always redrawn.
#
# Knobs, all optional:
#   BEFORE_REF / AFTER_REF   commits to compare      (default: the PR base, HEAD)
#   GRAPHS / DTYPES / STAGES / DIMS   what to sweep
#   OGB_ROOT                 where the OGB downloads live (default: data/ogb)
#   PYTHON                   interpreter             (default: python)
#   WORKDIR                  where the worktrees go  (default: /tmp/turbo-bench)
#   SKIP_BUILD=1             do not build the extension in each worktree
#   BENCH_CMD                how to invoke bench.py (see the T4 note below)
#
# BENCH_CMD exists because this repo does not build for every GPU: on sm_75 the
# bf16 paths of gatv2_kernel.cu do not compile, so measuring there needs a
# partial extension.  Point BENCH_CMD at a wrapper that arranges one and the
# sweep is unchanged otherwise.  The wrapper is invoked as
#
#     $BENCH_CMD <path to bench.py> <out.json> --graph ... --dtype ...
#
# with TURBO_REPO set to the checkout being measured.  bench.py is always this
# copy, never the one inside that checkout: the two sides of a comparison have
# to be measured by the same code (and the older commit has no bench.py at all).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"

BEFORE_REF="${BEFORE_REF:-603239d}"
AFTER_REF="${AFTER_REF:-HEAD}"
GRAPHS="${GRAPHS:-random skewed ogbn-arxiv ogbn-products}"
DTYPES="${DTYPES:-float16 float32}"
STAGES="${STAGES:-0 1 2 4}"
DIMS="${DIMS:-32,64,128}"
PLOT_DIM="${PLOT_DIM:-64}"
PYTHON="${PYTHON:-python}"
WORKDIR="${WORKDIR:-/tmp/turbo-bench}"
BENCH_CMD="${BENCH_CMD:-}"
OGB_ROOT="${OGB_ROOT:-data/ogb}"

RESULTS="$HERE/results"
PLOTS="$HERE/plots"
mkdir -p "$RESULTS" "$PLOTS" "$WORKDIR"

started=$(date +%s)
echo "sweep started $(date -Is)"
echo "  before=$BEFORE_REF  after=$AFTER_REF"
echo "  graphs=[$GRAPHS] dtypes=[$DTYPES] stages=[$STAGES] dims=$DIMS"
echo

# --- checkouts -------------------------------------------------------------
# A worktree per side rather than measuring the working tree: the sweep runs
# for an hour, and nothing it reports should depend on whether someone edited a
# file meanwhile.
prepare_state() {
    # Separate statements on purpose: `local` is a builtin, so every word on its
    # line is expanded before any of the assignments happen, and a later one
    # cannot refer to an earlier one.
    local name="$1"
    local ref="$2"
    local dir="$WORKDIR/$name"

    if [ ! -d "$dir" ]; then
        echo "== checking out $ref into $dir"
        git -C "$REPO" worktree add --detach "$dir" "$ref" || return 1
    fi
    if [ "${SKIP_BUILD:-0}" != "1" ]; then
        echo "== building the extension in $dir"
        (cd "$dir" && "$PYTHON" setup.py build_ext --inplace) || return 1
    fi
    echo "$dir"
}

BEFORE_DIR="$(prepare_state before "$BEFORE_REF" | tail -1)" || { echo "FATAL: cannot prepare the before state"; exit 1; }
AFTER_DIR="$(prepare_state after "$AFTER_REF" | tail -1)"    || { echo "FATAL: cannot prepare the after state"; exit 1; }
BEFORE_LABEL="$(git -C "$REPO" rev-parse --short "$BEFORE_REF") (до)"
AFTER_LABEL="$(git -C "$REPO" rev-parse --short "$AFTER_REF") (после)"

# --- measure ---------------------------------------------------------------
failed=0
measured=0
skipped=0

bench_one() {
    local state="$1"
    local dir="$2"
    local label="$3"
    local graph="$4"
    local dtype="$5"
    local stages="$6"
    local out="$RESULTS/${state}_${graph}_${dtype}_st${stages}.json"

    if [ -s "$out" ]; then
        skipped=$((skipped + 1))
        return 0
    fi

    echo "-- $state $graph $dtype stages=$stages  ($(date +%H:%M:%S))"
    local -a cmd
    if [ -n "$BENCH_CMD" ]; then
        read -r -a cmd <<< "$BENCH_CMD"
    else
        cmd=("$PYTHON")
    fi
    cmd+=("$HERE/bench.py")

    if TURBO_REPO="$dir" PYTHONPATH="$dir:${PYTHONPATH:-}" \
        "${cmd[@]}" "$out" --graph "$graph" --dtype "$dtype" --stages "$stages" \
        --dims "$DIMS" --label "$label" --ogb-root "$OGB_ROOT"
    then
        measured=$((measured + 1))
    else
        echo "!! FAILED: $state $graph $dtype stages=$stages"
        rm -f "$out"          # never leave a half-written file behind
        failed=$((failed + 1))
    fi
    echo
}

for graph in $GRAPHS; do
    for dtype in $DTYPES; do
        for stages in $STAGES; do
            bench_one before "$BEFORE_DIR" "$BEFORE_LABEL" "$graph" "$dtype" "$stages"
            bench_one after  "$AFTER_DIR"  "$AFTER_LABEL"  "$graph" "$dtype" "$stages"
        done
    done
done

echo "measured $measured, reused $skipped, failed $failed"
echo

# --- charts ----------------------------------------------------------------
plots=0
plot() {
    if "$PYTHON" "$HERE/plot.py" "$@"; then
        plots=$((plots + 1))
    else
        echo "!! chart failed: $*"
    fi
}

pair_args() {  # pair_args <state_a> <state_b> <dtype> <stages_a> <stages_b>
    local a="$1"
    local b="$2"
    local dtype="$3"
    local sa="$4"
    local sb="$5"
    local graph
    for graph in $GRAPHS; do
        local fa="$RESULTS/${a}_${graph}_${dtype}_st${sa}.json"
        local fb="$RESULTS/${b}_${graph}_${dtype}_st${sb}.json"
        [ -s "$fa" ] && [ -s "$fb" ] && printf -- '--pair\n%s\n%s\n' "$fa" "$fb"
    done
}

echo "== before/after, one chart per dtype x stages x kind"
for dtype in $DTYPES; do
    for stages in $STAGES; do
        for kind in fwd bwd; do
            mapfile -t pairs < <(pair_args before after "$dtype" "$stages" "$stages")
            [ "${#pairs[@]}" -gt 0 ] || continue
            plot "${pairs[@]}" --kind "$kind" --dim "$PLOT_DIM" \
                -o "$PLOTS/ba_${dtype}_st${stages}_${kind}_d${PLOT_DIM}.png" \
                --title "g-SpMM $kind, $dtype, d=$PLOT_DIM, stages=$stages: до и после"
        done
    done
done

echo "== the same at the other feature widths (headline config only)"
for dim in ${DIMS//,/ }; do
    [ "$dim" = "$PLOT_DIM" ] && continue
    for kind in fwd bwd; do
        mapfile -t pairs < <(pair_args before after float16 0 0)
        [ "${#pairs[@]}" -gt 0 ] || continue
        plot "${pairs[@]}" --kind "$kind" --dim "$dim" \
            -o "$PLOTS/ba_float16_st0_${kind}_d${dim}.png" \
            --title "g-SpMM $kind, float16, d=$dim, stages=0: до и после"
    done
done

echo "== pipelining: every stage count against stages=0, on the after state"
for dtype in $DTYPES; do
    for kind in fwd bwd; do
        for graph in $GRAPHS; do
            args=()
            for stages in $STAGES; do
                [ "$stages" = "0" ] && continue
                fa="$RESULTS/after_${graph}_${dtype}_st0.json"
                fb="$RESULTS/after_${graph}_${dtype}_st${stages}.json"
                [ -s "$fa" ] && [ -s "$fb" ] && args+=(--pair "$fa" "$fb")
            done
            [ "${#args[@]}" -gt 0 ] || continue
            plot "${args[@]}" --kind "$kind" --dim "$PLOT_DIM" --compare stages \
                -o "$PLOTS/stages_${dtype}_${graph}_${kind}_d${PLOT_DIM}.png" \
                --title "Пайплайнинг, $graph, $dtype, $kind, d=$PLOT_DIM: stages=0 против остальных" \
                --xlabel "Отношение времени stages=0 / stages=N ($kind). Меньше 1 — с пайплайном медленнее"
        done
    done
done

echo "== float16 against float32, on the after state"
for kind in fwd bwd; do
    args=()
    for graph in $GRAPHS; do
        fa="$RESULTS/after_${graph}_float32_st0.json"
        fb="$RESULTS/after_${graph}_float16_st0.json"
        [ -s "$fa" ] && [ -s "$fb" ] && args+=(--pair "$fa" "$fb")
    done
    [ "${#args[@]}" -gt 0 ] || continue
    plot "${args[@]}" --kind "$kind" --dim "$PLOT_DIM" --compare dtype \
        -o "$PLOTS/dtype_${kind}_d${PLOT_DIM}.png" \
        --title "float32 против float16, $kind, d=$PLOT_DIM (после правок)" \
        --xlabel "Ускорение: float32 / float16 ($kind). Больше 1 — half быстрее"
done

echo
echo "sweep finished $(date -Is), $(( ($(date +%s) - started) / 60 )) min"
echo "results in $RESULTS, $plots charts in $PLOTS"
[ "$failed" -eq 0 ] || echo "WARNING: $failed measurement(s) failed; re-run to retry just those"
exit 0
