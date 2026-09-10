#!/bin/bash
# Benchmark all 64 turbo_gnn GSDDMM ops (our own CUDA kernels, no DGL) via
# benchmark.py --backend cuda --aggr on every dataset config under
# configs/datasets/.
#
# Ops = the 32 DGL-style gsddmm names turbo_gnn generates, each in BOTH
# implementations:
#   - regular:  one CSR thread block per node, light/heavy degree buckets
#   - `_edge`:  one warp per edge over an explicit [E, 2] edge list
#
# Per dataset, by default: forward sweeps over 64 ops x 3 dtypes (fp32, fp16,
# bf16) written to out/cuda_gsddmm/<dataset>_cuda_gsddmm.csv (64 x 3 = 192
# rows), plus a combined out/cuda_gsddmm/all_datasets_cuda_gsddmm.csv at the
# end. See SUBSETTING below to run a single dtype.
#
# Forward only: the GSDDMM kernels have no backward pass, so their output is
# detached and a --mode backward run would time an empty graph.
#
# --dtype (not --amp) sets the precision: --amp only wraps the call in
# torch.autocast, which does not reach custom C++/CUDA ops, so the operands
# themselves must be materialized in the target dtype for the kernel to
# dispatch on it.
#
# --exact-iters makes --warmup/--iters mean *iterations* (5 warmup, 20 timed).
# Without it, benchmark.py delegates to triton.testing.do_bench, whose
# warmup/rep are MILLISECONDS. Note the tradeoff: do_bench also flushes L2
# between reps, so exact-iters numbers are more susceptible to cache reuse.
#
# Env: CUDA_VISIBLE_DEVICES="0" TORCH_CUDA_ARCH_LIST="8.0". Assumes the full
# 80 GB of GPU RAM is available. The GSDDMM kernels only accept feature dim D
# in {32, 64, 128, 256}, so on OOM the dim is lowered (256 -> 128 -> 64 -> 32)
# until EVERY dtype in $DTYPES succeeds at the same dim, keeping each dataset's
# CSV self-consistent and comparable across precisions.
#
# SUBSETTING -- these three are env-overridable:
#   DTYPES        default "fp32 fp16 bf16"  e.g. DTYPES=fp16 for one precision
#   FEATURE_DIMS  default "256 128 64 32"   e.g. FEATURE_DIMS=128 to pin a dim
#   OUT_DIR       default out/cuda_gsddmm   redirects CSVs, logs and overrides
#   DATASETS      default "" (= all)        e.g. DATASETS="cora ogbn_arxiv"
# DATASETS takes the dataset names as they appear in the CSV filenames (so
# `ls out/cuda_gsddmm/*.csv` lists the valid values), space- or
# comma-separated. An unrecognised name is a hard error listing the valid
# ones, rather than a run that silently benchmarks nothing. Note the combined
# table is still rebuilt from every CSV present in OUT_DIR, so after a
# filtered run it mixes the refreshed datasets with whatever was there before
# -- fine when only the kernels changed, misleading if DTYPES/FEATURE_DIMS
# differed between the runs.
# There is deliberately no --mode knob: these kernels have no backward, so the
# sweep is always forward-only (see above).
#
# Two traps when narrowing DTYPES -- which is why OUT_DIR is overridable:
#   1. Each dataset's CSV is deleted and rewritten from scratch, so a
#      DTYPES=fp16 run against the default OUT_DIR REPLACES a finished 192-row
#      table with 64 fp16 rows. Point OUT_DIR elsewhere to keep both.
#   2. The dim picked is the largest at which the SELECTED dtypes fit. fp32 is
#      the most memory-hungry of the three, so an fp16-only run can settle on a
#      LARGER dim than a full run did, and those numbers are then not
#      comparable with the 3-dtype tables. Pin FEATURE_DIMS to the dim the full
#      run used (the feature_dim column of its CSV) when you need to compare.
#
# So, a scratch fp16-only run that touches nothing existing:
#   DTYPES=fp16 FEATURE_DIMS=128 OUT_DIR=out/cuda_gsddmm_fp16 \
#       bash scripts/run_cuda_gsddmm_dataset_sweep.sh
#
# DISK: the datasets download into data/ and are large (web-traffic 14G,
# web-topics 9.2G, ogbn-products 4.2G, reddit+flickr ~2.6G, hm-* ~1.2G).
# Budget ~40 GB of free space before a full sweep; a full disk fails the
# remaining datasets in a way that looks like a download error.
#
# Skipped dataset configs (cannot be loaded at all -- same three the DGL
# sweep skips; these are dataset-loading failures, not backend limitations):
#   secondary/amazon_ratings.yaml (AmazonBook) - PyG returns heterogeneous
#       HeteroData; load_single_graph expects a homogeneous graph.
#   secondary/facebook.yaml (FacebookPagePage) - PyG downloads from
#       graphmining.ai, whose DNS no longer resolves.
#   secondary/lastfm_asia.yaml (LastFMAsia)    - same dead download host.
#
# Config overrides (generated into out/cuda_gsddmm/.overrides/):
#   reddit   - root=data reuses the pre-seeded raw npz files and avoids PyG's
#              flat data/processed/data.pt clashing with other root='data'
#              datasets (AmazonBook's HeteroData cache poisoned it before).
#   Flickr   - isolated root for the same reason.
#   CoraFull - loader requires masks; override sets allow_random_split: true.

set -u
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=".venv/bin/python3"
OUT_DIR="${OUT_DIR:-out/cuda_gsddmm}"
LOG_DIR="$OUT_DIR/logs"
OVERRIDE_DIR="$OUT_DIR/.overrides"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$OVERRIDE_DIR"

WARMUP=5
ITERS=20
DTYPES="${DTYPES:-fp32 fp16 bf16}"
# Feature dims to try, in order; the kernels reject anything outside
# {32, 64, 128, 256}, so this is the whole usable fallback chain below 256.
FEATURE_DIMS="${FEATURE_DIMS:-256 128 64 32}"
# Empty = every dataset. Commas are accepted as separators alongside spaces.
DATASETS="$(echo "${DATASETS:-}" | tr ',' ' ')"

# Every dataset config the sweep considers. Declared once so the DATASETS
# filter can be validated against it before any benchmarking starts.
ALL_YAMLS=(
    configs/datasets/main/*.yaml
    configs/datasets/secondary/*.yaml
    configs/datasets/graphland_remaining/*.yaml
    configs/datasets/pyg_cora.yaml
)

# Dataset configs that can never be benchmarked (see header).
SKIP_YAMLS=(
    "configs/datasets/secondary/amazon_ratings.yaml"
    "configs/datasets/secondary/facebook.yaml"
    "configs/datasets/secondary/lastfm_asia.yaml"
)

# ---------------------------------------------------------------------------
# All 64 turbo_gnn gsddmm ops:
#  - binary:  {lhs}_{add,sub,mul,div,dot}_{rhs} over the 6 MIXED ordered
#             member pairs (u-v, v-u, u-e, e-u, v-e, e-v); turbo_gnn generates
#             no same-member variants (those are dense ops and the kernel
#             static_asserts against them).
#  - copies:  copy_u, copy_v  (no copy_e: copying edge rows to edges is a
#             plain memcpy, so turbo_gnn does not generate it).
#  - each of the above again with an `_edge` suffix -> edge-parallel kernel.
# ---------------------------------------------------------------------------
BASE_OPS=""
for op in add sub mul div dot; do
    for pair in u:v v:u u:e e:u v:e e:v; do
        BASE_OPS="$BASE_OPS ${pair%%:*}_${op}_${pair##*:}"
    done
done
for target in u v; do
    BASE_OPS="$BASE_OPS copy_${target}"
done
ALL_OPS=""
for op in $BASE_OPS; do
    ALL_OPS="$ALL_OPS $op"
done
for op in $BASE_OPS; do
    ALL_OPS="$ALL_OPS ${op}_edge"
done
OPS="$(echo $ALL_OPS | tr ' ' ',')"
NUM_OPS="$(echo $ALL_OPS | wc -w)"

# ---------------------------------------------------------------------------
# Per-dataset config overrides (see header).
# ---------------------------------------------------------------------------
cat > "$OVERRIDE_DIR/reddit.yaml" <<'EOF'
dataset:
  source: pyg
  name: reddit
  root: data
EOF
cat > "$OVERRIDE_DIR/flickr.yaml" <<'EOF'
dataset:
  source: pyg
  name: Flickr
  root: data/solo/flickr
EOF
cat > "$OVERRIDE_DIR/corafull.yaml" <<'EOF'
dataset:
  source: pyg
  name: CoraFull
  root: data/solo/corafull
  allow_random_split: true
EOF

resolve_yaml() {
    # resolve_yaml <dataset name> <original yaml> -> yaml to actually use
    case "$1" in
        reddit)   echo "$OVERRIDE_DIR/reddit.yaml" ;;
        Flickr)   echo "$OVERRIDE_DIR/flickr.yaml" ;;
        CoraFull) echo "$OVERRIDE_DIR/corafull.yaml" ;;
        *)        echo "$2" ;;
    esac
}

is_skipped() {
    local yaml="$1" s
    for s in "${SKIP_YAMLS[@]}"; do
        [ "$yaml" = "$s" ] && return 0
    done
    return 1
}

yaml_name() {
    # yaml_name <yaml> -- the dataset name as used for CSV/log filenames.
    grep -m1 -oP 'name:\s*\K\S+' "$1" | tr -d "'\"" | tr '/-' '__'
}

is_selected() {
    # is_selected <dataset name> -- true if DATASETS is empty (run all) or
    # names this dataset.
    [ -z "$DATASETS" ] && return 0
    local d
    for d in $DATASETS; do
        [ "$1" = "$d" ] && return 0
    done
    return 1
}

is_oom() {
    # is_oom <logfile> -- did this run die from a GPU OOM? Only an OOM is worth
    # retrying at a smaller feature dim; a dataset that fails to download, or a
    # full disk, fails identically at every dim, so retrying just wastes time.
    grep -qiE "OutOfMemoryError|out of memory" "$1"
}

run_dtype() {
    # run_dtype <yaml> <name> <dtype> <d>
    # `yes |` auto-answers OGB's interactive download confirmation prompt.
    yes | "$PY" scripts/benchmark.py \
        --layer "$OPS" --backend cuda --aggr \
        --dataset "$1" --feature_dim "$4" --mode forward \
        --warmup "$WARMUP" --iters "$ITERS" --exact-iters \
        --dtype "$3" \
        --csv-out "$OUT_DIR/$2_cuda_gsddmm.csv" \
        > "$LOG_DIR/$2_cuda_gsddmm_$3_d$4.log" 2>&1
}

# Reject an unknown DATASETS entry now: a filter that matches nothing would
# otherwise look like a sweep that ran and found no work to do.
AVAILABLE=""
for yaml in "${ALL_YAMLS[@]}"; do
    [ -f "$yaml" ] || continue
    AVAILABLE="$AVAILABLE $(yaml_name "$yaml")"
done
if [ -n "$DATASETS" ]; then
    unknown=""
    for d in $DATASETS; do
        case " $AVAILABLE " in
            *" $d "*) ;;
            *) unknown="$unknown $d" ;;
        esac
    done
    if [ -n "$unknown" ]; then
        echo "ERROR: unknown dataset(s) in DATASETS:$unknown" >&2
        echo "Available dataset names:" >&2
        for a in $AVAILABLE; do echo "    $a" >&2; done
        exit 1
    fi
fi

echo "benchmarking $NUM_OPS ops (forward only) x $(echo $DTYPES | wc -w) dtypes [$DTYPES]"
echo "  dims tried: $FEATURE_DIMS | ${WARMUP} warmup / ${ITERS} timed iters | out: $OUT_DIR"
if [ -n "$DATASETS" ]; then
    echo "  datasets: $(echo $DATASETS | wc -w) selected [$DATASETS]"
else
    echo "  datasets: all $(echo $AVAILABLE | wc -w)"
fi

summary=""
declare -A SEEN=()
for yaml in "${ALL_YAMLS[@]}"; do
    [ -f "$yaml" ] || continue
    name="$(yaml_name "$yaml")"
    is_selected "$name" || continue
    if is_skipped "$yaml"; then
        echo "SKIP (unloadable): $yaml"
        summary="$summary\n$(basename "$yaml" .yaml): SKIPPED (unloadable)"
        continue
    fi
    csv="$OUT_DIR/${name}_cuda_gsddmm.csv"
    if [ -n "${SEEN[$name]:-}" ]; then
        echo "SKIP (duplicate of ${SEEN[$name]}): $yaml"
        continue
    fi
    SEEN[$name]="$yaml"
    yaml="$(resolve_yaml "$name" "$yaml")"

    rm -f "$csv"
    echo "================ $(date +%H:%M:%S)  $name  ($yaml) ================"

    ok_d=""
    for d in $FEATURE_DIMS; do
        all_dtypes_ok=1
        retry_smaller=0
        for dt in $DTYPES; do
            log="$LOG_DIR/${name}_cuda_gsddmm_${dt}_d${d}.log"
            if ! run_dtype "$yaml" "$name" "$dt" "$d"; then
                all_dtypes_ok=0
                if is_oom "$log"; then
                    echo "  $dt OOM at d=$d -- retrying at a smaller feature dim"
                    retry_smaller=1
                else
                    echo "  $dt failed at d=$d, not an OOM (see $log)"
                fi
                break
            fi
            echo "  ok: $dt d=$d"
        done
        if [ "$all_dtypes_ok" = 1 ]; then
            ok_d="$d"
            break
        fi
        # Partial rows from the failed dim would make the table inconsistent.
        rm -f "$csv"
        # Only an OOM gets a smaller-dim retry; anything else fails the dataset.
        [ "$retry_smaller" = 1 ] || break
    done

    if [ -z "$ok_d" ]; then
        echo "  FAILED: $name -- skipping dataset"
        summary="$summary\n$name: FAILED"
        continue
    fi
    echo "  OK: $name (d=$ok_d) -> $csv ($(($(wc -l < "$csv") - 1)) rows)"
    summary="$summary\n$name: OK d=$ok_d"
done

# Combined table: header from the first CSV, then all data rows.
combined="$OUT_DIR/all_datasets_cuda_gsddmm.csv"
rm -f "$combined"
first=1
for csv in "$OUT_DIR"/*_cuda_gsddmm.csv; do
    [ -f "$csv" ] || continue
    if [ "$first" = 1 ]; then
        cat "$csv" > "$combined"
        first=0
    else
        tail -n +2 "$csv" >> "$combined"
    fi
done

echo "================ summary ================"
echo -e "$summary"
if [ -f "$combined" ]; then
    echo "combined table: $combined ($(($(wc -l < "$combined") - 1)) rows)"
fi
