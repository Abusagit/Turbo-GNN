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
# Per dataset: forward sweeps over 64 ops x 3 dtypes (fp32, fp16, bf16) are
# written to out/cuda_gsddmm/<dataset>_cuda_gsddmm.csv (64 x 3 = 192 rows),
# plus a combined out/cuda_gsddmm/all_datasets_cuda_gsddmm.csv at the end.
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
# in {32, 64, 128, 256}, so on OOM the dim is lowered (128 -> 64 -> 32) until
# ALL THREE dtypes succeed at the same dim, keeping every dataset's CSV
# self-consistent and comparable across precisions.
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
OUT_DIR="out/cuda_gsddmm"
LOG_DIR="$OUT_DIR/logs"
OVERRIDE_DIR="$OUT_DIR/.overrides"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$OVERRIDE_DIR"

WARMUP=5
ITERS=20
DTYPES="fp32 fp16 bf16"
# Feature dims to try, in order; the kernels reject anything outside
# {32, 64, 128, 256}, so this is the whole usable fallback chain below 256.
FEATURE_DIMS="256 128 64 32"

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

echo "benchmarking $NUM_OPS ops x $(echo $DTYPES | wc -w) dtypes, ${WARMUP} warmup / ${ITERS} timed iters"

summary=""
declare -A SEEN=()
for yaml in configs/datasets/main/*.yaml configs/datasets/secondary/*.yaml \
            configs/datasets/graphland_remaining/*.yaml configs/datasets/pyg_cora.yaml; do
    [ -f "$yaml" ] || continue
    if is_skipped "$yaml"; then
        echo "SKIP (unloadable): $yaml"
        summary="$summary\n$(basename "$yaml" .yaml): SKIPPED (unloadable)"
        continue
    fi
    name="$(grep -m1 -oP 'name:\s*\K\S+' "$yaml" | tr -d "'\"" | tr '/-' '__')"
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
