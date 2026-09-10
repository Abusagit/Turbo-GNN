#!/bin/bash
# Benchmark the 32 simple DGL gsddmm ops (via benchmark.py --aggr, raw dgl.ops
# launched directly, no projections) on every dataset config under configs/datasets/.
#
# Per dataset: forward + backward sweeps over the 32 ops in each of 3 dtypes
# (fp32, fp16, bf16) are written to out/dgl_gspmm/<dataset>_dgl_ops.csv
# (32 ops x 2 modes x 3 dtypes = 192 rows), plus a combined
# out/dgl_gspmm/all_datasets_dgl_ops.csv at the end.
#
# THE OP SET is exactly the overlap with run_cuda_gsddmm_dataset_sweep.sh
# (see COMPARABILITY below); DGL's other simple ops are deliberately not
# benchmarked. CSVs collected before the trim to 32 ops hold all 57 -- filter
# conv_type to the 32 when comparing against newer per-dataset CSVs.
#
# PRECISION: set via --dtype, which materializes the operands themselves in the
# target precision, so DGL's kernels dispatch on it. Do NOT use --amp for this:
# it only wraps the call in torch.autocast, which does not reach the gspmm/gsddmm
# ops, so an --amp run would silently time fp32 under an fp16 label.
#
# Note all 32 ops share one process per (dtype, mode) so the dataset is loaded
# once. The tradeoff: if DGL lacks an fp16/bf16 kernel for even one op, that
# whole (dtype, mode) combination dies and the dataset is reported FAILED. Check
# the log named in the failure line to see which op raised before concluding the
# dataset itself is at fault.
#
# TIMING: deliberately IDENTICAL to run_cuda_gsddmm_dataset_sweep.sh so the two
# sweeps' ms_per_iter columns can be compared directly -- 5 warmup + 20 timed
# iterations, pinned with --exact-iters. Without that flag benchmark.py
# delegates to triton.testing.do_bench, whose warmup/rep are MILLISECONDS
# rather than counts, and the two sweeps would be measuring different things.
# Keep WARMUP/ITERS/--exact-iters in step across BOTH scripts.
#
# Caveat, applying EQUALLY to both sweeps: do_bench flushes L2 between reps and
# --exact-iters does not, so absolute numbers are optimistic for graphs whose
# operands fit in the A100's 40 MB L2 (the small datasets: cora, citeseer,
# pubmed). Ratios between implementations stay meaningful because both sides
# are measured the same way; if you want cold-cache absolutes, drop
# --exact-iters from BOTH scripts and re-run BOTH.
#
# COMPARABILITY with run_cuda_gsddmm_dataset_sweep.sh: join rows on
# (dataset, conv_type, dtype, feature_dim, mode). Every op benchmarked here is
# in the overlap set -- copy_u, copy_v and the 30 binary {lhs}_{op}_{rhs}
# names -- and each maps to TWO cuda-sweep rows: `<op>` (CSR node-bucketed) and
# `<op>_edge` (edge-parallel). DGL's copy_e is excluded (copying edge rows onto
# edges is a memcpy, so turbo_gnn does not generate it), and the 24 gspmm
# aggregations (copy_*_<reduce>, u_*_e_<reduce>) are excluded because they
# belong against reduction_aggr / spmm_aggr, NOT against gsddmm. Our gsddmm
# kernels are forward-only, so only mode=forward rows have a counterpart; the
# backward rows are reference data for when a gsddmm backward lands (set
# MODES="forward" to skip them and halve the runtime).
#
# Both scripts must also use the SAME FEATURE_DIMS ladder: a dataset that lands
# at a different dim on each side is not comparable. Note the existing tables
# in out/cuda_gsddmm/ were collected with a 128-first ladder, so re-run that
# sweep with the current 256-first ladder before comparing against this one.
#
# Env: CUDA_VISIBLE_DEVICES="0" TORCH_CUDA_ARCH_LIST="8.0". Assumes the full
# 80 GB of GPU RAM is available; if a dataset OOMs, the feature dim is lowered
# (256 -> 128 -> 64 -> 32) until ALL dtypes and BOTH modes succeed at the same
# dim, so every dataset's CSV is self-consistent and comparable across
# precisions. DGL itself accepts any feature dim, but this ladder is pinned to
# {32, 64, 128, 256} -- the only dims the turbo_gnn kernels accept -- so both
# sweeps can land on the same dim for the same dataset.
#
# PYTHON: DGL pins torch==2.4.0, so it cannot share the .venv built for the
# turbo_gnn CUDA extension (see `make install-bench` vs `make install-dev`).
# Point PY at a DGL-capable interpreter, e.g.:
#   PY=~/micromamba/envs/graph_ml/bin/python bash scripts/run_dgl_ops_dataset_sweep.sh
#
# FILTER: run only a subset of the datasets with
#   DATASETS="ogbn-arxiv cora" bash scripts/run_dgl_ops_dataset_sweep.sh
# Entries match the dataset name with the same '-' -> '_' normalization the
# script applies, so "city-reviews" and "city_reviews" both work. Use this to
# retry the datasets a down download host (e.g. Zenodo) failed on without
# re-benchmarking the ones that already succeeded: a filtered pass writes the
# same per-dataset CSVs, so the combined table picks everything up.
#
# DOWNLOADS: a (dtype, mode) run that dies with a download error (HTTP/gateway/
# connection/zip) is retried up to 3 times with a short pause; a dataset that
# still cannot be downloaded after 3 attempts fails and is skipped. Failures
# that are NOT download errors are not retried.
#
# DISK: the datasets download into data/ and are large (web-traffic 14G,
# web-topics 9.2G, ogbn-products 4.2G, reddit+flickr ~2.6G, hm-* ~1.2G).
# Budget ~40 GB of free space before a full sweep; a full disk fails the
# remaining datasets in a way that looks like a download error.
#
# Skipped dataset configs (cannot be benchmarked):
#   secondary/amazon_ratings.yaml (AmazonBook) - PyG returns heterogeneous
#       HeteroData; load_single_graph expects a homogeneous graph.
#   secondary/facebook.yaml (FacebookPagePage) - PyG downloads from
#       graphmining.ai, whose DNS no longer resolves.
#   secondary/lastfm_asia.yaml (LastFMAsia)    - same dead download host.
#
# Config overrides (generated into out/dgl_gspmm/.overrides/):
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

# Override to point at a DGL-capable interpreter (see PYTHON note above).
PY="${PY:-.venv/bin/python3}"
OUT_DIR="out/dgl_gspmm"
LOG_DIR="$OUT_DIR/logs"
OVERRIDE_DIR="$OUT_DIR/.overrides"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$OVERRIDE_DIR"

# Timing knobs -- MUST match run_cuda_gsddmm_dataset_sweep.sh (see TIMING above).
WARMUP=5
ITERS=20
DTYPES="fp32 fp16 bf16"
# Override with MODES="forward" to run only the rows our kernels can be
# compared against (they have no backward pass).
MODES="${MODES:-forward backward}"
FEATURE_DIMS="256 128 64 32"
# Optional subset filter -- see FILTER in the header. Normalized with the same
# '-' -> '_' mapping applied to dataset names, so both spellings match.
DATASETS="$(echo ${DATASETS:-} | tr -d "'" | tr '/-' '__')"

# Dataset configs that can never be benchmarked (see header).
SKIP_YAMLS=(
    "configs/datasets/secondary/amazon_ratings.yaml"
    "configs/datasets/secondary/facebook.yaml"
    "configs/datasets/secondary/lastfm_asia.yaml"
)

# ---------------------------------------------------------------------------
# The 32 simple dgl.ops gsddmm functions that overlap the turbo_gnn kernels:
#  - gsddmm copies:  copy_u, copy_v
#  - gsddmm binary:  {lhs}_{add,sub,mul,div,dot}_{rhs} over the 6 MIXED
#                    ordered pairs (u-v, u-e, v-u, v-e, e-u, e-v); DGL
#                    generates no same-operand variants.
# ---------------------------------------------------------------------------
GSDDMM_COPIES="copy_u copy_v"
GSDDMM_BINARY=""
for op in add sub mul div dot; do
    for pair in u:v u:e v:u v:e e:u e:v; do
        GSDDMM_BINARY="$GSDDMM_BINARY ${pair%%:*}_${op}_${pair##*:}"
    done
done
ALL_OPS="$GSDDMM_COPIES $GSDDMM_BINARY"
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
    # retrying at a smaller feature dim; a dataset that fails to download, an op
    # DGL has no kernel for in this dtype, or a full disk all fail identically
    # at every dim, so retrying just wastes time.
    grep -qiE "OutOfMemoryError|out of memory" "$1"
}

is_download_error() {
    # is_download_error <logfile> -- did this run die fetching the dataset?
    # HTTP/gateway/connection failures are worth up to 3 attempts (a flaky or
    # briefly-down download host); anything else is not retried.
    grep -qiE "HTTPError|URLError|Gateway Time-out|Connection|timed out|BadZipFile|IncompleteRead" "$1"
}

run_one() {
    # run_one <yaml> <name> <mode> <dtype> <d>
    # `yes |` auto-answers OGB's interactive download confirmation prompt.
    yes | "$PY" scripts/benchmark.py \
        --layer "$OPS" --backend dgl --aggr \
        --dataset "$1" --feature_dim "$5" --mode "$3" \
        --warmup "$WARMUP" --iters "$ITERS" --exact-iters \
        --dtype "$4" \
        --csv-out "$OUT_DIR/$2_dgl_ops.csv" \
        > "$LOG_DIR/$2_dgl_ops_$3_$4_d$5.log" 2>&1
}

echo "benchmarking $NUM_OPS ops x $(echo $DTYPES | wc -w) dtypes x $(echo $MODES | wc -w) modes" \
     "= $((NUM_OPS * $(echo $DTYPES | wc -w) * $(echo $MODES | wc -w))) rows per dataset," \
     "${WARMUP} warmup / ${ITERS} timed iters (exact)"

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
    csv="$OUT_DIR/${name}_dgl_ops.csv"
    if [ -n "${SEEN[$name]:-}" ]; then
        echo "SKIP (duplicate of ${SEEN[$name]}): $yaml"
        continue
    fi
    SEEN[$name]="$yaml"
    yaml="$(resolve_yaml "$name" "$yaml")"

    # Optional subset filter -- see FILTER in the header.
    if [ -n "$DATASETS" ]; then
        found=0
        for want in $DATASETS; do
            [ "$name" = "$want" ] && { found=1; break; }
        done
        [ "$found" = 1 ] || {
            echo "SKIP (not in DATASETS): $yaml"
            continue
        }
    fi

    rm -f "$csv"
    echo "================ $(date +%H:%M:%S)  $name  ($yaml) ================"

    ok_d=""
    for d in $FEATURE_DIMS; do
        combo_ok=1
        retry_smaller=0
        for dt in $DTYPES; do
            for mode in $MODES; do
                log="$LOG_DIR/${name}_dgl_ops_${mode}_${dt}_d${d}.log"
                # Up to 3 attempts when the failure is a download error (see
                # DOWNLOADS in the header); anything else fails immediately.
                combo_ok_run=0
                for attempt in 1 2 3; do
                    if run_one "$yaml" "$name" "$mode" "$dt" "$d"; then
                        combo_ok_run=1
                        break
                    fi
                    is_download_error "$log" || break
                    [ "$attempt" = 3 ] || {
                        echo "  $dt/$mode download error at d=$d (attempt $attempt/3) -- retrying"
                        sleep 15
                    }
                done
                if [ "$combo_ok_run" != 1 ]; then
                    combo_ok=0
                    if is_oom "$log"; then
                        echo "  $dt/$mode OOM at d=$d -- retrying at a smaller feature dim"
                        retry_smaller=1
                    else
                        echo "  $dt/$mode failed at d=$d, not an OOM (see $log)"
                    fi
                    break 2
                fi
                echo "  ok: $dt/$mode d=$d"
            done
        done
        if [ "$combo_ok" = 1 ]; then
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
combined="$OUT_DIR/all_datasets_dgl_ops.csv"
rm -f "$combined"
first=1
for csv in "$OUT_DIR"/*_dgl_ops.csv; do
    [ -f "$csv" ] || continue
    case "$csv" in *all_datasets*) continue ;; esac
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
