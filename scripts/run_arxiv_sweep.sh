#!/bin/bash
# Launch kernel_launch_comparison.py for selected configs.
#
# Usage:
#   # Run all configs in a directory:
#   bash scripts/run_arxiv_sweep.sh
#
#   # Run specific configs with a custom output suffix:
#   bash scripts/run_arxiv_sweep.sh --configs gt gatv2 --suffix v2
#
#   # Override backends and output dir:
#   bash scripts/run_arxiv_sweep.sh --configs gcn --backends "cuda dgl" --out-dir out/comparison
#
#   # Custom mode:
#   bash scripts/run_arxiv_sweep.sh --configs gt --mode layer --suffix layer_bench

# set -e

# ── Defaults ──
SCRIPT="scripts/kernel_launch_comparison.py"
CONFIG_DIR="configs/kernels_measurements/arxiv"
OUT_DIR="out/arxiv"
BACKENDS="cuda dgl pyg tcgnn triton_block_sparse"
TARGET_BACKEND="cuda"
MODE="aggr"
SUFFIX=""
CONFIGS=()  # empty = all *.yaml in CONFIG_DIR

# Map config filename -> conv_type
declare -A CONV_TYPES=(
    ["gt"]="gt"
    ["gatv2"]="gat_v2"
    ["gcn"]="gcn"
    ["min_aggr"]="min_aggr"
    ["max_aggr"]="max_aggr"
    ["mean_aggr"]="mean_aggr"
    ["sum_aggr"]="sum_aggr"
)

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CONFIGS+=("$1")
                shift
            done
            ;;
        --suffix)
            SUFFIX="$2"; shift 2 ;;
        --out-dir)
            OUT_DIR="$2"; shift 2 ;;
        --config-dir)
            CONFIG_DIR="$2"; shift 2 ;;
        --backends)
            BACKENDS="$2"; shift 2 ;;
        --target-backend)
            TARGET_BACKEND="$2"; shift 2 ;;
        --mode)
            MODE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --configs NAME [NAME ...]   Config basenames to run (default: all in config-dir)"
            echo "  --suffix STR                Append to output filenames (e.g. 'v2' -> gt_v2.csv)"
            echo "  --out-dir DIR               Output directory (default: out/arxiv)"
            echo "  --config-dir DIR            Config directory (default: configs/kernels_measurements/arxiv)"
            echo "  --backends STR              Space-separated backends (default: 'cuda dgl')"
            echo "  --target-backend STR        Target backend (default: cuda)"
            echo "  --mode STR                  'aggr' or 'layer' (default: aggr)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If no --configs given, discover all yaml files in CONFIG_DIR
if [ ${#CONFIGS[@]} -eq 0 ]; then
    for f in "$CONFIG_DIR"/*.yaml; do
        CONFIGS+=("$(basename "$f" .yaml)")
    done
fi

mkdir -p "$OUT_DIR"

# ── Build output name helper ──
make_out_name() {
    local base="$1"
    if [ -n "$SUFFIX" ]; then
        echo "${base}_${SUFFIX}"
    else
        echo "$base"
    fi
}

# ── Run ──
echo "Config dir:  $CONFIG_DIR"
echo "Output dir:  $OUT_DIR"
echo "Backends:    $BACKENDS"
echo "Mode:        $MODE"
echo "Configs:     ${CONFIGS[*]}"
echo "Suffix:      ${SUFFIX:-<none>}"
echo ""

for config_name in "${CONFIGS[@]}"; do
    config_file="$CONFIG_DIR/${config_name}.yaml"

    if [ ! -f "$config_file" ]; then
        echo "WARNING: Config not found: $config_file, skipping"
        continue
    fi

    conv_type="${CONV_TYPES[$config_name]}"
    if [ -z "$conv_type" ]; then
        echo "WARNING: No conv_type mapping for '$config_name', skipping"
        continue
    fi

    out_name=$(make_out_name "$config_name")
    out_csv="$OUT_DIR/${out_name}.csv"
    out_log="$OUT_DIR/${out_name}.log"

    echo "========================================"
    echo "Running: $conv_type ($config_file) -> $out_csv"
    echo "========================================"

    python "$SCRIPT" \
        --conv_type "$conv_type" \
        --backends $BACKENDS \
        --target_backend "$TARGET_BACKEND" \
        --mode "$MODE" \
        --conv_params_grid "$config_file" \
        --out "$out_csv" \
        2>&1 | tee "$out_log"

    echo "Done: $conv_type -> $out_csv"
    echo ""
done

echo "All sweeps complete. Results in $OUT_DIR/"
