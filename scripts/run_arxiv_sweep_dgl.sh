#!/bin/bash
# Launch kernel_launch_comparison.py for all configs in configs/kernels_measurements/arxiv/
# Results saved to out/arxiv/

set -e

SCRIPT="scripts/kernel_launch_comparison.py"
CONFIG_DIR="configs/kernels_measurements/arxiv"
OUT_DIR="out/arxiv"

mkdir -p "$OUT_DIR"

# Map config filename -> conv_type
declare -A CONV_TYPES=(
    ["gt"]="gt"
    ["gatv2"]="gat_v2"
    ["gcn"]="gcn"
    ["min_aggr"]="min_aggr"
)

BACKENDS="dgl"
TARGET_BACKEND="cuda"
MODE="aggr"

for config_file in "$CONFIG_DIR"/*.yaml; do
    basename=$(basename "$config_file" .yaml)
    conv_type="${CONV_TYPES[$basename]}"

    if [ -z "$conv_type" ]; then
        echo "WARNING: No conv_type mapping for $basename, skipping"
        continue
    fi

    out_csv="$OUT_DIR/${basename}.csv"
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
        2>&1 | tee "$OUT_DIR/${basename}.log"

    echo "Done: $conv_type -> $out_csv"
    echo ""
done

echo "All sweeps complete. Results in $OUT_DIR/"
