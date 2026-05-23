#!/bin/bash

set -e

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

SCRIPT="dev/sinfillo/bench_gatv2.py"
OUTDIR="dev/sinfillo/bench_results"

mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/bench_ncu_${TIMESTAMP}.txt"

NCU_DIR="$(pwd)/$OUTDIR/ncu_${TIMESTAMP}"
mkdir -p "$NCU_DIR"

PYTHON="$(pwd)/.venv/bin/python"
NCU="/usr/local/cuda/bin/ncu"
NCU_ARGS="--set full --kernel-name GATv2Forward_Kernel --launch-skip 5 --launch-count 3"
NCU_BENCH_ARGS="--num-nodes 100000 --avg-degree 16 --head-dim 128 --warmup 5 --repeats 5"

echo "===== Nsight Compute profiling =====" | tee "$OUTFILE"

echo "Profiling baseline (D=128, fp32)..." | tee -a "$OUTFILE"
sudo -E $NCU $NCU_ARGS -o "$NCU_DIR/baseline_D128_fp32" \
    $PYTHON "$SCRIPT" --no-pipeline $NCU_BENCH_ARGS 2>&1 | tee -a "$OUTFILE"

echo "Profiling pipeline (D=128, fp32)..." | tee -a "$OUTFILE"
sudo -E $NCU $NCU_ARGS -o "$NCU_DIR/pipeline_D128_fp32" \
    $PYTHON "$SCRIPT" --use-pipeline $NCU_BENCH_ARGS 2>&1 | tee -a "$OUTFILE"

echo "Profiling baseline (D=128, fp16)..." | tee -a "$OUTFILE"
sudo -E $NCU $NCU_ARGS -o "$NCU_DIR/baseline_D128_fp16" \
    $PYTHON "$SCRIPT" --no-pipeline $NCU_BENCH_ARGS --dtype float16 2>&1 | tee -a "$OUTFILE"

echo "Profiling pipeline (D=128, fp16)..." | tee -a "$OUTFILE"
sudo -E $NCU $NCU_ARGS -o "$NCU_DIR/pipeline_D128_fp16" \
    $PYTHON "$SCRIPT" --use-pipeline $NCU_BENCH_ARGS --dtype float16 2>&1 | tee -a "$OUTFILE"

# Fix ownership (sudo creates files as root)
sudo chown -R $USER:$USER "$NCU_DIR"

echo "" | tee -a "$OUTFILE"
echo "NCU reports saved to: $NCU_DIR/" | tee -a "$OUTFILE"
echo "View with: ncu-ui $NCU_DIR/baseline_D128_fp32.ncu-rep $NCU_DIR/pipeline_D128_fp32.ncu-rep" | tee -a "$OUTFILE"
