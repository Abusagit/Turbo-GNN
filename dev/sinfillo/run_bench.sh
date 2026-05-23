#!/bin/bash

set -e

SCRIPT="dev/sinfillo/bench_gatv2.py"
OUTDIR="dev/sinfillo/bench_results"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/bench_${TIMESTAMP}.txt"

# Make sure LD_LIBRARY_PATH is set
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

echo "Results will be saved to: $OUTFILE"
echo ""

# Header
{
    echo "================================================================="
    echo "GATv2 Forward Kernel Benchmark: baseline vs async pipeline"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "GPU:  $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'unknown')"
    echo "================================================================="
    echo ""
} | tee "$OUTFILE"

run() {
    local label="$1"
    shift
    echo "--- $label ---" | tee -a "$OUTFILE"
    python "$SCRIPT" --compare "$@" 2>&1 | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
}

# ----- Vary head_dim (fp32) -----
echo "===== Varying head_dim (fp32, N=100k, deg=16) =====" | tee -a "$OUTFILE"
run "D=32"  --head-dim 32
run "D=64"  --head-dim 64
run "D=128" --head-dim 128
run "D=256" --head-dim 256

# ----- Vary avg_degree (D=128, fp32) -----
echo "===== Varying avg_degree (fp32, N=100k, D=128) =====" | tee -a "$OUTFILE"
run "deg=4"  --head-dim 128 --avg-degree 4
run "deg=8"  --head-dim 128 --avg-degree 8
run "deg=16" --head-dim 128 --avg-degree 16
run "deg=64" --head-dim 128 --avg-degree 64

# ----- fp16 -----
echo "===== fp16 (N=100k, deg=16) =====" | tee -a "$OUTFILE"
run "D=64  fp16" --head-dim 64  --dtype float16
run "D=128 fp16" --head-dim 128 --dtype float16
run "D=256 fp16" --head-dim 256 --dtype float16

# ----- bf16 -----
echo "===== bf16 (N=100k, deg=16) =====" | tee -a "$OUTFILE"
run "D=128 bf16" --head-dim 128 --dtype bfloat16
run "D=256 bf16" --head-dim 256 --dtype bfloat16

# ----- Larger graphs -----
echo "===== Larger graphs (fp32, D=128, deg=16) =====" | tee -a "$OUTFILE"
run "N=500k"  --head-dim 128 --num-nodes 500000
run "N=1M"    --head-dim 128 --num-nodes 1000000

# ----- Larger graphs + sparse -----
echo "===== Larger graphs sparse (fp32, D=128, deg=4) =====" | tee -a "$OUTFILE"
run "N=500k deg=4" --head-dim 128 --num-nodes 500000  --avg-degree 4
run "N=1M   deg=4" --head-dim 128 --num-nodes 1000000 --avg-degree 4

# ----- More heads -----
echo "===== More heads (fp32, N=100k, deg=16) =====" | tee -a "$OUTFILE"
run "H=8 D=64"  --heads 8 --head-dim 64
run "H=8 D=128" --heads 8 --head-dim 128

# ----- Best-case combo: large graph + large D + fp16 + sparse -----
echo "===== Best-case combos =====" | tee -a "$OUTFILE"
run "N=500k D=256 fp16 deg=8"  --head-dim 256 --dtype float16 --num-nodes 500000  --avg-degree 8
run "N=1M   D=128 fp16 deg=4"  --head-dim 128 --dtype float16 --num-nodes 1000000 --avg-degree 4
run "N=500k D=128 fp32 deg=4"  --head-dim 128 --num-nodes 500000 --avg-degree 4

echo "=================================================================" | tee -a "$OUTFILE"
echo "Done! Results saved to: $OUTFILE" | tee -a "$OUTFILE"

echo "===== Nsight Compute profiling =====" | tee -a "$OUTFILE"

NCU_DIR="$OUTDIR/ncu_${TIMESTAMP}"
mkdir -p "$NCU_DIR"

NCU_ARGS="--set full --kernel-name GATv2Forward_Kernel --launch-skip 5 --launch-count 3"
NCU_BENCH_ARGS="--num-nodes 100000 --avg-degree 16 --head-dim 128 --warmup 5 --repeats 5"

echo "Profiling baseline (D=128, fp32, N=100k, deg=16)..." | tee -a "$OUTFILE"
ncu $NCU_ARGS -o "$NCU_DIR/baseline_D128_fp32" \
    python "$SCRIPT" --no-pipeline $NCU_BENCH_ARGS 2>&1 | tail -5 | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "Profiling pipeline (D=128, fp32, N=100k, deg=16)..." | tee -a "$OUTFILE"
ncu $NCU_ARGS -o "$NCU_DIR/pipeline_D128_fp32" \
    python "$SCRIPT" --use-pipeline $NCU_BENCH_ARGS 2>&1 | tail -5 | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "Profiling baseline (D=128, fp16, N=100k, deg=16)..." | tee -a "$OUTFILE"
ncu $NCU_ARGS -o "$NCU_DIR/baseline_D128_fp16" \
    python "$SCRIPT" --no-pipeline $NCU_BENCH_ARGS --dtype float16 2>&1 | tail -5 | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "Profiling pipeline (D=128, fp16, N=100k, deg=16)..." | tee -a "$OUTFILE"
ncu $NCU_ARGS -o "$NCU_DIR/pipeline_D128_fp16" \
    python "$SCRIPT" --use-pipeline $NCU_BENCH_ARGS --dtype float16 2>&1 | tail -5 | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "NCU reports saved to: $NCU_DIR/" | tee -a "$OUTFILE"
echo "View with: ncu-ui $NCU_DIR/baseline_D128_fp32.ncu-rep $NCU_DIR/pipeline_D128_fp32.ncu-rep" | tee -a "$OUTFILE"
