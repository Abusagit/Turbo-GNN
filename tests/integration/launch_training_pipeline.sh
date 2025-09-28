#!/bin/bash

echo "==========================================="
echo "Testing Training Pipeline"
echo "==========================================="

# Test 1: Train GCN on Cora with PyG backend
echo -e "\n[TEST 1] Training GCN on Cora (PyG backend)"
python -W ignore scripts/train.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --config configs/training/base.yaml \
    --out runs/test_pyg_cora

# Test 2: Benchmark different backends on same dataset
echo -e "\n[TEST 2] Benchmarking GCN layer across backends"
for backend in pyg dgl torch_native_gcn; do
    echo "Testing $backend..."
    python -W ignore scripts/benchmark.py \
        --layer gcn \
        --backend $backend \
        --num-nodes 100000 \
        --avg-degree 20 \
        --in-ch 512 \
        --out-ch 64 \
        --mode forward \
        --iters 100 \
        --warmup 20
done

# Test 3: Memory profiling with hooks
echo -e "\n[TEST 3] Training with memory profiling"
python -W ignore scripts/train.py \
    --dataset configs/datasets/pyg_cora.yaml \
    --model configs/models/gcn.yaml \
    --config configs/training/base.yaml \
    --profile configs/benchmarks/profile.yaml \
    --out runs/test_profile

# Test 4: Validate model checkpoint
echo -e "\n[TEST 4] Validation from checkpoint"
if [ -f "runs/test_pyg_cora/ckpts/best_model.pth" ]; then
    python scripts/validate.py \
        --dataset configs/datasets/pyg_cora.yaml \
        --model configs/models/gcn.yaml \
        --checkpoint runs/test_pyg_cora/ckpts/best_model.pth
else
    echo "No checkpoint found, skipping validation test"
fi

echo -e "\n==========================================="
echo "Training Pipeline Tests Complete"
echo "==========================================="