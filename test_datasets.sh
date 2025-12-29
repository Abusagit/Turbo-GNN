COMMON_PARAMS="--model configs/models/node_classifier_128_hidden_8_heads.yaml --profile configs/benchmarks/profile.yaml --out runs/test_dgl_with_comet  --config configs/training/base.yaml  --conv_type gt  --backend dgl"


DATASET_DIR="configs/datasets/main"

for dataset_cfg_file in "$DATASET_DIR"/*; do
    python scripts/train.py $COMMON_PARAMS --dataset "$dataset_cfg_file"
    echo "$dataset_cfg_file SUCCESS"
done
