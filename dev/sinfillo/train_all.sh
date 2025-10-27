# python scripts/train.py --dataset configs/datasets/pyg_wikics.yaml      --model configs/models/gcn_pyg.yaml --config configs/training/base.yaml --out runs/gcn_wikics
# python scripts/train.py --dataset configs/datasets/pyg_flickr.yaml      --model configs/models/gatv2_cugraph.yaml --config configs/training/base.yaml --out runs/gatv2_flickr

# python scripts/train.py --dataset configs/datasets/pyg_corafull.yaml        --model configs/models/gcn_dgl.yaml --config configs/training/base.yaml --out runs/gcn_corafull
python scripts/train.py --dataset configs/datasets/pyg_coauthor_cs.yaml      --model configs/models/gcn_dgl.yaml --config configs/training/base.yaml --out runs/gcn_coauthor_cs
python scripts/train.py --dataset configs/datasets/pyg_coauthor_physics.yaml --model configs/models/gcn_dgl.yaml --config configs/training/base.yaml --out runs/gcn_coauthor_physics
python scripts/train.py --dataset configs/datasets/pyg_amazon_computers.yaml --model configs/models/gcn_dgl.yaml --config configs/training/base.yaml --out runs/gcn_amazon_computers
python scripts/train.py --dataset configs/datasets/pyg_amazon_photo.yaml     --model configs/models/gcn_dgl.yaml --config configs/training/base.yaml --out runs/gcn_amazon_photo

# python scripts/train.py --dataset configs/datasets/ogbn_products.yaml  --model configs/models/gcn_pyg.yaml --config configs/training/base.yaml --out runs/gcn_ogbn_products
