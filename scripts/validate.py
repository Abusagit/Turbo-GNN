import argparse

import torch

import sys
sys.path.append("./")

from src.utils.scripts_utils import create_split_datasets_from_yaml, ensure_outdir, infer_graph_backend  # reserved for future outputs if needed

from src.data.loaders import LoaderConfig, build_dataloader
from src.models.config import build_model_from_yaml
from src.training.trainer import GNNTrainer, TrainingConfig
from src.utils.logger import get_logger

doc = """
Validation launcher script (using common helpers for consistency).

- Loads dataset and model from YAMLs
- Restores weights from a checkpoint (.pth with model_state_dict or raw state_dict)
- Evaluates on validation and test splits using trainer.validate(...)
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI args for validation.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Validate a trained model on val/test.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset YAML path.")
    p.add_argument("--model", type=str, required=True, help="Model YAML path.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint with model_state_dict.")
    p.add_argument("--batch-size", type=int, default=1, help="Loader batch size (single-graph = 1).")
    p.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    p.add_argument("--pin-memory", action="store_true", help="Enable pinned memory.")
    return p.parse_args()


def main() -> int:
    """Entry point for validation.

    Args:
        None

    Returns:
        int: 0 on success.
    """
    args = parse_args()
    logger = get_logger()

    tcfg = TrainingConfig(epochs=1, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    device = tcfg.device
    torch.set_default_device(device)

    graph_backend = infer_graph_backend(args.model)
    train_ds, val_ds, test_ds = create_split_datasets_from_yaml(args.dataset, graph_backend=graph_backend)
    in_dim = train_ds.sample.num_features
    num_classes = train_ds.sample.num_classes
    lc = LoaderConfig(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = build_dataloader(val_ds, lc)
    test_loader = build_dataloader(test_ds, lc)

    model = build_model_from_yaml(args.model, input_dim=in_dim, override_num_classes=num_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)


    trainer = GNNTrainer(model=model, config=tcfg)

    logger.info("Evaluating on validation set...")
    val_metrics = trainer.validate(val_loader)
    logger.info(f"val: {val_metrics}")

    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"test: {test_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
