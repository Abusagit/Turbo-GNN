import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

sys.path.append("./")

from src.data.datasets import GraphBackendOption
from src.data.loaders import LoaderConfig, build_dataloader
from src.models.config import build_model_from_yaml
from src.training.hooks import CheckpointHook, MemoryHook, MetricHook, ProfilerHook
from src.training.optimizer import OptimizerConfig, build_optimizer
from src.training.scheduler import SchedulerConfig, build_scheduler
from src.training.trainer import GNNTrainer, TrainingConfig
from src.utils.logger import get_logger
from src.utils.scripts_utils import (
    create_split_datasets_from_yaml,
    ensure_outdir,
    infer_graph_backend,
    merge_yaml_files,
    read_yaml,
    save_json,
)

doc = """
Train launcher script (updated to use scripts/_common.py).

- Merges one or more training YAMLs (order matters; later overrides earlier)
- Builds dataset + optional transforms from dataset YAML
- Builds model from model YAML (infers in_channels; overrides num_classes)
- Constructs optimizer + scheduler from merged training config
- Attaches hooks (metrics, checkpoints, memory; optional profiler)
- Trains and writes history JSON in the output directory
"""


# --------------------------- small helpers ---------------------------------- #


def _extract_training_cfg(full: dict[str, Any]) -> dict[str, Any]:
    """Extract the 'training' section from a merged config dict.

    Args:
        full (Dict[str, Any]): Merged configuration dictionary.

    Returns:
        Dict[str, Any]: The 'training' sub-dictionary (empty if missing).
    """
    return dict(full.get("training", {}))


def _extract_optimizer_cfg(full: dict[str, Any]) -> dict[str, Any]:
    """Extract the 'optimizer' section from a merged config dict.

    Args:
        full (Dict[str, Any]): Merged configuration dictionary.

    Returns:
        Dict[str, Any]: The 'optimizer' sub-dictionary (empty if missing).
    """
    return dict(full.get("optimizer", {}))


def _extract_scheduler_cfg(full: dict[str, Any]) -> dict[str, Any]:
    """Extract the 'scheduler' section from a merged config dict.

    Args:
        full (Dict[str, Any]): Merged configuration dictionary.

    Returns:
        Dict[str, Any]: The 'scheduler' sub-dictionary (empty if missing).
    """
    return dict(full.get("scheduler", {}))


# ------------------------------ build pipeline ------------------------------ #


def build_data(
    dataset_yaml: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    graph_backend: GraphBackendOption,
) -> tuple[Any, Any, Any, int, int]:
    """Create train/val/test loaders and infer dataset dimensions.

    Args:
        dataset_yaml (str | Path): Path to dataset (and transforms) YAML.
        batch_size (int): DataLoader batch size.
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): Whether to enable pinned memory.
        graph_backend (GraphBackendOption): infered graph representation type
    Returns:
        Tuple[Any, Any, Any, int, int]: Tuple of
            (train_loader, val_loader, test_loader, num_features, num_classes).
    """
    train_ds, val_ds, test_ds = create_split_datasets_from_yaml(str(dataset_yaml), graph_backend=graph_backend)
    num_features = train_ds.sample.num_features
    num_classes = train_ds.sample.num_classes

    lc = LoaderConfig(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = build_dataloader(train_ds, lc)
    val_loader = build_dataloader(val_ds, lc)
    test_loader = build_dataloader(test_ds, lc)
    return train_loader, val_loader, test_loader, num_features, num_classes


def build_opt_and_sched(
    model: torch.nn.Module,
    merged_cfg: dict[str, Any],
    *,
    steps_per_epoch: int | None,
    total_epochs: int,
) -> tuple[torch.optim.Optimizer, Any | None]:
    """Build optimizer and scheduler from merged config.

    Args:
        model (torch.nn.Module): Model whose parameters will be optimized.
        merged_cfg (Dict[str, Any]): Merged YAML configuration.
        steps_per_epoch (Optional[int]): Steps per epoch (for per-step schedulers).
        total_epochs (int): Total number of training epochs.

    Returns:
        Tuple[torch.optim.Optimizer, Optional[Any]]: (optimizer, scheduler).
    """
    opt_cfg = (
        OptimizerConfig(**_extract_optimizer_cfg(merged_cfg))
        if _extract_optimizer_cfg(merged_cfg)
        else OptimizerConfig()
    )
    optimizer = build_optimizer(model, opt_cfg)

    scheduler = None
    sch_d = _extract_scheduler_cfg(merged_cfg)
    if sch_d and sch_d.get("name", "none") != "none":
        scfg = SchedulerConfig(**sch_d)
        scheduler = build_scheduler(optimizer, scfg, steps_per_epoch=steps_per_epoch, total_epochs=total_epochs)

    return optimizer, scheduler


def build_trainer(
    model: torch.nn.Module,
    merged_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> GNNTrainer:
    """Construct `GNNTrainer` from merged training config.

    Args:
        model (torch.nn.Module): Model to train.
        merged_cfg (Dict[str, Any]): Merged configuration dictionary.

    Returns:
        GNNTrainer: Trainer instance (without optimizer/scheduler yet).
    """
    tcfg_dict = _extract_training_cfg(merged_cfg)
    tcfg = TrainingConfig(**tcfg_dict) if tcfg_dict else TrainingConfig()

    return GNNTrainer(model=model, config=tcfg, optimizer=optimizer, scheduler=scheduler)


# ------------------------------- CLI and main -------------------------------- #


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the train launcher.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    # TODO add autotune right in training

    p = argparse.ArgumentParser(description="Train a GNN with backend-agnostic stack.")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset YAML.")
    p.add_argument("--model", type=str, required=True, help="Path to model YAML.")
    p.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="One or more training YAMLs to merge (e.g., base.yaml, amp.yaml, typically from `config/training` dir).",
    )
    p.add_argument(
        "--profile", type=str, default=None, help="Optional profiler YAML (configs/benchmarks/profile.yaml)."
    )
    p.add_argument("--out", type=str, default="runs/train", help="Output directory.")
    p.add_argument("--record-snapshots", action="store_true", help="Flag to record memory snapshots")

    p.add_argument("--conv_type", type=str, required=True, help="Convolution type")
    p.add_argument("--backend", type=str, required=True, help="Backend type")

    return p.parse_args()


def main() -> int:
    """Entry point: YAML merge -> data/model -> trainer/opt/sched -> train.

    Args:
        None

    Returns:
        int: 0 on success.
    """
    args = parse_args()
    logger = get_logger()

    # merge training YAMLs (later overrides earlier)
    merged_cfg: dict[str, Any] = merge_yaml_files(args.config)
    outdir = ensure_outdir(args.out)

    # NOTE initialize memory hook as early as possible to collect ALL traces
    memory_hook = MemoryHook(
        measure_every=1,
        sample_batches=None,
        log_every=5,
        track_cpu_rss=True,
        sync_cuda=True,
        record_snapshots=args.record_snapshots,
        snapshot_dir=str(outdir / "memory_snahphots"),
    )

    # build data
    tcfg = _extract_training_cfg(merged_cfg)
    batch_size = int(tcfg.get("batch_size", 1))
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", True))

    torch.set_default_device(tcfg["device"])

    graph_backend = args.backend
    train_loader, val_loader, test_loader, in_dim, num_classes = build_data(
        args.dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        graph_backend=graph_backend,
    )

    # build model
    model = build_model_from_yaml(
        args.model,
        backend_to_override=args.backend,
        conv_type_to_override=args.conv_type,
        input_dim=in_dim,
        override_num_classes=num_classes,
    )
    print(model)

    # Optimizer + Scheduler
    steps_per_epoch = len(train_loader)
    total_epochs = int(tcfg.get("epochs", 1))
    optimizer, scheduler = build_opt_and_sched(
        model, merged_cfg, steps_per_epoch=steps_per_epoch, total_epochs=total_epochs
    )
    logger.info("Built Optimizer & Schedulers")

    # build trainer
    trainer = build_trainer(model, merged_cfg=merged_cfg, optimizer=optimizer, scheduler=scheduler)
    logger.info(f"Built data with graph representation: {graph_backend}\tBuilt Trainer\tBuild model")

    trainer.add_hook(
        CheckpointHook(
            checkpoint_dir=str(outdir / "ckpts"),
            save_interval=int(tcfg.get("checkpoint_interval", 10)),
            keep_last_n=5,
            save_best_only=False,
        )
    )

    trainer.add_hook(memory_hook)

    # trainer.add_hook(LRSchedulerStepHook(scheduler=...)) # TODO add LR scheduler
    if args.profile:
        prof_cfg = read_yaml(args.profile).get("profiler", {})
        trainer.add_hook(
            ProfilerHook(
                output_dir=str(outdir / "profiler"),
                wait=int(prof_cfg.get("wait", 1)),
                warmup=int(prof_cfg.get("warmup", 1)),
                active=int(prof_cfg.get("active", 3)),
                repeat=int(prof_cfg.get("repeat", 1)),
                profile_memory=bool(prof_cfg.get("profile_memory", True)),
                with_stack=bool(prof_cfg.get("with_stack", True)),
            )
        )

    comet_config = None
    params_for_comet = None
    if "comet_ml" in merged_cfg:
        params_for_comet = {}
        params_for_comet["dataset"] = read_yaml(args.dataset)["dataset"]["name"]
        model_config = read_yaml(args.model)
        params_for_comet["model"] = str(model_config)
        params_for_comet["conv_type"] = model_config["encoder"]["layers"][0]["conv_type"]
        params_for_comet["backend"] = model_config["encoder"]["layers"][0]["backend"]
        params_for_comet.update(_extract_optimizer_cfg(merged_cfg))  # add optimizer parameters

        comet_config = merged_cfg["comet_ml"]
        comet_config["ExperimentConfig"]["tags"].extend(
            [
                f'dataset: {params_for_comet["dataset"]}',
                f'conv_type: {params_for_comet["conv_type"]}',
                f'backend: {params_for_comet["backend"]}',
            ]
        )

    # should be at the end to log other hooks' metrics, e.g. memory
    trainer.add_hook(
        MetricHook(
            log_dir=str(outdir / "logs"),
            log_interval=int(tcfg.get("log_interval", 10)),
            comet_config=comet_config,
            params_for_comet=params_for_comet,
        )
    )

    # train
    logger.info("Starting training…")
    history = trainer.train(train_loader, val_loader, test_loader)
    save_json(outdir / "history.json", history)  # pretty JSON
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
