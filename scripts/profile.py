#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any, Dict

from scripts._common import read_yaml, ensure_outdir

from src.data.datasets import create_split_datasets_from_yaml
from src.data.loaders import LoaderConfig, build_dataloader
from src.models.config import build_model_from_yaml
from src.training.trainer import GNNTrainer, TrainingConfig
from src.training.optimizer import OptimizerConfig, build_optimizer
from src.training.scheduler import SchedulerConfig, build_scheduler
from src.training.hooks import ProfilerHook
from src.utils.logger import get_logger

doc = """
Profiling launcher script (updated to use scripts/_common.py).

- Reads a training YAML and a profiler YAML
- Builds dataset + model
- Creates trainer, optimizer, scheduler
- Attaches ProfilerHook configured from YAML
- Runs training for the given number of epochs (keep small for flame charts)
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI args for profiling.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(description="Profile training with torch.profiler.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset YAML path.")
    p.add_argument("--model", type=str, required=True, help="Model YAML path.")
    p.add_argument("--training", type=str, required=True, help="Training YAML path.")
    p.add_argument("--profile", type=str, required=True, help="Profiler YAML path.")
    p.add_argument("--out", type=str, default="runs/profile", help="Output directory.")
    return p.parse_args()


def main() -> int:
    """Entry point for profiling launcher.

    Args:
        None

    Returns:
        int: 0 on success.
    """
    args = parse_args()
    logger = get_logger()

    training_cfg: Dict[str, Any] = read_yaml(args.training).get("training", {})
    opt_cfg_d: Dict[str, Any] = read_yaml(args.training).get("optimizer", {})
    sch_cfg_d: Dict[str, Any] = read_yaml(args.training).get("scheduler", {})
    prof_cfg: Dict[str, Any] = read_yaml(args.profile).get("profiler", {})

    # Data
    train_ds, val_ds, _ = create_split_datasets_from_yaml(args.dataset)
    in_dim = train_ds.sample.num_features
    num_classes = train_ds.sample.num_classes
    lc = LoaderConfig(batch_size=int(training_cfg.get("batch_size", 1)),
                      num_workers=int(training_cfg.get("num_workers", 0)),
                      pin_memory=bool(training_cfg.get("pin_memory", True)))
    train_loader = build_dataloader(train_ds, lc)
    val_loader = build_dataloader(val_ds, lc)

    # Model
    model = build_model_from_yaml(args.model, input_dim=in_dim, override_num_classes=num_classes)

    # Trainer/Opt/Sched
    tcfg = TrainingConfig(**training_cfg) if training_cfg else TrainingConfig()
    trainer = GNNTrainer(model=model, config=tcfg)

    ocfg = OptimizerConfig(**opt_cfg_d) if opt_cfg_d else OptimizerConfig()
    optimizer = build_optimizer(model, ocfg)
    trainer.optimizer = optimizer

    if sch_cfg_d and sch_cfg_d.get("name", "none") != "none":
        scfg = SchedulerConfig(**sch_cfg_d)
        trainer.scheduler = build_scheduler(optimizer, scfg, steps_per_epoch=len(train_loader),
                                            total_epochs=int(training_cfg.get("epochs", 1)))

    # Profiler hook
    outdir = ensure_outdir(args.out)
    trainer.add_hook(ProfilerHook(
        output_dir=str(outdir / "profiler"),
        wait=int(prof_cfg.get("wait", 1)),
        warmup=int(prof_cfg.get("warmup", 1)),
        active=int(prof_cfg.get("active", 3)),
        repeat=int(prof_cfg.get("repeat", 1)),
        profile_memory=bool(prof_cfg.get("profile_memory", True)),
        with_stack=bool(prof_cfg.get("with_stack", True)),
    ))

    logger.info("Profiling run…")
    trainer.train(train_loader, val_loader, test_loader=None)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
