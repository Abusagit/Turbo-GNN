"""
Learning rate scheduler factory and warmup utilities.

This module provides typed configuration and builders for common schedulers,
including linear/cosine warmup wrappers.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ExponentialLR,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

doc = """
Schedulers factory for GNN training.

- SchedulerConfig: typed scheduler configuration
- build_scheduler: construct a scheduler for an optimizer
- WarmupScheduler: linear warmup wrapper around another scheduler
"""


@dataclass
class SchedulerConfig:
    """Configuration dataclass for learning rate schedulers.

    Attributes:
        name (str): 'none','step','multistep','exponential','cosine',
            'cosine_restart','onecycle','plateau','cosine_warmup'.
        step_size (int): Step size for StepLR.
        gamma (float): Multiplicative decay factor.
        milestones (Optional[List[int]]): Milestones for MultiStepLR.
        eta_min (float): Min LR for cosine annealing.
        T_max (Optional[int]): Period (epochs) for CosineAnnealingLR.
        T_0 (int): Initial restart period for CosineAnnealingWarmRestarts.
        T_mult (int): Restart period multiplier for CosineAnnealingWarmRestarts.
        warmup_epochs (int): Linear warmup in epochs (converted to steps if per-batch).
        warmup_steps (int): Linear warmup in steps (batches or epochs depending on step granularity).
        max_lr (Optional[float]): Max LR for OneCycleLR.
        pct_start (float): Fraction of cycle to increase LR (OneCycleLR).
        div_factor (float): Initial LR divisor (OneCycleLR).
        final_div_factor (float): Final LR divisor (OneCycleLR).
        step_on_batch (bool): If True, you must call scheduler.step() each batch.
    """
    name: str = "none"
    step_size: int = 30
    gamma: float = 0.1
    milestones: Optional[List[int]] = None
    eta_min: float = 0.0
    T_max: Optional[int] = None
    T_0: int = 10
    T_mult: int = 2
    warmup_epochs: int = 0
    warmup_steps: int = 0
    max_lr: Optional[float] = None
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    step_on_batch: bool = False


class WarmupScheduler(_LRScheduler):
    """Linear warmup wrapper around another scheduler.

    During warmup, LR increases linearly from 0 to base LR; afterwards, delegates
    stepping to the wrapped scheduler (if provided) on the same step granularity.

    Note:
        Do not wrap ReduceLROnPlateau — it requires metric-driven stepping.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        wrapped: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ) -> None:
        """Initialize WarmupScheduler.

        Args:
            optimizer (Optimizer): Optimizer to schedule.
            warmup_steps (int): Number of warmup steps (batches or epochs).
            wrapped (Optional[_LRScheduler]): Inner scheduler to delegate after warmup.
            last_epoch (int): Last epoch index.

        Returns:
            None
        """
        self.warmup_steps = max(0, int(warmup_steps))
        self.wrapped = wrapped
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rates for current step.

        Args:
            None

        Returns:
            List[float]: Learning rates per param group.
        """
        if self.last_epoch < self.warmup_steps and self.warmup_steps > 0:
            scale = float(self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        if self.wrapped is not None and hasattr(self.wrapped, "get_last_lr"):
            return list(self.wrapped.get_last_lr())
        return [base_lr for base_lr in self.base_lrs]

    def step(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Advance scheduler by one step.

        Args:
            *args (Any): Delegated to wrapped scheduler after warmup.
            **kwargs (Any): Delegated to wrapped scheduler after warmup.

        Returns:
            None
        """
        self.last_epoch += 1
        if self.last_epoch <= self.warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
            return
        if self.wrapped is not None:
            self.wrapped.step(*args, **kwargs)


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    *,
    total_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
) -> Optional[_LRScheduler]:
    """Build a learning rate scheduler for the optimizer.

    Args:
        optimizer (Optimizer): Target optimizer.
        cfg (SchedulerConfig): Scheduler configuration.
        total_epochs (Optional[int]): Total epochs (for cosine/onecycle).
        steps_per_epoch (Optional[int]): Batches per epoch (for per-batch or onecycle).

    Returns:
        Optional[_LRScheduler]: Configured scheduler, or None for 'none'.

    Notes:
        - Your `trainer.py` calls `scheduler.step()` once per epoch. If you set
          `cfg.step_on_batch=True` (e.g., for OneCycle), you must call `step()`
          each batch via a hook or script instead.
        - For ReduceLROnPlateau, call `scheduler.step(metric)` with your monitored metric.
    """
    name = cfg.name.lower()
    if name in ("", "none", "null"):
        return None

    base: Optional[_LRScheduler] = None

    if name == "step":
        base = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif name == "multistep":
        base = MultiStepLR(optimizer, milestones=cfg.milestones or [], gamma=cfg.gamma)
    elif name == "exponential":
        base = ExponentialLR(optimizer, gamma=cfg.gamma)
    elif name == "cosine":
        tmax = cfg.T_max if cfg.T_max is not None else total_epochs
        if tmax is None:
            raise ValueError("cosine scheduler requires T_max or total_epochs")
        base = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=cfg.eta_min)
    elif name in ("cosine_restart", "cosine_restarts"):
        base = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
    elif name == "onecycle":
        if steps_per_epoch is None or total_epochs is None or cfg.max_lr is None:
            raise ValueError("onecycle requires steps_per_epoch, total_epochs, and max_lr")
        base = OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.pct_start,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
            three_phase=False,
            anneal_strategy="cos",
        )
    elif name in ("cosine_warmup", "warmup_cosine"):
        tmax = cfg.T_max if cfg.T_max is not None else total_epochs
        if tmax is None:
            raise ValueError("cosine_warmup requires T_max or total_epochs")
        base = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=cfg.eta_min)
    else:
        raise KeyError(f"Unsupported scheduler '{cfg.name}'.")

    warmup_steps = 0
    if cfg.warmup_steps > 0:
        warmup_steps = cfg.warmup_steps
    elif cfg.warmup_epochs > 0:
        # If you step per epoch (your current trainer), warmup_epochs is fine.
        # If you later step per batch, convert to steps with steps_per_epoch.
        warmup_steps = cfg.warmup_epochs if steps_per_epoch is None else cfg.warmup_epochs * steps_per_epoch

    if warmup_steps > 0:
        return WarmupScheduler(optimizer, warmup_steps=warmup_steps, wrapped=base)
    return base
