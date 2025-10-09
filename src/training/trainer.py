"""
Main trainer class for graph neural network training.

This module implements the core training loop with support for hooks,
automatic mixed precision, and distributed training.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

__doc__ = """
GNN Trainer implementation.

This module provides a flexible trainer for graph neural networks with:
- Configurable training loops
- Hook system for extensibility
- Automatic mixed precision support
- Gradient accumulation
- Learning rate scheduling
- Progress tracking and logging
"""

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for GNN training.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        batch_size: Batch size for training
        accumulation_steps: Gradient accumulation steps
        use_amp: Whether to use automatic mixed precision
        clip_grad_norm: Maximum gradient norm for clipping
        patience: Early stopping patience
        device: Device to use for training
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for dataloaders
        log_interval: Interval for logging metrics
        checkpoint_interval: Interval for saving checkpoints
        profile: Whether to enable profiling
    """

    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    batch_size: int = 32
    accumulation_steps: int = 1
    use_amp: bool = False
    clip_grad_norm: float | None = None
    patience: int = 20
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 10
    profile: bool = False

    def __post_init__(self):
        torch.set_default_device(torch.device(self.device))


class GNNTrainer:
    """Trainer class for graph neural networks.

    This class implements a flexible training pipeline with support for
    various training configurations and extensibility through hooks.

    Attributes:
        model: The GNN model to train
        config: Training configuration
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        criterion: Loss function
        hooks: List of training hooks
        scaler: Gradient scaler for AMP
        device: Device for training
        best_val_score: Best validation score achieved
        current_epoch: Current training epoch
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: GNN model to train
            config: Training configuration
            optimizer: Optional optimizer (created if not provided)
            scheduler: Optional learning rate scheduler
            criterion: Optional loss function (CrossEntropyLoss if not provided)
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Initialize hooks
        self.hooks: list[Any] = []

        # Setup AMP if enabled
        self.scaler = GradScaler() if config.use_amp else None
        self.autocast_context = (
            autocast(enabled=config.use_amp, dtype=torch.bfloat16) if config.use_amp else nullcontext()
        )

        # Training state
        self.best_val_score: float = 0.0
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.early_stopping_counter: int = 0

    def add_hook(self, hook: Any) -> None:
        """Add a training hook.

        Args:
            hook: Hook object to add to training pipeline
        """
        self.hooks.append(hook)
        logger.info(f"Added hook: {hook.__class__.__name__}")

    def remove_hook(self, hook: Any) -> None:
        """Remove a training hook.

        Args:
            hook: Hook object to remove from training pipeline
        """
        if hook in self.hooks:
            self.hooks.remove(hook)
            logger.info(f"Removed hook: {hook.__class__.__name__}")

    def fire_event(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire an event to all registered hooks.

        Args:
            event: Name of the event to fire
            *args: Positional arguments to pass to hook handlers
            **kwargs: Keyword arguments to pass to hook handlers
        """
        for hook in self.hooks:
            if hasattr(hook, event):
                getattr(hook, event)(*args, **kwargs)

    def _backward(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Performs backward pass

        Args:
            loss (torch.Tensor): loss tensor with backward graph
            batch_idx (int): batch index to trach gradient accumulation
        """
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # grad accumulation
        if (batch_idx + 1) % self.config.accumulation_steps == 0:
            # grad clipping
            if self.config.clip_grad_norm:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

            # optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

    def train_epoch(
        self,
        dataloader: Any,
        progress_bar_ctx: Any,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Dataloader for training data
            progress_bar_ctx: tqdm progress bar handle

        Returns:
            Dictionary containing training metrics for the epoch
        """

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            # fier batch start event
            batch_idx = progress_bar_ctx.n
            self.fire_event("on_batch_start", batch, batch_idx)

            # Forward pass with optional AMP
            with self.autocast_context:
                output = self.model(batch["features"], batch["graph"])
                loss = self.criterion(output[batch["mask"]], batch["labels"][batch["mask"]])

                # scale loss for gradient accumulation
                loss = loss / self.config.accumulation_steps

            # fire forward end event
            self.fire_event("on_forward_end", output, loss)

            self._backward(loss, batch_idx)

            # fire batch end event
            self.fire_event("on_batch_end", loss, batch_idx)

            # update metrics
            total_loss += loss.item() * self.config.accumulation_steps
            pred = output[batch["mask"]].argmax(dim=-1)
            total_correct += (pred == batch["labels"][batch["mask"]]).sum().item()
            total_samples += batch["mask"].sum().item()

            # update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress_bar_ctx.set_postfix(
                    {
                        "train_loss": round(total_loss / (batch_idx + 1), 4),
                        "train_acc": round(total_correct / total_samples if total_samples > 0 else 0, 4),
                    }
                )
            progress_bar_ctx.update()

        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        }

        return metrics

    @torch.no_grad()
    def validate(self, dataloader: Any) -> dict[str, float]:
        """Validate the model.

        Args:
            dataloader: Dataloader for validation data

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            output = self.model(batch["features"], batch["graph"])
            loss = self.criterion(output[batch["mask"]], batch["labels"][batch["mask"]])

            total_loss += loss.item()
            pred = output[batch["mask"]].argmax(dim=-1)
            total_correct += (pred == batch["labels"][batch["mask"]]).sum().item()
            total_samples += batch["mask"].sum().item()

        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        }

        return metrics

    def train(
        self, train_loader: Any, val_loader: Any | None = None, test_loader: Any | None = None
    ) -> dict[str, list[float]]:
        """Execute the full training pipeline.

        Args:
            train_loader: Dataloader for training data
            val_loader: Optional dataloader for validation data
            test_loader: Optional dataloader for test data

        Returns:
            Dictionary containing training history
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # fire training start event
        self.fire_event("on_training_start", self.model, self.config)
        n_steps = len(train_loader) * self.config.epochs

        self.progress_bar = tqdm(total=n_steps, desc="Training...")

        with self.progress_bar as progress_bar_ctx:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch

                # fire epoch start event
                self.fire_event("on_epoch_start", epoch)

                # training
                train_metrics = self.train_epoch(train_loader, progress_bar_ctx)
                history["train_loss"].append(train_metrics["loss"])
                history["train_acc"].append(train_metrics["accuracy"])

                # validation
                val_metrics = {}
                if val_loader and progress_bar_ctx.n % 20 == 0:  # validation every 20 steps
                    val_metrics = self.validate(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_acc"].append(val_metrics["accuracy"])
                    progress_bar_ctx.set_postfix(
                        {"val_loss": round(val_metrics["loss"], 4), "val_acc": round(val_metrics["accuracy"], 4)}
                    )
                    # early stopping
                    if val_metrics["accuracy"] > self.best_val_score:
                        self.best_val_score = val_metrics["accuracy"]
                        self.early_stopping_counter = 0
                        self.fire_event("on_best_model", self.model, val_metrics)
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.config.patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break

                # learning rate scheduling
                if self.scheduler:
                    self.scheduler.step()

                # fire epoch end event
                self.fire_event("on_epoch_end", epoch, train_metrics, val_metrics)

                # logging
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
                if val_metrics:
                    logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

        # fire training end event
        self.fire_event("on_training_end", history)

        # test evaluation
        if test_loader:
            test_metrics = self.validate(test_loader)
            history["test_loss"] = test_metrics["loss"]
            history["test_acc"] = test_metrics["accuracy"]
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")

        return history

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch data to the target device.

        Args:
            batch: Batch dictionary containing tensors
            graph is already on GPU, move features, labels & masks

        Returns:
            Batch dictionary with tensors moved to device
        """
        return batch  # NOTE the code is legacy, everything is put on device during the dataset cunstruction
