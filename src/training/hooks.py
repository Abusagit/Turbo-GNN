"""
Hook system for training pipeline extensibility.

This module provides various hooks that can be attached to the training
pipeline for monitoring, profiling, checkpointing, and other extensions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import logging

from pathlib import Path
import time

__doc__ = """
Training hooks module for GNN benchmarking.

This module implements a hook system for the training pipeline that allows
for extensible monitoring and control. Hooks can be used for:
- Performance profiling
- Metric tracking
- Model checkpointing
- Custom logging
- Early stopping
- Learning rate scheduling

The hook system follows an event-driven pattern where hooks respond to
specific training events.
"""

logger = logging.getLogger(__name__)


class Hook(ABC):
    """Abstract base class for training hooks.
    
    Hooks provide a way to extend the training pipeline with custom
    functionality without modifying the core training loop.
    """
    
    @abstractmethod
    def on_training_start(
        self, 
        model: nn.Module, 
        config: Any
    ) -> None:
        """Called at the beginning of training.

        Args:
            model: The model being trained
            config: Training configuration
        """
        pass
    
    def on_training_end(
        self, 
        history: Dict[str, List[float]]
    ) -> None:
        """Called at the end of training.

        Args:
            history: Dictionary containing training history
        """
        pass
    
    def on_epoch_start(
        self, 
        epoch: int
    ) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
        """
        pass
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
        """
        pass
    
    def on_batch_start(
        self, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Called before processing each batch.

        Args:
            batch: Current batch data
            batch_idx: Batch index
        """
        pass
    
    def on_batch_end(
        self, 
        loss: torch.Tensor, 
        batch_idx: int
    ) -> None:
        """Called after processing each batch.

        Args:
            loss: Loss value for the batch
            batch_idx: Batch index
        """
        pass
    
    def on_forward_end(
        self, 
        output: torch.Tensor, 
        loss: torch.Tensor
    ) -> None:
        """Called after forward pass.

        Args:
            output: Model output
            loss: Computed loss
        """
        pass
    
    def on_best_model(
        self, 
        model: nn.Module, 
        metrics: Dict[str, float]
    ) -> None:
        """Called when a new best model is found.

        Args:
            model: The current best model
            metrics: Metrics of the best model
        """
        pass


class ProfilerHook(Hook):
    """Hook for PyTorch profiler integration.
    
    This hook enables detailed performance profiling of the training
    pipeline using PyTorch's built-in profiler.
    
    Attributes:
        output_dir: Directory to save profiling results
        wait: Number of steps to wait before profiling
        warmup: Number of warmup steps
        active: Number of active profiling steps
        repeat: Number of times to repeat the profiling cycle
        with_stack: Whether to record stack traces
        profile_memory: Whether to profile memory usage
        profiler: PyTorch profiler instance
    """
    
    def __init__(
        self,
        output_dir: str = "./profiling",
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        with_stack: bool = True,
        profile_memory: bool = True
    ) -> None:
        """Initialize the profiler hook.

        Args:
            output_dir: Directory to save profiling results
            wait: Number of steps to wait before profiling
            warmup: Number of warmup steps
            active: Number of active profiling steps
            repeat: Number of times to repeat the cycle
            with_stack: Whether to record stack traces
            profile_memory: Whether to profile memory usage
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.profiler: Optional[Any] = None
        self.step_count: int = 0
    
    def on_training_start(
        self, 
        model: nn.Module, 
        config: Any
    ) -> None:
        """Initialize the profiler at training start.

        Args:
            model: The model being trained
            config: Training configuration
        """
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        self.profiler = profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat
            ),
            on_trace_ready=tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        )
        self.profiler.__enter__()
        logger.info(f"Profiler started, writing to {self.output_dir}")
    
    def on_batch_end(
        self, 
        loss: torch.Tensor, 
        batch_idx: int
    ) -> None:
        """Step the profiler after each batch.

        Args:
            loss: Loss value for the batch
            batch_idx: Batch index
        """
        if self.profiler:
            self.profiler.step()
            self.step_count += 1
    
    def on_training_end(
        self, 
        history: Dict[str, List[float]]
    ) -> None:
        """Finalize the profiler at training end.

        Args:
            history: Dictionary containing training history
        """
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            logger.info(f"Profiling complete. Results saved to {self.output_dir}")


class MetricHook(Hook):
    """Hook for tracking and logging metrics.
    
    This hook tracks various metrics during training and provides
    logging and visualization capabilities.
    
    Attributes:
        log_dir: Directory for saving metric logs
        log_interval: Interval for logging metrics
        metrics: Dictionary storing all metrics
        start_time: Training start time
        epoch_start_time: Current epoch start time
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        log_interval: int = 10,
        use_wandb: bool = False,
        wandb_project: str = "gnn-benchmark"
    ) -> None:
        """Initialize the metric hook.

        Args:
            log_dir: Directory for saving logs
            log_interval: Interval for logging metrics
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        self.metrics: Dict[str, List[float]] = {}
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
    
    def on_training_start(
        self, 
        model: nn.Module, 
        config: Any
    ) -> None:
        """Initialize metric tracking at training start.

        Args:
            model: The model being trained
            config: Training configuration
        """
        self.start_time = time.time()
        
        if self.use_wandb:
            self.wandb.init(
                project=self.wandb_project,
                config=config.__dict__ if hasattr(config, '__dict__') else config
            )
            self.wandb.watch(model)
        
        logger.info("Metric tracking initialized")
    
    def on_epoch_start(
        self, 
        epoch: int
    ) -> None:
        """Record epoch start time.

        Args:
            epoch: Current epoch number
        """
        self.epoch_start_time = time.time()
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics at epoch end.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
        """
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Store metrics
        for key, value in train_metrics.items():
            self.metrics.setdefault(f"train_{key}", []).append(value)
        
        for key, value in val_metrics.items():
            self.metrics.setdefault(f"val_{key}", []).append(value)
        
        self.metrics.setdefault("epoch_time", []).append(epoch_time)
        
        # Log to W&B
        if self.use_wandb:
            log_dict = {
                "epoch": epoch,
                "epoch_time": epoch_time,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }
            self.wandb.log(log_dict)
        
        # Log to console
        if epoch % self.log_interval == 0:
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    def on_training_end(
        self, 
        history: Dict[str, List[float]]
    ) -> None:
        """Finalize metric tracking at training end.

        Args:
            history: Dictionary containing training history
        """
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save metrics to file
        import json
        metrics_file = self.log_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        if self.use_wandb:
            self.wandb.finish()


class CheckpointHook(Hook):
    """Hook for model checkpointing.
    
    This hook saves model checkpoints at regular intervals and when
    new best models are found.
    
    Attributes:
        checkpoint_dir: Directory for saving checkpoints
        save_interval: Interval for saving checkpoints
        keep_last_n: Number of recent checkpoints to keep
        best_model_path: Path to the best model checkpoint
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_interval: int = 10,
        keep_last_n: int = 5,
        save_best_only: bool = False
    ) -> None:
        """Initialize the checkpoint hook.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            save_interval: Interval for saving checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best_only: Whether to save only the best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best_only = save_best_only
        self.best_model_path: Optional[Path] = None
        self.checkpoints: List[Path] = []
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Save checkpoint at epoch end if needed.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
        """
        if not self.save_best_only and epoch % self.save_interval == 0:
            self._save_checkpoint(epoch, train_metrics, val_metrics)
    
    def on_best_model(
        self, 
        model: nn.Module, 
        metrics: Dict[str, float]
    ) -> None:
        """Save the best model checkpoint.

        Args:
            model: The current best model
            metrics: Metrics of the best model
        """
        if self.best_model_path and self.best_model_path.exists():
            self.best_model_path.unlink()
        
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.best_model_path)
        logger.info(f"Best model saved to {self.best_model_path}")
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Save a checkpoint file.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        checkpoint = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")


class EarlyStoppingHook(Hook):
    """Hook for early stopping based on validation metrics.
    
    This hook monitors validation metrics and triggers early stopping
    when no improvement is observed for a specified number of epochs.
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: Whether to maximize or minimize the metric
        best_score: Best score observed so far
        counter: Counter for patience
        stopped: Whether early stopping was triggered
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'max',
        metric_key: str = 'accuracy'
    ) -> None:
        """Initialize the early stopping hook.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for the metric
            metric_key: Key of the metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_key = metric_key
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.stopped: bool = False
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Check for early stopping at epoch end.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
        """
        if self.metric_key not in val_metrics:
            return
        
        current_score = val_metrics[self.metric_key]
        
        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == 'max':
                improved = current_score > self.best_score + self.min_delta
            else:
                improved = current_score < self.best_score - self.min_delta
            
            if improved:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                
                if self.counter >= self.patience:
                    self.stopped = True
                    logger.info(f"Early stopping triggered at epoch {epoch}")
