"""
Hook system for training pipeline extensibility.

This module provides various hooks that can be attached to the training
pipeline for monitoring, profiling, checkpointing, and other extensions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Literal, List
import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import logging

from pathlib import Path
import time

from ..benchmarking.memory import reset_cuda_peak_memory, capture_cuda_snapshot, current_process_rss_bytes, human_bytes


# # local import from benchmarking utilities (relative to `training/`)
# try:
#     from ..benchmarking.memory import (
#         reset_cuda_peak_memory,
#         capture_cuda_snapshot,
#         current_process_rss_bytes,
#         human_bytes,
#     )
# except Exception as _mem_import_err:
#     reset_cuda_peak_memory = None  # type: ignore[assignment]
#     capture_cuda_snapshot = None   # type: ignore[assignment]
#     current_process_rss_bytes = None  # type: ignore[assignment]
#     human_bytes = None  # type: ignore[assignment]


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
- Other stuff you come up with ;)

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


Mode = Literal["batch", "epoch", "plateau"]
SchedulerLike = Union[_LRScheduler, ReduceLROnPlateau]

class LRSchedulerStepHook(Hook):
    """Drive LR scheduling using hook events (batch/epoch/plateau).

    This hook advances the LR scheduler at the correct time *without*
    modifying the trainer. Supports:
      - mode="batch": step after each *optimizer update* (i.e., at the grad
        accumulation boundary).
      - mode="epoch": step once at the end of each epoch.

    AMP & Accumulation:
        With gradient accumulation, the hook steps only when
        (batch_idx + 1) % accumulate_steps == 0 (i.e., when you'd call
        optimizer.step()). If AMP overflows *skip* an optimizer step, this
        hook cannot detect that without a signal from the trainer.
        # TODO if needed, add such a signal later

    Args:
        scheduler (SchedulerLike): torch scheduler.
        mode (Literal["batch","epoch"]): Step timing.
        accumulate_steps (Optional[int]): Grad accumulation steps; if None, read
            from config.accumulation_steps on training start (defaults to 1).
        log_every (int): If > 0, prints current LR every N batches (batch mode)
            or at every epoch end (epoch/plateau modes).
    """

    def __init__(
        self,
        scheduler: SchedulerLike,
        *,
        mode: Mode = "epoch",
        accumulate_steps: Optional[int] = None,
        log_every: int = 0,
    ) -> None:
        self.scheduler = scheduler
        self.mode = mode
        self.accumulate_steps = int(accumulate_steps) if accumulate_steps is not None else None
        self.log_every = int(log_every)

        # Internal state
        self._model: Optional[nn.Module] = None
        self._config: Any = None
        self._batch_counter: int = 0
        self._epoch_counter: int = 0

    # ---------------- Hook API ----------------

    def on_training_start(self, model: nn.Module, config: Any) -> None:
        """Cache references and finalize accumulation steps."""
        self._model = model
        self._config = config
        if self.accumulate_steps is None and hasattr(config, "accumulation_steps"):
            try:
                self.accumulate_steps = int(getattr(config, "accumulation_steps"))
            except Exception:
                self.accumulate_steps = 1
        if not self.accumulate_steps or self.accumulate_steps < 1:
            self.accumulate_steps = 1
        self._batch_counter = 0
        self._epoch_counter = 0

    def on_batch_end(self, loss: torch.Tensor, batch_idx: int) -> None:
        """In 'batch' mode, step after accumulation boundary."""
        if self.mode == "batch":
            self._batch_counter += 1
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self._safe_step()
                if self.log_every and (self._batch_counter % self.log_every == 0):
                    self._log_lr(prefix=f"[batch {batch_idx+1}]")

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """In 'epoch' modes, step once per epoch."""
        self._epoch_counter += 1

        if self.mode == "epoch":
            self._safe_step()
            if self.log_every:
                self._log_lr(prefix=f"[epoch {epoch}]")

    def on_training_end(self, history: Dict[str, List[float]]) -> None:
        """Optionally log final LR on training end."""
        if self.log_every:
            self._log_lr(prefix="[training end]")

    # ---------------- Internals ----------------

    def _safe_step(self) -> None:
        """Step _LRScheduler instances (not Plateau)."""
        try:
            self.scheduler.step()
        except Exception:
            # Never fail training because of scheduler hiccups
            pass

    def _last_lr_list(self) -> Optional[List[float]]:
        """Return last LR list for logging, robust to scheduler variants."""
        try:
            if hasattr(self.scheduler, "get_last_lr"):
                return list(self.scheduler.get_last_lr())
        except Exception:
            pass
        try:
            opt: Optimizer = self.scheduler.optimizer  # type: ignore[attr-defined]
            return [pg.get("lr", None) for pg in opt.param_groups]
        except Exception:
            return None

    def _log_lr(self, prefix: str = "") -> None:
        """Log learning rate(s) with a minimal dependency footprint."""
        lrs = self._last_lr_list()
        if lrs is not None:
            print(f"{prefix} lr=" + ", ".join(f"{lr:.6g}" for lr in lrs if lr is not None))


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
        wandb_project: str = "gnn-benchmark", # TODO move from wandb
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
        
        if self.use_wandb:
            log_dict = {
                "epoch": epoch,
                "epoch_time": epoch_time,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }
            self.wandb.log(log_dict)
        
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


class MemoryHook(Hook):
    """Measure CUDA peak memory per batch and summarize per epoch.

    This hook resets CUDA peak memory at batch start and reads peak stats at the
    end of the batch, giving an accurate *per-batch* peak (forward+backward+opt).
    It aggregates epoch-level stats (max/avg) and injects them into train_metrics.

    CPU-only environments are supported: if `psutil` is installed, the hook will
    record process RSS deltas; CUDA stats will be zeros.

    Args:
        measure_every (int): Measure every N batches (default: 1 = every batch).
        sample_batches (Optional[int]): If set, only the first K measured batches
            per epoch are recorded (useful to limit overhead).
        log_every (int): If > 0, print memory after every N *measured* batches.
        track_cpu_rss (bool): If True, record RSS deltas per measured batch.
        sync_cuda (bool): If True, synchronize before snapshot at batch end.
    """

    def __init__(
        self,
        *,
        measure_every: int = 1,
        sample_batches: Optional[int] = None,
        log_every: int = 0,
        track_cpu_rss: bool = True,
        sync_cuda: bool = True,
    ) -> None:
        self.measure_every = max(1, int(measure_every))
        self.sample_batches = None if sample_batches is None else max(1, int(sample_batches))
        self.log_every = max(0, int(log_every))
        self.track_cpu_rss = bool(track_cpu_rss)
        self.sync_cuda = bool(sync_cuda)

        # State
        self._epoch_measured: int = 0
        self._batch_idx_in_epoch: int = 0
        self._cuda_available: bool = torch.cuda.is_available()
        self._rss_start: Optional[int] = None

        # Accumulators (per epoch)
        self._peaks_alloc: List[int] = []
        self._peaks_reserved: List[int] = []
        self._rss_deltas: List[int] = []

    # ---------------- Hook API ----------------

    def on_training_start(self, model: nn.Module, config: Any) -> None:
        """Initialize availability flags; no heavy work needed."""
        self._cuda_available = torch.cuda.is_available()
        logger.info(
            f"MemoryHook initialized (cuda={self._cuda_available}, "
            f"measure_every={self.measure_every}, sample_batches={self.sample_batches}, "
            f"log_every={self.log_every})"
        )

    def on_epoch_start(self, epoch: int) -> None:
        """Reset per-epoch accumulators."""
        self._epoch_measured = 0
        self._batch_idx_in_epoch = 0
        self._peaks_alloc.clear()
        self._peaks_reserved.clear()
        self._rss_deltas.clear()

    def on_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Reset CUDA peak stats and (optionally) capture starting RSS."""
        self._batch_idx_in_epoch = batch_idx
        if not self._should_measure_this_batch(batch_idx):
            return

        if self._cuda_available and reset_cuda_peak_memory is not None:
            try:
                reset_cuda_peak_memory()
            except Exception:
                pass

        if self.track_cpu_rss and current_process_rss_bytes is not None:
            self._rss_start = current_process_rss_bytes()
        else:
            self._rss_start = None

    def on_batch_end(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Read peak CUDA stats for this batch and aggregate."""
        if not self._should_measure_this_batch(batch_idx):
            return

        peak_alloc = 0
        peak_reserved = 0

        if self._cuda_available and capture_cuda_snapshot is not None:
            try:
                if self.sync_cuda:
                    torch.cuda.synchronize()
                snap = capture_cuda_snapshot()
                peak_alloc = int(getattr(snap, "max_allocated_bytes", 0))
                peak_reserved = int(getattr(snap, "max_reserved_bytes", 0))
            except Exception:
                pass

        self._peaks_alloc.append(peak_alloc)
        self._peaks_reserved.append(peak_reserved)

        if self.track_cpu_rss and current_process_rss_bytes is not None:
            try:
                end_rss = current_process_rss_bytes()
                if end_rss is not None and self._rss_start is not None:
                    self._rss_deltas.append(max(0, int(end_rss) - int(self._rss_start)))
            except Exception:
                pass

        self._epoch_measured += 1

        if self.log_every > 0 and (self._epoch_measured % self.log_every == 0):
            if human_bytes is not None:
                a = human_bytes(peak_alloc, binary=True)
                r = human_bytes(peak_reserved, binary=True)
                msg = f"[batch {batch_idx}] CUDA peak alloc={a}, reserved={r}"
                if self._rss_deltas:
                    msg += f", RSS Δ={human_bytes(self._rss_deltas[-1], binary=True)}"
                logger.info(msg)
            else:
                logger.info(f"[batch {batch_idx}] CUDA peak alloc={peak_alloc}B, reserved={peak_reserved}B")

    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Summarize per-epoch stats and inject into train_metrics."""
        if not self._peaks_alloc and not self._peaks_reserved and not self._rss_deltas:
            return

        def _to_mb(x: int) -> float:
            return float(x) / (1024.0 ** 2)

        peak_alloc_max = max(self._peaks_alloc) if self._peaks_alloc else 0
        peak_reserved_max = max(self._peaks_reserved) if self._peaks_reserved else 0
        peak_alloc_avg = int(sum(self._peaks_alloc) / max(1, len(self._peaks_alloc))) if self._peaks_alloc else 0
        peak_reserved_avg = int(sum(self._peaks_reserved) / max(1, len(self._peaks_reserved))) if self._peaks_reserved else 0

        rss_delta_max = max(self._rss_deltas) if self._rss_deltas else 0
        rss_delta_avg = int(sum(self._rss_deltas) / max(1, len(self._rss_deltas))) if self._rss_deltas else 0

        # inject into train_metrics (so other hooks / logs can pick it up)
        train_metrics["cuda_peak_alloc_mb_max"] = _to_mb(peak_alloc_max)
        train_metrics["cuda_peak_alloc_mb_avg"] = _to_mb(peak_alloc_avg)
        train_metrics["cuda_peak_reserved_mb_max"] = _to_mb(peak_reserved_max)
        train_metrics["cuda_peak_reserved_mb_avg"] = _to_mb(peak_reserved_avg)
        if self._rss_deltas:
            train_metrics["cpu_rss_delta_mb_max"] = _to_mb(rss_delta_max)
            train_metrics["cpu_rss_delta_mb_avg"] = _to_mb(rss_delta_avg)

        if human_bytes is not None:
            msg = (
                f"[epoch {epoch}] CUDA peak alloc: max={human_bytes(peak_alloc_max, binary=True)}, "
                f"avg={human_bytes(peak_alloc_avg, binary=True)}; "
                f"reserved: max={human_bytes(peak_reserved_max, binary=True)}, "
                f"avg={human_bytes(peak_reserved_avg, binary=True)}"
            )
            if self._rss_deltas:
                msg += (
                    f"; RSS Δ: max={human_bytes(rss_delta_max, binary=True)}, "
                    f"avg={human_bytes(rss_delta_avg, binary=True)}"
                )
            logger.info(msg)
        else:
            logger.info(
                f"[epoch {epoch}] CUDA peak alloc: max={peak_alloc_max}B avg={peak_alloc_avg}B; "
                f"reserved: max={peak_reserved_max}B avg={peak_reserved_avg}B; "
                f"RSS Δ: max={rss_delta_max}B avg={rss_delta_avg}B"
            )

    def _should_measure_this_batch(self, batch_idx: int) -> bool:
        """Return True if this batch should be measured."""
        if self.sample_batches is not None and self._epoch_measured >= self.sample_batches:
            return False
        return (batch_idx % self.measure_every) == 0
