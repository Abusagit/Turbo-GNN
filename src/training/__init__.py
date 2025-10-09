"""
Training pipeline module for graph neural networks.

This module provides comprehensive training infrastructure including trainers,
optimizers, schedulers, metrics, and hooks for profiling and monitoring.
"""

from .hooks import CheckpointHook, Hook, MetricHook, ProfilerHook
from .metrics import MetricTracker, compute_accuracy, compute_f1
from .trainer import GNNTrainer, TrainingConfig

__doc__ = """
Training pipeline for GNN benchmarking.

This module implements a flexible training pipeline with:
- Configurable trainers with hook system
- Automatic mixed precision support
- Profiling and monitoring capabilities
- Metric tracking and checkpointing
"""

__all__ = [
    "GNNTrainer",
    "TrainingConfig",
    "Hook",
    "ProfilerHook",
    "MetricHook",
    "CheckpointHook",
    "MetricTracker",
    "compute_accuracy",
    "compute_f1",
]
