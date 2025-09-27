"""
Training pipeline module for graph neural networks.

This module provides comprehensive training infrastructure including trainers,
optimizers, schedulers, metrics, and hooks for profiling and monitoring.
"""

from .trainer import GNNTrainer, TrainingConfig
from .hooks import Hook, ProfilerHook, MetricHook, CheckpointHook
from .metrics import MetricTracker, compute_accuracy, compute_f1

__doc__ = """
Training pipeline for GNN benchmarking.

This module implements a flexible training pipeline with:
- Configurable trainers with hook system
- Automatic mixed precision support
- Profiling and monitoring capabilities
- Metric tracking and checkpointing
- Distributed training support
"""

__all__ = [
    'GNNTrainer',
    'TrainingConfig',
    'Hook',
    'ProfilerHook',
    'MetricHook',
    'CheckpointHook',
    'MetricTracker',
    'compute_accuracy',
    'compute_f1'
]
