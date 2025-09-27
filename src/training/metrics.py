"""
Metrics module for evaluating graph neural network performance.

This module provides various metrics and tracking utilities for
monitoring model performance during training and evaluation.
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import logging

__doc__ = """
Metrics module for GNN benchmarking.

This module implements various metrics for evaluating GNN performance:
- Classification metrics (accuracy, F1, precision, recall)
- Regression metrics (MSE, MAE, R²)
- Graph-specific metrics (link prediction AUC, etc.)
- Metric tracking and aggregation utilities

The module provides both batch-wise and epoch-wise metric computation
with support for masked evaluation.
"""

logger = logging.getLogger(__name__)


class MetricTracker:
    """Tracker for accumulating and computing metrics over batches.
    
    This class accumulates predictions and labels across batches
    and computes metrics at the end of an epoch.
    
    Attributes:
        metrics_to_track: List of metric names to track
        predictions: Accumulated predictions
        labels: Accumulated ground truth labels
        losses: Accumulated loss values
        num_samples: Total number of samples
    """
    
    def __init__(
        self,
        metrics_to_track: List[str] = ['accuracy', 'f1_macro', 'f1_micro']
    ) -> None:
        """Initialize the metric tracker.

        Args:
            metrics_to_track: List of metric names to track
        """
        self.metrics_to_track = metrics_to_track
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated values."""
        self.predictions: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.losses: List[float] = []
        self.num_samples: int = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """Update tracker with batch results.

        Args:
            predictions: Model predictions for the batch
            labels: Ground truth labels for the batch
            loss: Optional loss value for the batch
            mask: Optional mask for valid samples
        """
        if mask is not None:
            predictions = predictions[mask]
            labels = labels[mask]
        
        self.predictions.append(predictions.detach().cpu())
        self.labels.append(labels.detach().cpu())
        
        if loss is not None:
            self.losses.append(loss)
        
        self.num_samples += len(labels)
    
    def compute(self) -> Dict[str, float]:
        """Compute all tracked metrics.

        Returns:
            Dictionary containing computed metrics
        """
        if not self.predictions:
            return {}
        
        all_preds = torch.cat(self.predictions, dim=0)
        all_labels = torch.cat(self.labels, dim=0)
        
        metrics = {}
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        if 'accuracy' in self.metrics_to_track:
            metrics['accuracy'] = compute_accuracy(all_preds, all_labels)
        
        if 'f1_macro' in self.metrics_to_track:
            metrics['f1_macro'] = compute_f1(all_preds, all_labels, average='macro')
        
        if 'f1_micro' in self.metrics_to_track:
            metrics['f1_micro'] = compute_f1(all_preds, all_labels, average='micro')
        
        if 'precision' in self.metrics_to_track:
            metrics['precision'] = compute_precision(all_preds, all_labels)
        
        if 'recall' in self.metrics_to_track:
            metrics['recall'] = compute_recall(all_preds, all_labels)
        
        if 'auc' in self.metrics_to_track and all_preds.dim() > 1:
            metrics['auc'] = compute_auc(all_preds, all_labels)
        
        return metrics


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        mask: Optional mask for valid samples

    Returns:
        Accuracy score between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    correct = (predictions == labels).float().sum()
    total = len(labels)
    
    return (correct / total).item() if total > 0 else 0.0


def compute_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    average: str = 'macro',
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute F1 score.

    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        average: Type of averaging ('micro', 'macro', 'weighted')
        mask: Optional mask for valid samples

    Returns:
        F1 score between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    return f1_score(labels, predictions, average=average, zero_division=0)


def compute_precision(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute precision score.

    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        mask: Optional mask for valid samples

    Returns:
        Precision score between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    true_positives = ((predictions == 1) & (labels == 1)).float().sum()
    predicted_positives = (predictions == 1).float().sum()
    
    if predicted_positives == 0:
        return 0.0
    
    return (true_positives / predicted_positives).item()


def compute_recall(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute recall score.

    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        mask: Optional mask for valid samples

    Returns:
        Recall score between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    true_positives = ((predictions == 1) & (labels == 1)).float().sum()
    actual_positives = (labels == 1).float().sum()
    
    if actual_positives == 0:
        return 0.0
    
    return (true_positives / actual_positives).item()


def compute_auc(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        mask: Optional mask for valid samples

    Returns:
        AUC score between 0 and 1
    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    # Convert to probabilities if needed
    if predictions.dim() > 1:
        predictions = F.softmax(predictions, dim=-1)
        # For multi-class, use probability of positive class
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1]
        else:
            # Multi-class AUC requires one-vs-rest
            return compute_multiclass_auc(predictions, labels)
    
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    try:
        return roc_auc_score(labels, predictions)
    except ValueError:
        # Handle case where only one class is present
        return 0.5


def compute_multiclass_auc(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """Compute multi-class AUC using one-vs-rest strategy.

    Args:
        predictions: Model predictions (probabilities) [N, C]
        labels: Ground truth labels [N]

    Returns:
        Average AUC score across all classes
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    n_classes = predictions.shape[1]
    
    # convert labels to one-hot encoding
    labels_onehot = np.eye(n_classes)[labels]
    
    auc_scores = []
    for i in range(n_classes):
        auc = roc_auc_score(labels_onehot[:, i], predictions[:, i])
        auc_scores.append(auc)
    
    return np.mean(auc_scores)


def compute_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute Mean Squared Error.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        mask: Optional mask for valid samples

    Returns:
        MSE value
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    mse = F.mse_loss(predictions, targets)
    return mse.item()


def compute_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute Mean Absolute Error.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        mask: Optional mask for valid samples

    Returns:
        MAE value
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    mae = F.l1_loss(predictions, targets)
    return mae.item()


def compute_r2_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute R² (coefficient of determination) score.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        mask: Optional mask for valid samples

    Returns:
        R² score
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()
