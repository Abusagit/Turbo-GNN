"""
Optimizers factory for the training pipeline.

This module implements utilities to build PyTorch optimizers with sensible
defaults for GNNs, including parameter grouping (no weight decay on bias
and normalization parameters) and optional fused/foreach support.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


doc = """
Optimizers factory for GNN training.

- OptimizerConfig: typed configuration for common optimizers
- build_optimizer: construct an optimizer for a model
- create_param_groups: build param groups with/without weight decay
"""


@dataclass
class OptimizerConfig:
    """Configuration dataclass for building optimizers.

    Attributes:
        name (str): Optimizer name ('adamw','adam','sgd','rmsprop','adagrad','lion').
        lr (float): Base learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        betas (Tuple[float, float]): (beta1, beta2) for Adam/AdamW/Lion.
        eps (float): Numerical stability epsilon.
        momentum (float): Momentum for SGD/RMSprop.
        nesterov (bool): Use Nesterov momentum for SGD.
        amsgrad (bool): Use AMSGrad variant for Adam/AdamW.
        foreach (Optional[bool]): Force foreach implementation (None keeps PyTorch default).
        fused (Optional[bool]): Request fused CUDA implementation if available (PyTorch≥2.0).
        no_decay_norm_bias (bool): Exclude weight decay on bias and norm/embedding params.
    """
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = False
    amsgrad: bool = False
    foreach: Optional[bool] = None
    fused: Optional[bool] = None
    no_decay_norm_bias: bool = True


def _is_norm_module(module: nn.Module) -> bool:
    """Return True if module is a normalization layer.

    Args:
        module (nn.Module): The module to check.

    Returns:
        bool: True if normalization-like module, False otherwise.
    """
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
        nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LocalResponseNorm,
        # nn.RMSNorm, # BUG wasn't added in pytorch 2.4
    )
    return isinstance(module, norm_types)


def _decay_filter(name: str, param: nn.Parameter, module: Optional[nn.Module]) -> bool:
    """Decide whether weight decay should be applied to a parameter.

    Args:
        name (str): Parameter name (e.g., 'weight', 'bias').
        param (nn.Parameter): The parameter tensor.
        module (Optional[nn.Module]): The parent module if available.

    Returns:
        bool: True if weight decay should be applied, False otherwise.
    """
    if not param.requires_grad:
        return False
    if name.endswith("bias"):
        return False
    if module is not None and _is_norm_module(module):
        return False
    if module is not None and isinstance(module, nn.Embedding):
        return False
    return True


def create_param_groups(model: nn.Module, weight_decay: float, *, no_decay_norm_bias: bool = True) -> List[Dict[str, Any]]:
    """Create parameter groups that separate decay vs no-decay params.

    Args:
        model (nn.Module): Model whose parameters to group.
        weight_decay (float): Weight decay for decay group.
        no_decay_norm_bias (bool): If True, exclude bias/Norm/Embedding from decay.

    Returns:
        List[Dict[str, Any]]: Two param groups:
            [{'params': [...], 'weight_decay': wd}, {'params': [...], 'weight_decay': 0.0}]
    """
    if not no_decay_norm_bias:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            _ = module_name  # name kept only for debugging/logging if needed
            if _decay_filter(pname, param, module):
                decay.append(param)
            else:
                no_decay.append(param)

    remaining = set(p for p in model.parameters() if p.requires_grad) - set(decay) - set(no_decay)
    if remaining:
        decay.extend(list(remaining))

    return [
        {"params": list({*decay}), "weight_decay": weight_decay},
        {"params": list({*no_decay}), "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    """Construct a PyTorch optimizer from config.

    Args:
        model (nn.Module): Model to optimize.
        cfg (OptimizerConfig): Optimizer configuration.

    Returns:
        Optimizer: Configured optimizer instance.

    Raises:
        KeyError: If the optimizer name is unsupported.
        ValueError: If Lion is requested but not installed.
    """
    param_groups = create_param_groups(model, cfg.weight_decay, no_decay_norm_bias=cfg.no_decay_norm_bias)
    name = cfg.name.lower()
    common_kwargs: Dict[str, Any] = {"lr": cfg.lr}
    if cfg.foreach is not None:
        common_kwargs["foreach"] = cfg.foreach
    if cfg.fused is not None:
        common_kwargs["fused"] = cfg.fused

    if name == "adamw":
        return torch.optim.AdamW(param_groups, betas=cfg.betas, eps=cfg.eps, amsgrad=cfg.amsgrad, **common_kwargs)
    if name == "adam":
        return torch.optim.Adam(param_groups, betas=cfg.betas, eps=cfg.eps, amsgrad=cfg.amsgrad, **common_kwargs)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=cfg.momentum, nesterov=cfg.nesterov, **common_kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(param_groups, momentum=cfg.momentum, eps=cfg.eps, **common_kwargs)
    if name == "adagrad":
        return torch.optim.Adagrad(param_groups, eps=cfg.eps, **common_kwargs)
    if name == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore
            return Lion(param_groups, betas=cfg.betas, weight_decay=cfg.weight_decay, lr=cfg.lr)
        except Exception as exc:
            raise ValueError("Lion optimizer requested but 'lion-pytorch' is not installed") from exc

    raise KeyError(f"Unsupported optimizer '{cfg.name}'.")
