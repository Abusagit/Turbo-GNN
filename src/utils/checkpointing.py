from typing import Any, Dict, Optional

import torch

doc = """
Model/optimizer checkpoint save/load helpers.
"""


def save_checkpoint(
    path: str,
    *,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint to disk.

    Args:
        path (str): Output file path (.pt or .pth).
        model_state (Dict[str, Any]): model.state_dict().
        optimizer_state (Optional[Dict[str, Any]]): optimizer.state_dict().
        scheduler_state (Optional[Dict[str, Any]]): scheduler.state_dict().
        scaler_state (Optional[Dict[str, Any]]): GradScaler state_dict().
        extra (Optional[Dict[str, Any]]): Any additional metadata.

    Returns:
        None: Saves to disk.
    """
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "scaler": scaler_state,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str,
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Load checkpoint and optionally restore states.

    Args:
        path (str): Checkpoint file path.
        model (Optional[torch.nn.Module]): If provided, loads into this model.
        optimizer (Optional[torch.optim.Optimizer]): If provided, loads into this optimizer.
        scheduler (Optional[Any]): If provided, loads into this scheduler.
        scaler (Optional[Any]): If provided, loads into this GradScaler.
        map_location (str | torch.device | None): map_location for torch.load.

    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location=map_location)
    if model is not None and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
