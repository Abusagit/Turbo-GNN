from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

doc = """
Unified DataLoader builder for GNN workloads (works with PyG/DGL/custom datasets).

For now it isn't used as we work with full-batch training
"""


@dataclass
class LoaderConfig:
    """Configuration for building DataLoader.

    Attributes:
        batch_size (int): Batch size for loading.
        num_workers (int): Number of worker processes (0 for main process).
        pin_memory (bool): Pin host memory when using CUDA.
        persistent_workers (bool): Keep workers alive across epochs (when num_workers>0).
        prefetch_factor (int): Number of batches prefetched per worker.
        drop_last (bool): Drop the last incomplete batch.
        shuffle (bool): Shuffle dataset each epoch.
    """
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = False
    shuffle: bool = False


def build_dataloader(ds: Dataset, cfg: LoaderConfig) -> DataLoader:
    """Build a DataLoader with sensible defaults for GNNs.

    Args:
        ds (Dataset): Dataset instance.
        cfg (LoaderConfig): Loader configuration.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=cfg.drop_last,
    )
