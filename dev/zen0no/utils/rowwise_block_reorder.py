import gc
from pathlib import Path

import dgl
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.data.datasets import DatasetConfig, load_single_graph

row_size = 16


dataset_names = [
    "artnet-views",
    "avazu-ctr",
    "city-roads-M",
    "hm-categories",
    "ogbn-arxiv",
    "ogbn-products",
    "tolokers-2",
    "twitch-views",
]

sources = [
    "pyg",
    "pyg",
    "pyg",
    "pyg",
    "ogbn",
    "ogbn",
    "pyg",
    "pyg",
]


def process_datasets(output_path: Path) -> None:
    for dataset_name, source in zip(dataset_names, sources):
        dataset_path = output_path / dataset_name.lower().replace("-", "_")
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset_config = DatasetConfig(
            source=source,
            name=dataset_name,
            conv_backend="dgl",
            root="data",
        )
        graph = load_single_graph(dataset_config)
        dgl_graph = dgl.graph((graph.edge_index[0], graph.edge_index[1]))
        dgl_graph = dgl_graph.to("cuda")
        # dgl_graph = dgl.reorder_graph(dgl_graph, node_permute_algo="metis", permute_config={"k": 1024})
        src_indices, dst_indices = dgl_graph.edges()
        views = split_by_rows(src_indices, dst_indices, row_size)
        columns_wize_stats = []
        for view in views:
            assert ((view[0] - view[0]).min() < row_size).all()
            dense_row_block = csr_to_dense(view[0], view[1], graph.num_nodes, row_size)
            columns_wize_stats_local = calculate_columns_wize_stats(dense_row_block)
            # Stats tensor needs to be size row_size + 1 to hold sum values 0 to row_size
            stats = torch.zeros(row_size + 1, device=src_indices.device, dtype=torch.int64)
            indices = columns_wize_stats_local[0].to(torch.int64)
            values = columns_wize_stats_local[1].to(torch.int64)
            # Filter out indices that are out of bounds (shouldn't happen, but be safe)
            valid_mask = (indices >= 0) & (indices <= row_size)
            if valid_mask.any():
                stats.scatter_add_(0, indices[valid_mask], values[valid_mask])
            columns_wize_stats.append(stats)
        stacked = torch.stack(columns_wize_stats).to(torch.float32)
        zero_to_ones_ration = (stacked[:, 0] / stacked.sum(dim=1)).mean(dim=0)
        stats = stacked[:, 1:].mean(dim=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=range(1, row_size + 1), y=stats.cpu().numpy(), ax=ax)

        # Add text annotation with zero_to_non_zero ratio
        ratio_text = f"Zero to Non-Zero Ratio: {zero_to_ones_ration.item():.4f}"
        ax.text(
            0.02,
            0.98,
            ratio_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        ax.set_xlabel("Column Index")
        ax.set_ylabel("Mean Count")
        ax.set_title(f"Row-wise Block Statistics - {dataset_name}")

        plot_path = dataset_path / "rowwise_stats.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {plot_path} with ratio: {zero_to_ones_ration.item():.4f}")

        del dgl_graph, src_indices, dst_indices, views, columns_wize_stats, stacked, zero_to_ones_ration, stats
        torch.cuda.empty_cache()
        gc.collect()


def split_by_rows(
    src_indices: torch.Tensor, dst_indices: torch.Tensor, row_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    splitted = src_indices.clone() // row_size
    boundaries = torch.cat([torch.tensor([True], device=src_indices.device), splitted[1:] != splitted[:-1]])
    idx = boundaries.nonzero(as_tuple=True)[0]
    idx = torch.cat([idx, torch.tensor([len(splitted)], device=src_indices.device)])
    return [(src_indices[idx[i] : idx[i + 1]], dst_indices[idx[i] : idx[i + 1]]) for i in range(len(idx) - 1)]


def calculate_columns_wize_stats(dense_row_block: torch.Tensor) -> torch.Tensor:
    sums = dense_row_block.sum(dim=0)
    return sums.unique(return_counts=True)


def csr_to_dense(src_indices: torch.Tensor, dst_indices: torch.Tensor, num_nodes: int, row_size: int) -> torch.Tensor:
    dense = torch.zeros(row_size, num_nodes, device=src_indices.device).view(-1)
    index_unwrapped = (src_indices - src_indices.min()) * num_nodes + dst_indices
    dense.scatter_(0, index_unwrapped, 1)
    return dense.view(row_size, num_nodes)


if __name__ == "__main__":
    process_datasets(Path("dev/zen0no/plots/rowwise_block"))
