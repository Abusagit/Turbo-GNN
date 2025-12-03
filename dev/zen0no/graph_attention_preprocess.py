from pathlib import Path

import dgl
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.data.datasets import DatasetConfig, load_single_graph


def plot_large_graph_thumbnail(
    src_indices: torch.Tensor, dst_indices: torch.Tensor, block_num: int, out_path: Path
) -> None:
    """Plot a thumbnail of a large graph.

    Args:
        edge_index (torch.Tensor): [2, E] long.
        block_size (int): Block size.
        out_path (str): Output path.
    """

    num_original_nodes = max(src_indices.max(), dst_indices.max()) + 1
    block_size = max(1, num_original_nodes // block_num)

    thumb_src_indices = src_indices.clone() // block_size
    thumb_dst_indices = dst_indices.clone() // block_size

    num_nodes = max(thumb_src_indices.max(), thumb_dst_indices.max()) + 1
    thumbnail_map = torch.zeros(num_nodes, num_nodes, dtype=torch.int32)

    for u, v in zip(thumb_src_indices, thumb_dst_indices):
        thumbnail_map[u, v] += 1

    sns.heatmap(thumbnail_map, cmap="viridis")
    print(f"Saving figure to {out_path}")
    plt.savefig(out_path)
    plt.close()


def reorder_and_plot(src_indices: torch.Tensor, dst_indices: torch.Tensor, block_size: int, out_path: str) -> None:
    original_path = out_path / "original.png"
    plot_large_graph_thumbnail(src_indices, dst_indices, block_size, original_path)

    dgl_graph = dgl.graph((src_indices, dst_indices))

    graph_perm = dgl.reorder_graph(dgl_graph, node_permute_algo="metis", permute_config={"k": 8192})
    src_indices, dst_indices = graph_perm.edges()

    reordered_path = out_path / "reordered.png"
    plot_large_graph_thumbnail(src_indices, dst_indices, block_size, reordered_path)


def process_datasets(output_path: Path) -> None:
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

    print(f"Processing {len(dataset_names)} datasets")

    for i, (dataset_name, source) in enumerate(zip(dataset_names, sources)):
        print(f"Processing {i + 1}/{len(dataset_names)}: {dataset_name} ({source})")
        dataset_path = output_path / dataset_name.lower().replace("-", "_")

        dataset_path.mkdir(parents=True, exist_ok=True)

        dataset_config = DatasetConfig(
            source=source,
            name=dataset_name,
            graph_backend="edge_list",
            root="data",
        )
        graph = load_single_graph(dataset_config)
        src_indices, dst_indices = graph.edge_index[0], graph.edge_index[1]
        reorder_and_plot(src_indices, dst_indices, block_size=256, out_path=dataset_path)


if __name__ == "__main__":
    out_path = Path("dev/zen0no/plots/adjency_matrix").resolve()
    process_datasets(out_path)
