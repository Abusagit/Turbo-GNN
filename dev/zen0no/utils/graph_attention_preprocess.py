from copy import deepcopy
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    num_original_nodes = max(src_indices.max(), dst_indices.max()) + 1
    block_size = max(1, num_original_nodes // block_num)

    src_indices = src_indices.clone() // block_size
    dst_indices = dst_indices.clone() // block_size
    num_nodes = max(src_indices.max(), dst_indices.max()) + 1
    linear_indices = src_indices * num_nodes + dst_indices
    counts = torch.bincount(linear_indices, minlength=num_nodes * num_nodes)
    thumbnail_map = counts.reshape(num_nodes, num_nodes).to(torch.float32) / block_size**2

    means = sparsity_hist(thumbnail_map)
    sns.lineplot(x=range(len(means)), y=means, ax=axes[2])
    axes[2].set_title("Sparsity histogram")

    sns.heatmap(thumbnail_map.cpu(), cmap="viridis", ax=axes[0])
    axes[0].set_title(f"Scale: {block_size}")
    sns.heatmap(thumbnail_map.cpu() != 0, cmap="viridis", ax=axes[1])
    axes[1].set_title(f"Scale: {block_size}")
    print(f"Saving figure to {out_path}")
    plt.savefig(out_path)
    plt.close()


def sparsity_hist(tensor: torch.Tensor) -> None:
    means = []
    diag_start = -tensor.size(0) + 1
    diag_end = tensor.size(0) - 1
    for i in range(diag_start, diag_end):
        means.append(tensor.diagonal(offset=i).to(torch.float32).mean())
    return torch.tensor(means)


def reorder_and_plot(src_indices: torch.Tensor, dst_indices: torch.Tensor, block_size: int, out_path: str) -> None:
    original_path = out_path / "original.png"
    plot_large_graph_thumbnail(src_indices, dst_indices, block_size, original_path)

    dgl_graph = dgl.graph((src_indices, dst_indices))

    partition_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    block_num = 256
    for partition_size in partition_sizes:
        dgl_copy = deepcopy(dgl_graph)
        graph_perm = dgl.reorder_graph(dgl_copy, node_permute_algo="metis", permute_config={"k": partition_size})
        src_indices, dst_indices = graph_perm.edges()
        reordered_path = out_path / f"reordered_{partition_size}.png"
        plot_large_graph_thumbnail(src_indices, dst_indices, block_num, reordered_path)

        del dgl_copy, graph_perm
        torch.cuda.empty_cache()


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
            conv_backend="dgl",
            root="data",
        )
        graph = load_single_graph(dataset_config)
        src_indices, dst_indices = graph.edge_index[0].to("cuda"), graph.edge_index[1].to("cuda")
        reorder_and_plot(src_indices, dst_indices, block_size=256, out_path=dataset_path)
        del graph, src_indices, dst_indices
        torch.cuda.empty_cache()


if __name__ == "__main__":
    out_path = Path("dev/zen0no/plots/adjency_matrix").resolve()
    process_datasets(out_path)
