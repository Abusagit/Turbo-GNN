import itertools
import json
import subprocess
from pathlib import Path


DATASETS = {
    "ogbn-arxiv": "configs/datasets/main/ogbn_arxiv.yaml",
    "tolokers-2": "configs/datasets/main/tolokers_2.yaml",
    "twitch-views": "configs/datasets/main/twitch_views.yaml",
}

# больше голов, убрать hidden dim = 32, графы с большим кол-вом соседей у вершин, снять профили на любом выбранном графе
CONFIGS = [
    (64, 1),
    (128, 1),
    (256, 1),
]

AMPS = ["none", "fp16", "bf16"]

PIPELINE_VARIANTS = [0, 1, 2, 4]  # 0 disables the pipeline (baseline)

RESULTS_DIR = Path("results/pipeline_sweep_after_tile_ops")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


for dataset_name, dataset_path in DATASETS.items():
    for (feature_dim, heads), amp, stages in itertools.product(
        CONFIGS,
        AMPS,
        PIPELINE_VARIANTS,
    ):
        head_dim = feature_dim
        mode_name = "baseline" if stages == 0 else f"pipeline_s{stages}"

        output_path = RESULTS_DIR / (
            f"{dataset_name}"
            f"_f{feature_dim}"
            f"_h{heads}"
            f"_d{head_dim}"
            f"_{amp}"
            f"_{mode_name}.json"
        )

        cmd = [
            "python3",
            "scripts/benchmark.py",
            "--layer",
            "gat_v2",
            "--backend",
            "cuda",
            "--dataset",
            dataset_path,
            "--feature_dim",
            str(feature_dim),
            "--heads",
            str(heads),
            "--mode",
            "forward",
            "--warmup",
            "20",
            "--iters",
            "100",
            "--amp",
            amp,
            "--json-out",
            str(output_path),
            "--pipeline-stages",
            str(stages),
        ]

        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
