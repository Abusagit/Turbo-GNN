"""Same graphs/grid as run_pipeline_sweep.py, but using the autotuner instead of
manually sweeping --pipeline-stages.

Each (dataset, feature_dim, heads, amp) combo gets a single benchmark.py run
with --autotune: conv.autotune() grid-searches warps/pipeline_stages/graph
bucketing and applies the best config before timing. Results are written into
the *same* results dir as run_pipeline_sweep.py (as "..._autotuned.json"), so
plot_pipeline_results.py can compare the autotuner's pick against the
manually-forced baseline/s1/s2/s4 runs already sitting there.
"""

import itertools
import subprocess
from pathlib import Path


DATASETS = {
    "ogbn-arxiv": "configs/datasets/main/ogbn_arxiv.yaml",
    "tolokers-2": "configs/datasets/main/tolokers_2.yaml",
    "twitch-views": "configs/datasets/main/twitch_views.yaml",
}

CONFIGS = [
    (64, 1),
    (128, 1),
    (256, 1),
]

AMPS = ["none", "fp16", "bf16"]

RESULTS_DIR = Path("results/pipeline_sweep_after_tile_ops_3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


for dataset_name, dataset_path in DATASETS.items():
    for (feature_dim, heads), amp in itertools.product(CONFIGS, AMPS):
        head_dim = feature_dim

        output_path = RESULTS_DIR / (
            f"{dataset_name}"
            f"_f{feature_dim}"
            f"_h{heads}"
            f"_d{head_dim}"
            f"_{amp}"
            f"_autotuned.json"
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
            "--autotune",
            "--autotune-warmup",
            "25",
            "--autotune-iters",
            "100",
        ]

        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
