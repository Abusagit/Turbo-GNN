import itertools
import os
import subprocess
import sys
from pathlib import Path


DATASET_NAME = "twitch-views"
DATASET_PATH = "configs/datasets/main/twitch_views.yaml"

HEAD_DIMS = [64, 128, 256]
PIPELINE_STAGES = [0, 2]  # 0 disables the pipeline (baseline)

OUTPUT_DIR = Path("results/ncu_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

for head_dim, pipeline_stages in itertools.product(
    HEAD_DIMS,
    PIPELINE_STAGES,
):
    variant = "baseline" if pipeline_stages == 0 else f"pipeline_s{pipeline_stages}"

    report = OUTPUT_DIR / (
        f"{DATASET_NAME}_d{head_dim}_{variant}"
    )

    cmd = [
        "/usr/local/cuda-12.9/bin/ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--kernel-name",
        "GATv2Forward_Kernel",
        "--launch-count",
        "1",
        "-f",
        "-o",
        str(report),
        sys.executable,
        "scripts/benchmark.py",
        "--layer",
        "gat_v2",
        "--backend",
        "cuda",
        "--dataset",
        DATASET_PATH,
        "--feature_dim",
        str(head_dim),
        "--heads",
        "1",
        "--mode",
        "forward",
        "--warmup",
        "2",
        "--iters",
        "1",
        "--amp",
        "none",
        "--pipeline-stages",
        str(pipeline_stages),
    ]

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
