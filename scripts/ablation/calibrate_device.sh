#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/ablation/calibrate_device.sh a100 reports/simulation-a100
# Extra arguments are passed to measure_kernel_times.py (e.g. --graphs cora pubmed ...).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
TARGET=${1:?Usage: calibrate_device.sh h100\|a100 OUTPUT_DIRECTORY [measurement options]}
OUT=${2:?Provide an output directory}
shift 2
case "$TARGET" in h100|a100) ;; *) echo 'Target must be h100 or a100' >&2; exit 2 ;; esac
mkdir -p "$OUT"
"$PY" - "$TARGET" <<'PY'
import sys
from pathlib import Path

import torch
import turbo_gnn._C as extension

expected = (Path.cwd() / "turbo_gnn").resolve()
actual = Path(extension.__file__).resolve()
if actual.parent != expected:
    raise SystemExit(
        f"CUDA extension loaded from {actual}, expected a build in {expected}. "
        "Rebuild this checkout with pip install -e . --no-build-isolation before measuring."
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
name = torch.cuda.get_device_name(0)
if sys.argv[1].upper() not in name.upper():
    raise SystemExit(f"Expected {sys.argv[1]}, found {name}; set CUDA_VISIBLE_DEVICES")
print(f"Calibrating {name}, {torch.cuda.get_device_properties(0).multi_processor_count} SMs", flush=True)
PY
"$PY" -u scripts/ablation/measure_kernel_times.py \
    --graphs cora citeseer pubmed city-roads-M artnet-exp tolokers-2 city-roads-L ogbn-arxiv \
    --convs gt gat_v2 min_aggr --passes forward --head-dims 128 256 \
    --heads 1 --quantile 0.99 --warmup 10 --iters 30 \
    "$@" --device 0 --out "$OUT/measurements.jsonl" 2>&1 | tee "$OUT/measurements.log"
"$PY" scripts/ablation/calibrate_cost_model.py \
    --from-json "$OUT/measurements.jsonl" --out "$OUT/cost_models.json" 2>&1 | tee "$OUT/calibration.log"
printf '\nCost models: %s/cost_models.json\n' "$OUT"
