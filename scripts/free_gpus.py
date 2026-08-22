"""Print the indices of GPUs with no other process on them.

Benchmark numbers taken on a shared GPU are worthless -- another tenant's kernels interleave
with ours and inflate the timings, and the contention is not constant, so the noise is not
even symmetric. This queries `nvidia-smi` for *compute processes*, not just memory: a process
that has a context open but is momentarily idle will still preempt us mid-measurement.

    CUDA_VISIBLE_DEVICES=$(python scripts/free_gpus.py --count 4) python scripts/....
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def busy_uuids() -> set[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def free_indices(max_mib: int) -> list[int]:
    busy = busy_uuids()
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    free = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, uuid, used = (f.strip() for f in line.split(","))
        if uuid not in busy and int(used) <= max_mib:
            free.append(int(idx))
    return free


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=0, help="require at least this many; exit 1 if short")
    p.add_argument("--max-mib", type=int, default=64, help="treat more used memory than this as busy")
    args = p.parse_args()

    free = free_indices(args.max_mib)
    if args.count and len(free) < args.count:
        print(f"only {len(free)} free GPU(s): {free}", file=sys.stderr)
        return 1
    print(",".join(str(i) for i in (free[: args.count] if args.count else free)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
