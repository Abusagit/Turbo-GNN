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


def busy_uuids(ignore_pids: set[int] | None = None) -> set[str]:
    """UUIDs of GPUs running a compute process, excluding processes we started ourselves.

    `ignore_pids` matters when polling *during* our own benchmark: the child is itself a compute
    process on that GPU, so without excluding it every poll reports contention against us.
    """
    ignore = ignore_pids or set()
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    busy = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, uuid = (f.strip() for f in line.split(","))
        if int(pid) not in ignore:
            busy.add(uuid)
    return busy


def free_indices(max_mib: int, ignore_pids: set[int] | None = None) -> list[int]:
    busy = busy_uuids(ignore_pids)
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
        # Memory held by a process we are ignoring should not count against us either.
        if uuid not in busy and (int(used) <= max_mib or ignore_pids):
            free.append(int(idx))
    return free


def is_free(index: int, max_mib: int = 64, ignore_pids: set[int] | None = None) -> bool:
    """Is this specific GPU free of other tenants right now?

    Checking once before a long run is not enough: a multi-hour benchmark can be invaded
    halfway through, and the resulting numbers look like real regressions rather than noise.
    Call this between runs, not just at launch.
    """
    return index in free_indices(max_mib, ignore_pids)


def wait_until_free(index: int, timeout_s: float = 1800.0, poll_s: float = 30.0, max_mib: int = 64) -> bool:
    """Block until `index` has no other compute process, or the timeout expires."""
    import time as _time

    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        if is_free(index, max_mib=max_mib):
            return True
        _time.sleep(poll_s)
    return is_free(index, max_mib=max_mib)


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
