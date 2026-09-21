"""Static occupancy audit of every compiled kernel instantiation.

Answers "which kernels cannot fill the SM, and what is stopping them" without running
anything: `cuobjdump -res-usage` reports registers and static shared memory per instantiation,
and the sm_80 occupancy rules turn that into resident warps per SM.

This is deliberately static. `ncu` is unavailable on this machine (ERR_NVGPUCTRPERM) and a
timing run needs an idle GPU, but register pressure is fixed at compile time, so the question
"is this kernel register-limited, and by how much" can be answered from the binary alone.

Dynamic shared memory is *not* visible here -- these kernels size it at the launch site -- so a
kernel reported as register-limited may in practice be shared-memory-limited. Treat the
register ceiling as an upper bound on occupancy, not a promise.

    python scripts/occupancy_audit.py                       # summary per kernel
    python scripts/occupancy_audit.py --kernel GATv2Forward # every instantiation of one
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from collections import Counter as collections_Counter
from collections import defaultdict
from pathlib import Path

# NVIDIA A100 (sm_80) per-SM limits.
REGS_PER_SM = 65536
MAX_WARPS_PER_SM = 64
MAX_BLOCKS_PER_SM = 32
REG_ALLOC_GRANULARITY = 256  # registers, allocated per warp
SMEM_PER_SM = 164 * 1024

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"


def warps_by_registers(regs_per_thread: int) -> int:
    """Resident warps per SM allowed by the register file alone."""
    if regs_per_thread <= 0:
        return MAX_WARPS_PER_SM
    per_warp = -(-(regs_per_thread * 32) // REG_ALLOC_GRANULARITY) * REG_ALLOC_GRANULARITY
    return min(MAX_WARPS_PER_SM, REGS_PER_SM // per_warp)


def occupancy(regs: int, threads_per_block: int, smem_per_block: int) -> tuple[float, str]:
    """Achievable warps/SM as a fraction of 64, and which resource binds."""
    wpb = max(1, -(-threads_per_block // 32))
    by_reg = warps_by_registers(regs) // wpb
    by_warp = MAX_WARPS_PER_SM // wpb
    by_block = MAX_BLOCKS_PER_SM
    by_smem = (SMEM_PER_SM // smem_per_block) if smem_per_block > 0 else 10**6
    blocks = max(0, min(by_reg, by_warp, by_block, by_smem))
    limits = {"registers": by_reg, "warps/blocks": min(by_warp, by_block), "shared mem": by_smem}
    binding = min(limits, key=lambda k: limits[k])
    return blocks * wpb / MAX_WARPS_PER_SM, binding


def parse(so_path: Path) -> list[dict]:
    out = subprocess.run([CUOBJDUMP, "-res-usage", str(so_path)], capture_output=True, text=True).stdout
    entries, name = [], None
    for line in out.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)", line)
        if m and name:
            entries.append(
                {
                    "mangled": name,
                    "regs": int(m.group(1)),
                    "stack": int(m.group(2)),
                    "static_smem": int(m.group(3)),
                }
            )
            name = None
    return entries


def demangle(mangled: str) -> str:
    """Base kernel name from an Itanium-mangled symbol.

    The length prefix says exactly how many characters the name is; matching greedily instead
    swallows the template mangling that follows and makes every instantiation look like its own
    kernel.
    """
    m = re.match(r"_Z(\d+)", mangled)
    if not m:
        return mangled
    n = int(m.group(1))
    start = m.end()
    return mangled[start : start + n]


def block_threads(mangled: str) -> int | None:
    """Threads per block, read off the template arguments where the kernel encodes them.

    These kernels take their warp count as a template parameter, so the block size is fixed per
    instantiation and can be recovered rather than guessed. Returns None when the pattern does
    not apply, in which case the caller falls back to evaluating a range.
    """
    # The scheduler kind is the first argument; the warp/row count is the next integer literal.
    m = re.search(r"ScheduleKindE\d+E(?:Lm|Li)(\d+)E", mangled)
    if not m:
        return None
    warps = int(m.group(1))
    return warps * 32 if 1 <= warps <= 32 else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--so", default=None, help="path to _C*.so (default: the one in turbo_gnn/)")
    p.add_argument("--kernel", default=None, help="show every instantiation whose name contains this")
    p.add_argument(
        "--threads", type=int, nargs="+", default=[32, 128, 256, 512], help="block sizes to evaluate each kernel at"
    )
    args = p.parse_args()

    so = Path(args.so) if args.so else next(Path("turbo_gnn").glob("_C*.so"))
    entries = parse(so)
    if not entries:
        raise SystemExit(f"no kernels found in {so} (is {CUOBJDUMP} present?)")

    by_kernel = defaultdict(list)
    for e in entries:
        by_kernel[demangle(e["mangled"])].append(e)

    if args.kernel:
        sel = {k: v for k, v in by_kernel.items() if args.kernel in k}
        for k, es in sorted(sel.items()):
            print(f"\n{k}  ({len(es)} instantiations)")
            print(f"  {'REG':>5}{'STACK':>7}{'sSMEM':>7}" + "".join(f"{'occ@' + str(t):>10}" for t in args.threads))
            for e in sorted(es, key=lambda e: -e["regs"])[:20]:
                occ_cells = "".join(
                    f"{occupancy(e['regs'], t, e['static_smem'])[0] * 100:>9.0f}%" for t in args.threads
                )
                print(f"  {e['regs']:>5}{e['stack']:>7}{e['static_smem']:>7}{occ_cells}")
        return 0

    # Where the block size is encoded in the template arguments, evaluate at that size rather
    # than at a guessed range -- the kernel can only ever run at the size it was compiled for.
    known = [(e, block_threads(e["mangled"])) for e in entries]
    resolved = [(e, t) for e, t in known if t]
    if resolved:
        per_entry = [occupancy(e["regs"], t, e["static_smem"]) for e, t in resolved]
        binds = collections_Counter(b for _, b in per_entry)
        vals = [o for o, _ in per_entry]
        print(f"At each instantiation's own compiled block size ({len(resolved):,} of {len(entries):,} resolvable):")
        print(
            f"  median occupancy {statistics.median(vals) * 100:.0f}%   "
            f"mean {statistics.mean(vals) * 100:.0f}%   "
            f"below 50%: {sum(v < 0.5 for v in vals):,}   at 100%: {sum(v >= 0.999 for v in vals):,}"
        )
        print(f"  binding resource: {dict(binds)}\n")

    print(f"{so}   {len(entries):,} instantiations across {len(by_kernel)} kernels")
    print("Register-limited occupancy on an A100. 'spill' counts instantiations with a stack frame.\n")
    print(
        f"  {'kernel':<46}{'n':>6}{'REG med':>9}{'REG max':>8}{'spill':>7}"
        + "".join(f"{'occ@' + str(t):>9}" for t in args.threads)
    )
    rows: list[tuple[str, int, float, int, int, list[float]]] = []
    for k, es in by_kernel.items():
        regs = [e["regs"] for e in es]
        med, mx = statistics.median(regs), max(regs)
        spill = sum(1 for e in es if e["stack"] > 0)
        occs = [statistics.median([occupancy(e["regs"], t, e["static_smem"])[0] for e in es]) for t in args.threads]
        rows.append((k, len(es), med, mx, spill, occs))
    for k, n, med, mx, spill, occs in sorted(rows, key=lambda r: r[5][1]):
        flag = " *" if occs[1] < 0.5 else ""
        print(f"  {k:<46}{n:>6}{med:>9.0f}{mx:>8}{spill:>7}" + "".join(f"{o * 100:>8.0f}%" for o in occs) + flag)
    print("\n  * median occupancy below 50% at a 128-thread block -- the first place to look.")
    print("  Occupancy here is the register ceiling only; dynamic shared memory can lower it further.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
