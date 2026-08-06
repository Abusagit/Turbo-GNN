"""Vec layout, VecOpsBase data movement, and SelectTW.

These are the parts of tile.cuh with no arithmetic in them, so everything here is
asserted bit-exact.
"""

from __future__ import annotations

import pytest
import torch
from conftest import DTYPES, VALID_NS, all_combos
from reference import TORCH_DTYPE, ref_select_tw

pytestmark = [pytest.mark.cuda, pytest.mark.csrc]

COMBOS = all_combos()
COMBO_IDS = [f"{dt}-N{n}" for n, dt in COMBOS]

# D values the GATv2/GT kernels actually dispatch, plus 96 as the known trap.
DISPATCHED_D = (32, 64, 128, 256)
TRAP_D = 96


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_vec_layout(bridge, n, dtype_name):
    """Vec<N,T> must be exactly N*sizeof(T) bytes, aligned to its own width."""
    mod = bridge.get("data", dtype_name)
    itemsize = torch.finfo(TORCH_DTYPE[dtype_name]).bits // 8
    entry = mod.vec_layout()[n]

    assert entry["size"] == n * itemsize, "Vec must be exactly its element span"
    assert entry["align"] == n * itemsize, "alignas(sizeof(num_type) * N) must hold"
    assert entry["wide_size"] == n * itemsize, "wide_t must cover the whole vector"


@pytest.mark.parametrize("dtype_name", list(DTYPES), ids=list(DTYPES))
def test_vec_rejects_oversized(bridge, dtype_name):
    """Vec caps at 128 bits, so max N is 16/sizeof(T) and nothing wider exists."""
    mod = bridge.get("data", dtype_name)
    itemsize = torch.finfo(TORCH_DTYPE[dtype_name]).bits // 8
    assert mod.max_n == 16 // itemsize
    assert max(VALID_NS[dtype_name]) == mod.max_n


@pytest.mark.parametrize("dtype_name", list(DTYPES), ids=list(DTYPES))
def test_select_tw_matches_intent(bridge, dtype_name):
    """SelectTW = min(ceil(D/32), 16/sizeof(T)), on a whole warp per row."""
    mod = bridge.get("data", dtype_name)
    table = mod.select_tw()

    for d in (*DISPATCHED_D, TRAP_D):
        got = table[d]["value"]
        assert got == ref_select_tw(d, dtype_name), f"SelectTW<{d}, {dtype_name}>"
        assert table[d]["threads_per_d"] == 32


@pytest.mark.parametrize("dtype_name", list(DTYPES), ids=list(DTYPES))
def test_select_tw_is_usable_as_vec_width(bridge, dtype_name):
    """Every dispatched D must yield a TW that Vec actually accepts.

    Vec requires a power-of-two N, so a TW like 3 would be unusable. D=96 is
    checked separately because it does produce 3 -- a trap for any future caller.
    """
    mod = bridge.get("data", dtype_name)
    table = mod.select_tw()

    for d in DISPATCHED_D:
        tw = table[d]["value"]
        assert tw.bit_count() == 1, f"SelectTW<{d}, {dtype_name}> = {tw} is not a power of two"
        assert tw in VALID_NS[dtype_name], f"TW={tw} is outside the tested N set"

    trap = table[TRAP_D]["value"]
    assert trap.bit_count() != 1, (
        f"SelectTW<{TRAP_D}, {dtype_name}> = {trap} is now a power of two; if D=96 became a "
        "supported feature width, add it to DISPATCHED_D"
    )


def _run(mod, op_name, n, dst, src):
    return mod.data_move(mod.op_codes()[op_name], n, dst, src)


@pytest.mark.parametrize("on_host", [False, True], ids=["device", "host"])
@pytest.mark.parametrize("op_name", ["transfer_vector", "transfer_scalars", "load__scalars", "store_scalars"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_copy_ops_are_faithful(bridge, op_name, n, dtype_name, on_host):
    """All four copy flavours move N scalars verbatim.

    Note the header's doc comments for load__scalars/store_scalars are swapped
    relative to their signatures; the signatures are what is tested here.
    """
    mod = bridge.get("data", dtype_name, on_host)
    dev = torch.device("cpu" if on_host else "cuda")
    dt = TORCH_DTYPE[dtype_name]
    m = 64

    src = torch.arange(1, m * n + 1, dtype=torch.float64).reshape(m, n).to(dt).to(dev)
    dst = torch.zeros((m, n), dtype=dt, device=dev)

    _run(mod, op_name, n, dst, src)
    assert torch.equal(dst, src), f"{op_name} did not copy the payload verbatim"


@pytest.mark.parametrize("on_host", [False, True], ids=["device", "host"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_get_zero(bridge, n, dtype_name, on_host):
    mod = bridge.get("data", dtype_name, on_host)
    dev = torch.device("cpu" if on_host else "cuda")
    dt = TORCH_DTYPE[dtype_name]
    m = 32

    src = torch.ones((m, n), dtype=dt, device=dev)
    dst = torch.full((m, n), 7.0, dtype=dt, device=dev)

    _run(mod, "get_zero", n, dst, src)
    assert not dst.any(), "get_zero must produce an all-zero vector"


@pytest.mark.parametrize("on_host", [False, True], ids=["device", "host"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_store_zero_writes_exactly_n_elements(bridge, n, dtype_name, on_host):
    """store_zero must zero N*sizeof(T) bytes and not a byte more.

    The extra trailing row is a canary: it is inside the allocation but outside the
    slice handed to the kernel, so an over-wide store shows up as a clobbered
    sentinel rather than as silent corruption.
    """
    mod = bridge.get("data", dtype_name, on_host)
    dev = torch.device("cpu" if on_host else "cuda")
    dt = TORCH_DTYPE[dtype_name]
    m = 32
    sentinel = 1.0

    buf = torch.full((m + 1, n), sentinel, dtype=dt, device=dev)
    target = buf[:m]
    assert target.is_contiguous()
    src = torch.ones((m, n), dtype=dt, device=dev)

    _run(mod, "store_zero", n, target, src)

    assert not buf[:m].any(), "store_zero left non-zero elements behind"
    assert bool((buf[m] == sentinel).all()), "store_zero overran its vector into the canary row"
