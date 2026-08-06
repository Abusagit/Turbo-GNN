"""convert_vec across every dtype pair, and write_row's cooperative row copy."""

from __future__ import annotations

import pathlib

import pytest
import torch
from conftest import DTYPES, VALID_NS, all_combos
from reference import TORCH_DTYPE, make_input

pytestmark = [pytest.mark.cuda, pytest.mark.csrc]

COMBOS = all_combos()
COMBO_IDS = [f"{dt}-N{n}" for n, dt in COMBOS]

DST_CODE = {"float": 0, "half": 1, "bf16": 2}
M = 128


def _itemsize(name: str) -> int:
    return torch.finfo(TORCH_DTYPE[name]).bits // 8


def _both_fit(n: int, src: str, dst: str) -> bool:
    return n * _itemsize(src) <= 16 and n * _itemsize(dst) <= 16


def _packed_path(n: int, src: str, dst: str) -> bool:
    """convert_vec's packed branch: both sides float/half/bf16 and N even."""
    return n % 2 == 0


# packed_convert's third branch tests dst_type where it means src_type, then re-tests
# dst_type inside, so half -> float and half -> bf16 fall through to
# __builtin_unreachable() -- which nvcc lowers to a trap, aborting the process. Only
# reachable on the packed path; N=1 goes through static_cast and is fine. These pairs are
# kept out of the main matrix and proven separately, in a subprocess, by
# test_convert_vec_half_source_packed_traps.
def _packed_convert_is_broken(n: int, src: str, dst: str) -> bool:
    return src == "half" and dst in ("float", "bf16") and _packed_path(n, src, dst)


CONVERT_CASES = [(n, src, dst) for n, src in COMBOS for dst in DTYPES if _both_fit(n, src, dst)]
CONVERT_IDS = [f"{src}->{dst}-N{n}" for n, src, dst in CONVERT_CASES]


@pytest.mark.parametrize(("n", "src_name", "dst_name"), CONVERT_CASES, ids=CONVERT_IDS)
def test_convert_vec(bridge, n, src_name, dst_name):
    """Elementwise round-to-nearest conversion, bit-exact.

    Both the packed intrinsics (__float22half2_rn and friends) and the scalar
    static_cast fallback round to nearest-even, so torch's own cast is an exact oracle.
    """

    mod = bridge.get("cvt", src_name)
    device = torch.device("cuda:0")

    src = make_input(M, n, src_name, device, seed=9000 + n, kind="special")
    got = mod.convert(n, src, DST_CODE[dst_name])
    want = src.to(TORCH_DTYPE[dst_name])

    assert got.dtype == want.dtype
    torch.testing.assert_close(got.double(), want.double(), rtol=0, atol=0, equal_nan=True)


def test_transfer_vector_wide_pun_is_not_miscompiled(bridge):
    """A 16-byte Vec must survive a convert_vec round trip.

    Minimal reproducer: bf16 -> half at N=8, where both Vec types are exactly 16 bytes so
    wide_t is `unsigned __int128`. N=2 and N=4 (uint32/uint64 wide_t) come out correct, and
    the device pass is correct at every N, which is what points at alias analysis rather
    than at the conversion maths.
    """

    mod = bridge.get("cvt", "bf16")
    src = torch.tensor([[1.0, -2.0, 0.5, 3.0, -0.25, 8.0, -16.0, 0.125]], dtype=torch.bfloat16, device="cuda:0")
    got = mod.convert(8, src, DST_CODE["half"])
    torch.testing.assert_close(got.double(), src.to(torch.float16).double(), rtol=0, atol=0)


@pytest.mark.parametrize(("n", "src_name"), COMBOS, ids=COMBO_IDS)
def test_convert_vec_same_type_is_a_copy(bridge, n, src_name):
    """The same-type early-out must move the payload verbatim."""

    mod = bridge.get("cvt", src_name)
    device = torch.device("cuda:0")

    src = make_input(M, n, src_name, device, seed=42, kind="special")
    got = mod.convert(n, src, DST_CODE[src_name])
    assert torch.equal(got, src)


@pytest.mark.parametrize("dtype_name", list(DTYPES), ids=list(DTYPES))
def test_convert_vec_rejects_oversized_destination(bridge, dtype_name):
    """A 16-bit -> fp32 conversion at N=8 cannot exist, and must say so.

    Vec caps at 128 bits, so Vec<8, float> does not exist. This is the same limit that
    stops atomic_add_scaled_f32 from working at full tile width; see
    test_tile_gatv2.py::test_atomic_add_scaled_f32_supports_full_tile_width.
    """
    if 8 not in VALID_NS[dtype_name] or _itemsize(dtype_name) == 4:
        pytest.skip("only the 16-bit types reach N=8")

    mod = bridge.get("cvt", dtype_name)
    src = make_input(4, 8, dtype_name, torch.device("cuda"), seed=1, kind="random")
    with pytest.raises(RuntimeError, match="16-byte cap"):
        mod.convert(8, src, DST_CODE["float"])


# ---------------------------------------------------------------------------
# write_row
# ---------------------------------------------------------------------------

# (row_width, worker_cnt) pairs the bridge instantiates.
WRITE_ROW_SHAPES = [(32, 32), (64, 32), (128, 32), (256, 32), (128, 64), (256, 128)]


def _copy_n(src_name: str, dst_name: str) -> int:
    """write_row's copy_N = 16 / max(sizeof(src), sizeof(dst))."""
    return 16 // max(_itemsize(src_name), _itemsize(dst_name))


@pytest.mark.parametrize(("row_width", "worker_cnt"), WRITE_ROW_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("dst_name", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("src_name", list(DTYPES), ids=list(DTYPES))
def test_write_row_copies_the_whole_row(bridge, src_name, dst_name, row_width, worker_cnt):
    """Every element of the row is copied and converted exactly once.

    The trailing slack in dst is a canary: write_row's tail guard is
    ``tile_id * copy_N < row_width``, so anything past row_width would show up here.
    """

    mod = bridge.get("cvt", src_name)
    device = torch.device("cuda")
    slack = 64

    src = (torch.arange(1, row_width + 1, dtype=torch.float64) / 8.0).to(TORCH_DTYPE[src_name]).to(device)
    out = mod.write_row_run(row_width, worker_cnt, src, DST_CODE[dst_name], row_width + slack)

    want = src.to(TORCH_DTYPE[dst_name])
    torch.testing.assert_close(out[:row_width].double(), want.double(), rtol=0, atol=0)
    assert not out[row_width:].any(), "write_row wrote past row_width into the canary region"


@pytest.mark.parametrize("dst_name", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("src_name", list(DTYPES), ids=list(DTYPES))
def test_write_row_does_not_overrun_a_ragged_row(bridge, src_name, dst_name):
    """A row_width that is not a multiple of copy_N must not be rounded up.

    The guard is ``tile_id * copy_N < row_width``, so the final tile is admitted as soon
    as it *starts* inside the row and then writes a full copy_N elements -- overrunning
    by up to copy_N-1. row_width=36 triggers this whenever copy_N is 8.
    """
    copy_n = _copy_n(src_name, dst_name)
    row_width = 36
    if row_width % copy_n == 0:
        pytest.skip(f"row_width={row_width} is a multiple of copy_N={copy_n}; nothing ragged to probe")

    mod = bridge.get("cvt", src_name)
    device = torch.device("cuda")
    slack = 64

    src = (torch.arange(1, row_width + 1, dtype=torch.float64) / 8.0).to(TORCH_DTYPE[src_name]).to(device)
    # The source must be readable for the whole final tile, or the kernel reads OOB on
    # its own account; pad it so the only thing under test is the *write* extent.
    padded = torch.zeros(row_width + copy_n, dtype=TORCH_DTYPE[src_name], device=device)
    padded[:row_width] = src

    out = mod.write_row_run(row_width, 32, padded, DST_CODE[dst_name], row_width + slack)

    want = src.to(TORCH_DTYPE[dst_name])
    torch.testing.assert_close(out[:row_width].double(), want.double(), rtol=0, atol=0)
    overrun = out[row_width:].nonzero().flatten().tolist()
    assert not overrun, (
        f"write_row overran a ragged row: copy_N={copy_n}, row_width={row_width}, "
        f"wrote {len(overrun)} extra element(s) at offsets {overrun[:8]}"
    )
