"""VecOpsFloatBase elementwise ops, every (N, num_type).

Each op is checked against a float64 reference at the strictness it deserves: bit-exact
for the single-rounding ops, 1 ulp for div/fma, relative for the approximate
transcendental intrinsics. See reference.py for the tolerance table.

Both branches of the header's ``if constexpr`` ladder are covered by construction:
N=1 always takes the scalar path, and 16-bit types with N in {2,4,8} take the packed
path. float never packs on this arch (``can_be_packed_new`` needs kCudaArch >= 1000).
"""

from __future__ import annotations

import pytest
import torch
from conftest import all_combos
from reference import TORCH_DTYPE, Paths, check_elementwise, make_input, ref_elementwise

pytestmark = [pytest.mark.cuda, pytest.mark.csrc]

COMBOS = all_combos()
COMBO_IDS = [f"{dt}-N{n}" for n, dt in COMBOS]

M = 256
NS = 0.2  # the GATv2 default negative slope
MUL_S = 1.5

UNARY = ["neg_", "log_", "exp_", "relu_"]
UNARY_SCALAR = ["scalar_mul_", "leaky_relu_"]
BINARY_WITH_SCALAR = ["leaky_relu_backward_"]
BINARY = [
    f"{base}{suffix}" for base in ("add", "sub", "mul", "div", "minimum", "maximum") for suffix in ("_", "_dst", "_ret")
]
TERNARY = ["fmam_", "fmaa_", "fma_dst", "fma_ret"]

ALL_OPS = [*UNARY, *UNARY_SCALAR, *BINARY_WITH_SCALAR, *BINARY, *TERNARY]

# ops whose first operand must be strictly positive
NEEDS_POSITIVE_A = {"log_"}
# ops whose second operand must stay away from zero
NEEDS_NONZERO_B = {"div_", "div_dst", "div_ret"}


def _scalar_for(op: str) -> float:
    if op == "scalar_mul_":
        return MUL_S
    if op in ("leaky_relu_", "leaky_relu_backward_"):
        return NS
    return 0.0


def _make_operands(op, n, dtype_name, device, kind, seed):
    a = make_input(
        M, n, dtype_name, device, seed=seed, kind=kind, positive=op in NEEDS_POSITIVE_A, nonzero=op in NEEDS_POSITIVE_A
    )
    b = make_input(M, n, dtype_name, device, seed=seed + 1, kind=kind, nonzero=op in NEEDS_NONZERO_B)
    c = make_input(M, n, dtype_name, device, seed=seed + 2, kind=kind)
    return a, b, c


@pytest.mark.parametrize("kind", ["random", "special"])
@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_elementwise(bridge, op, n, dtype_name, kind):
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    paths = Paths(n=n, name=dtype_name)
    s = _scalar_for(op)

    a, b, c = _make_operands(op, n, dtype_name, device, kind, seed=1000 + n)
    got = mod.elementwise(mod.op_codes()[op], n, a, b, c, s)

    ref = ref_elementwise(op, a.double(), b.double(), c.double(), s, dtype_name, paths)
    check_elementwise(got, ref, op, dtype_name, f"{op} {dtype_name} N={n} {kind}", paths)


@pytest.mark.parametrize("flavour", ["_", "_dst", "_ret"])
@pytest.mark.parametrize("base", ["add", "sub", "mul", "div", "minimum", "maximum"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_binary_flavours_agree(bridge, base, flavour, n, dtype_name):
    """The in-place, dst-out and by-value forms must be indistinguishable."""
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    op = f"{base}{flavour}"

    a, b, c = _make_operands(op, n, dtype_name, device, "random", seed=77)
    got = mod.elementwise(mod.op_codes()[op], n, a, b, c, 0.0)
    ref_inplace = mod.elementwise(mod.op_codes()[f"{base}_"], n, a, b, c, 0.0)

    assert torch.equal(got, ref_inplace), f"{op} disagrees with {base}_"


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_leaky_relu_at_zero(bridge, n, dtype_name):
    """leaky_relu_ must map +-0 to 0 regardless of which branch is live."""
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    dt = TORCH_DTYPE[dtype_name]

    a = torch.zeros((2, n), dtype=dt, device=device)
    a[1] = -0.0
    zeros = torch.zeros_like(a)

    got = mod.elementwise(mod.op_codes()["leaky_relu_"], n, a, zeros, zeros, NS)
    assert not got.any(), f"leaky_relu_(+-0) must be 0, got {got}"


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_leaky_relu_backward_boundary(bridge, n, dtype_name):
    """The derivative at exactly 0 differs between the two branches.

    The scalar path tests ``x >= 0`` and so returns dy; the packed path uses
    ``__hgt2`` (strict ``>``) and so returns dy*ns. Both are defensible choices for a
    subgradient, but they are not the same function -- this pins whichever branch the
    (N, dtype) pair actually selects, so the inconsistency cannot drift unnoticed.
    """
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    dt = TORCH_DTYPE[dtype_name]
    paths = Paths(n=n, name=dtype_name)

    a = torch.zeros((4, n), dtype=dt, device=device)
    dy = torch.full((4, n), 3.0, dtype=dt, device=device)
    got = mod.elementwise(mod.op_codes()["leaky_relu_backward_"], n, a, dy, dy, NS)

    expected = 3.0 * NS if paths.packed else 3.0
    want = torch.full_like(got, expected)
    torch.testing.assert_close(
        got.double(),
        want.double(),
        rtol=2e-2,
        atol=1e-3,
        msg=lambda s: f"at x=0 the {'packed' if paths.packed else 'scalar'} path should give {expected}: {s}",
    )


@pytest.mark.parametrize("op", ["minimum_", "maximum_"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_minmax_nan_propagation(bridge, op, n, dtype_name):
    """
    On device the header reaches ``__hmin``/``__hmax`` (16-bit) or ``fminf``/``fmaxf``
    (float), which are IEEE-754 minNum/maxNum and return the *non*-NaN operand. On the
    host ``kCudaArch == 0``, so float falls through to ``cuda::std::min``, a plain
    ``b < a ? b : a`` that propagates whichever operand the comparison happens to
    pick. This test states the actual behaviour per path rather than pretending they
    match.
    """

    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    dt = TORCH_DTYPE[dtype_name]

    a = torch.full((1, n), float("nan"), dtype=dt, device=device)
    b = torch.full((1, n), 1.0, dtype=dt, device=device)
    got = mod.elementwise(mod.op_codes()[op], n, a, b, b, 0.0)

    assert not got.isnan().any(), f"{op} on device should return the non-NaN operand"
    torch.testing.assert_close(got.double(), torch.ones_like(got).double(), rtol=0, atol=0)
