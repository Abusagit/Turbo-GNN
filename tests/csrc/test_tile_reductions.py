"""VecOpsFloatBase reductions: sum, weighted_sum, prod, min, max, dot_product.

Each reduction gets two independent checks, because "correct" means two things here:

* ``test_reduction_matches_header_order`` replays the header's own reduction tree and
  intermediate dtype step by step, asserted bit-exact. This pins the implementation.
* ``test_reduction_matches_fp64`` compares against the mathematically correct value with
  a tolerance derived from the worst-case accumulation error. This judges the design --
  a reduction can pass the first check and still be losing precision it need not lose.

The accumulator-pointer forms are seeded with a non-zero value so the accumulate
semantics (``*acc += ...`` and friends) are actually exercised rather than assumed.
"""

from __future__ import annotations

import pytest
import torch
from conftest import all_combos
from reference import (
    TORCH_DTYPE,
    Paths,
    assert_exact,
    assert_sum_fp64,
    assert_ulp,
    make_input,
    ref_reduce,
    ref_reduce_fp64,
    a_tols, r_tols,
)

pytestmark = [pytest.mark.cuda, pytest.mark.csrc]

COMBOS = all_combos()
COMBO_IDS = [f"{dt}-N{n}" for n, dt in COMBOS]

M = 256
ACC_INIT = 2.5
WEIGHT = 0.75

ALL_REDUCTIONS = [
    "sum_acc",
    "sum_ret",
    "weighted_sum_acc",
    "weighted_sum_ret",
    "prod_acc",
    "prod_ret",
    "min_acc",
    "min_ret",
    "max_acc",
    "max_ret",
    "dot_product_acc",
    "dot_product_ret",
]

ACC_DTYPES = [("float", False), ("double", True)]


def _run(mod, op, n, a, b, *, acc_init=ACC_INIT, w=WEIGHT, use_double=False):
    return mod.reduce(mod.red_codes()[op], n, a, b, acc_init, w, use_double)


@pytest.mark.parametrize("kind", ["random", "special"])
@pytest.mark.parametrize(("acc_name", "use_double"), ACC_DTYPES, ids=["acc=f32", "acc=f64"])
@pytest.mark.parametrize("op", ALL_REDUCTIONS)
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_reduction_matches_header_order(bridge, op, n, dtype_name, acc_name, use_double, kind):
    """Bit-exact against an order-faithful replay of the header's reduction tree."""
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    paths = Paths(n=n, name=dtype_name)

    cap = float(torch.finfo(TORCH_DTYPE[dtype_name]).max) ** 0.5 / 4.0
    a = make_input(M, n, dtype_name, device, seed=2000 + n, kind=kind, max_abs=cap)
    b = make_input(M, n, dtype_name, device, seed=2100 + n, kind=kind, max_abs=cap)

    got = _run(mod, op, n, a, b, use_double=use_double)
    ref = ref_reduce(op, a.double(), b.double(), ACC_INIT, WEIGHT, paths, acc_name)
    label = f"{op} {dtype_name} N={n} {acc_name} {kind}"

    if op.startswith("dot_product") or op == "weighted_sum_acc":
        # dot_product keeps its product in a register, so the compiler is free to
        # contract multiply-then-accumulate into an FMA -- one rounding instead of two.
        # That makes it slightly *more* accurate than the replay, not less. The residual
        # has to be bounded against the sum of |products|, not against the result: the
        # products can cancel, leaving a near-zero result whose ulp is far smaller than
        # the roundings that produced it.
        # weighted_sum_acc lands here too: its `*acc += sum * w` is a multiply feeding an
        # add, which the compiler contracts into a single-rounding FMA.
        seed = ACC_INIT if op.endswith("_acc") else 0.0
        terms = a.double() * b.double() if op.startswith("dot_product") else a.double() * WEIGHT
        assert_sum_fp64(got, ref, terms, dtype_name, label, seed=seed)
    else:
        assert_exact(got, ref, acc_name, label, ftz=paths.ftz, ftz_name=dtype_name)


@pytest.mark.parametrize("op", ALL_REDUCTIONS)
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_reduction_matches_fp64(bridge, op, n, dtype_name):
    """Close to the mathematically correct value, within the accumulation error budget.

    The budget is N ulp of the *accumulator* type: each of the N steps can contribute
    up to one rounding. Where the header accumulates in num_type rather than accum_t
    the budget has to be taken in num_type instead, which is exactly the precision
    question ``test_sum_accumulates_in_accum_t`` isolates.
    """
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    paths = Paths(n=n, name=dtype_name)

    # Well-conditioned inputs only: fp64 agreement is not a meaningful question for
    # deliberately overflowing or subnormal operands.
    a = make_input(M, n, dtype_name, device, seed=3000 + n, kind="random")
    b = make_input(M, n, dtype_name, device, seed=3100 + n, kind="random")

    got = _run(mod, op, n, a, b)
    ref = ref_reduce_fp64(op, a.double(), b.double(), ACC_INIT, WEIGHT)
    label = f"{op} {dtype_name} N={n} vs fp64"

    if op.startswith(("sum", "weighted_sum", "dot_product")):
        # The header accumulates these in num_type (tile.cuh:807), so the error is
        # bounded in the *narrow* type and scales with the terms, not the result.
        terms = (a.double() * b.double()) if op.startswith("dot_product") else a.double()
        seed = ACC_INIT if op.endswith("_acc") else 0.0
        scaled = terms * WEIGHT if op.startswith("weighted_sum") else terms
        assert_sum_fp64(got, ref, scaled, dtype_name, label, seed=seed)
    else:
        assert_ulp(got, ref, "float", label, max_ulp=float(n) + 1.0)


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_prod_uses_multiplicative_identity(bridge, n, dtype_name):
    """prod of an all-ones vector must be 1, not 0.

    Guards the accumulator seed: an additive-identity seed silently turns every product
    into zero, which is invisible in any test whose expected value happens to be 0.
    """
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    ones = torch.ones((8, n), dtype=TORCH_DTYPE[dtype_name], device=device)

    got = _run(mod, "prod_ret", n, ones, ones, acc_init=0.0)
    torch.testing.assert_close(got.double(), torch.ones(8, dtype=torch.float64, device=device), rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize("want_max", [False, True], ids=["min", "max"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_minmax_value_form_is_not_clamped_against_zero(bridge, n, dtype_name, want_max):
    """min/max of a same-signed vector must be an element of that vector."""
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    dt = TORCH_DTYPE[dtype_name]

    # All strictly positive for min, all strictly negative for max: in both cases the
    # true answer has the same sign as the input, so a zero seed cannot produce it.
    sign = -1.0 if want_max else 1.0
    a = sign * (torch.arange(1, 8 * n + 1, dtype=torch.float64).reshape(8, n) / 4.0).to(dt).to(device)

    got = _run(mod, "max_ret" if want_max else "min_ret", n, a, a, acc_init=0.0)
    want = a.double().max(dim=1).values if want_max else a.double().min(dim=1).values
    torch.testing.assert_close(got.double(), want, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize("dtype_name", ["half", "bf16"])
@pytest.mark.xfail(
    strict=True,
    reason="sum declares `num_type buf_acc{}` (tile.cuh:807), so it accumulates in the "
    "narrow type even when accum_t is float. With N=8 half terms of 1024 the running "
    "total leaves half's exactly-representable range and the result is wrong by more "
    "than rounding. Remove once the accumulator is accum_t.",
)
def test_sum_accumulates_in_accum_t(bridge, dtype_name):
    """A wide accum_t should protect the running total from the narrow input type."""
    n = 8
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    dt = TORCH_DTYPE[dtype_name]

    # 8 x 1024: representable individually, and the exact total (8192) is representable
    # too, but half's 11-bit significand cannot hold the odd partial sums along the way
    # once a +1 term is mixed in.
    a = torch.full((4, n), 1024.0, dtype=torch.float64)
    a[:, 0] = 1.0
    a_t = a.to(dt).to(device)

    got = _run(mod, "sum_ret", n, a_t, a_t, acc_init=0.0)
    want = a_t.double().sum(dim=1)  # exact in fp64
    torch.testing.assert_close(got.double(), want, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_acc_forms_accumulate_rather_than_overwrite(bridge, n, dtype_name):
    """The acc-pointer forms must combine with the incoming value, not replace it.

    Running the same vector twice into a live accumulator has to move it twice as far
    from the seed as running it once -- the property the GATv2/GT kernels rely on when
    they accumulate a row across tiles.
    """
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    a = make_input(M, n, dtype_name, device, seed=4242 + n, kind="random", positive=True)

    once = _run(mod, "sum_acc", n, a, a, acc_init=0.0).double()
    from_seed = _run(mod, "sum_acc", n, a, a, acc_init=ACC_INIT).double()

    torch.testing.assert_close(from_seed, once + ACC_INIT, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_dot_product_equals_sum_of_products(bridge, n, dtype_name):
    """dot_product must agree with mul_ followed by sum, since that is how it is built."""
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")

    a = make_input(M, n, dtype_name, device, seed=555 + n, kind="random")
    b = make_input(M, n, dtype_name, device, seed=666 + n, kind="random")

    dot = _run(mod, "dot_product_ret", n, a, b, acc_init=0.0)
    prod = mod.elementwise(mod.op_codes()["mul_"], n, a, b, b, 0.0)
    via_sum = _run(mod, "sum_ret", n, prod, prod, acc_init=0.0)

    # Not bit-equal: mul_ rounds each product through memory, whereas dot_product keeps
    # it in a register where multiply-then-accumulate can contract into a single-rounding
    # FMA. They must still agree to within a rounding per term, measured against the
    # magnitude of the products rather than of the (possibly cancelling) total.
    assert_sum_fp64(dot, via_sum.double(), a.double() * b.double(), dtype_name, f"dot vs sum(mul) N={n} {dtype_name}")


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_weighted_sum_equals_sum_times_weight(bridge, n, dtype_name):
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    a = make_input(M, n, dtype_name, device, seed=777 + n, kind="random")

    weighted = _run(mod, "weighted_sum_ret", n, a, a, acc_init=0.0).double()
    plain = _run(mod, "sum_ret", n, a, a, acc_init=0.0).double()

    torch.testing.assert_close(weighted, plain * WEIGHT, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])
