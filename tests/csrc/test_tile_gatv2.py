"""TileOps: read, make_ns, the GATv2 helpers, and atomic_add_scaled_f32.

The GATv2 maths these implement. The vectors are named as the kernels name them
(gatv2_kernel.cu): lv = dst-side / "left" (l_i below), r = src-side / "right" (r_j),
a = attention vector, ns = negative slope.

    e_ij    = sum_k a_k * LeakyReLU_ns(l_i[k] + r_j[k])
    alpha   = softmax_j(e_ij)
    h_i     = sum_j alpha_ij * r_j

and, for the backward pass, with t = LeakyReLU'_ns(l_i + r_j):

    grad_a += grad_e * LeakyReLU_ns(l_i + r_j)     (== grad_e * t * edge)
    grad_l += grad_e * t * a
    grad_r += alpha * grad_h  +  grad_e * t * a    (r_j is both a score input and the
                                                   aggregated message, hence two terms)

The reference implementations live in reference.py and follow the header's rounding:
local vectors stay in num_type, and the conversion to accum_t happens only where the
result meets the fp32 accumulators.
"""

from __future__ import annotations

import pytest
import torch
from conftest import DTYPES, VALID_NS, all_combos
from reference import (
    TORCH_DTYPE,
    Paths,
    assert_sum_fp64,
    make_input,
    ref_gatv2_accum_grad_al,
    ref_gatv2_accum_grad_r,
    ref_gatv2_dot_leaky_relu,
    a_tols, r_tols,
)

pytestmark = [pytest.mark.cuda, pytest.mark.csrc]

COMBOS = all_combos()
COMBO_IDS = [f"{dt}-N{n}" for n, dt in COMBOS]

M = 128
NS = 0.2

# ---------------------------------------------------------------------------
# TileOps::read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0, 3], ids=["start=0", "start=3"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_read_uses_vector_indexing(bridge, n, dtype_name, start):
    """read(arr, vec_idx) must fetch arr[vec_idx * TW .. +TW).

    Note the index convention differs from atomic_add_scaled_f32, which takes an element
    offset under the same parameter name -- see test_atomic_add_uses_element_offset.
    """
    mod = bridge.get("cvt", dtype_name)
    device = torch.device("cuda:0")

    total = (start + M) * n
    arr = (torch.arange(1, total + 1, dtype=torch.float64) / 16.0).to(TORCH_DTYPE[dtype_name]).to(device)

    got = mod.tile_read(n, arr, start, M)
    want = arr[start * n : (start + M) * n].reshape(M, n)
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# gatv2_dot_leaky_relu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random", "special"])
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_gatv2_dot_leaky_relu(bridge, n, dtype_name, kind):
    """e_partial = sum_k a_k * LeakyReLU_ns(l_k + r_k), one tile's worth."""
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    paths = Paths(n=n, name=dtype_name)

    cap = float(torch.finfo(TORCH_DTYPE[dtype_name]).max) ** 0.5 / 4.0
    lv = make_input(M, n, dtype_name, device, seed=6000 + n, kind=kind, max_abs=cap)
    r = make_input(M, n, dtype_name, device, seed=6100 + n, kind=kind, max_abs=cap)
    a = make_input(M, n, dtype_name, device, seed=6200 + n, kind=kind, max_abs=cap)

    got = mod.gatv2_dot_leaky_relu(n, lv, r, a, NS)
    ref = ref_gatv2_dot_leaky_relu(lv.double(), r.double(), a.double(), NS, paths)

    # A dot product internally, so the residual is bounded by the sum of |terms| rather
    # than by the (possibly cancelling) result.
    terms = a.double() * (lv.double() + r.double())
    assert_sum_fp64(got, ref, terms, dtype_name, f"gatv2_dot {dtype_name} N={n} {kind}")


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_gatv2_dot_leaky_relu_closed_form(bridge, n, dtype_name):
    """A hand-checkable case, independent of the reference implementation.

    With a = ones, lv + r = [2, -2, 2, -2, ...], LeakyReLU_0.2 gives [2, -0.4, ...], so
    the tile's contribution is N/2 * (2 - 0.4) = 0.8 * N for even N, and 2 for N == 1.
    """
    mod = bridge.get("ops", dtype_name)
    device = torch.device("cuda:0")
    dt = TORCH_DTYPE[dtype_name]

    pattern = torch.tensor([2.0 if k % 2 == 0 else -2.0 for k in range(n)], dtype=torch.float64)
    lv = pattern.repeat(4, 1).to(dt).to(device)
    r = torch.zeros((4, n), dtype=dt, device=device)
    a = torch.ones((4, n), dtype=dt, device=device)

    got = mod.gatv2_dot_leaky_relu(n, lv, r, a, NS)
    expected = sum(2.0 if k % 2 == 0 else -2.0 * NS for k in range(n))
    want = torch.full((4,), expected, dtype=torch.float64, device=device)

    torch.testing.assert_close(got.double(), want, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


# ---------------------------------------------------------------------------
# atomic_add_scaled_f32
# ---------------------------------------------------------------------------

ATOMIC_OK = [(n, dt) for n, dt in COMBOS if n * 4 <= 16]
ATOMIC_OK_IDS = [f"{dt}-N{n}" for n, dt in ATOMIC_OK]

ATOMIC_TOO_WIDE = [(n, dt) for n, dt in COMBOS if n * 4 > 16]
ATOMIC_TOO_WIDE_IDS = [f"{dt}-N{n}" for n, dt in ATOMIC_TOO_WIDE]


@pytest.mark.parametrize("vec_idx", [0, 5], ids=["idx=0", "idx=5"])
@pytest.mark.parametrize(("n", "dtype_name"), ATOMIC_OK, ids=ATOMIC_OK_IDS)
def test_atomic_add_uses_element_offset(bridge, n, dtype_name, vec_idx):
    """ptr[vec_idx + k] += scalar * v[k], in fp32, and nothing outside that window.

    The parameter is named vec_idx but is indexed as ``ptr[vec_idx + i]``, i.e. an
    element offset -- which matches the real caller (gt_backward.cu passes
    ``base_f = fv * TW``) but not TileOps::read's vector-index convention.
    """
    mod = bridge.get("cvt", dtype_name)
    device = torch.device("cuda")
    scalar = 2.5
    rows = 64

    v = make_input(rows, n, dtype_name, device, seed=7000 + n, kind="random")
    ptr = torch.zeros(vec_idx + n + 8, dtype=torch.float32, device=device)
    mod.atomic_add_scaled_f32(n, ptr, vec_idx, scalar, v)

    # Every row targets the same offset, so the slot accumulates the whole column sum.
    want = v.double().sum(dim=0) * scalar
    torch.testing.assert_close(ptr[vec_idx : vec_idx + n].double(), want, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])
    assert not ptr[:vec_idx].any(), "wrote before vec_idx"
    assert not ptr[vec_idx + n :].any(), "wrote past vec_idx + N"


@pytest.mark.parametrize(("n", "dtype_name"), ATOMIC_TOO_WIDE, ids=ATOMIC_TOO_WIDE_IDS)
def test_atomic_add_scaled_f32_supports_full_tile_width(bridge, n, dtype_name):
    """The widest tile a 16-bit type can have must still be able to accumulate in fp32."""
    mod = bridge.get("cvt", dtype_name)
    device = torch.device("cuda")

    v = make_input(8, n, dtype_name, device, seed=1, kind="random")
    ptr = torch.zeros(n, dtype=torch.float32, device=device)
    mod.atomic_add_scaled_f32(n, ptr, 0, 1.0, v)
    torch.testing.assert_close(ptr.double(), v.double().sum(dim=0), rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize("dtype_name", list(DTYPES), ids=list(DTYPES))
def test_atomic_add_is_contended(bridge, dtype_name):
    """Many threads hitting one slot must all land -- that is the point of the atomic."""
    n = min(4, max(VALID_NS[dtype_name]))
    mod = bridge.get("cvt", dtype_name)
    device = torch.device("cuda")
    rows = 4096

    v = torch.ones((rows, n), dtype=TORCH_DTYPE[dtype_name], device=device)
    ptr = torch.zeros(n, dtype=torch.float32, device=device)
    mod.atomic_add_scaled_f32(n, ptr, 0, 1.0, v)

    torch.testing.assert_close(
        ptr.double(), torch.full((n,), float(rows), dtype=torch.float64, device=device), rtol=r_tols[dtype_name], atol=a_tols[dtype_name]
    )


# ---------------------------------------------------------------------------
# make_ns and the gradient accumulators (device-only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ns", [0.2, 0.01, 1.0], ids=lambda v: f"ns={v}")
@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_make_ns_broadcasts_the_slope(bridge, n, dtype_name, ns):
    """make_ns must produce the negative slope in ns_t, readable as a num_type scalar."""
    mod = bridge.get("grad", dtype_name)
    got = mod.make_ns(n, ns)
    want = torch.tensor([ns], dtype=TORCH_DTYPE[dtype_name]).double().item()
    assert got == pytest.approx(want, rel=2e-2, abs=1e-3)


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_gatv2_accum_grad_al(bridge, n, dtype_name):
    """grad_a += grad_e * LeakyReLU(lv+r) and grad_l += grad_e * LeakyReLU'(lv+r) * a."""
    mod = bridge.get("grad", dtype_name)
    device = torch.device("cuda")
    paths = Paths(n=n, name=dtype_name)

    lv = make_input(M, n, dtype_name, device, seed=8000 + n, kind="random")
    r = make_input(M, n, dtype_name, device, seed=8100 + n, kind="random")
    a = make_input(M, n, dtype_name, device, seed=8200 + n, kind="random")
    ge = torch.linspace(-1.5, 1.5, M, dtype=torch.float32, device=device)

    ga0 = torch.full((M, n), 0.25, dtype=torch.float32, device=device)
    gl0 = torch.full((M, n), -0.5, dtype=torch.float32, device=device)
    ga, gl = mod.gatv2_accum_grad_al(n, ga0.clone(), gl0.clone(), ge, lv, r, a, NS)

    want_ga, want_gl = ref_gatv2_accum_grad_al(
        ga0.double(), gl0.double(), ge.double(), lv.double(), r.double(), a.double(), NS, paths
    )
    torch.testing.assert_close(ga.double(), want_ga, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])
    torch.testing.assert_close(gl.double(), want_gl, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_gatv2_accum_grad_r(bridge, n, dtype_name):
    """grad_r picks up both the aggregation term and the score term."""
    mod = bridge.get("grad", dtype_name)
    device = torch.device("cuda")
    paths = Paths(n=n, name=dtype_name)

    lv = make_input(M, n, dtype_name, device, seed=8300 + n, kind="random")
    r = make_input(M, n, dtype_name, device, seed=8400 + n, kind="random")
    a = make_input(M, n, dtype_name, device, seed=8500 + n, kind="random")
    gh = make_input(M, n, dtype_name, device, seed=8600 + n, kind="random")
    ge = torch.linspace(-1.5, 1.5, M, dtype=torch.float32, device=device)
    alpha = torch.linspace(0.05, 0.95, M, dtype=torch.float32, device=device)

    gr0 = torch.full((M, n), 0.125, dtype=torch.float32, device=device)
    gr = mod.gatv2_accum_grad_r(n, gr0.clone(), alpha, gh, ge, lv, r, a, NS)

    want = ref_gatv2_accum_grad_r(
        gr0.double(), alpha.double(), gh.double(), ge.double(), lv.double(), r.double(), a.double(), NS, paths
    )
    torch.testing.assert_close(gr.double(), want, rtol=r_tols[dtype_name], atol=a_tols[dtype_name])


@pytest.mark.parametrize(("n", "dtype_name"), COMBOS, ids=COMBO_IDS)
def test_gatv2_grad_al_matches_finite_differences(bridge, n, dtype_name):
    """d e_ij / d a and d e_ij / d lv, checked numerically against the forward pass.

    An independent check on the analytic gradients: the reference in reference.py could
    encode the same misunderstanding as the header, but a finite difference of
    gatv2_dot_leaky_relu cannot.
    """
    if dtype_name != "float":
        pytest.skip("finite differences need fp32 headroom to be meaningful")

    grad_mod = bridge.get("grad", dtype_name)
    fwd = bridge.get("ops", dtype_name)
    device = torch.device("cuda")
    rows = 32
    eps = 1e-2

    lv = make_input(rows, n, dtype_name, device, seed=91, kind="random")
    r = make_input(rows, n, dtype_name, device, seed=92, kind="random")
    a = make_input(rows, n, dtype_name, device, seed=93, kind="random")
    ge = torch.ones(rows, dtype=torch.float32, device=device)  # d/d e_ij, so grad_e = 1

    zeros = torch.zeros((rows, n), dtype=torch.float32, device=device)
    ga, gl = grad_mod.gatv2_accum_grad_al(n, zeros.clone(), zeros.clone(), ge, lv, r, a, NS)

    # LeakyReLU has a kink at 0, where the derivative is not defined and a central
    # difference just averages the two one-sided slopes. Only compare away from it: an
    # element whose edge sits within the perturbation of zero says nothing about whether
    # the analytic gradient is right.
    edge = (lv.double() + r.double()).abs()
    away_from_kink = edge > 20 * eps

    for k in range(n):
        for tensor, analytic, label in ((a, ga, "grad_a"), (lv, gl, "grad_l")):
            plus, minus = tensor.clone(), tensor.clone()
            plus[:, k] += eps
            minus[:, k] -= eps
            if label == "grad_a":
                e_plus = fwd.gatv2_dot_leaky_relu(n, lv, r, plus, NS)
                e_minus = fwd.gatv2_dot_leaky_relu(n, lv, r, minus, NS)
            else:
                e_plus = fwd.gatv2_dot_leaky_relu(n, plus, r, a, NS)
                e_minus = fwd.gatv2_dot_leaky_relu(n, minus, r, a, NS)
            numeric = (e_plus.double() - e_minus.double()) / (2 * eps)
            keep = away_from_kink[:, k]
            assert bool(keep.any()), "no sample points away from the kink; adjust the seed"
            torch.testing.assert_close(
                analytic[:, k].double()[keep],
                numeric[keep], rtol=r_tols[dtype_name], atol=a_tols[dtype_name],
                msg=lambda s, label=label, k=k: f"{label}[{k}] disagrees with finite differences: {s}",
            )
