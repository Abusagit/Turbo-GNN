"""Host-side float64 references, tolerances and input generators for the tile.cuh suite.

Two levels of expectation are used, deliberately:

* **fp64 ground truth** -- the mathematically correct value, computed in float64 and
  compared with an ulp-derived tolerance. This is what catches real mistakes.
* **order-faithful emulation** -- for the reductions, a step-by-step replay of the
  header's own reduction tree *and its intermediate dtype*, asserted bit-exact. A
  reduction can be perfectly implemented and still differ from fp64 by a lot (a
  half accumulator over 8 terms, say), so the two are reported separately: the
  emulation pins the implementation, the fp64 check judges the design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TORCH_DTYPE: dict[str, torch.dtype] = {
    "float": torch.float32,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    # accum_t can be double; it is never a num_type.
    "double": torch.float64,
}


def rnd(x: torch.Tensor, name: str) -> torch.Tensor:
    """Round a float64 tensor through ``name``'s precision and back to float64."""
    return x.to(TORCH_DTYPE[name]).double()


def is_half_fp(name: str) -> bool:
    """Mirrors ``is_half_fp_v`` -- true for the two 16-bit types."""
    return name in ("half", "bf16")


# Ops that perform real arithmetic, and so are subject to -ftz=true on the device: it
# flushes subnormal *operands* as well as subnormal results. The others only select or
# copy (relu/min/max keep their operand; leaky_relu_backward_ passes dy straight
# through), so no flushing happens and a subnormal survives.
FTZ_ARITH_OPS = {
    "scalar_mul_",
    "leaky_relu_",
    "log_",
    "exp_",
    *(f"{base}{suffix}" for base in ("add", "sub", "mul", "div") for suffix in ("_", "_dst", "_ret")),
    "fmam_",
    "fmaa_",
    "fma_dst",
    "fma_ret",
    # leaky_relu_backward_ is deliberately absent: it flushes only its comparison
    # operand, which ref_leaky_relu_backward handles itself.
}


def flush_subnormals(x: torch.Tensor, name: str) -> torch.Tensor:
    """Model -ftz=true: subnormals become zero, keeping their sign."""
    smallest = torch.finfo(TORCH_DTYPE[name]).smallest_normal
    return torch.where(x.abs() < smallest, torch.zeros_like(x) * x.sign(), x)


def _subnormal_dont_care(g: torch.Tensor, r: torch.Tensor, name: str) -> torch.Tensor:
    """Where both values sit in the subnormal neighbourhood, treat them as agreeing.

    --use_fast_math means the device is not IEEE-compliant down there and is not
    self-consistent about it either: -ftz=true flushes arithmetic results, __fdividef
    zeroes quotients near the boundary, but a pure select (leaky_relu_backward_ passing
    dy straight through) does no arithmetic and so preserves the subnormal. Rather than
    encode each of those, the suite keeps subnormals in the inputs -- they are good at
    finding UB -- and declines to pin the exact result in that one band.
    """
    smallest = torch.finfo(TORCH_DTYPE[name]).smallest_normal
    band = 2.0 * smallest
    return (g.abs() < band) & (r.abs() < band)


@dataclass(frozen=True)
class Paths:
    """Which ``if constexpr`` branch the header takes for a given (N, dtype).

    ``packed`` mirrors ``can_be_packed``; ``packed_new`` mirrors ``can_be_packed_new``,
    which additionally admits float but only from ``kCudaArch >= 1000`` -- never true
    on the sm_80/86/89/90 arch list this project builds, nor in the host pass.
    """

    n: int
    name: str
    cuda_arch: int = 800

    @property
    def packed(self) -> bool:
        return is_half_fp(self.name) and self.n % 2 == 0

    @property
    def ftz(self) -> bool:
        """Whether single-precision subnormals are flushed to zero.

        setup.py compiles with --use_fast_math, which implies -ftz=true: on the device,
        float denormals are flushed at both the inputs and the result. The host pass is
        plain g++ and keeps them, and 16-bit denormals are preserved in hardware.
        """
        return self.name == "float"

    @property
    def packed_new(self) -> bool:
        float_packs = self.name == "float" and self.cuda_arch >= 1000
        return (float_packs or is_half_fp(self.name)) and self.n % 2 == 0


# ---------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------


def ulp_of(x: torch.Tensor, name: str) -> torch.Tensor:
    """Width of one ulp of ``name`` at each magnitude in ``x`` (a float64 tensor)."""
    fi = torch.finfo(TORCH_DTYPE[name])
    ax = x.abs().clamp(min=fi.smallest_normal)
    binade = torch.floor(torch.log2(ax))
    return torch.exp2(binade) * fi.eps


# Relative tolerance for the approximate intrinsics. exp/log go through hexp/hlog on
# the device but cuda::std:: on the host, so they are not bit-comparable.
REL_TOL: dict[tuple[str, str], float] = {
    ("exp_", "float"): 2e-6,
    ("exp_", "half"): 2e-3,
    ("exp_", "bf16"): 2e-2,
    ("log_", "float"): 2e-6,
    ("log_", "half"): 2e-3,
    ("log_", "bf16"): 2e-2,
}


def _report(got: torch.Tensor, ref: torch.Tensor, ok: torch.Tensor, label: str) -> None:
    if bool(ok.all()):
        return
    bad = (~ok).nonzero()
    lines = [f"{label}: {bad.shape[0]} of {ok.numel()} elements outside tolerance"]
    for row in bad[:8].tolist():
        idx = tuple(row)
        lines.append(f"  at {idx}: got={got[idx].item()!r} want={ref[idx].item()!r}")
    if bad.shape[0] > 8:
        lines.append(f"  ... and {bad.shape[0] - 8} more")
    raise AssertionError("\n".join(lines))


def _agree_nonfinite(g: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    both_nan = torch.isnan(g) & torch.isnan(r)
    both_inf = torch.isinf(g) & torch.isinf(r) & (torch.sign(g) == torch.sign(r))
    return both_nan | both_inf


def assert_exact(
    got: torch.Tensor, ref64: torch.Tensor, name: str, label: str, ftz: bool = False, ftz_name: str | None = None
) -> None:
    """``got`` must equal the fp64 reference rounded once into ``name``.

    Valid only for ops that are single-rounding-correct: data movement, neg, min/max,
    relu, and add/sub/mul.
    """
    g = got.double()
    want = rnd(ref64, name)
    ok = (g == want) | _agree_nonfinite(g, want)
    if ftz:
        # The band is a property of the flushed type (always single precision), which is
        # not necessarily the accumulator's type.
        ok = ok | _subnormal_dont_care(g, want, ftz_name or name)
    _report(g, want, ok, f"{label} (expected bit-exact)")


def assert_ulp(
    got: torch.Tensor, ref64: torch.Tensor, name: str, label: str, max_ulp: float = 1.0, ftz: bool = False
) -> None:
    g = got.double()
    tol = ulp_of(ref64, name) * max_ulp
    ok = ((g - ref64).abs() <= tol) | _agree_nonfinite(g, ref64)
    if ftz:
        ok = ok | _subnormal_dont_care(g, ref64, name)
    _report(g, ref64, ok, f"{label} (within {max_ulp} ulp)")


def assert_rel(
    got: torch.Tensor, ref64: torch.Tensor, name: str, label: str, rtol: float, rel_floor: float = 0.0
) -> None:
    """Relative check, with an optional floor on the scale the tolerance is taken against.

    ``rel_floor`` matters for log: near x==1 the result goes to zero while the argument
    does not, so a purely relative bound would demand accuracy the intrinsic cannot
    deliver. __logf is accurate to a few ulp *of the argument's scale*, so the floor
    expresses the right error model rather than just loosening the number.
    """
    g = got.double()
    scale = ref64.abs().clamp(min=rel_floor)
    tol = scale * rtol + torch.finfo(TORCH_DTYPE[name]).smallest_normal
    ok = ((g - ref64).abs() <= tol) | _agree_nonfinite(g, ref64)
    _report(g, ref64, ok, f"{label} (rtol={rtol}, floor={rel_floor})")


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------


def _special_pool(name: str) -> list[float]:
    """Values chosen to hit the boundaries the header's branches actually turn on."""
    fi = torch.finfo(TORCH_DTYPE[name])
    return [
        0.0,
        -0.0,  # leaky_relu / leaky_relu_backward boundary (>= vs __hgt2)
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        3.75,
        -3.75,
        fi.eps,
        -fi.eps,
        fi.smallest_normal,
        -fi.smallest_normal,
        fi.smallest_normal / 2.0,  # subnormal
        -fi.smallest_normal / 2.0,
        fi.max / 2.0,
        -fi.max / 2.0,
    ]


def make_input(
    m: int,
    n: int,
    name: str,
    device: torch.device,
    *,
    seed: int = 0,
    kind: str = "random",
    positive: bool = False,
    nonzero: bool = False,
    max_abs: float | None = None,
) -> torch.Tensor:
    """A deterministic ``[m, n]`` input tensor of dtype ``name``.

    ``kind="random"`` gives well-conditioned values; ``kind="special"`` tiles the
    boundary pool; ``kind="extreme"`` mixes in inf/-inf/nan.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dt = TORCH_DTYPE[name]

    if kind == "random":
        x = torch.rand((m, n), generator=gen, dtype=torch.float64) * 3.5 + 0.25
        if not positive:
            sign = torch.randint(0, 2, (m, n), generator=gen, dtype=torch.float64) * 2 - 1
            x = x * sign
    elif kind == "special":
        pool = torch.tensor(_special_pool(name), dtype=torch.float64)
        idx = torch.randint(0, pool.numel(), (m, n), generator=gen)
        x = pool[idx]
        if positive:
            x = x.abs().clamp(min=torch.finfo(dt).smallest_normal)
    elif kind == "extreme":
        pool = torch.tensor([*_special_pool(name), float("inf"), float("-inf"), float("nan")], dtype=torch.float64)
        idx = torch.randint(0, pool.numel(), (m, n), generator=gen)
        x = pool[idx]
    else:
        raise ValueError(f"unknown kind {kind!r}")

    if nonzero:
        # Keep magnitudes away from zero so division stays finite.
        floor = torch.finfo(dt).smallest_normal * 8
        x = torch.where(x.abs() < floor, torch.full_like(x, 0.75), x)

    if max_abs is not None:
        # Cap the top of the range without disturbing the small-magnitude boundaries.
        # Reductions use this so that pairwise products stay representable: an overflow
        # to +-inf inside a sum makes the result order-dependent, which is a property of
        # the input rather than of the code under test. Overflow itself is covered by the
        # elementwise mul_/prod tests.
        x = x.clamp(min=-max_abs, max=max_abs)

    return x.to(dt).to(device)


# ---------------------------------------------------------------------------
# elementwise references (all arithmetic rounded through num_type, as the header does)
# ---------------------------------------------------------------------------


def ref_leaky_relu(a: torch.Tensor, ns: float, name: str) -> torch.Tensor:
    """max(a,0) + ns*min(a,0), with the header's rounding at each step."""
    ns_r = rnd(torch.tensor(ns, dtype=torch.float64), name)
    hi = torch.clamp(a, min=0.0)
    lo = torch.clamp(a, max=0.0)
    return rnd(hi + rnd(ns_r * lo, name), name)


def ref_leaky_relu_backward(a: torch.Tensor, dy: torch.Tensor, ns: float, name: str, paths: Paths) -> torch.Tensor:
    """d/dx of leaky_relu, times dy.

    Two subtleties, both observed rather than assumed:

    * The branches genuinely disagree at exactly 0. The scalar path tests ``x >= 0`` and
      returns dy; the packed path uses ``__hgt2`` (strict ``>``) and returns dy*ns.
    * Under -ftz the *comparison* operand is flushed, so a negative subnormal x compares
      as ``-0.0 >= 0`` and takes the pass-through branch -- while dy is not flushed,
      because passing it through involves no arithmetic.
    """
    ns_r = rnd(torch.tensor(ns, dtype=torch.float64), name)
    if paths.ftz:
        a = flush_subnormals(a, name)
    if paths.packed:
        mask = (a > 0).double()
        diff = rnd(rnd(torch.tensor(1.0, dtype=torch.float64), name) - ns_r, name)
        factor = rnd(mask * diff + ns_r, name)  # packed_fma -> single rounding
        return rnd(dy * factor, name)
    return torch.where(a >= 0, dy, rnd(dy * ns_r, name))


def ref_elementwise(
    op: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: float, name: str, paths: Paths
) -> torch.Tensor:
    """fp64 reference for one elementwise op. Inputs are float64 views of the dtype."""
    s_r = rnd(torch.tensor(s, dtype=torch.float64), name)
    if paths.ftz and op in FTZ_ARITH_OPS:
        a, b, c = (flush_subnormals(t, name) for t in (a, b, c))
        return flush_subnormals(_ref_elementwise_body(op, a, b, c, s, s_r, name, paths), name)
    return _ref_elementwise_body(op, a, b, c, s, s_r, name, paths)


def _ref_elementwise_body(
    op: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: float, s_r: torch.Tensor, name: str, paths: Paths
) -> torch.Tensor:
    if op == "neg_":
        return rnd(-a, name)
    if op == "log_":
        return torch.log(a)
    if op == "exp_":
        return torch.exp(a)
    if op == "relu_":
        return torch.clamp(a, min=0.0)
    if op == "scalar_mul_":
        return rnd(a * s_r, name)
    if op == "leaky_relu_":
        return ref_leaky_relu(a, s, name)
    if op == "leaky_relu_backward_":
        return ref_leaky_relu_backward(a, b, s, name, paths)

    base = op.rstrip("_").removesuffix("_dst").removesuffix("_ret")
    if base == "add":
        return rnd(a + b, name)
    if base == "sub":
        return rnd(a - b, name)
    if base == "mul":
        return rnd(a * b, name)
    if base == "div":
        if paths.ftz:
            # --use_fast_math turns float division into __fdividef, which the CUDA math
            # docs define as delivering zero whenever |y| > 2^126.
            return torch.where(b.abs() > 2.0**126, torch.zeros_like(a), rnd(a / b, name))
        return rnd(a / b, name)
    if base == "minimum":
        return torch.minimum(a, b)
    if base == "maximum":
        return torch.maximum(a, b)

    # fma family: a single rounding, which is what both __hfma2 and a contracted
    # scalar a*b+c produce. The 1-ulp tolerance covers a non-contracted build.
    if op in ("fmam_", "fma_dst", "fma_ret"):
        return rnd(a * b + c, name)
    if op == "fmaa_":
        return rnd(b * c + a, name)

    raise ValueError(f"no reference for op {op!r}")


EXACT_OPS = {
    "neg_",
    "relu_",
    "scalar_mul_",
    "add_",
    "add_dst",
    "add_ret",
    "sub_",
    "sub_dst",
    "sub_ret",
    "mul_",
    "mul_dst",
    "mul_ret",
    "minimum_",
    "minimum_dst",
    "minimum_ret",
    "maximum_",
    "maximum_dst",
    "maximum_ret",
    "leaky_relu_",
    "leaky_relu_backward_",
}
ULP_OPS = {"div_", "div_dst", "div_ret", "fmam_", "fmaa_", "fma_dst", "fma_ret"}
REL_OPS = {"exp_", "log_"}


def check_elementwise(got: torch.Tensor, ref64: torch.Tensor, op: str, name: str, label: str, paths: Paths) -> None:
    """Apply the right strictness for ``op``."""
    if op in REL_OPS:
        # log's error is absolute in the argument's scale, not in the result's.
        floor = 1.0 if op == "log_" else 0.0
        assert_rel(got, ref64, name, label, REL_TOL[(op, name)], rel_floor=floor)
    elif op in ULP_OPS:
        # float division under --use_fast_math becomes __fdividef: ~2 ulp, and it zeroes
        # quotients near the subnormal boundary.
        max_ulp = 2.0 if (name == "float" and op.startswith("div")) else 1.0
        assert_ulp(got, ref64, name, label, max_ulp, ftz=paths.ftz)
    elif op in EXACT_OPS:
        assert_exact(got, ref64, name, label, ftz=paths.ftz)
    else:
        assert_ulp(got, ref64, name, label, 1.0, ftz=paths.ftz)


def assert_sum_fp64(
    got: torch.Tensor, ref64: torch.Tensor, terms: torch.Tensor, name: str, label: str, seed: float = 0.0
) -> None:
    """Compare a summation against fp64 with a term-scaled error bound.

    A relative bound on the *result* is the wrong model for a sum: when the terms
    cancel, the result can be near zero while the intermediate partial sums -- and so
    the roundings -- are not. The standard bound scales with the sum of magnitudes:

        |computed - exact|  <=  steps * eps * (|seed| + sum |a_i|)
    """
    fi = torch.finfo(TORCH_DTYPE[name])
    steps = terms.shape[1] + 1
    scale = terms.abs().sum(dim=1) + abs(seed)
    tol = steps * fi.eps * scale + fi.smallest_normal
    g = got.double()
    ok = ((g - ref64).abs() <= tol) | _agree_nonfinite(g, ref64)
    _report(g, ref64, ok, f"{label} (term-scaled, {steps} steps)")


# ---------------------------------------------------------------------------
# order-faithful reduction emulation
# ---------------------------------------------------------------------------


def _pairs(v: torch.Tensor, i: int) -> torch.Tensor:
    return v[:, 2 * i : 2 * i + 2]


def _step(x: torch.Tensor, name: str, paths: Paths) -> torch.Tensor:
    """Round one reduction step into ``name``, flushing subnormals if the build does."""
    r = rnd(x, name)
    return flush_subnormals(r, name) if (paths.ftz and name == "float") else r


def emul_sum_num(v: torch.Tensor, paths: Paths) -> torch.Tensor:
    """``buf_acc`` from tile.cuh's ``sum``.

    Note it accumulates in *num_type*, not accum_t -- that is the header's choice and
    the reason the fp64 cross-check is reported separately.
    """
    n, name = paths.n, paths.name
    if paths.packed_new and n >= 4:
        acc = _pairs(v, 0).clone()
        for i in range(1, n // 2):
            acc = _step(acc + _pairs(v, i), name, paths)
        return _step(acc[:, 0] + acc[:, 1], name, paths)
    acc = torch.zeros(v.shape[0], dtype=torch.float64, device=v.device)
    for i in range(n):
        acc = _step(acc + v[:, i], name, paths)
    return acc


def emul_prod(v: torch.Tensor, paths: Paths, acc_name: str) -> torch.Tensor:
    """``buf_acc`` from tile.cuh's ``prod``. Unlike sum, this one accumulates in accum_t."""
    n, name = paths.n, paths.name
    if paths.packed_new and n >= 4:
        acc = _pairs(v, 0).clone()
        for i in range(1, n // 2):
            acc = _step(acc * _pairs(v, i), name, paths)
        return _step(rnd(acc[:, 0], acc_name) * rnd(acc[:, 1], acc_name), acc_name, paths)
    acc = torch.ones(v.shape[0], dtype=torch.float64, device=v.device)
    for i in range(n):
        acc = _step(acc * rnd(v[:, i], acc_name), acc_name, paths)
    return acc


def emul_minmax(v: torch.Tensor, paths: Paths, want_max: bool) -> torch.Tensor:
    """``buf_acc`` from tile.cuh's ``min``/``max``. Seeded from ``buf[0]``, in num_type."""
    n = paths.n
    pick = torch.maximum if want_max else torch.minimum
    if paths.packed and n >= 4:
        acc = _pairs(v, 0).clone()
        for i in range(1, n // 2):
            acc = pick(acc, _pairs(v, i))
        return pick(acc[:, 0], acc[:, 1])
    acc = v[:, 0].clone()
    for i in range(1, n):
        acc = pick(acc, v[:, i])
    return acc


def ref_reduce(
    op: str, a: torch.Tensor, b: torch.Tensor, acc_init: float, w: float, paths: Paths, acc_name: str
) -> torch.Tensor:
    """Order-faithful expectation for one reduction, as a float64 tensor of shape [M]."""
    name = paths.name
    if paths.ftz:
        # -ftz flushes subnormal operands, for the comparisons in min/max just as much as
        # for the arithmetic in sum/prod/dot.
        a, b = flush_subnormals(a, name), flush_subnormals(b, name)
    init = rnd(torch.tensor(acc_init, dtype=torch.float64), acc_name).item()
    w_r = rnd(torch.tensor(w, dtype=torch.float64), acc_name)

    if op in ("sum_acc", "sum_ret"):
        buf = emul_sum_num(a, paths)
        start = init if op == "sum_acc" else 0.0
        return rnd(start + rnd(buf, acc_name), acc_name)

    if op in ("weighted_sum_acc", "weighted_sum_ret"):
        buf = rnd(rnd(emul_sum_num(a, paths), acc_name) * w_r, acc_name)
        start = init if op == "weighted_sum_acc" else 0.0
        return rnd(start + buf, acc_name)

    if op in ("prod_acc", "prod_ret"):
        buf = emul_prod(a, paths, acc_name)
        # The value-returning form seeds 1; the acc form multiplies into *acc.
        start = init if op == "prod_acc" else 1.0
        return rnd(start * buf, acc_name)

    if op in ("min_acc", "min_ret", "max_acc", "max_ret"):
        want_max = op.startswith("max")
        buf = emul_minmax(a, paths, want_max)
        pick = torch.maximum if want_max else torch.minimum
        # The value-returning form seeds accum_t{} == 0, so the result is clamped
        # against zero -- modelled here so the test states the current behaviour.
        start = init if op.endswith("_acc") else 0.0
        # AdOps<num_type>::min/max takes num_type, so the fp32 accumulator is
        # narrowed through num_type before the comparison.
        start_t = rnd(torch.full_like(buf, start), name)
        return rnd(pick(start_t, buf), name)

    if op in ("dot_product_acc", "dot_product_ret"):
        prod = _step(a * b, name, paths)  # mul_ in num_type
        buf = emul_sum_num(prod, paths)
        start = init if op == "dot_product_acc" else 0.0
        return rnd(start + rnd(buf, acc_name), acc_name)

    raise ValueError(f"no reference for reduction {op!r}")


def ref_reduce_fp64(op: str, a: torch.Tensor, b: torch.Tensor, acc_init: float, w: float) -> torch.Tensor:
    """The mathematically correct value, ignoring the header's intermediate precision."""
    if op in ("sum_acc", "sum_ret"):
        return (acc_init if op.endswith("_acc") else 0.0) + a.sum(dim=1)
    if op in ("weighted_sum_acc", "weighted_sum_ret"):
        return (acc_init if op.endswith("_acc") else 0.0) + a.sum(dim=1) * w
    if op in ("prod_acc", "prod_ret"):
        return (acc_init if op.endswith("_acc") else 1.0) * a.prod(dim=1)
    if op in ("min_acc", "min_ret"):
        return torch.minimum(a.min(dim=1).values, torch.full_like(a[:, 0], acc_init if op.endswith("_acc") else 0.0))
    if op in ("max_acc", "max_ret"):
        return torch.maximum(a.max(dim=1).values, torch.full_like(a[:, 0], acc_init if op.endswith("_acc") else 0.0))
    if op in ("dot_product_acc", "dot_product_ret"):
        return (acc_init if op.endswith("_acc") else 0.0) + (a * b).sum(dim=1)
    raise ValueError(f"no fp64 reference for reduction {op!r}")


# ---------------------------------------------------------------------------
# GATv2 references
# ---------------------------------------------------------------------------


def ref_gatv2_dot_leaky_relu(
    lv: torch.Tensor, r: torch.Tensor, a: torch.Tensor, ns: float, paths: Paths
) -> torch.Tensor:
    """e_partial = sum_k a_k * LeakyReLU_ns(l_k + r_k), rounded as the header rounds."""
    name = paths.name
    if paths.ftz:
        lv, r, a = (flush_subnormals(t, name) for t in (lv, r, a))
    edge = _step(lv + r, name, paths)
    act = _step(ref_leaky_relu(edge, ns, name), name, paths)
    prod = _step(act * a, name, paths)
    return rnd(emul_sum_num(prod, paths), "float")


def ref_gatv2_accum_grad_al(
    ga0: torch.Tensor,
    gl0: torch.Tensor,
    ge: torch.Tensor,
    lv: torch.Tensor,
    r: torch.Tensor,
    a: torch.Tensor,
    ns: float,
    paths: Paths,
) -> tuple[torch.Tensor, torch.Tensor]:
    """grad_a += grad_e * LeakyReLU(lv+r);  grad_l += grad_e * LeakyReLU'(lv+r) * a.

    Local vectors stay in num_type; the conversion to accum_t happens only when
    combining with the fp32 accumulators. ``ge`` is rounded through num_type first,
    which tile.cuh:1105 does explicitly.
    """
    name = paths.name
    edge = rnd(lv + r, name)
    ge_vec = rnd(ge.unsqueeze(1).expand_as(edge).contiguous(), name)
    buf = ref_leaky_relu_backward(edge, ge_vec, ns, name, paths)
    ga = rnd(ga0 + rnd(buf, "float") * rnd(edge, "float"), "float")
    gl = rnd(gl0 + rnd(buf, "float") * rnd(a, "float"), "float")
    return ga, gl


def ref_gatv2_accum_grad_r(
    gr0: torch.Tensor,
    alpha: torch.Tensor,
    gh: torch.Tensor,
    ge: torch.Tensor,
    lv: torch.Tensor,
    r: torch.Tensor,
    a: torch.Tensor,
    ns: float,
    paths: Paths,
) -> torch.Tensor:
    """grad_r += alpha*grad_h + grad_e * LeakyReLU'(lv+r) * a.

    r_j appears both inside the score and as the aggregated message, hence two terms.
    """
    name = paths.name
    edge = rnd(lv + r, name)
    ge_vec = rnd(ge.unsqueeze(1).expand_as(edge).contiguous(), name)
    alpha_vec = rnd(alpha.unsqueeze(1).expand_as(edge).contiguous(), name)
    buf = ref_leaky_relu_backward(edge, ge_vec, ns, name, paths)
    out = rnd(rnd(buf, "float") * rnd(a, "float") + gr0, "float")
    return rnd(out + rnd(alpha_vec, "float") * rnd(gh, "float"), "float")


def ref_select_tw(d: int, name: str) -> int:
    """min(ceil(D / 32), 16 / sizeof(T)) -- SelectTW's documented intent."""
    itemsize = torch.finfo(TORCH_DTYPE[name]).bits // 8
    return min((d + 31) // 32, 16 // itemsize)
