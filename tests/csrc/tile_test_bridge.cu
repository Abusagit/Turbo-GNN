// pybind bridge for exercising csrc/common/tile.cuh from pytest.
//
// Compiled once per (group, dtype, pass) triple, selected by preprocessor macros. The
// split keeps a combination that currently fails to instantiate from taking down the
// rows it does not affect. See tests/csrc/conftest.py.
//
//   TGNN_DTYPE : 0 = float, 1 = half, 2 = nv_bfloat16
//   TGNN_GROUP : 0 = data, 1 = ops, 2 = cvt, 3 = grad
//   TGNN_HOST  : 0 = compile the workers for the device, 1 = for the host
//
// Two structural notes, both learned the hard way:
//
//  * Every runtime -> compile-time N dispatch goes through a *template* impl function.
//    `if constexpr` only discards its untaken branch inside a template, so guarding
//    Vec<N,T>'s 16-byte limit in a plain function still instantiates the oversized Vec
//    and hard-fails the build.
//
//  * TGNN_HOST is a module dimension rather than a runtime flag. tile.cuh's
//    __host__ __device__ members call unqualified names from their dependent base
//    (transfer_vector, add_, mul_, ...). nvcc's device frontend resolves those; g++
//    does not, since ordinary lookup never searches a dependent base. Compiling both
//    passes in one TU would therefore make every device row fail for a host-only
//    reason.
//
// Includes only "common/tile.cuh", never "common.cuh": the latter also pulls in the
// legacy common_.cuh, which declares an incompatible template of the same name.

#include <torch/extension.h>

#include <string>
#include <vector>

#include "common/tile.cuh"

#ifndef TGNN_DTYPE
#error "TGNN_DTYPE must be defined (0=float, 1=half, 2=nv_bfloat16)"
#endif
#ifndef TGNN_GROUP
#error "TGNN_GROUP must be defined (0=data, 1=ops, 2=cvt, 3=grad)"
#endif
#ifndef TGNN_HOST
#define TGNN_HOST 0
#endif

#if TGNN_DTYPE == 0
using num_t                        = float;
static constexpr auto kTorchDtype  = torch::kFloat32;
static constexpr char kDtypeName[] = "float";
#elif TGNN_DTYPE == 1
using num_t                        = half;
static constexpr auto kTorchDtype  = torch::kFloat16;
static constexpr char kDtypeName[] = "half";
#elif TGNN_DTYPE == 2
using num_t                        = nv_bfloat16;
static constexpr auto kTorchDtype  = torch::kBFloat16;
static constexpr char kDtypeName[] = "nv_bfloat16";
#else
#error "bad TGNN_DTYPE"
#endif

#if TGNN_HOST
#define TGNN_WORKER __host__
static constexpr bool kOnHost = true;
#else
#define TGNN_WORKER __device__
static constexpr bool kOnHost = false;
#endif

// Largest N that Vec<N, num_t> permits (Vec caps at 16 bytes).
static constexpr size_t kMaxN = 16 / sizeof(num_t);

template <size_t N, typename T>
static constexpr bool vec_fits = (N * sizeof(T) <= 16);

namespace {

// Dispatch a runtime N to FN<N>(...). FN must guard `vec_fits` itself.
#define TGNN_DISPATCH_N(nval, FN, ...)                            \
    do {                                                          \
        switch (nval) {                                           \
            case 1: FN<1>(__VA_ARGS__); break;                    \
            case 2: FN<2>(__VA_ARGS__); break;                    \
            case 4: FN<4>(__VA_ARGS__); break;                    \
            case 8: FN<8>(__VA_ARGS__); break;                    \
            default: TORCH_CHECK(false, "unsupported N: ", nval);  \
        }                                                         \
    } while (0)

#define TGNN_N_TOO_WIDE(N)                                                                                     \
    TORCH_CHECK(                                                                                               \
        false, "N=", N, " is not instantiable for ", kDtypeName, ": Vec<", N, ", ", kDtypeName,                 \
        "> exceeds the 16-byte cap"                                                                            \
    )

[[maybe_unused]] int64_t grid_for(int64_t m, int64_t block) { return (m + block - 1) / block; }
[[maybe_unused]] constexpr int64_t kBlock = 128;

void check_vec_tensor(const torch::Tensor& t, int64_t n, const char* what) {
    TORCH_CHECK(t.defined(), what, ": tensor is undefined");
    TORCH_CHECK(t.is_contiguous(), what, ": tensor must be contiguous");
    TORCH_CHECK(t.scalar_type() == kTorchDtype, what, ": expected dtype ", kDtypeName);
    TORCH_CHECK(t.dim() == 2 && t.size(1) == n, what, ": expected shape [M, ", n, "]");
    TORCH_CHECK(t.is_cpu() == kOnHost, what, ": tensor must live ", kOnHost ? "on the CPU" : "on the GPU");
}

template <size_t N>
Vec<N, num_t>* vec_ptr(const torch::Tensor& t) {
    return reinterpret_cast<Vec<N, num_t>*>(t.data_ptr());
}
template <size_t N>
const Vec<N, num_t>* cvec_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const Vec<N, num_t>*>(t.data_ptr());
}

}  // namespace

// ===========================================================================
// group "data": Vec layout, SelectTW, VecOpsBase data movement
// ===========================================================================
#if TGNN_GROUP == 0

namespace {

enum DataOp : int {
    OP_STORE_ZERO = 0,
    OP_GET_ZERO,
    OP_LOAD_SCALARS,
    OP_STORE_SCALARS,
    OP_TRANSFER_SCALARS,
    OP_TRANSFER_VECTOR,
};

template <size_t N>
TGNN_WORKER void apply_data(int op, Vec<N, num_t>* dst, const Vec<N, num_t>* src) {
    using Ops = VecOpsBase<N, num_t>;
    switch (op) {
        case OP_STORE_ZERO:
            Ops::store_zero(dst);
            break;
        case OP_GET_ZERO:
            *dst = Ops::get_zero();
            break;
        // load__scalars(num_type* dst, vec_t const* src): vector -> scalar array.
        // (The header's doc comment describes the opposite direction; the signature is
        // authoritative and is what we test.)
        case OP_LOAD_SCALARS:
            Ops::load__scalars(reinterpret_cast<num_t*>(dst), src);
            break;
        // store_scalars(vec_t* dst, num_type const* src): scalar array -> vector.
        case OP_STORE_SCALARS:
            Ops::store_scalars(dst, reinterpret_cast<const num_t*>(src));
            break;
        case OP_TRANSFER_SCALARS:
            Ops::transfer_scalars(reinterpret_cast<num_t*>(dst), reinterpret_cast<const num_t*>(src));
            break;
        case OP_TRANSFER_VECTOR:
            Ops::transfer_vector(dst, src);
            break;
    }
}

#if !TGNN_HOST
template <size_t N>
__global__ void k_data(int op, Vec<N, num_t>* dst, const Vec<N, num_t>* src, int64_t m) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) apply_data<N>(op, dst + i, src + i);
}
#endif

template <size_t N>
void data_move_impl(int op, const torch::Tensor& dst, const torch::Tensor& src, int64_t m) {
    if constexpr (vec_fits<N, num_t>) {
#if TGNN_HOST
        for (int64_t i = 0; i < m; ++i) apply_data<N>(op, vec_ptr<N>(dst) + i, cvec_ptr<N>(src) + i);
#else
        k_data<N><<<grid_for(m, kBlock), kBlock>>>(op, vec_ptr<N>(dst), cvec_ptr<N>(src), m);
        C10_CUDA_CHECK(cudaGetLastError());
#endif
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

template <size_t N>
void layout_impl(py::dict& out) {
    if constexpr (vec_fits<N, num_t>) {
        py::dict e;
        e["size"]        = sizeof(Vec<N, num_t>);
        e["align"]       = alignof(Vec<N, num_t>);
        e["wide_size"]   = sizeof(typename Vec<N, num_t>::wide_t);
        out[py::int_(N)] = e;
    }
}

}  // namespace

torch::Tensor data_move(int64_t op, int64_t n, torch::Tensor dst, torch::Tensor src) {
    check_vec_tensor(dst, n, "dst");
    check_vec_tensor(src, n, "src");
    TORCH_CHECK(dst.size(0) == src.size(0), "dst and src must have the same M");
    TGNN_DISPATCH_N(n, data_move_impl, static_cast<int>(op), dst, src, dst.size(0));
    return dst;
}

py::dict vec_layout() {
    py::dict out;
    layout_impl<1>(out);
    layout_impl<2>(out);
    layout_impl<4>(out);
    layout_impl<8>(out);
    return out;
}

// SelectTW for the D values the kernels dispatch, plus D=96 (the non-power-of-two trap).
py::dict select_tw() {
    py::dict out;
#define TGNN_STW(D)                                             \
    {                                                           \
        py::dict e;                                             \
        e["value"]         = SelectTW<D, num_t>::value;         \
        e["threads_per_d"] = SelectTW<D, num_t>::threads_per_d; \
        out[py::int_(D)]   = e;                                 \
    }
    TGNN_STW(32) TGNN_STW(64) TGNN_STW(96) TGNN_STW(128) TGNN_STW(256)
#undef TGNN_STW
    return out;
}

py::dict op_codes() {
    py::dict d;
    d["store_zero"]       = static_cast<int>(OP_STORE_ZERO);
    d["get_zero"]         = static_cast<int>(OP_GET_ZERO);
    d["load__scalars"]    = static_cast<int>(OP_LOAD_SCALARS);
    d["store_scalars"]    = static_cast<int>(OP_STORE_SCALARS);
    d["transfer_scalars"] = static_cast<int>(OP_TRANSFER_SCALARS);
    d["transfer_vector"]  = static_cast<int>(OP_TRANSFER_VECTOR);
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("data_move", &data_move);
    m.def("vec_layout", &vec_layout);
    m.def("select_tw", &select_tw);
    m.def("op_codes", &op_codes);
    m.attr("dtype_name") = kDtypeName;
    m.attr("on_host")    = kOnHost;
    m.attr("max_n")      = static_cast<int64_t>(kMaxN);
}

#endif  // TGNN_GROUP == 0

// ===========================================================================
// group "ops": elementwise + reductions + gatv2_dot_leaky_relu
// ===========================================================================
#if TGNN_GROUP == 1

namespace {

enum EwOp : int {
    OP_NEG = 0,
    OP_LOG,
    OP_EXP,
    OP_RELU,
    OP_SCALAR_MUL,
    OP_LEAKY_RELU,
    OP_LEAKY_RELU_BWD,
    OP_ADD_I,
    OP_ADD_D,
    OP_ADD_R,
    OP_SUB_I,
    OP_SUB_D,
    OP_SUB_R,
    OP_MUL_I,
    OP_MUL_D,
    OP_MUL_R,
    OP_DIV_I,
    OP_DIV_D,
    OP_DIV_R,
    OP_MIN_I,
    OP_MIN_D,
    OP_MIN_R,
    OP_MAX_I,
    OP_MAX_D,
    OP_MAX_R,
    OP_FMAM_I,
    OP_FMAA_I,
    OP_FMA_D,
    OP_FMA_R,
};

template <size_t N>
TGNN_WORKER void apply_ew(
    int op, Vec<N, num_t>* out, const Vec<N, num_t>* a, const Vec<N, num_t>* b, const Vec<N, num_t>* c, num_t s
) {
    using Ops = VecOpsFloatBase<N, num_t>;
    // The in-place flavours mutate their first argument, so seed `out` from `a` and let
    // them work on `out`. One uniform ABI then covers all three flavours.
    *out = *a;
    switch (op) {
        case OP_NEG: Ops::neg_(out); break;
        case OP_LOG: Ops::log_(out); break;
        case OP_EXP: Ops::exp_(out); break;
        case OP_RELU: Ops::relu_(out); break;

        case OP_SCALAR_MUL: Ops::scalar_mul_(out, s); break;
        case OP_LEAKY_RELU: Ops::leaky_relu_(out, s); break;
        case OP_LEAKY_RELU_BWD: Ops::leaky_relu_backward_(out, b, s); break;

        case OP_ADD_I: Ops::add_(out, b); break;
        case OP_ADD_D: Ops::add(out, a, b); break;
        case OP_ADD_R: *out = Ops::add(a, b); break;
        case OP_SUB_I: Ops::sub_(out, b); break;
        case OP_SUB_D: Ops::sub(out, a, b); break;
        case OP_SUB_R: *out = Ops::sub(a, b); break;
        case OP_MUL_I: Ops::mul_(out, b); break;
        case OP_MUL_D: Ops::mul(out, a, b); break;
        case OP_MUL_R: *out = Ops::mul(a, b); break;
        case OP_DIV_I: Ops::div_(out, b); break;
        case OP_DIV_D: Ops::div(out, a, b); break;
        case OP_DIV_R: *out = Ops::div(a, b); break;
        case OP_MIN_I: Ops::minimum_(out, b); break;
        case OP_MIN_D: Ops::minimum(out, a, b); break;
        case OP_MIN_R: *out = Ops::minimum(a, b); break;
        case OP_MAX_I: Ops::maximum_(out, b); break;
        case OP_MAX_D: Ops::maximum(out, a, b); break;
        case OP_MAX_R: *out = Ops::maximum(a, b); break;

        case OP_FMAM_I: Ops::fmam_(out, b, c); break;
        case OP_FMAA_I: Ops::fmaa_(out, b, c); break;
        case OP_FMA_D: Ops::fma(out, a, b, c); break;
        case OP_FMA_R: *out = Ops::fma(a, b, c); break;
    }
}

enum RedOp : int {
    OP_SUM_ACC = 0,
    OP_SUM_RET,
    OP_WSUM_ACC,
    OP_WSUM_RET,
    OP_PROD_ACC,
    OP_PROD_RET,
    OP_RMIN_ACC,
    OP_RMIN_RET,
    OP_RMAX_ACC,
    OP_RMAX_RET,
    OP_DOT_ACC,
    OP_DOT_RET,
};

template <size_t N, typename acc_t>
TGNN_WORKER void apply_red(
    int op, acc_t* out, const Vec<N, num_t>* a, const Vec<N, num_t>* b, acc_t acc_init, acc_t w
) {
    using Ops = VecOpsFloatBase<N, num_t>;
    acc_t acc = acc_init;
    switch (op) {
        case OP_SUM_ACC: Ops::sum(&acc, a); break;
        case OP_SUM_RET: acc = Ops::template sum<acc_t>(a); break;
        case OP_WSUM_ACC: Ops::weighted_sum(&acc, w, a); break;
        case OP_WSUM_RET: acc = Ops::template weighted_sum<acc_t>(w, a); break;
        case OP_PROD_ACC: Ops::prod(&acc, a); break;
        case OP_PROD_RET: acc = Ops::template prod<acc_t>(a); break;
        case OP_RMIN_ACC: Ops::min(&acc, a); break;
        case OP_RMIN_RET: acc = Ops::template min<acc_t>(a); break;
        case OP_RMAX_ACC: Ops::max(&acc, a); break;
        case OP_RMAX_RET: acc = Ops::template max<acc_t>(a); break;
        case OP_DOT_ACC: Ops::dot_product(&acc, a, b); break;
        case OP_DOT_RET: acc = Ops::template dot_product<acc_t>(a, b); break;
    }
    *out = acc;
}

template <size_t N>
TGNN_WORKER void apply_gatv2_dot(
    float* out, const Vec<N, num_t>* l, const Vec<N, num_t>* r, const Vec<N, num_t>* a, float ns
) {
    *out = TileOps<N, num_t, float>::gatv2_dot_leaky_relu(*l, *r, *a, ns);
}

#if !TGNN_HOST
template <size_t N>
__global__ void k_ew(
    int op, Vec<N, num_t>* out, const Vec<N, num_t>* a, const Vec<N, num_t>* b, const Vec<N, num_t>* c, num_t s,
    int64_t m
) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) apply_ew<N>(op, out + i, a + i, b + i, c + i, s);
}

template <size_t N, typename acc_t>
__global__ void k_red(
    int op, acc_t* out, const Vec<N, num_t>* a, const Vec<N, num_t>* b, acc_t acc_init, acc_t w, int64_t m
) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) apply_red<N, acc_t>(op, out + i, a + i, b + i, acc_init, w);
}

template <size_t N>
__global__ void k_gatv2_dot(
    float* out, const Vec<N, num_t>* l, const Vec<N, num_t>* r, const Vec<N, num_t>* a, float ns, int64_t m
) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) apply_gatv2_dot<N>(out + i, l + i, r + i, a + i, ns);
}
#endif

template <size_t N>
void ew_impl(
    int op, const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& c,
    double s, int64_t m
) {
    if constexpr (vec_fits<N, num_t>) {
        // Round the scalar through num_t exactly as a kernel would.
        const num_t sv = static_cast<num_t>(static_cast<float>(s));
#if TGNN_HOST
        for (int64_t i = 0; i < m; ++i) {
            apply_ew<N>(op, vec_ptr<N>(out) + i, cvec_ptr<N>(a) + i, cvec_ptr<N>(b) + i, cvec_ptr<N>(c) + i, sv);
        }
#else
        k_ew<N><<<grid_for(m, kBlock), kBlock>>>(
            op, vec_ptr<N>(out), cvec_ptr<N>(a), cvec_ptr<N>(b), cvec_ptr<N>(c), sv, m
        );
        C10_CUDA_CHECK(cudaGetLastError());
#endif
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

template <size_t N, typename acc_t>
void red_run(
    int op, const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, double acc_init, double w,
    int64_t m
) {
    const auto ai = static_cast<acc_t>(acc_init);
    const auto wv = static_cast<acc_t>(w);
#if TGNN_HOST
    for (int64_t i = 0; i < m; ++i) {
        apply_red<N, acc_t>(op, out.template data_ptr<acc_t>() + i, cvec_ptr<N>(a) + i, cvec_ptr<N>(b) + i, ai, wv);
    }
#else
    k_red<N, acc_t><<<grid_for(m, kBlock), kBlock>>>(
        op, out.template data_ptr<acc_t>(), cvec_ptr<N>(a), cvec_ptr<N>(b), ai, wv, m
    );
    C10_CUDA_CHECK(cudaGetLastError());
#endif
}

template <size_t N>
void red_impl(
    int op, const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, double acc_init, double w,
    bool use_double, int64_t m
) {
    if constexpr (vec_fits<N, num_t>) {
        if (use_double) {
            red_run<N, double>(op, out, a, b, acc_init, w, m);
        } else {
            red_run<N, float>(op, out, a, b, acc_init, w, m);
        }
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

template <size_t N>
void gatv2_dot_impl(
    const torch::Tensor& out, const torch::Tensor& l, const torch::Tensor& r, const torch::Tensor& a, double ns,
    int64_t m
) {
    if constexpr (vec_fits<N, num_t>) {
        const auto nsv = static_cast<float>(ns);
#if TGNN_HOST
        for (int64_t i = 0; i < m; ++i) {
            apply_gatv2_dot<N>(
                out.data_ptr<float>() + i, cvec_ptr<N>(l) + i, cvec_ptr<N>(r) + i, cvec_ptr<N>(a) + i, nsv
            );
        }
#else
        k_gatv2_dot<N><<<grid_for(m, kBlock), kBlock>>>(
            out.data_ptr<float>(), cvec_ptr<N>(l), cvec_ptr<N>(r), cvec_ptr<N>(a), nsv, m
        );
        C10_CUDA_CHECK(cudaGetLastError());
#endif
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

}  // namespace

torch::Tensor elementwise(int64_t op, int64_t n, torch::Tensor a, torch::Tensor b, torch::Tensor c, double s) {
    check_vec_tensor(a, n, "a");
    const int64_t m = a.size(0);
    if (!b.defined()) b = torch::zeros_like(a);
    if (!c.defined()) c = torch::zeros_like(a);
    check_vec_tensor(b, n, "b");
    check_vec_tensor(c, n, "c");
    TORCH_CHECK(b.size(0) == m && c.size(0) == m, "a, b, c must share M");

    auto out = torch::empty_like(a);
    TGNN_DISPATCH_N(n, ew_impl, static_cast<int>(op), out, a, b, c, s, m);
    return out;
}

torch::Tensor reduce(
    int64_t op, int64_t n, torch::Tensor a, torch::Tensor b, double acc_init, double w, bool use_double
) {
    check_vec_tensor(a, n, "a");
    const int64_t m = a.size(0);
    if (!b.defined()) b = torch::ones_like(a);
    check_vec_tensor(b, n, "b");
    TORCH_CHECK(b.size(0) == m, "a and b must share M");

    auto out = torch::empty({m}, a.options().dtype(use_double ? torch::kFloat64 : torch::kFloat32));
    TGNN_DISPATCH_N(n, red_impl, static_cast<int>(op), out, a, b, acc_init, w, use_double, m);
    return out;
}

torch::Tensor gatv2_dot_leaky_relu(int64_t n, torch::Tensor l, torch::Tensor r, torch::Tensor a, double ns) {
    check_vec_tensor(l, n, "l");
    check_vec_tensor(r, n, "r");
    check_vec_tensor(a, n, "a");
    const int64_t m = l.size(0);
    TORCH_CHECK(r.size(0) == m && a.size(0) == m, "l, r, a must share M");

    auto out = torch::empty({m}, l.options().dtype(torch::kFloat32));
    TGNN_DISPATCH_N(n, gatv2_dot_impl, out, l, r, a, ns, m);
    return out;
}

py::dict op_codes() {
    py::dict d;
#define TGNN_EW(name, code) d[name] = static_cast<int>(code)
    TGNN_EW("neg_", OP_NEG);
    TGNN_EW("log_", OP_LOG);
    TGNN_EW("exp_", OP_EXP);
    TGNN_EW("relu_", OP_RELU);
    TGNN_EW("scalar_mul_", OP_SCALAR_MUL);
    TGNN_EW("leaky_relu_", OP_LEAKY_RELU);
    TGNN_EW("leaky_relu_backward_", OP_LEAKY_RELU_BWD);
    TGNN_EW("add_", OP_ADD_I);
    TGNN_EW("add_dst", OP_ADD_D);
    TGNN_EW("add_ret", OP_ADD_R);
    TGNN_EW("sub_", OP_SUB_I);
    TGNN_EW("sub_dst", OP_SUB_D);
    TGNN_EW("sub_ret", OP_SUB_R);
    TGNN_EW("mul_", OP_MUL_I);
    TGNN_EW("mul_dst", OP_MUL_D);
    TGNN_EW("mul_ret", OP_MUL_R);
    TGNN_EW("div_", OP_DIV_I);
    TGNN_EW("div_dst", OP_DIV_D);
    TGNN_EW("div_ret", OP_DIV_R);
    TGNN_EW("minimum_", OP_MIN_I);
    TGNN_EW("minimum_dst", OP_MIN_D);
    TGNN_EW("minimum_ret", OP_MIN_R);
    TGNN_EW("maximum_", OP_MAX_I);
    TGNN_EW("maximum_dst", OP_MAX_D);
    TGNN_EW("maximum_ret", OP_MAX_R);
    TGNN_EW("fmam_", OP_FMAM_I);
    TGNN_EW("fmaa_", OP_FMAA_I);
    TGNN_EW("fma_dst", OP_FMA_D);
    TGNN_EW("fma_ret", OP_FMA_R);
#undef TGNN_EW
    return d;
}

py::dict red_codes() {
    py::dict d;
    d["sum_acc"]          = static_cast<int>(OP_SUM_ACC);
    d["sum_ret"]          = static_cast<int>(OP_SUM_RET);
    d["weighted_sum_acc"] = static_cast<int>(OP_WSUM_ACC);
    d["weighted_sum_ret"] = static_cast<int>(OP_WSUM_RET);
    d["prod_acc"]         = static_cast<int>(OP_PROD_ACC);
    d["prod_ret"]         = static_cast<int>(OP_PROD_RET);
    d["min_acc"]          = static_cast<int>(OP_RMIN_ACC);
    d["min_ret"]          = static_cast<int>(OP_RMIN_RET);
    d["max_acc"]          = static_cast<int>(OP_RMAX_ACC);
    d["max_ret"]          = static_cast<int>(OP_RMAX_RET);
    d["dot_product_acc"]  = static_cast<int>(OP_DOT_ACC);
    d["dot_product_ret"]  = static_cast<int>(OP_DOT_RET);
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise", &elementwise);
    m.def("reduce", &reduce);
    m.def("gatv2_dot_leaky_relu", &gatv2_dot_leaky_relu);
    m.def("op_codes", &op_codes);
    m.def("red_codes", &red_codes);
    m.attr("dtype_name") = kDtypeName;
    m.attr("on_host")    = kOnHost;
    m.attr("max_n")      = static_cast<int64_t>(kMaxN);
}

#endif  // TGNN_GROUP == 1

// ===========================================================================
// group "cvt": convert_vec, write_row, TileOps::read, atomic_add_scaled_f32
// ===========================================================================
#if TGNN_GROUP == 2

namespace {

template <size_t N, typename dst_t>
TGNN_WORKER void apply_convert(Vec<N, dst_t>* dst, const Vec<N, num_t>* src) {
    convert_vec<N, dst_t, num_t>(dst, src);
}

template <size_t N>
TGNN_WORKER void apply_read(Vec<N, num_t>* out, const num_t* arr, size_t vec_idx) {
    *out = TileOps<N, num_t, float>::read(arr, vec_idx);
}

#if !TGNN_HOST
template <size_t N, typename dst_t>
__global__ void k_convert(Vec<N, dst_t>* dst, const Vec<N, num_t>* src, int64_t m) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) apply_convert<N, dst_t>(dst + i, src + i);
}

template <size_t N>
__global__ void k_read(Vec<N, num_t>* out, const num_t* arr, int64_t start, int64_t m) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) apply_read<N>(out + i, arr, static_cast<size_t>(start + i));
}

// Every row targets the same vec_idx, so the atomicAdd is genuinely contended.
template <size_t N>
__global__ void k_atomic(float* ptr, int64_t vec_idx, float scalar, const Vec<N, num_t>* v, int64_t m) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) TileOps<N, num_t, float>::atomic_add_scaled_f32(ptr, static_cast<size_t>(vec_idx), scalar, v[i]);
}

template <typename dst_t, size_t row_width, size_t worker_cnt>
__global__ void k_write_row(dst_t* dst, const num_t* src) {
    write_row<dst_t, num_t, row_width, worker_cnt>(dst, threadIdx.x, src);
}
#endif  // !TGNN_HOST

template <size_t N, typename dst_t>
void convert_run(const torch::Tensor& out, const torch::Tensor& src, int64_t m) {
    if constexpr (vec_fits<N, num_t> && vec_fits<N, dst_t>) {
        auto* d = reinterpret_cast<Vec<N, dst_t>*>(out.data_ptr());
#if TGNN_HOST
        for (int64_t i = 0; i < m; ++i) apply_convert<N, dst_t>(d + i, cvec_ptr<N>(src) + i);
#else
        k_convert<N, dst_t><<<grid_for(m, kBlock), kBlock>>>(d, cvec_ptr<N>(src), m);
        C10_CUDA_CHECK(cudaGetLastError());
#endif
    } else {
        TORCH_CHECK(
            false, "convert_vec<", N, ", dst, ", kDtypeName, "> is not instantiable: one of the two Vec<", N,
            ", ...> types exceeds the 16-byte cap"
        );
    }
}

template <size_t N>
void convert_impl(int64_t dst_code, const torch::Tensor& out, const torch::Tensor& src, int64_t m) {
    if (dst_code == 0) {
        convert_run<N, float>(out, src, m);
    } else if (dst_code == 1) {
        convert_run<N, half>(out, src, m);
    } else {
        convert_run<N, nv_bfloat16>(out, src, m);
    }
}

template <size_t N>
void read_impl(const torch::Tensor& out, const torch::Tensor& arr, int64_t start, int64_t m) {
    if constexpr (vec_fits<N, num_t>) {
        const auto* base = reinterpret_cast<const num_t*>(arr.data_ptr());
#if TGNN_HOST
        for (int64_t i = 0; i < m; ++i) apply_read<N>(vec_ptr<N>(out) + i, base, static_cast<size_t>(start + i));
#else
        k_read<N><<<grid_for(m, kBlock), kBlock>>>(vec_ptr<N>(out), base, start, m);
        C10_CUDA_CHECK(cudaGetLastError());
#endif
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

#if !TGNN_HOST
template <size_t N>
void atomic_impl(const torch::Tensor& ptr, int64_t vec_idx, double scalar, const torch::Tensor& v, int64_t m) {
    // atomic_add_scaled_f32 stages through Vec<N, float>, so *that* must fit, not just
    // Vec<N, num_t>. For the 16-bit types at N=8 it cannot be instantiated -- and
    // SelectTW<256, half> == 8 makes that a reachable configuration, not a hypothetical.
    if constexpr (vec_fits<N, num_t> && vec_fits<N, float>) {
        k_atomic<N><<<grid_for(m, kBlock), kBlock>>>(
            ptr.data_ptr<float>(), vec_idx, static_cast<float>(scalar), cvec_ptr<N>(v), m
        );
        C10_CUDA_CHECK(cudaGetLastError());
    } else {
        TORCH_CHECK(
            false, "atomic_add_scaled_f32 is not instantiable for N=", N, " with ", kDtypeName,
            ": it stages through Vec<", N, ", float>, which exceeds the 16-byte cap"
        );
    }
}
#endif

}  // namespace

torch::Tensor convert(int64_t n, torch::Tensor src, int64_t dst_code) {
    check_vec_tensor(src, n, "src");
    const int64_t m = src.size(0);
    auto dst_dtype  = dst_code == 0 ? torch::kFloat32 : (dst_code == 1 ? torch::kFloat16 : torch::kBFloat16);
    auto out        = torch::empty({m, n}, src.options().dtype(dst_dtype));
    TGNN_DISPATCH_N(n, convert_impl, dst_code, out, src, m);
    return out;
}

torch::Tensor tile_read(int64_t n, torch::Tensor arr, int64_t start, int64_t m) {
    TORCH_CHECK(arr.is_contiguous() && arr.scalar_type() == kTorchDtype, "arr: contiguous ", kDtypeName, " expected");
    TORCH_CHECK(arr.is_cpu() == kOnHost, "arr is on the wrong device");
    TORCH_CHECK((start + m) * n <= arr.numel(), "arr too small for the requested vec_idx range");
    auto out = torch::empty({m, n}, arr.options());
    TGNN_DISPATCH_N(n, read_impl, out, arr, start, m);
    return out;
}

#if !TGNN_HOST
torch::Tensor atomic_add_scaled_f32(int64_t n, torch::Tensor ptr, int64_t vec_idx, double scalar, torch::Tensor v) {
    check_vec_tensor(v, n, "v");
    TORCH_CHECK(ptr.is_contiguous() && ptr.scalar_type() == torch::kFloat32 && ptr.is_cuda(), "ptr: cuda fp32");
    TORCH_CHECK(vec_idx + n <= ptr.numel(), "ptr too small");
    TGNN_DISPATCH_N(n, atomic_impl, ptr, vec_idx, scalar, v, v.size(0));
    return ptr;
}

// One block of worker_cnt threads: write_row ends in __syncthreads(), so every worker
// must reach it. dst is over-allocated by the caller so overruns are detectable.
torch::Tensor write_row_run(
    int64_t row_width, int64_t worker_cnt, torch::Tensor src, int64_t dst_code, int64_t dst_len
) {
    TORCH_CHECK(src.is_contiguous() && src.scalar_type() == kTorchDtype && src.is_cuda(), "src: cuda ", kDtypeName);
    auto dst_dtype = dst_code == 0 ? torch::kFloat32 : (dst_code == 1 ? torch::kFloat16 : torch::kBFloat16);
    auto out       = torch::zeros({dst_len}, src.options().dtype(dst_dtype));
    const auto* s  = reinterpret_cast<const num_t*>(src.data_ptr());

    // row_width=36 is deliberately not a multiple of copy_N, to probe the tail guard.
#define TGNN_WR_CASE(dst_t, RW, WC)                                                         \
    if (row_width == RW && worker_cnt == WC) {                                              \
        k_write_row<dst_t, RW, WC><<<1, WC>>>(reinterpret_cast<dst_t*>(out.data_ptr()), s);  \
        C10_CUDA_CHECK(cudaGetLastError());                                                 \
        return out;                                                                         \
    }

#define TGNN_WR_GRID(dst_t)       \
    TGNN_WR_CASE(dst_t, 32, 32)   \
    TGNN_WR_CASE(dst_t, 64, 32)   \
    TGNN_WR_CASE(dst_t, 128, 32)  \
    TGNN_WR_CASE(dst_t, 256, 32)  \
    TGNN_WR_CASE(dst_t, 128, 64)  \
    TGNN_WR_CASE(dst_t, 256, 128) \
    TGNN_WR_CASE(dst_t, 36, 32)

    if (dst_code == 0) {
        TGNN_WR_GRID(float)
    } else if (dst_code == 1) {
        TGNN_WR_GRID(half)
    } else {
        TGNN_WR_GRID(nv_bfloat16)
    }
#undef TGNN_WR_GRID
#undef TGNN_WR_CASE

    TORCH_CHECK(false, "write_row not instantiated for row_width=", row_width, " worker_cnt=", worker_cnt);
    return out;  // unreachable
}
#endif  // !TGNN_HOST

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convert", &convert);
    m.def("tile_read", &tile_read);
#if !TGNN_HOST
    m.def("atomic_add_scaled_f32", &atomic_add_scaled_f32);
    m.def("write_row_run", &write_row_run);
#endif
    m.attr("dtype_name") = kDtypeName;
    m.attr("on_host")    = kOnHost;
    m.attr("max_n")      = static_cast<int64_t>(kMaxN);
}

#endif  // TGNN_GROUP == 2

// ===========================================================================
// group "grad": make_ns, gatv2_accum_grad_al, gatv2_accum_grad_r
// (device-only: make_ns is __device__, and the accumulators are global memory)
// ===========================================================================
#if TGNN_GROUP == 3

namespace {

template <size_t N>
__global__ void k_make_ns(float* out, float ns) {
    auto v = TileOps<N, num_t, float>::make_ns(ns);
    // ns_t is declared as a scalar; read it back through num_t to compare.
    out[0] = static_cast<float>(*reinterpret_cast<const num_t*>(&v));
}

template <size_t N>
void make_ns_impl(const torch::Tensor& out, double ns) {
    if constexpr (vec_fits<N, num_t>) {
        k_make_ns<N><<<1, 1>>>(out.data_ptr<float>(), static_cast<float>(ns));
        C10_CUDA_CHECK(cudaGetLastError());
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

template <size_t N>
__global__ void k_grad_al(
    float* ga, float* gl, const float* ge, const Vec<N, num_t>* l, const Vec<N, num_t>* r, const Vec<N, num_t>* a,
    float ns, int64_t m
) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) TileOps<N, num_t, float>::gatv2_accum_grad_al(ga + i * N, gl + i * N, ge[i], l[i], r[i], a[i], ns);
}

template <size_t N>
void grad_al_impl(
    const torch::Tensor& ga, const torch::Tensor& gl, const torch::Tensor& ge, const torch::Tensor& l,
    const torch::Tensor& r, const torch::Tensor& a, double ns, int64_t m
) {
    if constexpr (vec_fits<N, num_t>) {
        k_grad_al<N><<<grid_for(m, kBlock), kBlock>>>(
            ga.data_ptr<float>(), gl.data_ptr<float>(), ge.data_ptr<float>(), cvec_ptr<N>(l), cvec_ptr<N>(r),
            cvec_ptr<N>(a), static_cast<float>(ns), m
        );
        C10_CUDA_CHECK(cudaGetLastError());
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

template <size_t N>
__global__ void k_grad_r(
    float* gr, const float* alpha, const Vec<N, num_t>* gh, const float* ge, const Vec<N, num_t>* l,
    const Vec<N, num_t>* r, const Vec<N, num_t>* a, float ns, int64_t m
) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < m) {
        TileOps<N, num_t, float>::gatv2_accum_grad_r(gr + i * N, alpha[i], gh[i], ge[i], l[i], r[i], a[i], ns);
    }
}

template <size_t N>
void grad_r_impl(
    const torch::Tensor& gr, const torch::Tensor& alpha, const torch::Tensor& gh, const torch::Tensor& ge,
    const torch::Tensor& l, const torch::Tensor& r, const torch::Tensor& a, double ns, int64_t m
) {
    if constexpr (vec_fits<N, num_t>) {
        k_grad_r<N><<<grid_for(m, kBlock), kBlock>>>(
            gr.data_ptr<float>(), alpha.data_ptr<float>(), cvec_ptr<N>(gh), ge.data_ptr<float>(), cvec_ptr<N>(l),
            cvec_ptr<N>(r), cvec_ptr<N>(a), static_cast<float>(ns), m
        );
        C10_CUDA_CHECK(cudaGetLastError());
    } else {
        TGNN_N_TOO_WIDE(N);
    }
}

}  // namespace

double make_ns(int64_t n, double ns) {
    auto out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    TGNN_DISPATCH_N(n, make_ns_impl, out, ns);
    return out.cpu().item<double>();
}

std::vector<torch::Tensor> gatv2_accum_grad_al(
    int64_t n, torch::Tensor ga, torch::Tensor gl, torch::Tensor ge, torch::Tensor l, torch::Tensor r,
    torch::Tensor a, double ns
) {
    check_vec_tensor(l, n, "l");
    check_vec_tensor(r, n, "r");
    check_vec_tensor(a, n, "a");
    const int64_t m = l.size(0);
    TORCH_CHECK(ga.sizes() == gl.sizes() && ga.size(0) == m && ga.size(1) == n, "ga/gl must be [M, N]");
    TORCH_CHECK(ga.scalar_type() == torch::kFloat32 && gl.scalar_type() == torch::kFloat32, "ga/gl must be fp32");
    TGNN_DISPATCH_N(n, grad_al_impl, ga, gl, ge, l, r, a, ns, m);
    return {ga, gl};
}

torch::Tensor gatv2_accum_grad_r(
    int64_t n, torch::Tensor gr, torch::Tensor alpha, torch::Tensor gh, torch::Tensor ge, torch::Tensor l,
    torch::Tensor r, torch::Tensor a, double ns
) {
    check_vec_tensor(l, n, "l");
    check_vec_tensor(r, n, "r");
    check_vec_tensor(a, n, "a");
    check_vec_tensor(gh, n, "gh");
    const int64_t m = l.size(0);
    TORCH_CHECK(gr.size(0) == m && gr.size(1) == n && gr.scalar_type() == torch::kFloat32, "gr must be fp32 [M, N]");
    TGNN_DISPATCH_N(n, grad_r_impl, gr, alpha, gh, ge, l, r, a, ns, m);
    return gr;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_ns", &make_ns);
    m.def("gatv2_accum_grad_al", &gatv2_accum_grad_al);
    m.def("gatv2_accum_grad_r", &gatv2_accum_grad_r);
    m.attr("dtype_name") = kDtypeName;
    m.attr("on_host")    = kOnHost;
    m.attr("max_n")      = static_cast<int64_t>(kMaxN);
}

#endif  // TGNN_GROUP == 3
