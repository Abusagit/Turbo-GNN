#pragma once

#include "common/misc.cuh"


// =============================================================================
// TileOps<VW, cuda_t> — vectorized load/compute/store traits for forward kernel
// =============================================================================

union alignas(2) Packed16bit {
    uint8_t _b8[2];
    uint16_t _b16;
};

union alignas(4) Packed32bit {
    uint8_t _b8[4];
    uint16_t _b16[2];
    uint32_t _b32;
};

union alignas(8) Packed64bit {
    uint8_t _b8[8];
    uint16_t _b16[4];
    uint32_t _b32[2];
    uint64_t _b64;
};

union alignas(16) Packed128bit {
    uint8_t _b8[16];
    uint16_t _b16[8];
    uint32_t _b32[4];
    uint64_t _b64[2];
    uint4 _b128;
};


// Vec2 instructions

template<typename num_type>
struct Vec2 {
    static_assert(sizeof(num_type) <= 8, "Vec2 is for types, that are no larger than 64 bit.");
    num_type x, y;
};

template <typename num_type>
__device__ __forceinline__ Vec2<num_type> load_vec2(num_type const *const __restrict__ ptr) {
    return *reinterpret_cast<Vec2<num_type> const *>(ptr);
}

template <typename num_type>
__device__ __forceinline__ void store_vec2(num_type *const __restrict__ ptr, Vec2<num_type> val) {
    *reinterpret_cast<Vec2<num_type> *>(ptr) = val;
}

// Vec2Ops: type-generic packed operations for 16-bit types

template <typename cuda_t>
struct Vec2Ops;

template <>
struct Vec2Ops<float> {
    using vec2_t = float2;
    static __device__ __forceinline__ vec2_t get_zero() { return {0.0f, 0.0f}; }
    static __device__ __forceinline__ vec2_t add(vec2_t a, vec2_t b) { return {a.x + b.x, a.y + b.y}; }
    static __device__ __forceinline__ vec2_t mul(vec2_t a, vec2_t b) { return {a.x * b.x, a.y * b.y}; }
    static __device__ __forceinline__ vec2_t max2(vec2_t a, vec2_t b) { return {fmaxf(a.x, b.x), fmaxf(a.y, b.y)}; }
    static __device__ __forceinline__ vec2_t min2(vec2_t a, vec2_t b) { return {fminf(a.x, b.x), fminf(a.y, b.y)}; }
    static __device__ __forceinline__ float2 to_float2(vec2_t v) { return v; }
    static __device__ __forceinline__ vec2_t from_float2(float2 v) { return v; }
    static __device__ __forceinline__ vec2_t fma(vec2_t a, vec2_t b, vec2_t c) { return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)}; }
    static __device__ __forceinline__ vec2_t leaky_relu(vec2_t x, vec2_t neg_slope) {
        vec2_t z = get_zero();
        return add(max2(x, z), mul(neg_slope, min2(x, z)));
    }
};

template <>
struct Vec2Ops<__half> {
    using vec2_t = __half2;
    static __device__ __forceinline__ vec2_t get_zero() { return __float2half2_rn(0.0f); }
    static __device__ __forceinline__ vec2_t add(vec2_t a, vec2_t b) { return __hadd2(a, b); }
    static __device__ __forceinline__ vec2_t mul(vec2_t a, vec2_t b) { return __hmul2(a, b); }
    static __device__ __forceinline__ vec2_t from_float(float v) { return __float2half2_rn(v); }
    static __device__ __forceinline__ vec2_t max2(vec2_t a, vec2_t b) { return __hmax2(a, b); }
    static __device__ __forceinline__ vec2_t min2(vec2_t a, vec2_t b) { return __hmin2(a, b); }
    static __device__ __forceinline__ float2 to_float2(vec2_t v) { return __half22float2(v); }
    static __device__ __forceinline__ vec2_t from_float2(float2 v) { return __float22half2_rn(v); }
    static __device__ __forceinline__ vec2_t fma(vec2_t a, vec2_t b, vec2_t c) { return __hfma2(a, b, c); }
    static __device__ __forceinline__ vec2_t leaky_relu(vec2_t x, vec2_t neg_slope) {
        vec2_t z = get_zero();
        return __hadd2(__hmax2(x, z), __hmul2(neg_slope, __hmin2(x, z)));
    }
};

template <>
struct Vec2Ops<__nv_bfloat16> {
    using vec2_t = __nv_bfloat162;
    static __device__ __forceinline__ vec2_t get_zero() { return __float2bfloat162_rn(0.0f); }
    static __device__ __forceinline__ vec2_t add(vec2_t a, vec2_t b) { return __hadd2(a, b); }
    static __device__ __forceinline__ vec2_t mul(vec2_t a, vec2_t b) { return __hmul2(a, b); }
    static __device__ __forceinline__ vec2_t from_float(float v) { return __float2bfloat162_rn(v); }
    static __device__ __forceinline__ vec2_t max2(vec2_t a, vec2_t b) { return __hmax2(a, b); }
    static __device__ __forceinline__ vec2_t min2(vec2_t a, vec2_t b) { return __hmin2(a, b); }
    static __device__ __forceinline__ float2 to_float2(vec2_t v) { return __bfloat1622float2(v); }
    static __device__ __forceinline__ vec2_t from_float2(float2 v) { return __float22bfloat162_rn(v); }
    static __device__ __forceinline__ vec2_t fma(vec2_t a, vec2_t b, vec2_t c) { return __hfma2(a, b, c); }
    static __device__ __forceinline__ vec2_t leaky_relu(vec2_t x, vec2_t neg_slope) {
        vec2_t z = get_zero();
        return __hadd2(__hmax2(x, z), __hmul2(neg_slope, __hmin2(x, z)));
    }
};

// Vec8: 128-bit load/store for 16-bit types = 8 scalars = 4 vec2 pairs

template<typename cuda_t>
struct Vec8 {
    static_assert(sizeof(cuda_t) <= 2, "Vec8 only for 16-bit types or smaller");
};

template <typename cuda_t>
requires(sizeof(cuda_t) <= 2)
struct alignas(16) Vec8<cuda_t> {
    using vec2_t = typename Vec2Ops<cuda_t>::vec2_t;
    vec2_t v[4];
};

template <typename cuda_t>
requires(sizeof(cuda_t) == 1)
struct alignas(8) Vec8<cuda_t> {
    using vec2_t = typename Vec2Ops<cuda_t>::vec2_t;
    vec2_t v[4];
};

template <typename cuda_t>
__device__ __forceinline__ Vec8<cuda_t> load_vec8(cuda_t const *const __restrict__ ptr) {
    return *reinterpret_cast<Vec8<cuda_t> const *>(ptr);
}

template <typename cuda_t>
__device__ __forceinline__ void store_vec8(cuda_t *const __restrict__ ptr, const Vec8<cuda_t>& val) {
    *reinterpret_cast<uint4 *>(ptr) = *reinterpret_cast<uint4 const *>(&val);
}

// template (undefined)
template <int tile_width, typename num_type>
struct TileOps;

// template <typename num_type>
//     requires(sizeof(num_type) == 2)
// struct TileOps<1, num_type> {
//     using vec_t = num_type;

//     static __device__ __forceinline__ vec_t load(float const *const __restrict__ ptr, int vec_idx) { return ptr[vec_idx]; }
// };

// --- VW=1, float: scalar loads ---
template <typename T>
struct TileOps<1, T> {
    static_assert(sizeof(T) <= 16);
    using vec_t                       = T;
    using ns_t                        = T;
    static constexpr int TW = 1;

    static __device__ __forceinline__ vec_t load(float const *const __restrict__ ptr, int vec_idx) { return ptr[vec_idx]; }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return ns; }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        float s = leaky_relu_elementwise(l + r, ns);
        return s * a;
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) { return a * b; }
    static __device__ __forceinline__ void weighted_accum(float *const __restrict__ acc, float w, vec_t r) { acc[0] = fmaf(w, r, acc[0]); }
    static __device__ __forceinline__ void gatv2_accum_grad_al(
        float *const __restrict__ ga, float *const __restrict__ gl, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
        float edge = l + r;
        float tder = leaky_relu_der_elementwise(edge, ns);
        float t_ij = tder * edge;
        ga[0]      = fmaf(ge, t_ij, ga[0]);
        gl[0]      = fmaf(ge * tder, a, gl[0]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(
        float *const __restrict__ gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
        float edge = l + r;
        float tder = leaky_relu_der_elementwise(edge, ns);
        gr[0]      = fmaf(alpha, gh, gr[0]);
        gr[0]      = fmaf(ge * tder, a, gr[0]);
    }
    static __device__ __forceinline__ void write(
        float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc, float inv_sum
    ) {
        out[vec_idx] = acc[0] * inv_sum;
    }
    static __device__ __forceinline__ void write_typed(float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        out[vec_idx] = acc[0];
    }
    static __device__ __forceinline__ void write_float(float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        out[vec_idx] = acc[0];
    }
    static __device__ __forceinline__ void write_zero(float *out, int vec_idx) { out[vec_idx] = 0.0f; }

    // --- generic element access ---
    static __device__ __forceinline__ float extract(vec_t v, int /*i*/) { return v; }
    static __device__ __forceinline__ float extract_float(vec_t v, int /*i*/) { return v; }
    static __device__ __forceinline__ void store_vec(float *ptr, int vec_idx, vec_t v) { ptr[vec_idx] = v; }
    static __device__ __forceinline__ vec_t build(float const *const __restrict__ arr) { return arr[0]; }
    static __device__ __forceinline__ vec_t build_from_float(float const *const __restrict__ arr) { return arr[0]; }

    // --- GT backward: float32 atomic add of scalar * vec ---
    static __device__ __forceinline__ void atomic_add_scaled_f32(float *const __restrict__ ptr, int base_f, float scalar, vec_t v) {
        atomicAdd(&ptr[base_f], scalar * v);
    }
};

// --- VW=4, float: float4 loads ---
template <>
struct TileOps<4, float> {
    using vec_t                       = float4;
    using ns_t                        = float;
    static constexpr int TW = 4;

    static __device__ __forceinline__ vec_t load(float const *const __restrict__ ptr, int vec_idx) {
        return reinterpret_cast<float4 const *>(ptr)[vec_idx];
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return ns; }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        float4 sum = make_float4(
            leaky_relu_elementwise(l.x + r.x, ns),
            leaky_relu_elementwise(l.y + r.y, ns),
            leaky_relu_elementwise(l.z + r.z, ns),
            leaky_relu_elementwise(l.w + r.w, ns)
        );
        return dot_product_f4(sum, a);
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) { return dot_product_f4(a, b); }
    static __device__ __forceinline__ void weighted_accum(float *const __restrict__ acc, float w, vec_t r) {
        acc[0] = fmaf(w, r.x, acc[0]);
        acc[1] = fmaf(w, r.y, acc[1]);
        acc[2] = fmaf(w, r.z, acc[2]);
        acc[3] = fmaf(w, r.w, acc[3]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(
        float *const __restrict__ ga, float *const __restrict__ gl, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
        float4 edge = f4_add(l, r);
        float4 tder = f4_leaky_relu_der(edge, ns);
        float4 t_ij = f4_mul(tder, edge);
        f4_fma(*reinterpret_cast<float4 *>(ga), ge, t_ij);
        f4_fma(*reinterpret_cast<float4 *>(gl), ge, f4_mul(tder, a));
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(
        float *const __restrict__ gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
        float4 edge = f4_add(l, r);
        float4 tder = f4_leaky_relu_der(edge, ns);
        f4_fma(*reinterpret_cast<float4 *>(gr), alpha, gh);
        f4_fma(*reinterpret_cast<float4 *>(gr), ge, f4_mul(tder, a));
    }
    static __device__ __forceinline__ void write(
        float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc, float inv_sum
    ) {
        reinterpret_cast<float4 *>(out)[vec_idx] = make_float4(acc[0] * inv_sum, acc[1] * inv_sum, acc[2] * inv_sum, acc[3] * inv_sum);
    }
    static __device__ __forceinline__ void write_typed(float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        reinterpret_cast<float4 *>(out)[vec_idx] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    static __device__ __forceinline__ void write_float(float *const __restrict__ out, int vec_idx, float const *__restrict__ acc) {
        reinterpret_cast<float4 *>(out)[vec_idx] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    static __device__ __forceinline__ void write_zero(float *const __restrict__ out, int vec_idx) {
        reinterpret_cast<float4 *>(out)[vec_idx] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // --- generic element access ---
    static __device__ __forceinline__ float extract(vec_t v, int i) { return (&v.x)[i]; }
    static __device__ __forceinline__ float extract_float(vec_t v, int i) { return (&v.x)[i]; }
    static __device__ __forceinline__ void store_vec(float *const __restrict__ ptr, int vec_idx, vec_t v) {
        reinterpret_cast<float4 *>(ptr)[vec_idx] = v;
    }
    static __device__ __forceinline__ vec_t build(float const *const __restrict__ arr) { return {arr[0], arr[1], arr[2], arr[3]}; }
    static __device__ __forceinline__ vec_t build_from_float(float const *const __restrict__ arr) { return {arr[0], arr[1], arr[2], arr[3]}; }

    // --- GT backward: float32 atomic add of scalar * vec ---
    static __device__ __forceinline__ void atomic_add_scaled_f32(float *const __restrict__ ptr, int base_f, float scalar, vec_t v) {
        atomicAdd(&ptr[base_f + 0], scalar * v.x);
        atomicAdd(&ptr[base_f + 1], scalar * v.y);
        atomicAdd(&ptr[base_f + 2], scalar * v.z);
        atomicAdd(&ptr[base_f + 3], scalar * v.w);
    }
};

// --- VW=2, half/bf16: vec2 loads ---
template <typename cuda_t>
struct TileOps<2, cuda_t> {
    using Ops                         = Vec2Ops<cuda_t>;
    using vec2_t                      = typename Ops::vec2_t;
    using vec_t                       = vec2_t;
    using ns_t                        = vec2_t;
    static constexpr int TW = 2;

    static __device__ __forceinline__ vec2_t get_zero() { return Ops::from_float(0.0f); }

    static __device__ __forceinline__ vec_t load(cuda_t const *const __restrict__ ptr, int vec_idx) {
        return *reinterpret_cast<const vec2_t *>(&ptr[vec_idx * TW]);
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return Ops::from_float(ns); }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        vec2_t sum  = Ops::add(l, r);
        vec2_t act  = Ops::leaky_relu(sum, ns);
        vec2_t prod = Ops::mul(act, a);
        float2 pf   = Ops::to_float2(prod);  // TODO maybe do it in vec2_t and than
                                             // cast to float after the summation?
        return pf.x + pf.y;
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) {
        vec2_t prod = Ops::mul(a, b);
        float2 pf   = Ops::to_float2(prod);
        return pf.x + pf.y;
    }
    static __device__ __forceinline__ void weighted_accum(float *const __restrict__ acc, float w, vec_t r) {
        // maybe cast weight to vec_t?
        float2 rf = Ops::to_float2(r);
        acc[0]    = fmaf(w, rf.x, acc[0]);
        acc[1]    = fmaf(w, rf.y, acc[1]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(
        float *const __restrict__ ga, float *const __restrict__ gl, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
        float2 lf   = Ops::to_float2(l);
        float2 rf   = Ops::to_float2(r);
        float2 af   = Ops::to_float2(a);
        float edge0 = lf.x + rf.x;
        float edge1 = lf.y + rf.y;
        float tder0 = leaky_relu_der_elementwise(edge0, ns);
        float tder1 = leaky_relu_der_elementwise(edge1, ns);
        ga[0]       = fmaf(ge, tder0 * edge0, ga[0]);
        ga[1]       = fmaf(ge, tder1 * edge1, ga[1]);
        gl[0]       = fmaf(ge * tder0, af.x, gl[0]);
        gl[1]       = fmaf(ge * tder1, af.y, gl[1]);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(
        float *const __restrict__ gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
        float2 lf   = Ops::to_float2(l);
        float2 rf   = Ops::to_float2(r);
        float2 af   = Ops::to_float2(a);
        float2 ghf  = Ops::to_float2(gh);
        float edge0 = lf.x + rf.x;
        float edge1 = lf.y + rf.y;
        float tder0 = leaky_relu_der_elementwise(edge0, ns);
        float tder1 = leaky_relu_der_elementwise(edge1, ns);
        gr[0]       = fmaf(alpha, ghf.x, gr[0]);
        gr[0]       = fmaf(ge * tder0, af.x, gr[0]);
        gr[1]       = fmaf(alpha, ghf.y, gr[1]);
        gr[1]       = fmaf(ge * tder1, af.y, gr[1]);
    }
    static __device__ __forceinline__ void write(
        cuda_t *const __restrict__ out, int vec_idx, float const *const __restrict__ acc, float inv_sum
    ) {
        float2 of                                      = make_float2(acc[0] * inv_sum, acc[1] * inv_sum);
        *reinterpret_cast<vec2_t *>(&out[vec_idx * 2]) = Ops::from_float2(of);
    }
    static __device__ __forceinline__ void write_typed(cuda_t *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        *reinterpret_cast<vec2_t *>(&out[vec_idx * 2]) = Ops::from_float2(make_float2(acc[0], acc[1]));
    }
    static __device__ __forceinline__ void write_float(float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        *reinterpret_cast<float2 *>(&out[vec_idx * 2]) = make_float2(acc[0], acc[1]);
    }
    static __device__ __forceinline__ void write_zero(cuda_t *const __restrict__ out, int vec_idx) {
        *reinterpret_cast<vec2_t *>(&out[vec_idx * 2]) = Ops::from_float(0.0f);
    }

    // --- generic element access ---
    static __device__ __forceinline__ cuda_t extract(vec_t v, int i) { return reinterpret_cast<cuda_t const *>(&v)[i]; }
    static __device__ __forceinline__ float extract_float(vec_t v, int i) { return cuda_to_float(reinterpret_cast<cuda_t const *>(&v)[i]); }
    static __device__ __forceinline__ void store_vec(cuda_t *const __restrict__ ptr, int vec_idx, vec_t v) {
        *reinterpret_cast<vec2_t *>(&ptr[vec_idx * TW]) = v;
    }
    static __device__ __forceinline__ vec_t build(cuda_t const *const __restrict__ arr) { return *reinterpret_cast<vec2_t const *>(arr); }
    static __device__ __forceinline__ vec_t build_from_float(float const *const __restrict__ arr) {
        return Ops::from_float2(make_float2(arr[0], arr[1]));
    }

    // --- GT backward: float32 atomic add of scalar * vec ---
    static __device__ __forceinline__ void atomic_add_scaled_f32(float *const __restrict__ ptr, int base_f, float scalar, vec_t v) {
        float2 vf = Ops::to_float2(v);
        atomicAdd(&ptr[base_f], scalar * vf.x);
        atomicAdd(&ptr[base_f + 1], scalar * vf.y);
    }
};

// --- VW=8, half/bf16: Vec8 (128-bit) loads ---
template <typename cuda_t>
struct TileOps<8, cuda_t> {
    using Ops                         = Vec2Ops<cuda_t>;
    using vec2_t                      = typename Ops::vec2_t;
    using vec_t                       = Vec8<cuda_t>;
    using ns_t                        = vec2_t;

    static constexpr int TW = 8;
    static constexpr float4 zero_bits = {0.f, 0.f, 0.f, 0.f};

    static __device__ __forceinline__ vec_t load(cuda_t const *const __restrict__ ptr, int vec_idx) {
        return load_vec8(&ptr[vec_idx * TW]);
    }
    static __device__ __forceinline__ ns_t make_ns(float ns) { return Ops::from_float(ns); }

    static __device__ __forceinline__ float gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {
        float dot = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            vec2_t sum  = Ops::add(l.v[p], r.v[p]);
            vec2_t act  = Ops::leaky_relu(sum, ns);
            vec2_t prod = Ops::mul(act, a.v[p]);
            float2 pf   = Ops::to_float2(prod);
            dot += pf.x + pf.y;
        }
        return dot;
    }
    static __device__ __forceinline__ float dot_product(vec_t a, vec_t b) {
        float dot = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            vec2_t prod = Ops::mul(a.v[p], b.v[p]);
            float2 pf   = Ops::to_float2(prod);
            dot += pf.x + pf.y;
        }
        return dot;
    }
    static __device__ __forceinline__ void weighted_accum(float *const __restrict__ acc, float w, vec_t r) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 rf      = Ops::to_float2(r.v[p]);
            acc[p * 2]     = fmaf(w, rf.x, acc[p * 2]);
            acc[p * 2 + 1] = fmaf(w, rf.y, acc[p * 2 + 1]);
        }
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(
        float *const __restrict__ ga, float *const __restrict__ gl, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 lf     = Ops::to_float2(l.v[p]);
            float2 rf     = Ops::to_float2(r.v[p]);
            float2 af     = Ops::to_float2(a.v[p]);
            float edge0   = lf.x + rf.x;
            float edge1   = lf.y + rf.y;
            float tder0   = leaky_relu_der_elementwise(edge0, ns);
            float tder1   = leaky_relu_der_elementwise(edge1, ns);
            ga[p * 2]     = fmaf(ge, tder0 * edge0, ga[p * 2]);
            ga[p * 2 + 1] = fmaf(ge, tder1 * edge1, ga[p * 2 + 1]);
            gl[p * 2]     = fmaf(ge * tder0, af.x, gl[p * 2]);
            gl[p * 2 + 1] = fmaf(ge * tder1, af.y, gl[p * 2 + 1]);
        }
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(
        float *const __restrict__ gr, float alpha, vec_t gh, float ge, vec_t l, vec_t r, vec_t a, float ns
    ) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 lf     = Ops::to_float2(l.v[p]);
            float2 rf     = Ops::to_float2(r.v[p]);
            float2 af     = Ops::to_float2(a.v[p]);
            float2 ghf    = Ops::to_float2(gh.v[p]);
            float edge0   = lf.x + rf.x;
            float edge1   = lf.y + rf.y;
            float tder0   = leaky_relu_der_elementwise(edge0, ns);
            float tder1   = leaky_relu_der_elementwise(edge1, ns);
            gr[p * 2]     = fmaf(alpha, ghf.x, gr[p * 2]);
            gr[p * 2]     = fmaf(ge * tder0, af.x, gr[p * 2]);
            gr[p * 2 + 1] = fmaf(alpha, ghf.y, gr[p * 2 + 1]);
            gr[p * 2 + 1] = fmaf(ge * tder1, af.y, gr[p * 2 + 1]);
        }
    }
    static __device__ __forceinline__ void write(
        cuda_t *const __restrict__ out, int vec_idx, float const *const __restrict__ acc, float inv_sum
    ) {
        Vec8<cuda_t> out_v8;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            out_v8.v[p] = Ops::from_float2(make_float2(acc[p * 2] * inv_sum, acc[p * 2 + 1] * inv_sum));
        }
        store_vec8(&out[vec_idx * 8], out_v8);
    }
    static __device__ __forceinline__ void write_typed(cuda_t *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        Vec8<cuda_t> out_v8;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            out_v8.v[p] = Ops::from_float2(make_float2(acc[p * 2], acc[p * 2 + 1]));
        }
        store_vec8(&out[vec_idx * 8], out_v8);
    }
    static __device__ __forceinline__ void write_float(float *const __restrict__ out, int vec_idx, float const *const __restrict__ acc) {
        reinterpret_cast<float4 *>(&out[vec_idx * 8])[0] = reinterpret_cast<float4 const *>(acc)[0];
        reinterpret_cast<float4 *>(&out[vec_idx * 8])[1] = reinterpret_cast<float4 const *>(acc)[1];
    }
    static __device__ __forceinline__ void write_zero(cuda_t *const __restrict__ out, int vec_idx) {
        Vec8<cuda_t> zero_v;
#pragma unroll
        for (int p = 0; p < 4; ++p) zero_v.v[p] = Ops::get_zero();
        store_vec8(&out[vec_idx * 8], zero_v);
    }

    // --- generic element access ---
    static __device__ __forceinline__ cuda_t extract(vec_t v, int i) {
        int pair = i / 2, elem = i % 2;
        return reinterpret_cast<cuda_t const *>(&v.v[pair])[elem];
    }
    static __device__ __forceinline__ float extract_float(vec_t v, int i) {
        int pair = i / 2, elem = i % 2;
        return cuda_to_float(reinterpret_cast<cuda_t const *>(&v.v[pair])[elem]);
    }
    static __device__ __forceinline__ void store_vec(cuda_t *const __restrict__ ptr, int vec_idx, vec_t v) {
        store_vec8(&ptr[vec_idx * TW], v);
    }
    static __device__ __forceinline__ vec_t build(cuda_t const *const __restrict__ arr) {
        vec_t result;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            result.v[p] = *reinterpret_cast<vec2_t const *>(&arr[p * 2]);
        }
        return result;
    }
    static __device__ __forceinline__ vec_t build_from_float(float const *const __restrict__ arr) {
        vec_t result;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            result.v[p] = Ops::from_float2(make_float2(arr[p * 2], arr[p * 2 + 1]));
        }
        return result;
    }

    // --- GT backward: float32 atomic add of scalar * vec ---
    static __device__ __forceinline__ void atomic_add_scaled_f32(float *const __restrict__ ptr, int base_f, float scalar, vec_t v) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 vf = Ops::to_float2(v.v[p]);
            atomicAdd(&ptr[base_f + p * 2], scalar * vf.x);
            atomicAdd(&ptr[base_f + p * 2 + 1], scalar * vf.y);
        }
    }
};
