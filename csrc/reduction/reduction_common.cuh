#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "common.cuh"
#include "common/row_shape.cuh"

// ================================================================================================
// Neighbourhood reduction (reduction_aggr) for any feature dim D:
//
//     out[v, f] = reduce over the edges e of CSR row v of X[col_idx[e], f],   reduce in {min, max, sum}
//
// Min and max also record arg[v, f], the source node of the edge that attained the extremum -- of
// equal candidates, the first in edge order -- for the backward, which routes each output's
// gradient to that one source. A row with no candidate (no edges, or NaN at every edge) gives 0 and
// an invalid arg; NaN is never selected. Sums accumulate in fp32 and track no arg.
//
// Definitions shared by the forward (reduction_forward.cuh) and the backward
// (reduction_backward.cuh).
// ================================================================================================

namespace turbo_gnn::reduction {

template <int CODE>
using Shape = rows::Shape<CODE>;

// "No edge yet": the position of a running min/max that has taken no candidate.
inline constexpr uint32_t kNoPos = 0xffffffffu;

// The comparison reducers. Values are compared as fp32, which is exact for every input dtype.
template <ReductionOp OP>
struct Extremum {
    static_assert(OP == ReductionOp::MIN || OP == ReductionOp::MAX, "min and max only");
    static constexpr float kIdentity = OP == ReductionOp::MIN ? INFINITY : -INFINITY;

    // Whether x replaces the incumbent on a walk that meets the edges last to first: equal values
    // do, so the earliest edge stays; NaN never does.
    static __device__ __forceinline__ bool takes(float x, float best) { return OP == ReductionOp::MIN ? x <= best : x >= best; }
};

template <ReductionOp OP>
inline constexpr bool kTracksArg = OP != ReductionOp::SUM;

// ------------------------------------------------------------------------------------------------
// Packed candidates. The slices of a heavy row fold their min/max into it with one 64-bit
// atomicMin/atomicMax per element, on a key whose order is the candidates' order:
//
//   bits 63..32  the value as an order-preserving unsigned (negatives bit-inverted, the sign bit set
//                on the rest), every zero and subnormal mapped to +0's -- the kernels compare with
//                flush-to-zero, so those are all equal there too
//   bits 31..0   the edge's position in the row; for max, inverted, so that atomicMax prefers the
//                earlier edge of two equal values as atomicMin does
//
// The key of each candidate is unique, so the result does not depend on the order the atomics land
// in. A zero result is re-read from X, which restores -0 and subnormals.
// ------------------------------------------------------------------------------------------------
template <ReductionOp OP>
inline constexpr unsigned long long kPackedEmpty = OP == ReductionOp::MIN ? ~0ull : 0ull;

template <ReductionOp OP>
__device__ __forceinline__ unsigned long long pack_candidate(float v, uint32_t pos) {
    uint32_t bits = __float_as_uint(v);
    if ((bits & 0x7f800000u) == 0) {
        bits = 0;  // +-0 and subnormals compare equal to +0 under flush-to-zero
    }
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    const uint32_t low     = OP == ReductionOp::MIN ? pos : ~pos;
    return (static_cast<unsigned long long>(ordered) << 32) | low;
}

// The value and position of a key other than kPackedEmpty<OP>.
template <ReductionOp OP>
__device__ __forceinline__ float unpack_value(unsigned long long key) {
    const uint32_t ordered = static_cast<uint32_t>(key >> 32);
    return __uint_as_float((ordered & 0x80000000u) ? (ordered & 0x7fffffffu) : ~ordered);
}

template <ReductionOp OP>
__device__ __forceinline__ uint32_t unpack_pos(unsigned long long key) {
    const uint32_t low = static_cast<uint32_t>(key);
    return OP == ReductionOp::MIN ? low : ~low;
}

template <ReductionOp OP>
__device__ __forceinline__ void fold_candidate(unsigned long long *const addr, unsigned long long key) {
    if constexpr (OP == ReductionOp::MIN) {
        atomicMin(addr, key);
    } else {
        atomicMax(addr, key);
    }
}

// VEC 4- or 8-byte values to dst, in 16-byte pieces where they are that wide. dst must be aligned
// to min(16, VEC * sizeof(T)) bytes and vals to 16.
template <size_t VEC, typename T>
__device__ __forceinline__ void store_words(T *const __restrict__ dst, T const (&vals)[VEC]) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "4- or 8-byte values");
    constexpr size_t kBytes = VEC * sizeof(T);
    if constexpr (kBytes >= 16) {
#pragma unroll
        for (size_t i = 0; i < kBytes / 16; ++i) {
            reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<uint4 const *>(vals)[i];
        }
    } else {
#pragma unroll
        for (size_t i = 0; i < VEC; ++i) {
            dst[i] = vals[i];
        }
    }
}

// The inverse of store_words.
template <size_t VEC, typename T>
__device__ __forceinline__ void load_words(T (&vals)[VEC], T const *const __restrict__ src) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "4- or 8-byte values");
    constexpr size_t kBytes = VEC * sizeof(T);
    if constexpr (kBytes >= 16) {
#pragma unroll
        for (size_t i = 0; i < kBytes / 16; ++i) {
            reinterpret_cast<uint4 *>(vals)[i] = reinterpret_cast<uint4 const *>(src)[i];
        }
    } else {
#pragma unroll
        for (size_t i = 0; i < VEC; ++i) {
            vals[i] = src[i];
        }
    }
}

template <FloatingNum cuda_t, IntegralNum index_t>
struct ForwardParams {
    cuda_t const *X;   // [N_src, D] rows, unit stride over D
    int64_t stride_x;  // row stride of X, in elements
    uint32_t D;
    uint32_t nvec;    // vectors per row
    uint32_t passes;  // passes per row: 1 unless the row is longer than one pass of the shape

    index_t const *row_ptr;
    index_t const *col_idx;

    // Work items, in launch order: num_slices slices of heavy rows, then num_light light rows, each
    // times `passes`.
    int64_t num_slices;
    int64_t num_light;
    index_t const *light_nodes;
    index_t const *heavy_nodes;
    int const *chunk_node;   // [num_slices]: the slice's slot in heavy_nodes
    int const *chunk_start;  // [num_slices]: its first edge, relative to the row
    uint32_t slice_size;

    cuda_t *out;                 // [N_dst, D], light rows
    index_t *arg;                // [N_dst, D], light rows; min/max only
    float *part;                 // [num_slices, D]: per-slice sums; sum only
    unsigned long long *packed;  // [num_heavy, D]: the heavy rows' packed candidates; min/max only
};

}  // namespace turbo_gnn::reduction
