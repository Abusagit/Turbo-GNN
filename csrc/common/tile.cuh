#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

#include "common/adaptive_ops.cuh"
#include "common/misc.cuh"
#include "common/traits.cuh"

// ==================================================================================
// SelectTW: pick widest TW where all threads are working but not wider than 128 bits
// ==================================================================================

template <int D_CONST, typename cuda_t, int THREADS_PER_D = kWarpSize>
struct SelectTW {
   private:
    static consteval int calculate_tile_width(size_t type_size, size_t d, size_t thread_count) {
        size_t elems_per_thread = (d + thread_count - 1) / thread_count;

        return std::min(elems_per_thread, 16 / type_size);  // 16 bytes is the most wide load/store
    }

   public:
    static_assert(THREADS_PER_D == kWarpSize, "For now only whole warp count can operate on a single row.");
    static_assert(std::popcount(sizeof(cuda_t)) == 1, "Only types with size of power of 2 are supported.");

    static constexpr int threads_per_d = THREADS_PER_D;
    static constexpr int value         = calculate_tile_width(sizeof(cuda_t), D_CONST, THREADS_PER_D);
};

// Vec data struct

template <size_t N, typename num_type>
struct alignas(sizeof(num_type) * N) Vec {
    static constexpr size_t max_vec_size_bytes = 16;

    static_assert(sizeof(num_type) * N <= max_vec_size_bytes, "Vec can be at most 128 bit wide");
    static_assert(std::popcount(N) == 1, "Vec element count should be a power of 2");
    static_assert(std::popcount(sizeof(num_type)) == 1, "Vec element size must be a power of 2 in bytes");

    num_type data[N];
    using wide_t = deduce_uint_type_t<N * sizeof(num_type)>;

    __device__ __forceinline__ num_type operator[](size_t n) const { return data[n]; }
    __device__ __forceinline__ num_type& operator[](size_t n) { return data[n]; }
};

// Operations with vecs
template <size_t N, typename num_type>
struct VecOpsBase {
    using vec_t  = Vec<N, num_type>;  // TODO: Change to N, num_type
    using wide_t = vec_t::wide_t;

    static __device__ __forceinline__ void store_zero(vec_t *const __restrict__ dst) {
        if constexpr (sizeof(num_type) * N == 1) {
            *reinterpret_cast<uint8_t *>(dst) = 0;
        } else if constexpr (sizeof(num_type) * N == 2) {
            *reinterpret_cast<uint16_t *>(dst) = 0;
        } else if constexpr (sizeof(num_type) * N == 4) {
            *reinterpret_cast<uint32_t *>(dst) = 0;
        } else if constexpr (sizeof(num_type) * N == 8) {
            *reinterpret_cast<uint64_t *>(dst) = 0;
        } else if constexpr (sizeof(num_type) * N == 16) {
            *reinterpret_cast<unsigned __int128 *>(dst) = 0;
        } else {
            __builtin_unreachable();
        }
    }
    static constexpr __device__ __forceinline__ vec_t get_zero() { return vec_t{}; };

    // Loads N scalars from src vector to the address, pointed by dst
    static constexpr __device__ __forceinline__ void load__scalars(num_type *const __restrict__ dst, vec_t const *const __restrict__ src) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
    // Loads N scalars from src location to the dst vector
    static constexpr __device__ __forceinline__ void store_scalars(vec_t *const __restrict__ dst, num_type const *const __restrict__ src) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
    // Copies N scalars from src location into dst location
    static constexpr __device__ __forceinline__ void transfer_scalars(
        num_type *const __restrict__ dst, num_type const *const __restrict__ src
    ) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
    // Copies a vector from src to dst
    static constexpr __device__ __forceinline__ void transfer_vector(vec_t *const __restrict__ dst, vec_t const *const __restrict__ src) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
};

template <size_t N, FloatingNum num_type>
struct VecOpsFloatBase : VecOpsBase<N, num_type> {
   private:
    // N > 1 is implicit, because of Vec properties
    static constexpr bool can_be_packed = is_half_fp_v<num_type> && (N % 2 == 0);
    static constexpr bool can_be_packed_new =
        ((std::is_same_v<std::remove_cvref_t<num_type>, float> && kCudaArch >= 1000 || is_half_fp_v<num_type>) && (N % 2 == 0));

    using adops_ = AdOps<num_type>;

   public:
    using vec_t  = VecOpsBase<N, num_type>::vec_t;
    using wide_t = vec_t::wide_t;

    // Unary elementwise ops
    static constexpr __device__ void neg_(vec_t *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = adops_::packed_neg(reinterpret_cast<packed_t const *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = -buf[i];
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __device__ void log_(vec_t *const __restrict__ src)
        requires(sizeof(num_type) >= 2 && sizeof(num_type) <= 8)
    {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = adops_::packed_log(reinterpret_cast<packed_t const *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = adops_::log(buf[i]);
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __device__ void exp_(vec_t *const __restrict__ src)
        requires(sizeof(num_type) >= 2 && sizeof(num_type) <= 8)
    {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = adops_::packed_exp(reinterpret_cast<packed_t const *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = adops_::exp(buf[i]);
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __device__ void scalar_mul_(vec_t *const __restrict__ src, num_type s) {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed_new) {
            using packed_t    = deduce_packed_type_t<num_type>;
            packed_t packed_s = adops::broadcast_scalar_to_packed<num_type, num_type>(s);
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = adops_::packed_mul(packed_s, reinterpret_cast<packed_t const *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] *= s;
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __device__ void relu_(vec_t *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t          = deduce_packed_type_t<num_type>;
            constexpr packed_t zero = packed_t{};

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                packed_t& val = reinterpret_cast<packed_t *>(buf.data)[i];
                val           = adops_::packed_max(zero, val);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = adops_::max(buf[i], num_type{});
            }
        }

        transfer_vector(src, &buf);
    }
    static constexpr __device__ void leaky_relu_(vec_t *const __restrict__ src, num_type ns) {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
            constexpr packed_t packed_zero{};
            const packed_t packed_ns = adops::broadcast_scalar_to_packed<num_type, num_type>(ns);

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                packed_t& val = reinterpret_cast<packed_t *>(buf.data)[i];
                val           = adops_::packed_add(
                    adops_::packed_max(val, packed_zero), adops_::packed_mul(packed_ns, adops_::packed_min(val, packed_zero))
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = adops_::max(buf[i], num_type{}) + ns * adops_::min(buf[i], num_type{});
            }
        }

        transfer_vector(src, &buf);
    }
    static constexpr __device__ void leaky_relu_backward_(vec_t *const __restrict__ src, vec_t const *const __restrict__ dy, num_type ns) {
        vec_t buf, buf_y;
        transfer_vector(&buf, src);
        transfer_vector(&buf_y, dy);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
            constexpr packed_t packed_zero{};
            const packed_t packed_diff = adops::broadcast_scalar_to_packed<num_type, num_type>(static_cast<num_type>(1.0f) - ns);
            const packed_t packed_ns   = adops::broadcast_scalar_to_packed<num_type, num_type>(ns);

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                packed_t& val  = reinterpret_cast<packed_t *>(buf.data)[i];
                packed_t val_y = reinterpret_cast<packed_t const *>(buf_y.data)[i];

                val = adops_::packed_mul(val_y, adops_::packed_fma(__hgt2(val, packed_zero), packed_diff, packed_ns));
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = buf[i] >= num_type{} ? buf_y[i] : buf_y[i] * ns;
            }
        }

        transfer_vector(src, &buf);
    }

    // Binary elementwise ops
    // Add
    static constexpr __device__ void add_(vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_add(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] += buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void add(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_add(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] += buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t add(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_add(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] += buf1[i];
            }
        }

        return buf0;
    }

    // Sub
    static constexpr __device__ void sub_(vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_sub(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] -= buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void sub(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_sub(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] -= buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t sub(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_sub(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] -= buf1[i];
            }
        }

        return buf0;
    }

    // Mul
    static constexpr __device__ void mul_(vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_mul(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] *= buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void mul(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_mul(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] *= buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t mul(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_mul(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] *= buf1[i];
            }
        }

        return buf0;
    }

    // Div
    static constexpr __device__ void div_(vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_div(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] /= buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void div(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_div(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] /= buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t div(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_div(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] /= buf1[i];
            }
        }

        return buf0;
    }

    // FMA
    static constexpr __device__ void fmam_(
        vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1, vec_t const *const __restrict__ src2
    ) {
        vec_t buf0, buf1, buf2;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);
        transfer_vector(&buf2, src2);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] = adops_::packed_fma(
                    reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                    reinterpret_cast<packed_t const *>(buf2.data)[i]
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf0[i] * buf1[i] + buf2[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void fmaa_(
        vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1, vec_t const *const __restrict__ src2
    ) {
        vec_t buf0, buf1, buf2;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);
        transfer_vector(&buf2, src2);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] = adops_::packed_fma(
                    reinterpret_cast<packed_t const *>(buf1.data)[i], reinterpret_cast<packed_t const *>(buf2.data)[i],
                    reinterpret_cast<packed_t const *>(buf0.data)[i]
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf2[i] * buf1[i] + buf0[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void fma(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1,
        vec_t const *const __restrict__ src2
    ) {
        vec_t buf0, buf1, buf2;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);
        transfer_vector(&buf2, src2);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] = adops_::packed_fma(
                    reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                    reinterpret_cast<packed_t const *>(buf2.data)[i]
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf0[i] * buf1[i] + buf2[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t fma(
        vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1, vec_t const *const __restrict__ src2
    ) {
        vec_t buf0, buf1, buf2;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);
        transfer_vector(&buf2, src2);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] = adops_::packed_fma(
                    reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                    reinterpret_cast<packed_t const *>(buf2.data)[i]
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf0[i] * buf1[i] + buf2[i];
            }
        }

        return buf0;
    }

    // Min
    static constexpr __device__ void minimum_(vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_min(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = adops_::min(buf0[i], buf1[i]);
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void minimum(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_min(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = adops_::min(buf0[i], buf1[i]);
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t minimum(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_min(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = adops_::min(buf0[i], buf1[i]);
            }
        }

        return buf0;
    }

    // Max
    static constexpr __device__ void maximum_(vec_t *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_max(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = adops_::max(buf0[i], buf1[i]);
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __device__ void maximum(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_max(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = adops_::max(buf0[i], buf1[i]);
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __device__ vec_t maximum(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    adops_::packed_max(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = adops_::max(buf0[i], buf1[i]);
            }
        }

        return buf0;
    }

    // Reduction ops
    // Sum
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void sum(accum_t *const __restrict__ acc, vec_t const *const __restrict__ src) {
        num_type buf_acc{};
        vec_t buf;

        transfer_vector(&buf, src);

        if constexpr (can_be_packed_new && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_add(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
            }

            buf_acc = double_buf_acc.x + double_buf_acc.y;
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf_acc += buf[i];
            }
        }

        *acc += static_cast<accum_t>(buf_acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t sum(vec_t const *const __restrict__ src) {
        accum_t acc{};
        sum(&acc, src);
        return acc;
    }

    // Weighted sum
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void weighted_sum(accum_t *const __restrict__ acc, accum_t w, vec_t const *const __restrict__ src) {
        accum_t buf_acc = sum<accum_t>(src);
        buf_acc *= w;
        *acc += buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t weighted_sum(accum_t w, vec_t const *const __restrict__ src) {
        accum_t acc{};
        sum(&acc, src);
        acc *= w;
        return acc;
    }

    // Prod
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void prod(accum_t *const __restrict__ acc, vec_t const *const __restrict__ src) {
        accum_t buf_acc = static_cast<accum_t>(1.0f);
        vec_t buf;

        transfer_vector(&buf, src);

        if constexpr (can_be_packed_new && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_mul(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
            }

            buf_acc = static_cast<accum_t>(double_buf_acc.x) * static_cast<accum_t>(double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf_acc *= static_cast<accum_t>(buf[i]);
            }
        }

        *acc *= buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t prod(vec_t const *const __restrict__ src) {
        accum_t acc = static_cast<accum_t>(1.0f);
        prod(&acc, src);
        return acc;
    }

    // Min
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void min(accum_t *const __restrict__ acc, vec_t const *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);
        num_type buf_acc = buf[0];

        if constexpr (can_be_packed && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_min(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
            }
            buf_acc = adops_::min(double_buf_acc.x, double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 1; i < N; ++i) {
                buf_acc = adops_::min(buf_acc, buf[i]);
            }
        }

        *acc = adops_::min(*acc, buf_acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t min(vec_t const *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);
        num_type buf_acc = buf[0];

        if constexpr (can_be_packed && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_min(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
            }
            buf_acc = adops_::min(double_buf_acc.x, double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 1; i < N; ++i) {
                buf_acc = adops_::min(buf_acc, buf[i]);
            }
        }

        return static_cast<accum_t>(buf_acc);
    }

    // Max
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void max(accum_t *const __restrict__ acc, vec_t const *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);
        num_type buf_acc = buf[0];

        if constexpr (can_be_packed && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_max(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
            }
            buf_acc = adops_::max(double_buf_acc.x, double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 1; i < N; ++i) {
                buf_acc = adops_::max(buf_acc, buf[i]);
            }
        }

        *acc = adops_::max(*acc, buf_acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t max(vec_t const *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);
        num_type buf_acc = buf[0];

        if constexpr (can_be_packed && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_max(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
            }
            buf_acc = adops_::max(double_buf_acc.x, double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 1; i < N; ++i) {
                buf_acc = adops_::max(buf_acc, buf[i]);
            }
        }

        return static_cast<accum_t>(buf_acc);
    }

    // Dot product
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void dot_product(
        accum_t *const __restrict__ acc, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        accum_t buf_acc{};
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        mul_(&buf0, &buf1);
        sum(&buf_acc, &buf0);

        *acc += buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t dot_product(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        accum_t acc{};
        dot_product(&acc, src0, src1);
        return acc;
    }
};

template <size_t N, FloatingNum dst_type, FloatingNum src_type>
constexpr __device__ __forceinline__ void convert_vec(
    Vec<N, dst_type> *const __restrict__ dst, Vec<N, src_type> const *const __restrict__ src
) {
    if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, std::remove_cvref_t<src_type>>) {
        VecOpsFloatBase<N, src_type>::transfer_vector(dst, src);
        return;
    }

    using SrcOps = VecOpsFloatBase<N, src_type>;
    using DstOps = VecOpsFloatBase<N, dst_type>;

    typename SrcOps::vec_t in_buf;
    typename DstOps::vec_t out_buf;
    SrcOps::transfer_vector(&in_buf, src);

    if constexpr (((std::is_same_v<std::remove_cvref_t<src_type>, float> || is_half_fp_v<src_type>) &&
                      (std::is_same_v<std::remove_cvref_t<dst_type>, float> || is_half_fp_v<dst_type>) && (N % 2 == 0))) {
        using packed_src_t = deduce_packed_type_t<src_type>;
        using packed_dst_t = deduce_packed_type_t<dst_type>;

#pragma unroll
        for (size_t i = 0; i < N / 2; ++i) {
            reinterpret_cast<packed_dst_t *>(out_buf.data)[i] =
                adops::packed_convert<dst_type, src_type>(reinterpret_cast<packed_src_t *>(in_buf.data)[i]);
        }
    } else {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            out_buf[i] = static_cast<dst_type>(in_buf[i]);
        }
    }

    DstOps::transfer_vector(dst, &out_buf);
}

// Something like cooperative kernel-like function for copying the whole row with conversions
// row_width is D size
// worker_cnt is total numer of workers(threads), working on the whole row
template <FloatingNum dst_type, FloatingNum src_type, size_t row_width, size_t worker_cnt>
static __device__ void write_row(dst_type *const __restrict__ dst, size_t worker_idx, src_type const *const __restrict__ src) {
    constexpr size_t copy_N = Vec<1, src_type>::max_vec_size_bytes / std::max(sizeof(src_type), sizeof(dst_type));
    __builtin_assume(copy_N % 2 == 0 && copy_N > 1);
    constexpr size_t total_copies_per_worker = (row_width + copy_N * worker_cnt - 1) / (copy_N * worker_cnt);

    using src_vec_type = Vec<copy_N, src_type>;
    using dst_vec_type = Vec<copy_N, dst_type>;

    constexpr size_t unroll_k = 4;

    for (size_t i = 0; i < (total_copies_per_worker + unroll_k - 1) / unroll_k; ++i) {
#pragma unroll
        for (size_t j = 0; j < unroll_k; ++j) {
            const size_t tile_id = worker_cnt * (i * unroll_k + j) + worker_idx;
            if (tile_id * copy_N < row_width) [[likely]] {
                convert_vec<copy_N, dst_type, src_type>(&reinterpret_cast<dst_vec_type *>(dst)[tile_id], &reinterpret_cast<src_vec_type const *>(src)[tile_id]);
            }
        }
    }

    __syncthreads();
}

// Operations with whole tiles

template <size_t N, FloatingNum num_type, FloatingNum accum_t = float>
struct TileOps : VecOpsFloatBase<N, num_type> {
   private:
    static consteval auto deduce_ns_t() {
        if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, half>) {
            return half2{};
        } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, nv_bfloat16>) {
            return nv_bfloat162{};
        } else {
            return num_type{};
        }
    }

   public:
    static_assert(sizeof(accum_t) >= sizeof(num_type), "Accumulator type must not be smaller than basic num type");

    using scal_t = num_type;
    using vec_t  = Vec<N, num_type>;
    using wide_t = VecOpsFloatBase<N, num_type>::wide_t;
    using ns_t   = decltype(deduce_ns_t());

    static constexpr int TW = N;

    // Common
    static __device__ __forceinline__ vec_t read(num_type const *const __restrict__ src_arr, size_t vec_idx) {
        return *reinterpret_cast<vec_t const *>(&src_arr[vec_idx * TW]);
    }

    // GATv2
    static __device__ __forceinline__ ns_t make_ns(accum_t ns) {
        // Static casts to float are temporary. If somebody wants to fix it, be my guest.
        if constexpr (std::is_same_v<half, std::remove_cvref_t<num_type>>) {
            __float2half2_rn(static_cast<float>(ns));
        } else if constexpr (std::is_same_v<nv_bfloat16, std::remove_cvref_t<num_type>>) {
            return __float2bfloat162_rn(static_cast<float>(ns));
        } else {
            return static_cast<num_type>(ns);
        }
    }
    static __device__ __forceinline__ accum_t gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, accum_t ns) {
        add_(&l, &r);
        leaky_relu_(&l, ns);
        mul_(&l, &a);
        return sum<accum_t>(&l);
    }
    static __device__ __forceinline__ void gatv2_accum_grad_al(
        accum_t *const __restrict__ ga, accum_t *const __restrict__ gl, accum_t ge, vec_t l, vec_t r, vec_t a, accum_t ns
    ) {
        Vec<N, num_type> ge_vec;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            ge_vec[i] = static_cast<num_type>(ge);
        }

        add_(&l, &r);
        Vec<N, num_type> buf_vec = l;

        leaky_relu_backward_(&buf_vec, &ge_vec, ns);

        constexpr size_t compact_N  = std::min(N, sizeof(accum_t));
        constexpr size_t repeat_cnt = N / compact_N;
        using vec_compact_t         = Vec<compact_N, num_type>;
        using FloatOps              = VecOpsFloatBase<compact_N, accum_t>;

#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            Vec<compact_N, accum_t> l_out, bv_out;
            convert_vec<compact_N, accum_t, num_type>(&l_out, &reinterpret_cast<vec_compact_t const *>(&l)[i]);
            convert_vec<compact_N, accum_t, num_type>(&bv_out, &reinterpret_cast<vec_compact_t const *>(&buf_vec)[i]);

            FloatOps::fmaa_(&reinterpret_cast<Vec<compact_N, accum_t> *>(ga)[i], &bv_out, &l_out);
            FloatOps::fmaa_(&reinterpret_cast<Vec<compact_N, accum_t> *>(gl)[i], &bv_out, &l_out);
        }
    }
    static __device__ __forceinline__ void gatv2_accum_grad_r(
        accum_t *const __restrict__ gr, accum_t alpha, vec_t gh, accum_t ge, vec_t l, vec_t r, vec_t a, accum_t ns
    ) {
        Vec<N, num_type> ge_vec, alpha_vec;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            ge_vec[i]    = static_cast<num_type>(ge);
            alpha_vec[i] = static_cast<num_type>(alpha);
        }

        add_(&l, &r);
        leaky_relu_backward_(&l, &ge_vec, ns);
        mul_(&l, &a);
        fmaa_(&l, &gh, &alpha_vec);

        constexpr size_t compact_N  = std::min(N, sizeof(accum_t));
        constexpr size_t repeat_cnt = N / compact_N;
        using vec_compact_t         = Vec<compact_N, num_type>;
        using FloatOps              = VecOpsFloatBase<compact_N, accum_t>;

#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            Vec<compact_N, accum_t> l_out;
            convert_vec<compact_N, accum_t, num_type>(&l_out, &reinterpret_cast<vec_compact_t const *>(&l)[i]);

            FloatOps::add_(&reinterpret_cast<Vec<compact_N, accum_t> *>(gr)[i], &l_out);
        }
    }
    template <FloatingNum atomic_t = float>
    static __device__ __forceinline__ void atomic_add_scaled_f32(atomic_t *const __restrict__ ptr, size_t vec_idx, atomic_t scalar, vec_t v) {
        static_assert(std::is_same_v<std::remove_cvref_t<atomic_t>, float>, "atomic add is only implememnted for float atomic type.");

        constexpr size_t compact_N  = std::min(N, sizeof(atomic_t));
        constexpr size_t repeat_cnt = N / compact_N;
        using vec_compact_t         = Vec<compact_N, num_type>;
        using FloatOps              = VecOpsFloatBase<compact_N, atomic_t>;

        Vec<compact_N, atomic_t> buf;
#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            convert_vec<compact_N, atomic_t, num_type>(&buf, &reinterpret_cast<vec_compact_t const *>(&v)[i]);
            FloatOps::scalar_mul_(&buf, scalar);
#pragma unroll
            for (size_t j = 0; j < compact_N; ++j) {
                atomicAdd(&ptr[vec_idx + i * compact_N + j], buf[j]);
            }
        }
    }
};
