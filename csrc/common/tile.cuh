#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

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

    __host__ __device__ __tile__ __forceinline__ num_type operator[](size_t n) const { return data[n]; }
    __host__ __device__ __tile__ __forceinline__ num_type& operator[](size_t n) { return data[n]; }
};

// Operations with vecs
template <size_t N, typename num_type>
struct VecOpsBase {
    using vec_t  = Vec<N, num_type>;
    using wide_t = vec_t::wide_t;

    static __host__ __device__ __tile__ __forceinline__ void store_zero(vec_t *const __restrict__ dst) {
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
    static constexpr __host__ __device__ __tile__ __forceinline__ vec_t get_zero() { return vec_t{}; };

    // Loads N scalars from src vector to the address, pointed by dst
    static constexpr __host__ __device__ __tile__ __forceinline__ void load__scalars(
        num_type *const __restrict__ dst, vec_t const *const __restrict__ src
    ) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
    // Loads N scalars from src location to the dst vector
    static constexpr __host__ __device__ __tile__ __forceinline__ void store_scalars(
        vec_t *const __restrict__ dst, num_type const *const __restrict__ src
    ) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
    // Copies N scalars from src location into dst location
    static constexpr __host__ __device__ __tile__ __forceinline__ void transfer_scalars(
        num_type *const __restrict__ dst, num_type const *const __restrict__ src
    ) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
    // Copies a vector from src to dst
    static constexpr __host__ __device__ __tile__ __forceinline__ void transfer_vector(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src
    ) {
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

   public:
    using vec_t  = VecOpsBase<N, num_type>::vec_t;
    using wide_t = VecOpsBase<N, num_type>::wide_t;

    // Unary elementwise ops
    static constexpr __host__ __device__ __tile__ void neg_(vec_t *const __restrict__ src) {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = __hneg2(reinterpret_cast<packed_t *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] = -buf[i];
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __host__ __device__ __tile__ void log_(vec_t *const __restrict__ src)
        requires(sizeof(num_type) >= 2 && sizeof(num_type) <= 8)
    {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = h2log(reinterpret_cast<packed_t *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                if constexpr (is_half_fp_v<num_type>) {
                    buf[i] = hlog(buf[i]);
                } else if constexpr (sizeof(num_type) == 4) {
                    buf[i] = __logf(buf[i]);
                } else {
                    buf[i] = cuda::std::log(buf[i]);
                }
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __host__ __device__ __tile__ void exp_(vec_t *const __restrict__ src)
        requires(sizeof(num_type) >= 2 && sizeof(num_type) <= 8)
    {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = h2exp(reinterpret_cast<packed_t *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                if constexpr (is_half_fp_v<num_type>) {
                    buf[i] = hexp(buf[i]);
                } else if constexpr (sizeof(num_type) == 4) {
                    buf[i] = __expf(buf[i]);
                } else {
                    buf[i] = cuda::std::exp(buf[i]);
                }
            }
        }

        transfer_vector(src, &buf);
    }

    static constexpr __host__ __device__ __tile__ void scalar_mul_(vec_t *const __restrict__ src, num_type s) {
        vec_t buf;
        transfer_vector(&buf, src);

        if constexpr (can_be_packed) {
            using packed_t    = deduce_packed_type_t<num_type>;
            packed_t packed_s = broadcast_scalar_to_packed<num_type, num_type>(s);
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf.data)[i] = __hmul2(packed_s, reinterpret_cast<packed_t *>(buf.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf[i] *= s;
            }
        }

        transfer_vector(src, &buf);
    }

    // Binary elementwise ops
    // Add
    static constexpr __host__ __device__ __tile__ void add_(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __hadd2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __fadd2_rn(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] += buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __host__ __device__ __tile__ void add(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __hadd2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __fadd2_rn(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] += buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __host__ __device__ __tile__ vec_t add(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __hadd2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __fadd2_rn(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else {
                    __builtin_unreachable();
                }
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
    static constexpr __host__ __device__ __tile__ void sub_(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    __hsub2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] -= buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __host__ __device__ __tile__ void sub(
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
                    __hsub2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] -= buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __host__ __device__ __tile__ vec_t sub(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    __hsub2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
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
    static constexpr __host__ __device__ __tile__ void mul_(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __hmul2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __fmul2_rn(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] *= buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __host__ __device__ __tile__ void mul(
        vec_t *const __restrict__ dst, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __hmul2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __fmul2_rn(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] *= buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __host__ __device__ __tile__ vec_t mul(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed_new) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __hmul2(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] =
                        __fmul2_rn(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
                } else {
                    __builtin_unreachable();
                }
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
    static constexpr __host__ __device__ __tile__ void div_(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    __h2div(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] /= buf1[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __host__ __device__ __tile__ void div(
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
                    __h2div(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] /= buf1[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __host__ __device__ __tile__ vec_t div(vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1) {
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        if constexpr (can_be_packed) {
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(buf0.data)[i] =
                    __h2div(reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i]);
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
    static constexpr __host__ __device__ __tile__ void fma_(
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
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] = __hfma2(
                        reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                        reinterpret_cast<packed_t const *>(buf2.data)[i]
                    );
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] = __ffma2_rn(
                        reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                        reinterpret_cast<packed_t const *>(buf2.data)[i]
                    );
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf0[i] * buf1[i] + buf2[i];
            }
        }

        transfer_vector(src0, &buf0);
    }
    static constexpr __host__ __device__ __tile__ void fma(
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
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] = __hfma2(
                        reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                        reinterpret_cast<packed_t const *>(buf2.data)[i]
                    );
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] = __ffma2_rn(
                        reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                        reinterpret_cast<packed_t const *>(buf2.data)[i]
                    );
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf0[i] * buf1[i] + buf2[i];
            }
        }

        transfer_vector(dst, &buf0);
    }
    static constexpr __host__ __device__ __tile__ vec_t fma(
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
                if constexpr (sizeof(num_type) == 2) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] = __hfma2(
                        reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                        reinterpret_cast<packed_t const *>(buf2.data)[i]
                    );
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    reinterpret_cast<packed_t *>(buf0.data)[i] = __ffma2_rn(
                        reinterpret_cast<packed_t const *>(buf0.data)[i], reinterpret_cast<packed_t const *>(buf1.data)[i],
                        reinterpret_cast<packed_t const *>(buf2.data)[i]
                    );
                } else {
                    __builtin_unreachable();
                }
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                buf0[i] = buf0[i] * buf1[i] + buf2[i];
            }
        }

        return buf0;
    }

    // Reduction ops
    // Sum
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __host__ __device__ __tile__ void sum(accum_t *const __restrict__ acc, vec_t const *const __restrict__ src) {
        num_type buf_acc{};
        vec_t buf;

        transfer_vector(&buf, src);

        if constexpr (can_be_packed_new && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = *reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    __hadd2(double_buf_acc, *reinterpret_cast<packed_t const *>(&buf)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    __fadd2_rn(double_buf_acc, *reinterpret_cast<packed_t const *>(&buf)[i]);
                } else {
                    __builtin_unreachable();
                }
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
    static constexpr __host__ __device__ __tile__ accum_t sum(vec_t const *const __restrict__ src) {
        accum_t acc{};
        sum(&acc, src);
        return acc;
    }

    // Weighted sum
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __host__ __device__ __tile__ void weighted_sum(
        accum_t *const __restrict__ acc, accum_t w, vec_t const *const __restrict__ src
    ) {
        accum_t buf_acc = sum(src);
        buf_acc *= w;
        *acc = buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __host__ __device__ __tile__ accum_t weighted_sum(accum_t w, vec_t const *const __restrict__ src) {
        accum_t acc{};
        sum(&acc, src);
        acc *= w;
        return acc;
    }

    // Prod
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __host__ __device__ __tile__ void prod(accum_t *const __restrict__ acc, vec_t const *const __restrict__ src) {
        accum_t buf_acc{1};
        vec_t buf;

        transfer_vector(&buf, src);

        if constexpr (can_be_packed_new && N >= 4) {
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = *reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                if constexpr (sizeof(num_type) == 2) {
                    __hmul2(double_buf_acc, *reinterpret_cast<packed_t const *>(&buf)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, float>) {
                    __fmul2_rn(double_buf_acc, *reinterpret_cast<packed_t const *>(&buf)[i]);
                } else {
                    __builtin_unreachable();
                }
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
    static constexpr __host__ __device__ __tile__ accum_t prod(vec_t const *const __restrict__ src) {
        accum_t acc{};
        prod(&acc, src);
        return acc;
    }

    // Dot product
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __host__ __device__ __tile__ void dot_product(
        accum_t *const __restrict__ acc, vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        accum_t buf_acc{};
        vec_t buf0, buf1;

        transfer_vector(&buf0, src0);
        transfer_vector(&buf1, src1);

        mul_(buf0, buf1);
        sum(&buf_acc, &buf0);

        *acc += buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __host__ __device__ __tile__ accum_t dot_product(
        vec_t const *const __restrict__ src0, vec_t const *const __restrict__ src1
    ) {
        accum_t acc{};
        dot_product(&acc, src0, src1);
        return acc;
    }
};

template <size_t N, FloatingNum dst_type, FloatingNum src_type>
constexpr __host__ __device__ __tile__ __forceinline__ void convert_vec(
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
            if constexpr (std::is_same_v<std::remove_cvref_t<src_type>, float>) {
                if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, nv_bfloat16>) {
                    reinterpret_cast<packed_dst_t *>(out_buf.data)[i] = __float22bfloat162_rn(reinterpret_cast<packed_src_t *>(in_buf.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, half>) {
                    reinterpret_cast<packed_dst_t *>(out_buf.data)[i] = __float22half2_rn(reinterpret_cast<packed_src_t *>(in_buf.data)[i]);
                } else {
                    __builtin_unreachable();
                }
            } else if constexpr (std::is_same_v<std::remove_cvref_t<src_type>, nv_bfloat16>) {
                if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, float>) {
                    reinterpret_cast<packed_dst_t *>(out_buf.data)[i] = __bfloat1622float2(reinterpret_cast<packed_src_t *>(in_buf.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, half>) {
                    reinterpret_cast<packed_dst_t *>(out_buf.data)[i] =
                        __float22half2_rn(__bfloat1622float2(reinterpret_cast<packed_src_t *>(in_buf.data)[i]));
                } else {
                    __builtin_unreachable();
                }
            } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, half>) {
                if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, float>) {
                    reinterpret_cast<packed_dst_t *>(out_buf.data)[i] = __half22float2(reinterpret_cast<packed_src_t *>(in_buf.data)[i]);
                } else if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, nv_bfloat16>) {
                    reinterpret_cast<packed_dst_t *>(out_buf.data)[i] =
                        __float22bfloat162_rn(__half22float2(reinterpret_cast<packed_src_t *>(in_buf.data)[i]));
                } else {
                    __builtin_unreachable();
                }
            } else {
                __builtin_unreachable();
            }
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
template <FloatingNum dst_type, FloatingNum src_type, size_t row_width, size_t worker_cnt>
static __host__ __device__ __tile__ __forceinline__ void write_row(
    dst_type *const __restrict__ dst, size_t worker_num, src_type const *const __restrict__ src
) {
    static_assert(sizeof(src_type) >= sizeof(dst_type));
    constexpr size_t copy_N = Vec<1, src_type>::max_vec_size_bytes / sizeof(src_type);
    __builtin_assume(copy_N % 2 == 0 && copy_N > 1);
    constexpr size_t total_copies = (row_width + copy_N - 1) / copy_N;

    using src_vec_type = Vec<copy_N, src_type>;
    using dst_vec_type = Vec<copy_N, dst_type>;

#pragma unroll
    for (size_t i = 0; i < total_copies; ++i) {
        const size_t tile_id = worker_cnt * i + worker_num;
        if (tile_id * copy_N < row_width) {
            convert_vec(&reinterpret_cast<dst_vec_type *>(dst)[tile_id], &reinterpret_cast<src_vec_type const *>(src)[tile_id]);
        }
    }
}

// Operations with whole tiles

template <size_t N, FloatingNum num_type, FloatingNum accum_t = float>
struct TileOps : VecOpsFloatBase<N, num_type> {
   private:
    static consteval auto deduce_ns_t() {
        if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, half>) {
            return half{};
        } else if constexpr (std::is_same_v<std::remove_cvref_t<num_type>, nv_bfloat16>) {
            return nv_bfloat16{};
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
    static __host__ __device__ __tile__ __forceinline__ vec_t read(num_type const *const __restrict__ src_arr, size_t vec_idx) {
        return *reinterpret_cast<vec_t const *>(&src_arr[vec_idx * TW]);
    }

    // GATv2
    static __device__ __forceinline__ ns_t make_ns(float ns) {
        if constexpr (std::is_same_v<half, std::remove_cvref_t<num_type>>) {
            __float2half2_rn(ns);
        } else if constexpr (std::is_same_v<nv_bfloat16, std::remove_cvref_t<num_type>>) {
            return __float2bfloat162_rn(ns);
        } else {
            return static_cast<num_type>(ns);
        }
    }
    static __host__ __device__ __tile__ __forceinline__ accum_t gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, ns_t ns) {}
};
