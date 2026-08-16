#pragma once

#include <bit>
#include <type_traits>

#include "common/adaptive_ops.cuh"
#include "common/misc.cuh"
#include "common/traits.cuh"

// Vec data struct

template <size_t N, typename num_type>
struct alignas(sizeof(num_type) * N) Vec {
    static constexpr size_t max_vec_size_bytes = 16;

    static_assert(sizeof(num_type) * N <= max_vec_size_bytes, "Vec can be at most 128 bit wide");
    static_assert(std::popcount(N) == 1, "Vec element count should be a power of 2");
    static_assert(std::popcount(sizeof(num_type)) == 1, "Vec element size must be a power of 2 in bytes");

    num_type data[N];
    using vec_t  = Vec<N, num_type>;
    using wide_t = deduce_uint_type_t<N * sizeof(num_type)>;

    Vec() noexcept = default;
    __device__ Vec(num_type num) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] = num;
        }
    }
    Vec(const Vec& other) noexcept            = default;
    Vec(Vec&& other) noexcept                 = default;
    Vec& operator=(const Vec& other) noexcept = default;
    Vec& operator=(Vec&& other) noexcept      = default;
    ~Vec() noexcept                           = default;

    __device__ num_type operator[](size_t n) const noexcept { return data[n]; }
    __device__ num_type& operator[](size_t n) noexcept { return data[n]; }

    __device__ void store_zero_() noexcept { *reinterpret_cast<wide_t *>(data) = 0; }
    static constexpr __device__ vec_t get_zero() { return vec_t{}; };

    // Loads N scalars to this vector
    constexpr __device__ void load_scalars(num_type const *const __restrict__ src) noexcept {
        *reinterpret_cast<wide_t *>(this->data) = *reinterpret_cast<wide_t const *>(src);
    }
    // Loads N scalars from this vector into dst
    constexpr __device__ void store_scalars(num_type *const __restrict__ dst) const noexcept {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(this->data);
    }
    // Copies a vector from src to dst
    static constexpr __device__ void transfer_vector(vec_t *const __restrict__ dst, vec_t const *const __restrict__ src) {
        *reinterpret_cast<wide_t *>(dst) = *reinterpret_cast<wide_t const *>(src);
    }
};

static_assert(std::is_trivially_copyable_v<Vec<8, nv_bfloat16>>);
static_assert(std::is_trivially_default_constructible_v<Vec<8, nv_bfloat16>>);
static_assert(std::is_trivially_destructible_v<Vec<8, nv_bfloat16>>);

// ==================================================================================
// SelectTW: pick widest TW where all threads are working but not wider than 128 bits
// ==================================================================================

template <int D_CONST, typename cuda_t, int THREADS_PER_D = kWarpSize>
struct SelectTW {
   private:
    static consteval int calculate_tile_width(size_t type_size, size_t d, size_t thread_count) {
        size_t elems_per_thread = (d + thread_count - 1) / thread_count;

        return std::min(elems_per_thread, Vec<1, float>::max_vec_size_bytes / type_size);  // 16 bytes is the most wide load/store
    }

   public:
    static_assert(THREADS_PER_D == kWarpSize, "For now only whole warp count can operate on a single row.");
    static_assert(std::popcount(sizeof(cuda_t)) == 1, "Only types with size of power of 2 are supported.");

    static constexpr int threads_per_d = THREADS_PER_D;
    static constexpr int value         = calculate_tile_width(sizeof(cuda_t), D_CONST, THREADS_PER_D);
};

// Operations with vecs
template <size_t N, FloatingNum num_type>
struct alignas(sizeof(num_type) * N) VecFloat: Vec<N, num_type> {
   private:
    template <typename T>
    static consteval bool can_be_packed_b() {
        return is_half_fp_v<T>;
    }
    template <typename T>
    static consteval bool can_be_packed_new_b() {
        return std::is_same_v<std::remove_cvref_t<T>, float> && kCudaArch >= 1000 || is_half_fp_v<T>;
    }

    // N > 1 is implicit, because of Vec properties
    static constexpr bool can_be_packed     = can_be_packed_b<num_type>() && (N % 2 == 0);
    static constexpr bool can_be_packed_new = can_be_packed_new_b<num_type>() && (N % 2 == 0);

    using adops_ = AdOps<num_type>;

   public:
    using vec_t  = VecFloat<N, num_type>;
    using wide_t = vec_t::wide_t;

    // Unary elementwise ops
    constexpr __device__ void neg_() noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_neg(reinterpret_cast<packed_t const *>(this->data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = -this->data[i];
            }
        }
    }

    constexpr __device__ void log_() noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_log(reinterpret_cast<packed_t const *>(this->data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::log(this->data[i]);
            }
        }
    }

    constexpr __device__ void exp_() noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_exp(reinterpret_cast<packed_t const *>(this->data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::exp(this->data[i]);
            }
        }
    }

    // Scalar add
    constexpr __device__ void scalar_add_(num_type s) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_            = AdOpsPacked<num_type>;
            using packed_t          = deduce_packed_type_t<num_type>;
            const packed_t packed_s = adops::broadcast_scalar_to_packed<num_type, num_type>(s);
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_add(packed_s, reinterpret_cast<packed_t const *>(this->data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] += s;
            }
        }
    }
    constexpr __device__ vec_t& operator+=(num_type s) noexcept {
        scalar_add_(s);
        return *this;
    }
    friend constexpr __device__ vec_t operator+(vec_t vec, num_type s) noexcept {
        vec += s;
        return vec;
    }
    friend constexpr __device__ vec_t operator+(num_type s, vec_t vec) noexcept {
        vec += s;
        return vec;
    }
    // Scalar sub
    constexpr __device__ void scalar_sub_(num_type s) noexcept { scalar_add_(-s); }
    constexpr __device__ vec_t& operator-=(num_type s) noexcept {
        scalar_sub_(s);
        return *this;
    }
    friend constexpr __device__ vec_t operator-(vec_t vec, num_type s) noexcept {
        vec -= s;
        return vec;
    }
    // Scalar mul
    constexpr __device__ void scalar_mul_(num_type s) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_            = AdOpsPacked<num_type>;
            using packed_t          = deduce_packed_type_t<num_type>;
            const packed_t packed_s = adops::broadcast_scalar_to_packed<num_type, num_type>(s);
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_mul(packed_s, reinterpret_cast<packed_t const *>(this->data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] *= s;
            }
        }
    }
    constexpr __device__ vec_t& operator*=(num_type s) noexcept {
        scalar_mul_(s);
        return *this;
    }
    friend constexpr __device__ vec_t operator*(vec_t vec, num_type s) noexcept {
        vec *= s;
        return vec;
    }
    friend constexpr __device__ vec_t operator*(num_type s, vec_t vec) noexcept {
        vec *= s;
        return vec;
    }
    // Scalar div
    constexpr __device__ void scalar_div_(num_type s) noexcept { scalar_mul_(static_cast<num_type>(1.0f) / s); }
    constexpr __device__ vec_t& operator/=(num_type s) noexcept {
        scalar_div_(s);
        return *this;
    }
    friend constexpr __device__ vec_t operator/(vec_t vec, num_type s) noexcept {
        vec /= s;
        return vec;
    }

    constexpr __device__ void relu_() noexcept {
        if constexpr (can_be_packed) {
            using adops_            = AdOpsPacked<num_type>;
            using packed_t          = deduce_packed_type_t<num_type>;
            constexpr packed_t zero = packed_t{};

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                packed_t& val = reinterpret_cast<packed_t *>(this->data)[i];
                val           = adops_::packed_max(zero, val);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::max(this->data[i], num_type{});
            }
        }
    }

    constexpr __device__ void leaky_relu_(num_type ns) noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
            constexpr packed_t packed_zero{};
            const packed_t packed_ns = adops::broadcast_scalar_to_packed<num_type, num_type>(ns);

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                packed_t& val = reinterpret_cast<packed_t *>(this->data)[i];
                val           = adops_::packed_add(
                    adops_::packed_max(val, packed_zero), adops_::packed_mul(packed_ns, adops_::packed_min(val, packed_zero))
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::max(this->data[i], num_type{}) + ns * adops_::min(this->data[i], num_type{});
            }
        }
    }

    constexpr __device__ void leaky_relu_backward_(vec_t dy, num_type ns) noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
            constexpr packed_t packed_zero{};
            const packed_t packed_diff = adops::broadcast_scalar_to_packed<num_type, num_type>(static_cast<num_type>(1.0f) - ns);
            const packed_t packed_ns   = adops::broadcast_scalar_to_packed<num_type, num_type>(ns);

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                packed_t& val  = reinterpret_cast<packed_t *>(this->data)[i];
                packed_t val_y = reinterpret_cast<packed_t const *>(dy.data)[i];

                val = adops_::packed_mul(val_y, adops_::packed_fma(__hgt2(val, packed_zero), packed_diff, packed_ns));
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = this->data[i] >= num_type{} ? dy[i] : dy[i] * ns;
            }
        }
    }

    // Binary elementwise ops
    // Add
    constexpr __device__ void add_(vec_t other) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_add(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] += other[i];
            }
        }
    }
    static constexpr __device__ vec_t add(vec_t src0, vec_t src1) noexcept {
        src0.add_(src1);
        return src0;
    }
    constexpr __device__ vec_t& operator+=(vec_t other) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_add(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] += other[i];
            }
        }

        return *this;
    }
    friend constexpr __device__ vec_t operator+(vec_t first, vec_t other) noexcept {
        first += other;
        return first;
    }

    // Sub
    constexpr __device__ void sub_(vec_t other) noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_sub(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] -= other[i];
            }
        }
    }
    static constexpr __device__ vec_t sub(vec_t src0, vec_t src1) noexcept {
        src0.sub_(src1);
        return src0;
    }
    constexpr __device__ vec_t& operator-=(vec_t other) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_sub(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] -= other[i];
            }
        }

        return *this;
    }
    friend constexpr __device__ vec_t operator-(vec_t first, vec_t other) noexcept {
        first -= other;
        return first;
    }

    // Mul
    constexpr __device__ void mul_(vec_t other) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_mul(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] *= other[i];
            }
        }
    }
    static constexpr __device__ vec_t mul(vec_t src0, vec_t src1) noexcept {
        src0.mul_(src1);
        return src0;
    }
    constexpr __device__ vec_t& operator*=(vec_t other) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_mul(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] *= other[i];
            }
        }

        return *this;
    }
    friend constexpr __device__ vec_t operator*(vec_t first, vec_t other) noexcept {
        first *= other;
        return first;
    }

    // Div
    constexpr __device__ void div_(vec_t other) noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_div(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] /= other[i];
            }
        }
    }
    static constexpr __device__ vec_t div(vec_t src0, vec_t src1) noexcept {
        src0.div_(src1);
        return src0;
    }
    constexpr __device__ vec_t& operator/=(vec_t other) {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_div(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] /= other[i];
            }
        }

        return *this;
    }
    friend constexpr __device__ vec_t operator/(vec_t first, vec_t other) noexcept {
        first /= other;
        return first;
    }

    // FMA
    constexpr __device__ void fmam_(vec_t src1, vec_t src2) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_fma(
                    reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(src1.data)[i],
                    reinterpret_cast<packed_t const *>(src2.data)[i]
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::fma(this->data[i], src1[i], src2[i]);
            }
        }
    }
    constexpr __device__ void fmaa_(vec_t src0, vec_t src1) noexcept {
        if constexpr (can_be_packed_new) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] = adops_::packed_fma(
                    reinterpret_cast<packed_t const *>(src0.data)[i], reinterpret_cast<packed_t const *>(src1.data)[i],
                    reinterpret_cast<packed_t const *>(this->data)[i]
                );
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::fma(src0[i], src1[i], this->data[i]);
            }
        }
    }
    static constexpr __device__ vec_t fma(vec_t src0, vec_t src1, vec_t src2) noexcept {
        src2.fmaa_(src0, src1);
        return src2;
    }

    // Min
    constexpr __device__ void minimum_(vec_t other) noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_min(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::min(this->data[i], other[i]);
            }
        }
    }
    static constexpr __device__ void minimum(vec_t *const __restrict__ dst, vec_t src0, vec_t src1) noexcept {
        src0.minimum_(src1);
        transfer_vector(dst, &src0);
    }
    static constexpr __device__ vec_t minimum(vec_t src0, vec_t src1) noexcept {
        src0.minimum_(src1);
        return src0;
    }

    // Max
    constexpr __device__ void maximum_(vec_t other) noexcept {
        if constexpr (can_be_packed) {
            using adops_   = AdOpsPacked<num_type>;
            using packed_t = deduce_packed_type_t<num_type>;
#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_t *>(this->data)[i] =
                    adops_::packed_max(reinterpret_cast<packed_t const *>(this->data)[i], reinterpret_cast<packed_t const *>(other.data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                this->data[i] = adops_::max(this->data[i], other[i]);
            }
        }
    }
    static constexpr __device__ void maximum(vec_t *const __restrict__ dst, vec_t src0, vec_t src1) noexcept {
        src0.maximum_(src1);
        transfer_vector(dst, &src0);
    }
    static constexpr __device__ vec_t maximum(vec_t src0, vec_t src1) noexcept {
        src0.maximum_(src1);
        return src0;
    }

    // Reduction ops
    // Sum
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void sum_(accum_t *const __restrict__ acc) const noexcept {
        // Accumulate in accum_t: chunks are widened via convert_vec (exact for accum_t wider than
        // num_type), then reduced with accum_t arithmetic. 16-bit pair-trees lose low bits on
        // every step; accum_t accumulation keeps the result within one accum_t rounding.
        constexpr size_t compact_N  = std::min(N, VecFloat<1, float>::max_vec_size_bytes / std::max(sizeof(accum_t), sizeof(num_type)));
        constexpr size_t repeat_cnt = N / compact_N;

        using NumVec   = VecFloat<compact_N, num_type>;
        using AccumVec = VecFloat<compact_N, accum_t>;

        accum_t buf_acc{};
        AccumVec buf;

#pragma unroll
        for (size_t j = 0; j < repeat_cnt; ++j) {
            buf = reinterpret_cast<NumVec const *>(this->data)[j].template convert_vec<accum_t>();

            if constexpr (can_be_packed_new && can_be_packed_new_b<accum_t>() && compact_N >= 4) {
                using AccOps   = AdOpsPacked<accum_t>;
                using packed_t = deduce_packed_type_t<accum_t>;

                packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
                for (size_t i = 1; i < compact_N / 2; ++i) {
                    double_buf_acc = AccOps::packed_add(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
                }

                buf_acc += double_buf_acc.x + double_buf_acc.y;
            } else {
#pragma unroll
                for (size_t i = 0; i < compact_N; ++i) {
                    buf_acc += buf[i];
                }
            }
        }

        *acc += buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t sum_() const noexcept {
        accum_t acc{};
        this->sum_<accum_t>(&acc);
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void sum(accum_t *const __restrict__ acc, vec_t vec) {
        vec.sum_<accum_t>(acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t sum(vec_t vec) {
        return vec.sum_<accum_t>();
    }

    // Weighted sum
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void weighted_sum_(accum_t *const __restrict__ acc, accum_t w) const noexcept {
        accum_t buf_acc = this->sum_<accum_t>();
        buf_acc *= w;
        *acc += buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t weighted_sum_(accum_t w) const noexcept {
        accum_t acc = this->sum_<accum_t>();
        acc *= w;
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void weighted_sum(accum_t *const __restrict__ acc, accum_t w, vec_t vec) {
        vec.weighted_sum_<accum_t>(acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t weighted_sum(accum_t w, vec_t vec) {
        return vec.weighted_sum_<accum_t>(w);
    }

    // Weighted accumulate (elementwise): acc[i] += w * src[i].
    // NOT the same as weighted_sum, which is a reduction (*acc += w * sum(src)). The attention
    // kernels accumulate a scaled row (V, dO, K, Q) into a per-position accumulator and need the
    // elementwise form — this is the legacy TileOps::weighted_accum. Each element is one
    // single-rounding fma in accum_t, matching the legacy fmaf(w, r, acc).
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void weighted_accum_(accum_t *const __restrict__ acc, accum_t w) const noexcept {
        constexpr size_t compact_N  = std::min(N, VecFloat<1, float>::max_vec_size_bytes / std::max(sizeof(accum_t), sizeof(num_type)));
        constexpr size_t repeat_cnt = N / compact_N;
        using NumVec                = VecFloat<compact_N, num_type>;
        using AccumVec              = VecFloat<compact_N, accum_t>;

        AccumVec src_out, w_vec(w);

#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            src_out = reinterpret_cast<NumVec const *>(this->data)[i].template convert_vec<accum_t>();
            reinterpret_cast<AccumVec *>(acc)[i].fmaa_(src_out, w_vec);
        }
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t weighted_accum_(accum_t w) const noexcept {
        accum_t acc{};
        this->weighted_accum_<accum_t>(&acc, w);
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void weighted_accum(accum_t *const __restrict__ acc, accum_t w, vec_t vec) {
        vec.weighted_accum_<accum_t>(acc, w);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t weighted_accum(accum_t w, vec_t vec) {
        return vec.weighted_accum_<accum_t>(w);
    }

    // Prod
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void prod_(accum_t *const __restrict__ acc) const noexcept {
        constexpr size_t compact_N  = std::min(N, VecFloat<1, float>::max_vec_size_bytes / std::max(sizeof(accum_t), sizeof(num_type)));
        constexpr size_t repeat_cnt = N / compact_N;

        using NumVec    = VecFloat<compact_N, num_type>;
        using AccumVec  = VecFloat<compact_N, accum_t>;
        using AccOps    = AdOps<accum_t>;
        accum_t buf_acc = static_cast<accum_t>(1.0f);
        AccumVec buf;

        for (size_t j = 0; j < repeat_cnt; ++j) {
            buf = reinterpret_cast<NumVec const *>(this->data)[j].template convert_vec<accum_t>();

            if constexpr (can_be_packed_new && can_be_packed_new_b<accum_t>() && compact_N >= 4) {
                using AccOps            = AdOpsPacked<accum_t>;
                using packed_t          = deduce_packed_type_t<accum_t>;
                packed_t double_buf_acc = reinterpret_cast<packed_t const *>(&buf)[0];

#pragma unroll
                for (size_t i = 1; i < compact_N / 2; ++i) {
                    double_buf_acc = AccOps::packed_mul(double_buf_acc, reinterpret_cast<packed_t const *>(&buf)[i]);
                }

                buf_acc *= double_buf_acc.x * double_buf_acc.y;
            } else {
#pragma unroll
                for (size_t i = 0; i < compact_N; ++i) {
                    buf_acc *= buf[i];
                }
            }
        }

        *acc *= buf_acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t prod_() const noexcept {
        accum_t acc = static_cast<accum_t>(1.0f);
        this->prod_<accum_t>(&acc);
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void prod(accum_t *const __restrict__ acc, vec_t vec) {
        vec.prod_<accum_t>(acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t prod(vec_t vec) {
        return vec.prod_<accum_t>();
    }

    // Min
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void min_(accum_t *const __restrict__ acc) const noexcept {
        num_type buf_acc = this->data[0];

        if constexpr (can_be_packed && N >= 4) {
            using adops_            = AdOpsPacked<num_type>;
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(this->data)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_min(double_buf_acc, reinterpret_cast<packed_t const *>(this->data)[i]);
            }
            buf_acc = adops_::min(double_buf_acc.x, double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 1; i < N; ++i) {
                buf_acc = adops_::min(buf_acc, this->data[i]);
            }
        }

        *acc = adops_::min(*acc, buf_acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t min_() const noexcept {
        accum_t acc = static_cast<accum_t>(this->data[0]);
        this->min_<accum_t>(&acc);
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void min(accum_t *const __restrict__ acc, vec_t vec) {
        vec.min_<accum_t>(acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t min(vec_t vec) {
        return vec.min_<accum_t>();
    }

    // Max
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void max_(accum_t *const __restrict__ acc) const noexcept {
        num_type buf_acc = this->data[0];

        if constexpr (can_be_packed && N >= 4) {
            using adops_            = AdOpsPacked<num_type>;
            using packed_t          = deduce_packed_type_t<num_type>;
            packed_t double_buf_acc = reinterpret_cast<packed_t const *>(this->data)[0];

#pragma unroll
            for (size_t i = 1; i < N / 2; ++i) {
                double_buf_acc = adops_::packed_max(double_buf_acc, reinterpret_cast<packed_t const *>(this->data)[i]);
            }
            buf_acc = adops_::max(double_buf_acc.x, double_buf_acc.y);
        } else {
#pragma unroll
            for (size_t i = 1; i < N; ++i) {
                buf_acc = adops_::max(buf_acc, this->data[i]);
            }
        }

        *acc = adops_::max(*acc, buf_acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t max_() const noexcept {
        accum_t acc = static_cast<accum_t>(this->data[0]);
        this->max_<accum_t>(&acc);
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void max(accum_t *const __restrict__ acc, vec_t vec) {
        vec.max_<accum_t>(acc);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t max(vec_t vec) {
        return vec.max_<accum_t>();
    }

    // Dot product
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ void dot_product_(accum_t *const __restrict__ acc, vec_t other) const noexcept {
        other *= *this;
        *acc += other.sum_<accum_t>();
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    constexpr __device__ accum_t dot_product_(vec_t other) const noexcept {
        accum_t acc{};
        dot_product_<accum_t>(&acc, other);
        return acc;
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ void dot_product(accum_t *const __restrict__ acc, vec_t src0, vec_t src1) {
        src0.dot_product_<accum_t>(acc, src1);
    }
    template <FloatingNum accum_t>
        requires(sizeof(accum_t) >= sizeof(num_type))
    static constexpr __device__ accum_t dot_product(vec_t src0, vec_t src1) {
        return src0.dot_product_<accum_t>(src1);
    }

    template <FloatingNum dst_type>
    constexpr __device__ VecFloat<N, dst_type> convert_vec() const noexcept {
        using DstOps = VecFloat<N, dst_type>;
        DstOps out_buf;

        if constexpr (std::is_same_v<std::remove_cvref_t<dst_type>, std::remove_cvref_t<num_type>>) {
            VecFloat<N, num_type>::transfer_vector(&out_buf, this);
            return out_buf;
        }

        if constexpr (((std::is_same_v<std::remove_cvref_t<num_type>, float> || is_half_fp_v<num_type>) &&
                          (std::is_same_v<std::remove_cvref_t<dst_type>, float> || is_half_fp_v<dst_type>) && (N % 2 == 0))) {
            using packed_src_t = deduce_packed_type_t<num_type>;
            using packed_dst_t = deduce_packed_type_t<dst_type>;

#pragma unroll
            for (size_t i = 0; i < N / 2; ++i) {
                reinterpret_cast<packed_dst_t *>(out_buf.data)[i] =
                    adops::packed_convert<dst_type, num_type>(reinterpret_cast<packed_src_t const *>(this->data)[i]);
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
                out_buf[i] = static_cast<dst_type>(this->data[i]);
            }
        }

        return out_buf;
    }
};

// Something like cooperative kernel-like function for copying the whole row with conversions
// row_width is D size
// worker_cnt is total numer of workers(threads), working on the whole row
template <FloatingNum dst_type, FloatingNum src_type, size_t row_width, size_t worker_cnt>
static __device__ void write_row(dst_type *const __restrict__ dst, size_t worker_idx, src_type const *const __restrict__ src) {
    constexpr size_t copy_N                  = VecFloat<1, src_type>::max_vec_size_bytes / std::max(sizeof(src_type), sizeof(dst_type));
    constexpr size_t total_copies_per_worker = (row_width + copy_N * worker_cnt - 1) / (copy_N * worker_cnt);

    using src_vec_type = VecFloat<copy_N, src_type>;
    using dst_vec_type = VecFloat<copy_N, dst_type>;

    constexpr size_t unroll_k = 4;

    for (size_t i = 0; i < (total_copies_per_worker + unroll_k - 1) / unroll_k; ++i) {
#pragma unroll
        for (size_t j = 0; j < unroll_k; ++j) {
            const size_t tile_id = worker_cnt * (i * unroll_k + j) + worker_idx;
            if (tile_id * copy_N < row_width) [[likely]] {
                reinterpret_cast<dst_vec_type *>(dst)[tile_id] =
                    reinterpret_cast<src_vec_type const *>(src)[tile_id].template convert_vec<dst_type>();
            }
        }
    }

    __syncthreads();
}

static_assert(std::is_trivially_copyable_v<VecFloat<4, float>>);
static_assert(std::is_trivially_default_constructible_v<VecFloat<4, float>>);
static_assert(std::is_trivially_destructible_v<VecFloat<4, float>>);

// Operations with whole tiles

template <size_t N, FloatingNum num_type, FloatingNum accum_t = float>
struct TileOps {
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
    using vec_t  = VecFloat<N, num_type>;
    using wide_t = vec_t::wide_t;
    using ns_t   = decltype(deduce_ns_t());

    static constexpr int TW = N;

    // Common
    static __device__ vec_t read(num_type const *const __restrict__ src_arr, size_t vec_idx) {
        return *reinterpret_cast<vec_t const *>(&src_arr[vec_idx * TW]);
    }
    static __device__ void write_zero(num_type *const __restrict__ dst_arr, size_t vec_idx) {
        reinterpret_cast<vec_t *>(&dst_arr[vec_idx * TW])->store_zero_();
    }
    static __device__ void write(num_type *const __restrict__ dst_arr, size_t vec_idx, vec_t src_val) {
        *reinterpret_cast<wide_t *>(&dst_arr[vec_idx * TW]) = *reinterpret_cast<wide_t const *>(&src_val);
    }
    static __device__ void write_convert_to_accum(accum_t *const __restrict__ dst, num_type const *const __restrict__ src) {
        constexpr size_t compact_N  = std::min(N, VecFloat<1, num_type>::max_vec_size_bytes / std::max(sizeof(num_type), sizeof(accum_t)));
        constexpr size_t repeat_cnt = N / compact_N;

        for (size_t i = 0; i < repeat_cnt; ++i) {
            reinterpret_cast<VecFloat<compact_N, accum_t> *>(dst)[i] =
                reinterpret_cast<VecFloat<compact_N, num_type> const *>(src)[i].template convert_vec<accum_t>();
        }
    }
    static __device__ void write_convert_from_accum(num_type *const __restrict__ dst, accum_t const *const __restrict__ src) {
        constexpr size_t compact_N  = std::min(N, VecFloat<1, num_type>::max_vec_size_bytes / std::max(sizeof(num_type), sizeof(accum_t)));
        constexpr size_t repeat_cnt = N / compact_N;

        for (size_t i = 0; i < repeat_cnt; ++i) {
            reinterpret_cast<VecFloat<compact_N, num_type> *>(dst)[i] =
                reinterpret_cast<VecFloat<compact_N, accum_t> const *>(src)[i].template convert_vec<num_type>();
        }
    }

    // GATv2
    static __device__ ns_t make_ns(accum_t ns) {
        // Static casts to float are temporary. If somebody wants to fix it, be my guest.
        if constexpr (std::is_same_v<half, std::remove_cvref_t<num_type>>) {
            return __float2half2_rn(static_cast<float>(ns));
        } else if constexpr (std::is_same_v<nv_bfloat16, std::remove_cvref_t<num_type>>) {
            return __float2bfloat162_rn(static_cast<float>(ns));
        } else {
            return static_cast<num_type>(ns);
        }
    }
    static __device__ accum_t gatv2_dot_leaky_relu(vec_t l, vec_t r, vec_t a, accum_t ns) {
        l += r;
        l.leaky_relu_(ns);
        l *= a;

        return l.template sum_<accum_t>();
    }
    static __device__ void gatv2_accum_grad_al(
        accum_t *const __restrict__ ga, accum_t *const __restrict__ gl, accum_t ge, vec_t l, vec_t r, vec_t a, accum_t ns
    ) {
        VecFloat<N, num_type> ge_vec(static_cast<num_type>(ge));
        l += r;
        VecFloat<N, num_type> buf_vec = l;

        buf_vec.leaky_relu_backward_(ge_vec, ns);

        constexpr size_t compact_N  = std::min(N, sizeof(accum_t));
        constexpr size_t repeat_cnt = N / compact_N;
        using vec_compact_t         = VecFloat<compact_N, num_type>;
        using AccumVec              = VecFloat<compact_N, accum_t>;

#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            AccumVec l_out, bv_out, a_out;
            l_out  = reinterpret_cast<vec_compact_t const *>(&l)[i].template convert_vec<accum_t>();
            bv_out = reinterpret_cast<vec_compact_t const *>(&buf_vec)[i].template convert_vec<accum_t>();
            a_out  = reinterpret_cast<vec_compact_t const *>(&a)[i].template convert_vec<accum_t>();

            // ga += (ge * t) * edge,  gl += (ge * t) * a
            reinterpret_cast<VecFloat<compact_N, accum_t> *>(ga)[i].fmaa_(bv_out, l_out);
            reinterpret_cast<VecFloat<compact_N, accum_t> *>(gl)[i].fmaa_(bv_out, a_out);
        }
    }
    static __device__ void gatv2_accum_grad_r(
        accum_t *const __restrict__ gr, accum_t alpha, vec_t gh, accum_t ge, vec_t l, vec_t r, vec_t a, accum_t ns
    ) {
        VecFloat<N, num_type> ge_vec(static_cast<num_type>(ge)), alpha_vec(static_cast<num_type>(alpha));

        l += r;
        l.leaky_relu_backward_(ge_vec, ns);
        l *= a;
        l.fmaa_(gh, alpha_vec);

        constexpr size_t compact_N  = std::min(N, sizeof(accum_t));
        constexpr size_t repeat_cnt = N / compact_N;
        using vec_compact_t         = VecFloat<compact_N, num_type>;
        using AccumVec              = VecFloat<compact_N, accum_t>;

#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            AccumVec l_out = reinterpret_cast<vec_compact_t const *>(&l)[i].template convert_vec<accum_t>();
            reinterpret_cast<VecFloat<compact_N, accum_t> *>(gr)[i] += l_out;
        }
    }
    template <FloatingNum atomic_t = float>
    static __device__ void atomic_add_scaled_f32(atomic_t *const __restrict__ ptr, size_t vec_idx, atomic_t scalar, vec_t v) {
        static_assert(std::is_same_v<std::remove_cvref_t<atomic_t>, float>, "atomic add is only implememnted for float atomic type.");

        constexpr size_t compact_N  = std::min(N, sizeof(atomic_t));
        constexpr size_t repeat_cnt = N / compact_N;
        using vec_compact_t         = VecFloat<compact_N, num_type>;
        using AtomicVec             = VecFloat<compact_N, atomic_t>;

        VecFloat<compact_N, atomic_t> buf;
#pragma unroll
        for (size_t i = 0; i < repeat_cnt; ++i) {
            buf = reinterpret_cast<vec_compact_t const *>(&v)[i].template convert_vec<atomic_t>();
            buf *= scalar;
#pragma unroll
            for (size_t j = 0; j < compact_N; ++j) {
                atomicAdd(&ptr[vec_idx + i * compact_N + j], buf[j]);
            }
        }
    }
};
