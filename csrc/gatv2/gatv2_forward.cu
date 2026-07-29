#pragma once

#include <cstdint>
#include <cuda/pipeline>

#include "common.cuh"

// =============================================================================
// GATv2 Kernel with CSR Graph Format
// =============================================================================

template <
    int WARPS_PER_BLOCK, int D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float, bool USE_PIPELINE = false,
    int NUM_STAGES = 2>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) GATv2Forward_Kernel(
    size_t N,
    size_t H,
    size_t D,
    const cuda_t *__restrict__ d_l,
    const cuda_t *__restrict__ d_r,
    int64_t stride_l_n,
    int64_t stride_l_h,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr,
    const index_t *__restrict__ d_col_idx,
    const index_t *__restrict__ node_indices,  // node indirection
    const cuda_t *__restrict__ d_attn_vec,
    cuda_t *__restrict__ d_h_out,
    float *__restrict__ d_logsumexp_out,
    float negative_slope
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;                                                  // Accumulatores used by one thread

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    using vec_t = typename Tile::vec_t;

    const int node_i  = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h  = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) [[unlikely]] {
        return;
    }

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    cuda_t *h_out_base = d_h_out + ((int64_t)node_i * H + head_h) * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < TILES; v += kWarpSize) {
                Tile::write_zero(h_out_base, v);
            }
            if (lane == 0) {
                d_logsumexp_out[(int64_t)node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    const cuda_t *l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t *a_base = d_attn_vec + head_h * D_CONST;

    static_assert(!USE_PIPELINE || NUM_STAGES >= 2, "pipeline needs >= 2 stages");

    // Shared memory layout:
    //   l_sh:      D_CONST * sizeof(cuda_t)                              -- read-only
    //   r_dbuf:    WARPS_PER_BLOCK * NUM_STAGES * D_CONST * sizeof(cuda_t) -- per-warp ping-pong for async r[j], only when USE_PIPELINE
    //   warp_out:  WARPS_PER_BLOCK * D_CONST * sizeof(accum_t)           -- per-warp output accum
    //   warp_max:  WARPS_PER_BLOCK * sizeof(accum_t)                     -- per-warp softmax max
    //   warp_sum:  WARPS_PER_BLOCK * sizeof(accum_t)                     -- per-warp softmax sum_exp
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *l_sh   = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *r_dbuf = l_sh + D_CONST;  // only meaningful when USE_PIPELINE

    constexpr size_t r_dbuf_bytes = USE_PIPELINE ? WARPS_PER_BLOCK * NUM_STAGES * D_CONST * sizeof(cuda_t) : 0;
    accum_t *warp_out             = reinterpret_cast<accum_t *>(sh_raw + D_CONST * sizeof(cuda_t) + r_dbuf_bytes);
    accum_t *warp_max             = warp_out + WARPS_PER_BLOCK * D_CONST;
    accum_t *warp_sum             = warp_max + WARPS_PER_BLOCK;

    accum_t *my_out = warp_out + warp_id * D_CONST;

    // Cooperative load of l into shared memory using all threads
    {
        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4 *l_src4   = reinterpret_cast<const float4 *>(l_base);
        float4 *l_sh4          = reinterpret_cast<float4 *>(l_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kWarpSize) {
            l_sh4[i] = l_src4[i];
        }
    }
    __syncthreads();

    // Per-warp register accumulators
    accum_t h_acc[ACCS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
        h_acc[i] = accum_t{};
    }

    OnlineSoftmaxState softmax_state;

    if constexpr (USE_PIPELINE) {
        constexpr int R_BYTES  = D_CONST * (int)sizeof(cuda_t);
        constexpr int F4_PER_R = R_BYTES / 16;

        const int loop_iters =
            (num_neighbors > warp_id) ? (num_neighbors - warp_id + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK : 0;

        if (loop_iters > 0) {
            cuda_t *rows[NUM_STAGES];
#pragma unroll
            for (int s = 0; s < NUM_STAGES; ++s) {
                rows[s] = r_dbuf + (warp_id * NUM_STAGES + s) * D_CONST;
            }

            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            auto async_copy_r = [&](cuda_t *dst, const cuda_t *src) {
                for (int i = lane; i < F4_PER_R; i += kWarpSize) {
                    cuda::memcpy_async(
                        reinterpret_cast<char *>(dst) + i * 16, reinterpret_cast<const char *>(src) + i * 16, cuda::aligned_size_t<16>(16), pipe
                    );
                }
            };

            auto prefetch = [&](int it) {
                pipe.producer_acquire();
                if (it < loop_iters) {
                    const int k         = warp_id + it * WARPS_PER_BLOCK;
                    const index_t j     = d_col_idx[edge_start + static_cast<index_t>(k)];
                    const cuda_t *r_src = d_r + j * stride_r_n + head_h * stride_r_h;
                    async_copy_r(rows[it % NUM_STAGES], r_src);
                }
                pipe.producer_commit();
            };

#pragma unroll
            for (int s = 0; s < NUM_STAGES - 1; ++s) {
                prefetch(s);
            }

            for (int iter = 0; iter < loop_iters; ++iter) {
                prefetch(iter + NUM_STAGES - 1);

                cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
                cuda_t *r_cur = rows[iter % NUM_STAGES];

                accum_t dot_lane{};
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        const vec_t lv = Tile::read(l_sh, v);
                        const vec_t rv = Tile::read(r_cur, v);
                        const vec_t av = Tile::read(a_base, v);
                        dot_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, negative_slope);
                    }
                }
                const accum_t dot = warp_reduce_sum(dot_lane);

                const accum_t rescale = softmax_state.update(dot);
#pragma unroll
                for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                    h_acc[i] *= rescale;
                }

                const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        const vec_t rv = Tile::read(r_cur, v);
                        rv.template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                    }
                }

                pipe.consumer_release();
            }
        }
    } else {
        // Warp-strided neighbor loop
        for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
            index_t neighbor_j   = d_col_idx[edge_start + static_cast<index_t>(k)];
            const cuda_t *r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

            accum_t dot_lane{};
#pragma unroll
            for (int t = 0; t < TILES_PER_THREAD; ++t) {
                int v = lane + kWarpSize * t;
                if (v < TILES) {
                    const vec_t lv = Tile::read(l_sh, v);
                    const vec_t rv = Tile::read(r_base, v);
                    const vec_t av = Tile::read(a_base, v);
                    dot_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, negative_slope);
                }
            }
            const accum_t dot = warp_reduce_sum(dot_lane);

            const accum_t rescale = softmax_state.update(dot);
#pragma unroll
            for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                h_acc[i] *= rescale;
            }

            const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
            for (int t = 0; t < TILES_PER_THREAD; ++t) {
                int v = lane + kWarpSize * t;
                if (v < TILES) {
                    const vec_t rv = Tile::read(r_base, v);

                    rv.template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                }
            }
        }
    }

// Write per-warp results to shared memory
#pragma unroll
    for (int t = 0; t < TILES_PER_THREAD; ++t) {
        const int v = lane + kWarpSize * t;
        if (v < TILES) {
            constexpr size_t compact_N  = std::min<size_t>(TW, VecFloat<1, cuda_t>::max_vec_size_bytes / std::max(sizeof(cuda_t), sizeof(accum_t)));
            constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
            for (size_t i = 0; i < repeat_cnt; ++i) {
                TileOps<compact_N, accum_t>::write(
                    my_out, v * repeat_cnt + i, reinterpret_cast<VecFloat<compact_N, accum_t> const *>(h_acc)[t * repeat_cnt + i]
                );
            }
        }
    }

    if (lane == 0) {
        warp_max[warp_id] = softmax_state.max_val;
        warp_sum[warp_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // Cross-warp online-softmax reduction (warp 0 only)
    if (warp_id == 0) {
        accum_t global_max = -FLT_MAX;
        accum_t global_sum{};
        accum_t inv_sum{};

        if (lane == 0) {
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_max = AccumOps::max(global_max, warp_max[w]);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_sum = AccumOps::fma(warp_sum[w], AccumOps::exp(warp_max[w] - global_max), global_sum);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                warp_sum[w] = AccumOps::exp(warp_max[w] - global_max);
            }
            inv_sum                                       = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
            d_logsumexp_out[(int64_t)node_i * H + head_h] = (global_sum > 0.0f) ? (global_max + AccumOps::log(global_sum)) : -INFINITY;
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

// Combine all warps' outputs with proper rescaling
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                accum_t combined[TW];
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    combined[ep]    = accum_t{};
                    const int d_idx = v * TW + ep;
#pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                        combined[ep] = AccumOps::fma(warp_sum[w], warp_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_convert_from_accum(&h_out_base[v * TW], combined);
            }
        }
    }
}
