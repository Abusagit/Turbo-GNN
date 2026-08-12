#pragma once

#include "common/tile.cuh"

// =============================================================================
// GATv2 Kernel with CSR Graph Format
// =============================================================================

template <int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
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

    using Tile = TileOps<TW, cuda_t>;

    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

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

    // Shared memory layout:
    //   l_sh:      D_CONST * sizeof(cuda_t)                        -- read-only
    //   warp_out:  WARPS_PER_BLOCK * D_CONST * sizeof(float)       -- per-warp output accum
    //   warp_max:  WARPS_PER_BLOCK * sizeof(float)                 -- per-warp softmax max
    //   warp_sum:  WARPS_PER_BLOCK * sizeof(float)                 -- per-warp softmax sum_exp
    extern __shared__ uint8_t sh_raw[];
    cuda_t *l_sh    = reinterpret_cast<cuda_t *>(sh_raw);
    float *warp_out = reinterpret_cast<float *>(sh_raw + D_CONST * sizeof(cuda_t));
    float *warp_max = warp_out + WARPS_PER_BLOCK * D_CONST;
    float *warp_sum = warp_max + WARPS_PER_BLOCK;

    float *my_out = warp_out + warp_id * D_CONST;

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

    ns_t ns = Tile::make_ns(negative_slope);

    // Per-warp register accumulators
    float h_acc[ACCS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
        h_acc[i] = 0.f;
    }

    OnlineSoftmaxState softmax_state;

    // Warp-strided neighbor loop
    for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
        index_t neighbor_j   = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t *r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float dot_lane = 0.f;
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv = Tile::load(l_sh, v);
                vec_t rv = Tile::load(r_base, v);
                vec_t av = Tile::load(a_base, v);
                dot_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
            }
        }
        float dot = warp_reduce_sum(dot_lane);

        float rescale = softmax_state.update(dot);
#pragma unroll
        for (int i = 0; i < ACCS_PER_THREAD; ++i) {
            h_acc[i] *= rescale;
        }

        float contrib = __expf(dot - softmax_state.max_val);
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t rv = Tile::load(r_base, v);
                Tile::weighted_accum(&h_acc[t * TW], contrib, rv);
            }
        }
    }

// Write per-warp results to shared memory
#pragma unroll
    for (int t = 0; t < TILES_PER_THREAD; ++t) {
        int v = lane + kWarpSize * t;
        if (v < TILES) {
            Tile::write_float(my_out, v, &h_acc[t * TW]);
        }
    }

    if (lane == 0) {
        warp_max[warp_id] = softmax_state.max_val;
        warp_sum[warp_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // Cross-warp online-softmax reduction (warp 0 only)
    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        float global_sum = 0.0f;
        float inv_sum    = 0.0f;

        if (lane == 0) {
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_max = fmaxf(global_max, warp_max[w]);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_sum = fmaf(warp_sum[w], __expf(warp_max[w] - global_max), global_sum);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                warp_sum[w] = __expf(warp_max[w] - global_max);
            }
            inv_sum                                       = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
            d_logsumexp_out[(int64_t)node_i * H + head_h] = (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY;
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

// Combine all warps' outputs with proper rescaling
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                float combined[TW];
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    combined[ep] = 0.0f;
                    int d_idx    = v * TW + ep;
#pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                        combined[ep] = fmaf(warp_sum[w], warp_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_typed(h_out_base, v, combined);
            }
        }
    }
}
