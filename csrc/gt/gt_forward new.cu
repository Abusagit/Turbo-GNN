#include <cstdint>

#include "common.cuh"

template <int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize) GraphAttentionForward_CSR_MH_v2_D_new( // no-format
    int N, int H,
    const cuda_t *__restrict__ Q, const cuda_t *__restrict__ K, const cuda_t *__restrict__ V,
    int64_t stride_q_n, int64_t stride_q_h,
    int64_t stride_k_n, int64_t stride_k_h,
    int64_t stride_v_n, int64_t stride_v_h,
    const index_t *__restrict__ row_ptr, const index_t *__restrict__ col_idx,
    const index_t *__restrict__ node_indices,  // node indirection: node_i = node_indices[blockIdx.x]
    cuda_t *__restrict__ O, int64_t stride_o_n, int64_t stride_o_h,
    float *__restrict__ logsumexp, float scale
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be multiple of 32 for this fast path");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;                                                  // Accumulatores used by one thread

    using Tile = TileOps<TW, cuda_t>;

    const int node_i = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h = blockIdx.y;

    const int lane_id     = threadIdx.x;
    const int warp_id     = threadIdx.y;
    const int neighbor_id = threadIdx.z;

    if (node_i >= N || head_h >= H) [[unlikely]] {
        return;
    }

    const index_t edge_start = row_ptr[node_i];
    const index_t edge_end   = row_ptr[node_i + 1];
    const int num_neighbors  = static_cast<int>(edge_end - edge_start);

    // Shared memory layout (unchanged):
    // k_shared[D_CONST] as cuda_t
    // warp_out[WARPS_PER_BLOCK * D_CONST] as float
    // warp_max[WARPS_PER_BLOCK] as float
    // warp_sum[WARPS_PER_BLOCK] as float
    extern __shared__ uint8_t sh_raw[];
    cuda_t *const k_shared = reinterpret_cast<cuda_t *>(sh_raw);
    float *const warp_out  = reinterpret_cast<float *>(sh_raw + D_CONST * sizeof(cuda_t));
    float *const warp_max  = warp_out + WARPS_PER_BLOCK * D_CONST;
    float *const warp_sum  = warp_max + WARPS_PER_BLOCK;

    float *const my_out = warp_out + warp_id * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            cuda_t *out_base = O + node_i * stride_o_n + head_h * stride_o_h;
            for (int vi = lane_id; vi < TILES; vi += kWarpSize) {
                Tile::write_zero(out_base, vi);
            }
            if (lane_id == 0) {
                logsumexp[node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    // cooperative load of K_i via 128-bit transactions (unchanged)
    // TODO: make separate function
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);  // remainder is guaranteed to be zero
        constexpr int NUM_K_LOADS  = D_CONST / ELEMS_PER_F4;
        const cuda_t *k_base       = K + node_i * stride_k_n + head_h * stride_k_h;
        const float4 *k_src        = reinterpret_cast<const float4 *>(k_base);
        float4 *k_sh               = reinterpret_cast<float4 *>(k_shared);
        for (int i = threadIdx.x; i < NUM_K_LOADS; i += WARPS_PER_BLOCK * kWarpSize) {
            k_sh[i] = k_src[i];
        }
    }
    __syncthreads();

    OnlineSoftmaxState softmax_state;

    float o_acc[ACCS_PER_THREAD] = {0};

    // neighbor loop
    for (int e = warp_id; e < num_neighbors; e += WARPS_PER_BLOCK) {
        const index_t j = col_idx[edge_start + e];

        const cuda_t *q_base = Q + j * stride_q_n + head_h * stride_q_h;
        const cuda_t *v_base = V + j * stride_v_n + head_h * stride_v_h;

        // Q·K dot product (uses improved dot_product with native mul)
        float s_partial = 0.0f;
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < TILES) {
                auto kv = Tile::load(k_shared, vi);
                auto qv = Tile::load(q_base, vi);
                s_partial += Tile::dot_product(kv, qv);
            }
        }

        const float score      = warp_reduce_sum(s_partial) * scale;
        const float correction = softmax_state.update(score);
        const float w          = __expf(score - softmax_state.max_val);

// V accumulation (keeps fmaf via weighted_accum)
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < TILES) {
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    o_acc[t * TW + ep] *= correction;
                }
                auto vv = Tile::load(v_base, vi);
                Tile::weighted_accum(&o_acc[t * TW], w, vv);
            }
        }
    }

// write per-warp results to float32 shared
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int vi = lane_id + t * kWarpSize;
        if (vi < TILES) {
            Tile::write_float(my_out, vi, &o_acc[t * TW]);
        }
    }

    if (lane_id == 0) {
        warp_max[warp_id] = softmax_state.max_val;
        warp_sum[warp_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // cross-warp reduction (warp 0 only)
    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        float global_sum = 0.0f;
        float inv_sum    = 0.0f;

        if (lane_id == 0) {
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
                warp_sum[w] = __expf(warp_max[w] - global_max);  // scale_w
            }

            inv_sum = fmaxf(1.0f / global_sum, 0.0f);
            // This is correct for node that has at least 1 neighbour
            // (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY; - always correct
            logsumexp[node_i * H + head_h] = fmaxf(global_max + logf(global_sum), -INFINITY);
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

        // cross-warp output write (uses write_typed for vec2 stores)
        cuda_t *out_base = O + node_i * stride_o_n + head_h * stride_o_h;
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < TILES) {
                float combined[TW] = {0};
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    int d_idx = vi * TW + ep;
#pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                        combined[ep] = fmaf(warp_sum[w], warp_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_typed(out_base, vi, combined);
            }
        }
    }
}