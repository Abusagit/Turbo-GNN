#include <cstdint>

#include "common.cuh"

template <int WARPS_PER_N, int N_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_N * N_PER_BLOCK * kWarpSize) GraphAttentionForward_CSR_MH_v2_D( // no-format
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

    constexpr int TW = TW_SELECTOR::value;  // Tile width
    static_assert(D_CONST % TW == 0, "Per-head features dim should be divisible by Tile width");
    constexpr int TILES = D_CONST / TW;  // Total tiles count
    constexpr int TILES_PER_THREAD =
        (TILES + (TW_SELECTOR::threads_per_d * WARPS_PER_N) - 1) / (TW_SELECTOR::threads_per_d * WARPS_PER_N);  // Tiles per thread
    constexpr int ACCS_PER_THREAD = TW * TILES_PER_THREAD;  // Accumulatores used by one thread

    using Tile = TileOps<TW, cuda_t>;

    const int node_i = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h = blockIdx.y;

    __builtin_assume(threadIdx.y < static_cast<unsigned>(WARPS_PER_N));
    __builtin_assume(threadIdx.z < static_cast<unsigned>(N_PER_BLOCK));
    const int lane_id = threadIdx.x;
    __builtin_assume(lane_id < static_cast<int>(kWarpSize));
    constexpr int lane_cnt            = kWarpSize;
    const int warp_id                 = threadIdx.y;
    constexpr int neighbor_warp_cnt   = WARPS_PER_N;
    const int block_neighbor_id       = threadIdx.z;
    constexpr int neighbor_block_size = N_PER_BLOCK;

    if (node_i >= N || head_h >= H) [[unlikely]] {
        return;
    }

    const index_t edge_start = row_ptr[node_i];
    const index_t edge_end   = row_ptr[node_i + 1];
    const int num_neighbors  = static_cast<int>(edge_end - edge_start);

    // Shared memory layout (unchanged):
    // k_shared[D_CONST] as cuda_t
    // neighbor_out[neighbor_block_size * D_CONST] as float
    // neighbor_max[neighbor_block_size] as float
    // neighbor_sum[neighbor_block_size] as float
    // warp_sum_storage[neighbor_block_size * neighbor_warp_cnt] as float
    extern __shared__ uint8_t sh_raw[];
    cuda_t *const k_shared        = reinterpret_cast<cuda_t *>(sh_raw);  // Loading K_i into shared memory, because it's the same in one block
    float *const neighbor_out     = reinterpret_cast<float *>(sh_raw + D_CONST * sizeof(cuda_t));  // Outs for each neighbor in one block
    float *const neighbor_max     = neighbor_out + neighbor_block_size * D_CONST;  // Space to store local neighbor maximums of scores
    float *const neighbor_sum     = neighbor_max + neighbor_block_size;            // Space to store local neighbor sums of score
    float *const warp_sum_storage = neighbor_sum + neighbor_block_size;            // Space to store warp sums to agregate them later

    float *const my_out = neighbor_out + block_neighbor_id * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) [[unlikely]] {
        if (block_neighbor_id == 0) {
            cuda_t *out_base = O + node_i * stride_o_n + head_h * stride_o_h;
            for (int vi = warp_id * lane_cnt + lane_id; vi < TILES; vi += lane_cnt * neighbor_warp_cnt) {
                Tile::write_zero(out_base, vi);
            }
            if (warp_id == 0 && lane_id == 0) {
                logsumexp[node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    // cooperative load of K_i via 128-bit transactions (unchanged)
    // TODO: make separate function
    if (block_neighbor_id == 0) {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);  // remainder is guaranteed to be zero
        static_assert(D_CONST % ELEMS_PER_F4 == 0, "Per-head feature dim should be divisible by 8");
        constexpr int NUM_K_LOADS  = D_CONST / ELEMS_PER_F4;
        cuda_t const *const k_base = K + node_i * stride_k_n + head_h * stride_k_h;
        float4 const *const k_src  = reinterpret_cast<float4 const *>(k_base);
        float4 *const k_sh         = reinterpret_cast<float4 *>(k_shared);
        for (int i = warp_id * lane_cnt + lane_id; i < NUM_K_LOADS; i += neighbor_warp_cnt * lane_cnt) {
            k_sh[i] = k_src[i];
        }
    }
    __syncthreads();

    OnlineSoftmaxState softmax_state;

    float o_acc[ACCS_PER_THREAD] = {0};

    // neighbor loop
    const int rounds = (num_neighbors + neighbor_block_size - 1) / neighbor_block_size;
    for (int r = 0; r < rounds; ++r) {
        const int neighbor_id = r * neighbor_block_size + block_neighbor_id;
        const bool active = neighbor_id < num_neighbors;
        const index_t j   = active ? col_idx[edge_start + neighbor_id] : index_t{0};

        const cuda_t *q_base = Q + j * stride_q_n + head_h * stride_q_h;
        const cuda_t *v_base = V + j * stride_v_n + head_h * stride_v_h;

        float s_partial = 0.0f;

        // Q*K dot product (uses improved dot_product with native mul)
        if (active) {
#pragma unroll
            for (int tile_id = warp_id * lane_cnt + lane_id; tile_id < TILES; tile_id += lane_cnt * neighbor_warp_cnt) {
                const typename Tile::vec_t kv = Tile::load(k_shared, tile_id);
                const typename Tile::vec_t qv = Tile::load(q_base, tile_id);
                s_partial += Tile::dot_product(kv, qv);
            }
        }

        float score;
        if constexpr (neighbor_warp_cnt > 1) {
            if (active) {
                auto local_score = warp_reduce_sum(s_partial);
                if (lane_id == 0) {
                    warp_sum_storage[(r & 1) * neighbor_warp_cnt * neighbor_block_size + block_neighbor_id * neighbor_warp_cnt + warp_id] = local_score;
                }
            }
            __syncthreads();

            if (active) {
                float score_ = 0;
#pragma unroll
                for (int i = block_neighbor_id * neighbor_warp_cnt; i < (block_neighbor_id + 1) * neighbor_warp_cnt; ++i) {
                    score_ += warp_sum_storage[(r & 1) * neighbor_warp_cnt * neighbor_block_size + i];
                }
                score = score_ * scale;
            }
        } else {
            if (active) {
                score = warp_reduce_sum(s_partial) * scale;
            }
        }

        if (!active) {
            break;
        }

        const float correction = softmax_state.update(score);
        const float w          = __expf(score - softmax_state.max_val);

        // V accumulation (keeps fmaf via weighted_accum)
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            const int vi = warp_id * lane_cnt + lane_id + lane_cnt * neighbor_warp_cnt * t;
            if (vi < TILES) [[likely]] {
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    o_acc[t * TW + ep] *= correction;
                }
                const typename Tile::vec_t vv = Tile::load(v_base, vi);
                Tile::weighted_accum(&o_acc[t * TW], w, vv);
            }
        }
    }

// write per-warp results to float32 shared
#pragma unroll
    for (int t = 0; t < TILES_PER_THREAD; ++t) {
        const int vi = warp_id * lane_cnt + lane_id + lane_cnt * neighbor_warp_cnt * t;
        if (vi < TILES) [[likely]] {
            Tile::write_float(my_out, vi, &o_acc[t * TW]);
        }
    }

    if (lane_id == 0 && warp_id == 0) {
        neighbor_max[block_neighbor_id] = softmax_state.max_val;
        neighbor_sum[block_neighbor_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // cross-warp reduction (warp 0 only)
    if (warp_id == 0 && block_neighbor_id == 0) {
        float global_max = -FLT_MAX;
        float global_sum = 0.0f;
        float inv_sum    = 0.0f;

        if (lane_id == 0) {
            // #pragma unroll
            for (int w = 0; w < neighbor_block_size; ++w) {
                global_max = fmaxf(global_max, neighbor_max[w]);
            }
            // #pragma unroll
            for (int w = 0; w < neighbor_block_size; ++w) {
                global_sum = fmaf(neighbor_sum[w], __expf(neighbor_max[w] - global_max), global_sum);
            }
            // #pragma unroll
            for (int w = 0; w < neighbor_block_size; ++w) {
                neighbor_sum[w] = __expf(neighbor_max[w] - global_max);  // scale_w
            }

            inv_sum = fmaxf(1.0f / global_sum, 0.0f);
            // This is correct for node that has at least 1 neighbour
            // (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY; - always correct
            logsumexp[node_i * H + head_h] = fmaxf(global_max + logf(global_sum), -INFINITY);
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

        // cross-neighbor output write (uses write_typed for vec2 stores)
        cuda_t *const out_base = O + node_i * stride_o_n + head_h * stride_o_h;
#pragma unroll
        for (int t = 0; t < (TILES + lane_cnt - 1) / lane_cnt; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < TILES) [[likely]] {
                float combined[TW] = {0};
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    int d_idx = vi * TW + ep;
#pragma unroll
                    for (int w = 0; w < neighbor_block_size; ++w) {
                        combined[ep] = fmaf(neighbor_sum[w], neighbor_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_typed(out_base, vi, combined);
            }
        }
    }
}
