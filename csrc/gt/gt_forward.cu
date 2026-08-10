#include <cstdint>

#include "common.cuh"

template <size_t WARPS_PER_N, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(WARPS_PER_N * N_PER_BLOCK * kWarpSize) GraphAttentionForward_CSR_MH_v2_D( // no-format
    size_t N, size_t H,
    const cuda_t *__restrict__ Q, const cuda_t *__restrict__ K, const cuda_t *__restrict__ V,
    int64_t stride_q_n, int64_t stride_q_h,
    int64_t stride_k_n, int64_t stride_k_h,
    int64_t stride_v_n, int64_t stride_v_h,
    const index_t *__restrict__ row_ptr, const index_t *__restrict__ col_idx,
    const index_t *__restrict__ node_indices,  // node indirection: node_i = node_indices[blockIdx.x]
    cuda_t *__restrict__ O, int64_t stride_o_n, int64_t stride_o_h,
    accum_t *__restrict__ logsumexp, accum_t scale
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be multiple of 32 for this fast path");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr size_t TW = TW_SELECTOR::value;  // Tile width
    static_assert(D_CONST % TW == 0, "Per-head features dim should be divisible by Tile width");
    constexpr size_t TILES = D_CONST / TW;  // Total tiles count
    constexpr size_t TILES_PER_THREAD =
        (TILES + (TW_SELECTOR::threads_per_d * WARPS_PER_N) - 1) / (TW_SELECTOR::threads_per_d * WARPS_PER_N);  // Tiles per thread
    constexpr size_t ACCS_PER_THREAD = TW * TILES_PER_THREAD;  // Accumulatores used by one thread

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    const size_t node_i = static_cast<size_t>(node_indices[blockIdx.x]);
    const size_t head_h = blockIdx.y;

    __builtin_assume(threadIdx.y < static_cast<unsigned>(WARPS_PER_N));
    __builtin_assume(threadIdx.z < static_cast<unsigned>(N_PER_BLOCK));
    const size_t lane_id = threadIdx.x;
    __builtin_assume(lane_id < static_cast<size_t>(kWarpSize));
    constexpr size_t lane_cnt            = kWarpSize;
    const size_t warp_id                 = threadIdx.y;
    constexpr size_t neighbor_warp_cnt   = WARPS_PER_N;
    const size_t block_neighbor_id       = threadIdx.z;
    constexpr size_t neighbor_block_size = N_PER_BLOCK;

    if (node_i >= N || head_h >= H) [[unlikely]] {
        return;
    }

    const index_t edge_start   = row_ptr[node_i];
    const index_t edge_end     = row_ptr[node_i + 1];
    const size_t num_neighbors = static_cast<int>(edge_end - edge_start);

    // Shared memory layout. Ordered so that every array written through a vector
    // (Vec<N>/float4) store starts on a 16-byte boundary; the scalar-only arrays go last:
    // k_shared[D_CONST] as cuda_t                                    -- float4 loads, needs 16B
    // neighbor_out[neighbor_block_size * D_CONST] as accum_t         -- Vec<compact_N> stores, needs up to 16B
    // warp_sum_storage[2 * neighbor_block_size * neighbor_warp_cnt] as accum_t -- scalar, double-buffered on (r & 1)
    // neighbor_max[neighbor_block_size] as accum_t                   -- scalar
    // neighbor_sum[neighbor_block_size] as accum_t                   -- scalar
    //
    // D_CONST * sizeof(cuda_t) is a multiple of 16 for every supported (D, dtype),
    // so neighbor_out lands 16B-aligned; putting the scalar arrays first would offset
    // it by a single accum_t and misalign the wide stores below.
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *const k_shared      = reinterpret_cast<cuda_t *>(sh_raw);  // Loading K_i into shared memory, because it's the same in one block
    accum_t *const neighbor_out = reinterpret_cast<accum_t *>(sh_raw + D_CONST * sizeof(cuda_t));  // Outs for each neighbor in one block
    accum_t *const warp_sum_storage =
        neighbor_out + neighbor_block_size * D_CONST;  // Space to store warp sums to agregate them later (2 round buffers)
    accum_t *const neighbor_max =
        warp_sum_storage + 2 * neighbor_block_size * neighbor_warp_cnt;  // Space to store local neighbor maximums of scores
    accum_t *const neighbor_sum = neighbor_max + neighbor_block_size;    // Space to store local neighbor sums of score

    accum_t *const my_out = neighbor_out + block_neighbor_id * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) [[unlikely]] {
        if (block_neighbor_id == 0) {
            cuda_t *out_base = O + node_i * stride_o_n + head_h * stride_o_h;
            for (size_t vi = warp_id * lane_cnt + lane_id; vi < TILES; vi += lane_cnt * neighbor_warp_cnt) {
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
        constexpr size_t ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);  // remainder is guaranteed to be zero
        static_assert(D_CONST % ELEMS_PER_F4 == 0, "Per-head feature dim should be divisible by 8");
        constexpr size_t NUM_K_LOADS = D_CONST / ELEMS_PER_F4;
        cuda_t const *const k_base   = K + node_i * stride_k_n + head_h * stride_k_h;
        float4 const *const k_src    = reinterpret_cast<float4 const *>(k_base);
        float4 *const k_sh           = reinterpret_cast<float4 *>(k_shared);
        for (size_t i = warp_id * lane_cnt + lane_id; i < NUM_K_LOADS; i += neighbor_warp_cnt * lane_cnt) {
            k_sh[i] = k_src[i];
        }
    }
    __syncthreads();

    OnlineSoftmaxState softmax_state;

    accum_t o_acc[ACCS_PER_THREAD] = {0};

    // neighbor loop
    const size_t rounds = (num_neighbors + neighbor_block_size - 1) / neighbor_block_size;
    for (size_t r = 0; r < rounds; ++r) {
        const size_t neighbor_id = r * neighbor_block_size + block_neighbor_id;
        const bool active        = neighbor_id < num_neighbors;
        const index_t j          = active ? col_idx[edge_start + neighbor_id] : index_t{0};

        const cuda_t *q_base = Q + j * stride_q_n + head_h * stride_q_h;
        const cuda_t *v_base = V + j * stride_v_n + head_h * stride_v_h;

        accum_t s_partial{};

        // Q*K dot product (uses improved dot_product with native mul)
        if (active) {
#pragma unroll
            for (size_t tile_id = warp_id * lane_cnt + lane_id; tile_id < TILES; tile_id += lane_cnt * neighbor_warp_cnt) {
                const typename Tile::vec_t kv = Tile::read(k_shared, tile_id);
                const typename Tile::vec_t qv = Tile::read(q_base, tile_id);
                Tile::dot_product(&s_partial, &kv, &qv);
            }
        }

        accum_t score;
        if constexpr (neighbor_warp_cnt > 1) {
            if (active) {
                auto local_score = warp_reduce_sum(s_partial);
                if (lane_id == 0) {
                    warp_sum_storage[(r & 1) * neighbor_warp_cnt * neighbor_block_size + block_neighbor_id * neighbor_warp_cnt + warp_id] =
                        local_score;
                }
            }
            __syncthreads();

            if (active) {
                accum_t score_{};
#pragma unroll
                for (size_t i = block_neighbor_id * neighbor_warp_cnt; i < (block_neighbor_id + 1) * neighbor_warp_cnt; ++i) {
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

        __syncthreads();

        const accum_t correction = softmax_state.update(score);
        const accum_t w          = AccumOps::exp(score - softmax_state.max_val);

        // V accumulation (keeps fmaf via weighted_accum)
#pragma unroll
        for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
            const size_t vi = warp_id * lane_cnt + lane_id + lane_cnt * neighbor_warp_cnt * t;
            if (vi < TILES) [[likely]] {
#pragma unroll
                for (size_t ep = 0; ep < TW; ++ep) {
                    o_acc[t * TW + ep] *= correction;
                }
                const typename Tile::vec_t vv = Tile::read(v_base, vi);
                Tile::weighted_accum(&o_acc[t * TW], w, &vv);
            }
        }
    }

    __syncthreads();

// write per-warp results to float32 shared
#pragma unroll
    for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
        const size_t vi = warp_id * lane_cnt + lane_id + lane_cnt * neighbor_warp_cnt * t;
        if (vi < TILES) [[likely]] {
            constexpr size_t compact_N  = std::min(TW, Vec<1, cuda_t>::max_vec_size_bytes / std::max(sizeof(cuda_t), sizeof(accum_t)));
            constexpr size_t repeat_cnt = TW / compact_N;

            for (size_t i = 0; i < repeat_cnt; ++i) {
                TileOps<compact_N, accum_t>::write(
                    my_out, vi * repeat_cnt + i, &reinterpret_cast<Vec<compact_N, accum_t> const *>(o_acc)[t * repeat_cnt + i]
                );
            }
        }
    }

    if (lane_id == 0 && warp_id == 0) {
        neighbor_max[block_neighbor_id] = softmax_state.max_val;
        neighbor_sum[block_neighbor_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // cross-warp reduction (warp 0 only)
    if (warp_id == 0 && block_neighbor_id == 0) {
        accum_t global_max = -FLT_MAX;
        accum_t global_sum{};
        accum_t inv_sum{};

        if (lane_id == 0) {
            for (size_t w = 0; w < neighbor_block_size; ++w) {
                global_max = AccumOps::max(global_max, neighbor_max[w]);
            }
            for (size_t w = 0; w < neighbor_block_size; ++w) {
                global_sum = AccumOps::fma(neighbor_sum[w], AccumOps::exp(neighbor_max[w] - global_max), global_sum);
            }
            for (size_t w = 0; w < neighbor_block_size; ++w) {
                neighbor_sum[w] = AccumOps::exp(neighbor_max[w] - global_max);  // scale_w
            }

            inv_sum = AccumOps::max(1.0f / global_sum, 0.0f);
            // This is correct for node that has at least 1 neighbour
            // (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY; - always correct
            logsumexp[node_i * H + head_h] = AccumOps::max(global_max + AccumOps::log(global_sum), -INFINITY);
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

        // cross-neighbor output write (uses write_typed for vec2 stores)
        cuda_t *const out_base = O + node_i * stride_o_n + head_h * stride_o_h;
#pragma unroll
        for (size_t t = 0; t < (TILES + lane_cnt - 1) / lane_cnt; ++t) {
            const size_t vi = lane_id + t * kWarpSize;
            if (vi < TILES) [[likely]] {
                accum_t combined[TW] = {0};
#pragma unroll
                for (size_t ep = 0; ep < TW; ++ep) {
                    size_t d_idx = vi * TW + ep;
#pragma unroll
                    for (size_t w = 0; w < neighbor_block_size; ++w) {
                        combined[ep] = AccumOps::fma(neighbor_sum[w], neighbor_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_convert_from_accum(&out_base[vi * TW], combined);
            }
        }
    }
}
