#pragma once

#include "common/tile.cuh"

// =============================================================================
// Unified GATv2 Backward AL kernel (computes grad_a, grad_l, G)
// =============================================================================
template <int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) GATv2Backward_AL(
    size_t N, size_t H, size_t D, const cuda_t *__restrict__ grad_h, int64_t stride_gh_n, int64_t stride_gh_h, const cuda_t *__restrict__ d_l,
    int64_t stride_l_n, int64_t stride_l_h, const cuda_t *__restrict__ d_r, int64_t stride_r_n, int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr, const index_t *__restrict__ d_col_idx,
    const index_t *__restrict__ node_indices,  // node indirection
    const cuda_t *__restrict__ d_attn_vec,     // [H, D]
    const float *__restrict__ d_logsumexp,     // [N, H]
    float negative_slope,
    float *__restrict__ grad_a,   // [N, H, D] always float32
    cuda_t *__restrict__ grad_l,  // [N, H, D]
    float *__restrict__ d_G       // [N, H]
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread

    using Tile  = TileOps<TW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    const int node_i  = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h  = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    // Shared memory layout:
    //   li_sh:      D_CONST * sizeof(cuda_t)                       -- read-only
    //   ghi_sh:     D_CONST * sizeof(cuda_t)                       -- read-only
    //   warp_grada: WARPS_PER_BLOCK * D_CONST * sizeof(float)      -- per-warp
    //   warp_gradl: WARPS_PER_BLOCK * D_CONST * sizeof(float)      -- per-warp
    //   warp_G:     WARPS_PER_BLOCK * sizeof(float)                -- per-warp G partial
    //   G_broadcast: sizeof(float)                                  -- broadcast slot
    extern __shared__ uint8_t sh_raw[];
    cuda_t *li_sh      = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *ghi_sh     = li_sh + D_CONST;
    float *warp_grada  = reinterpret_cast<float *>(ghi_sh + D_CONST);
    float *warp_gradl  = warp_grada + WARPS_PER_BLOCK * D_CONST;
    float *warp_G      = warp_gradl + WARPS_PER_BLOCK * D_CONST;
    float *G_broadcast = warp_G + WARPS_PER_BLOCK;

    float *my_grada = warp_grada + warp_id * D_CONST;
    float *my_gradl = warp_gradl + warp_id * D_CONST;

    cuda_t *grad_l_base = grad_l + ((int64_t)(node_i * H + head_h) * D_CONST);
    float *grad_a_base  = grad_a + ((int64_t)(node_i * H + head_h) * D_CONST);

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < TILES; v += kWarpSize) {
                Tile::write_zero(grad_l_base, v);
            }
            constexpr int f4_count_f = D_CONST / 4;
            float4 *ga_f4            = reinterpret_cast<float4 *>(grad_a_base);
            for (int i = lane; i < f4_count_f; i += kWarpSize) {
                ga_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            if (lane == 0) {
                d_G[node_i * H + head_h] = 0.f;
            }
        }
        return;
    }

    float L_i = d_logsumexp[node_i * H + head_h];

    const cuda_t *li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t *ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const cuda_t *a_base   = d_attn_vec + head_h * D_CONST;

    // Zero per-warp accumulators and cooperatively load li, ghi
    {
        constexpr int f4_count_f = D_CONST / 4;
        float4 *my_grada_f4      = reinterpret_cast<float4 *>(my_grada);
        float4 *my_gradl_f4      = reinterpret_cast<float4 *>(my_gradl);
        for (int i = lane; i < f4_count_f; i += kWarpSize) {
            my_grada_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            my_gradl_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        constexpr int f4_count   = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4 *li_src_f4  = reinterpret_cast<const float4 *>(li_base);
        const float4 *ghi_src_f4 = reinterpret_cast<const float4 *>(ghi_base);
        float4 *li_sh_f4         = reinterpret_cast<float4 *>(li_sh);
        float4 *ghi_sh_f4        = reinterpret_cast<float4 *>(ghi_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kWarpSize) {
            li_sh_f4[i]  = li_src_f4[i];
            ghi_sh_f4[i] = ghi_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    // pass 1: compute G_{i,h} = sum_j alpha_ij * <grad_h_i, r_j> (warp-strided)
    float G_partial = 0.f;

    for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
        index_t neighbor_j    = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t *rj_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij = recompute_alpha(e_ij, L_i);
        G_partial      = fmaf(alpha_ij, p_ij, G_partial);
    }

    // Cross-warp reduction for G
    if (lane == 0) warp_G[warp_id] = G_partial;
    __syncthreads();

    float G_i_h = 0.f;
    if (warp_id == 0 && lane == 0) {
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) G_i_h += warp_G[w];
        *G_broadcast             = G_i_h;
        d_G[node_i * H + head_h] = G_i_h;
    }
    __syncthreads();
    G_i_h = *G_broadcast;

    // pass 2: accumulate gradients (warp-strided)
    for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
        index_t neighbor_j    = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t *rj_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij  = recompute_alpha(e_ij, L_i);
        float grad_e_ij = alpha_ij * (p_ij - G_i_h);

#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv   = Tile::load(li_sh, v);
                vec_t rv   = Tile::load(rj_base, v);
                vec_t av   = Tile::load(a_base, v);
                int base_f = v * TW;
                Tile::gatv2_accum_grad_al(&my_grada[base_f], &my_gradl[base_f], grad_e_ij, lv, rv, av, negative_slope);
            }
        }
    }

    // Cross-warp reduction: warp 0 sums all per-warp accumulators
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                int base_f = v * TW;
                float ga_sum[TW];
                float gl_sum[TW];
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    ga_sum[ep] = 0.f;
                    gl_sum[ep] = 0.f;
                }
#pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
#pragma unroll
                    for (int ep = 0; ep < TW; ++ep) {
                        ga_sum[ep] += warp_grada[w * D_CONST + base_f + ep];
                        gl_sum[ep] += warp_gradl[w * D_CONST + base_f + ep];
                    }
                }
                Tile::write_typed(grad_l_base, v, gl_sum);
                Tile::write_float(grad_a_base, v, ga_sum);
            }
        }
    }
}

// =============================================================================
// Unified GATv2 Backward R kernel (computes grad_r)
// =============================================================================
template <int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) GATv2Backward_R(
    size_t N, size_t H, size_t D, const cuda_t *__restrict__ grad_h, int64_t stride_gh_n, int64_t stride_gh_h, const cuda_t *__restrict__ d_l,
    int64_t stride_l_n, int64_t stride_l_h, const cuda_t *__restrict__ d_r, int64_t stride_r_n, int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr_T, const index_t *__restrict__ d_col_idx_T,
    const index_t *__restrict__ node_indices,  // node indirection
    const cuda_t *__restrict__ d_attn_vec,     // [H, D]
    const float *__restrict__ d_logsumexp,     // [N, H]
    const float *__restrict__ d_G,             // [N, H]
    float negative_slope,
    cuda_t *__restrict__ grad_r  // [N, H, D]
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread

    using Tile  = TileOps<TW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    const int node_j  = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h  = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    if (node_j >= static_cast<int>(N) || head_h >= static_cast<int>(H)) return;

    index_t edge_start = d_row_ptr_T[node_j];
    index_t edge_end   = d_row_ptr_T[node_j + 1];
    int num_incoming   = static_cast<int>(edge_end - edge_start);

    // Shared memory layout:
    //   rj_sh:       D_CONST * sizeof(cuda_t)                      -- read-only
    //   warp_gradr:  WARPS_PER_BLOCK * D_CONST * sizeof(float)     -- per-warp
    extern __shared__ uint8_t sh_raw[];
    cuda_t *rj_sh     = reinterpret_cast<cuda_t *>(sh_raw);
    float *warp_gradr = reinterpret_cast<float *>(rj_sh + D_CONST);

    float *my_gradr = warp_gradr + warp_id * D_CONST;

    cuda_t *grad_r_base = grad_r + ((int64_t)(node_j * H + head_h) * D_CONST);

    // Handle isolated nodes
    if (num_incoming == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < TILES; v += kWarpSize) {
                Tile::write_zero(grad_r_base, v);
            }
        }
        return;
    }

    const cuda_t *rj_base = d_r + node_j * stride_r_n + head_h * stride_r_h;
    const cuda_t *a_base  = d_attn_vec + head_h * D_CONST;

    // Zero per-warp accumulators and cooperatively load rj
    {
        constexpr int f4_count_f = D_CONST / 4;
        float4 *my_gradr_f4      = reinterpret_cast<float4 *>(my_gradr);
        for (int i = lane; i < f4_count_f; i += kWarpSize) {
            my_gradr_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        constexpr int f4_count  = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4 *rj_src_f4 = reinterpret_cast<const float4 *>(rj_base);
        float4 *rj_sh_f4        = reinterpret_cast<float4 *>(rj_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kWarpSize) {
            rj_sh_f4[i] = rj_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    // Warp-strided edge loop
    for (int idx = warp_id; idx < num_incoming; idx += WARPS_PER_BLOCK) {
        index_t node_i         = d_col_idx_T[edge_start + static_cast<index_t>(idx)];
        const cuda_t *li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
        const cuda_t *ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;

        float L_i_h = d_logsumexp[node_i * H + head_h];
        float G_i_h = d_G[node_i * H + head_h];

        float e_lane = 0.f;
        float p_lane = 0.f;
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv  = Tile::load(li_base, v);
                vec_t rv  = Tile::load(rj_sh, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_base, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij  = recompute_alpha(e_ij, L_i_h);
        float grad_e_ij = alpha_ij * (p_ij - G_i_h);

#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv   = Tile::load(li_base, v);
                vec_t rv   = Tile::load(rj_sh, v);
                vec_t av   = Tile::load(a_base, v);
                vec_t ghv  = Tile::load(ghi_base, v);
                int base_f = v * TW;
                Tile::gatv2_accum_grad_r(&my_gradr[base_f], alpha_ij, ghv, grad_e_ij, lv, rv, av, negative_slope);
            }
        }
    }

    // Cross-warp reduction: warp 0 sums per-warp accumulators
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                int base_f = v * TW;
                float gr_sum[TW];
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) gr_sum[ep] = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
#pragma unroll
                    for (int ep = 0; ep < TW; ++ep) gr_sum[ep] += warp_gradr[w * D_CONST + base_f + ep];
                }
                Tile::write_typed(grad_r_base, v, gr_sum);
            }
        }
    }
}

template <int grad_A_reduce_row_chunk_size, typename cuda_t>
__global__ void __launch_bounds__(kWarpSize *kWarpSize) ReduceGradAKernel(
    size_t N, size_t H, size_t D,

    const float *__restrict__ grad_a,         // [N, H, D] always float32
    float *__restrict__ d_grad_a_reduced_out  // [H, D] output in float32
) {
    // head inbex
    int head_h = blockIdx.z;  // 0..H-1

    // define feature chunk and node chunk to reduce
    int row_chunk_start     = grad_A_reduce_row_chunk_size * blockIdx.x;
    int feature_chunk_start = blockDim.y * blockIdx.y;

    // define thread-specific indices and feature locations
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int fx = feature_chunk_start + tx;

    // define shared memory chunk and accumulatur
    __shared__ float tile_reduce[kWarpSize][kWarpSize + 1];
    __shared__ float result_accum[kWarpSize];

    float accum = 0.0f;

    // looped logic across row chunks:
    const int row_chunk_end = min(static_cast<int>(N), (int)(row_chunk_start + grad_A_reduce_row_chunk_size));
    for (int base_row_offset = row_chunk_start; base_row_offset < row_chunk_end; base_row_offset += blockDim.y) {
        int row_to_load = base_row_offset + ty;  // node index
        if (row_to_load < static_cast<int>(N) && fx < (int)D && head_h < static_cast<int>(H)) {
            // grad_a layout: [N, H, D] contiguous
            // idx = (n * H + h) * D + d
            size_t idx = ((size_t)row_to_load * H + (size_t)head_h) * D + (size_t)fx;

            tile_reduce[tx][ty] = grad_a[idx];
        } else {
            tile_reduce[tx][ty] = 0.0f;
        }

        __syncthreads();

        // transpose tile for warp-level reduction:
        //   * each warp (fixed ty) reduces over rows for one feature (fx)
        float value = tile_reduce[ty][tx];
        accum += warp_reduce_sum(value);
    }
    // each first lane in a warp write its results into the sshared memory for the first warp to finally reduce it into HBM:
    if (tx == 0) {
        result_accum[ty] = accum;
    }

    __syncthreads();
    // now  threads with ty==0 and tx selecting feature within chunk
    // write out the final reduced result

    if (ty == 0 && fx < (int)D && head_h < static_cast<int>(H)) {
        // output layout: [H, D] contiguous
        size_t out_idx = (size_t)head_h * D + (size_t)fx;
        atomicAdd(d_grad_a_reduced_out + out_idx, result_accum[tx]);
    }
}

// =============================================================================
// Undirected GATv2 backward: G computation kernel (extracts pass 1 of AL)
// =============================================================================
template <int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kWarpSize) GATv2Backward_G_Kernel(
    size_t N, size_t H, size_t D, const cuda_t *__restrict__ grad_h, int64_t stride_gh_n, int64_t stride_gh_h, const cuda_t *__restrict__ d_l,
    int64_t stride_l_n, int64_t stride_l_h, const cuda_t *__restrict__ d_r, int64_t stride_r_n, int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr, const index_t *__restrict__ d_col_idx,
    const cuda_t *__restrict__ d_attn_vec,  // [H, D]
    const float *__restrict__ d_logsumexp,  // [N, H]
    float negative_slope,
    float *__restrict__ d_G  // [N, H] output
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread

    using Tile  = TileOps<TW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kWarpSize;

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    if (num_neighbors == 0) {
        if (lane == 0) d_G[node_i * H + head_h] = 0.f;
        return;
    }

    float L_i = d_logsumexp[node_i * H + head_h];

    // Shared memory: li_sh + ghi_sh
    extern __shared__ uint8_t sh_raw[];
    cuda_t *li_sh  = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *ghi_sh = li_sh + D_CONST;

    const cuda_t *li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t *ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const cuda_t *a_base   = d_attn_vec + head_h * D_CONST;

    // Load li, ghi via 128-bit transactions
    {
        constexpr int f4_count   = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4 *li_src_f4  = reinterpret_cast<const float4 *>(li_base);
        const float4 *ghi_src_f4 = reinterpret_cast<const float4 *>(ghi_base);
        float4 *li_sh_f4         = reinterpret_cast<float4 *>(li_sh);
        float4 *ghi_sh_f4        = reinterpret_cast<float4 *>(ghi_sh);
        for (int i = lane; i < f4_count; i += kWarpSize) {
            li_sh_f4[i]  = li_src_f4[i];
            ghi_sh_f4[i] = ghi_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    float G_i_h = 0.f;
    for (int k = 0; k < num_neighbors; ++k) {
        index_t neighbor_j    = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t *rj_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij = recompute_alpha(e_ij, L_i);
        G_i_h          = fmaf(alpha_ij, p_ij, G_i_h);
    }

    if (lane == 0) {
        d_G[node_i * H + head_h] = G_i_h;
    }
}

// =============================================================================
// Undirected GATv2 backward: fused ALR kernel
// Computes grad_a[i], grad_l[i] (forward direction) and grad_r[i] (reverse
// direction) in a single pass over forward CSR neighbors.
// Requires G[j] to be pre-computed globally.
// =============================================================================
template <int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kWarpSize) GATv2Backward_ALR_Undirected(
    size_t N, size_t H, size_t D, const cuda_t *__restrict__ grad_h, int64_t stride_gh_n, int64_t stride_gh_h, const cuda_t *__restrict__ d_l,
    int64_t stride_l_n, int64_t stride_l_h, const cuda_t *__restrict__ d_r, int64_t stride_r_n, int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr, const index_t *__restrict__ d_col_idx,
    const cuda_t *__restrict__ d_attn_vec,  // [H, D]
    const float *__restrict__ d_logsumexp,  // [N, H]
    const float *__restrict__ d_G,          // [N, H] (pre-computed)
    float negative_slope,
    float *__restrict__ grad_a,   // [N, H, D] always float32
    cuda_t *__restrict__ grad_l,  // [N, H, D]
    cuda_t *__restrict__ grad_r   // [N, H, D]
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread

    using Tile  = TileOps<TW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kWarpSize;

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    // Shared memory layout:
    //   li_sh:      D_CONST * sizeof(cuda_t)   -- l[i]
    //   ri_sh:      D_CONST * sizeof(cuda_t)   -- r[i]
    //   ghi_sh:     D_CONST * sizeof(cuda_t)   -- grad_h[i]
    //   grada_sh:   D_CONST * sizeof(float)    -- accumulator for grad_a[i]
    //   gradli_sh:  D_CONST * sizeof(float)    -- accumulator for grad_l[i]
    //   gradri_sh:  D_CONST * sizeof(float)    -- accumulator for grad_r[i]
    extern __shared__ uint8_t sh_raw[];
    cuda_t *li_sh    = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *ri_sh    = li_sh + D_CONST;
    cuda_t *ghi_sh   = ri_sh + D_CONST;
    float *grada_sh  = reinterpret_cast<float *>(ghi_sh + D_CONST);
    float *gradli_sh = grada_sh + D_CONST;
    float *gradri_sh = gradli_sh + D_CONST;

    cuda_t *grad_l_base = grad_l + ((int64_t)(node_i * H + head_h) * D_CONST);
    cuda_t *grad_r_base = grad_r + ((int64_t)(node_i * H + head_h) * D_CONST);
    float *grad_a_base  = grad_a + ((int64_t)(node_i * H + head_h) * D_CONST);

    // Handle isolated nodes: write zeros
    if (num_neighbors == 0) {
        for (int v = lane; v < TILES; v += kWarpSize) {
            Tile::write_zero(grad_l_base, v);
            Tile::write_zero(grad_r_base, v);
        }
        constexpr int f4_count_f = D_CONST / 4;
        float4 *ga_f4            = reinterpret_cast<float4 *>(grad_a_base);
        for (int i = lane; i < f4_count_f; i += kWarpSize) {
            ga_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        return;
    }

    float L_i   = d_logsumexp[node_i * H + head_h];
    float G_i_h = d_G[node_i * H + head_h];

    const cuda_t *li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t *ri_base  = d_r + node_i * stride_r_n + head_h * stride_r_h;
    const cuda_t *ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const cuda_t *a_base   = d_attn_vec + head_h * D_CONST;

    // Zero accumulators and load li, ri, ghi via 128-bit transactions
    {
        constexpr int f4_count_f = D_CONST / 4;
        float4 *grada_f4         = reinterpret_cast<float4 *>(grada_sh);
        float4 *gradli_f4        = reinterpret_cast<float4 *>(gradli_sh);
        float4 *gradri_f4        = reinterpret_cast<float4 *>(gradri_sh);
        for (int i = lane; i < f4_count_f; i += kWarpSize) {
            grada_f4[i]  = make_float4(0.f, 0.f, 0.f, 0.f);
            gradli_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            gradri_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        constexpr int f4_count   = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4 *li_src_f4  = reinterpret_cast<const float4 *>(li_base);
        const float4 *ri_src_f4  = reinterpret_cast<const float4 *>(ri_base);
        const float4 *ghi_src_f4 = reinterpret_cast<const float4 *>(ghi_base);
        float4 *li_sh_f4         = reinterpret_cast<float4 *>(li_sh);
        float4 *ri_sh_f4         = reinterpret_cast<float4 *>(ri_sh);
        float4 *ghi_sh_f4        = reinterpret_cast<float4 *>(ghi_sh);
        for (int i = lane; i < f4_count; i += kWarpSize) {
            li_sh_f4[i]  = li_src_f4[i];
            ri_sh_f4[i]  = ri_src_f4[i];
            ghi_sh_f4[i] = ghi_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    for (int k = 0; k < num_neighbors; ++k) {
        index_t neighbor_j = d_col_idx[edge_start + static_cast<index_t>(k)];

        const cuda_t *rj_base  = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;
        const cuda_t *lj_base  = d_l + neighbor_j * stride_l_n + head_h * stride_l_h;
        const cuda_t *ghj_base = grad_h + neighbor_j * stride_gh_n + head_h * stride_gh_h;

        // ── Forward direction: score(i,j) = a^T . LeakyReLU(l[i] + r[j]) ──
        float e_fwd_lane = 0.f;
        float p_fwd_lane = 0.f;
        // ── Reverse direction: score(j,i) = a^T . LeakyReLU(l[j] + r[i]) ──
        float e_rev_lane = 0.f;
        float p_rev_lane = 0.f;

#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv  = Tile::load(li_sh, v);    // l[i]
                vec_t rv  = Tile::load(rj_base, v);  // r[j]
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);  // grad_h[i]

                vec_t ljv  = Tile::load(lj_base, v);   // l[j]
                vec_t riv  = Tile::load(ri_sh, v);     // r[i]
                vec_t ghjv = Tile::load(ghj_base, v);  // grad_h[j]

                // Forward: e(i,j) and <grad_h[i], r[j]>
                e_fwd_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_fwd_lane += Tile::dot_product(ghv, rv);

                // Reverse: e(j,i) and <grad_h[j], r[i]>
                e_rev_lane += Tile::gatv2_dot_leaky_relu(ljv, riv, av, ns);
                p_rev_lane += Tile::dot_product(ghjv, riv);
            }
        }

        float e_fwd = warp_reduce_sum(e_fwd_lane);
        float p_fwd = warp_reduce_sum(p_fwd_lane);
        float e_rev = warp_reduce_sum(e_rev_lane);
        float p_rev = warp_reduce_sum(p_rev_lane);

        // Forward: alpha(i,j), grad_e(i,j)
        float alpha_fwd  = recompute_alpha(e_fwd, L_i);
        float grad_e_fwd = alpha_fwd * (p_fwd - G_i_h);

        // Reverse: alpha(j,i), grad_e(j,i) — uses L[j] and G[j]
        float L_j        = d_logsumexp[neighbor_j * H + head_h];
        float G_j_h      = d_G[neighbor_j * H + head_h];
        float alpha_rev  = recompute_alpha(e_rev, L_j);
        float grad_e_rev = alpha_rev * (p_rev - G_j_h);

// Accumulate gradients
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                vec_t lv   = Tile::load(li_sh, v);
                vec_t rv   = Tile::load(rj_base, v);
                vec_t av   = Tile::load(a_base, v);
                int base_f = v * Tile::TW;

                // Forward: grad_a[i], grad_l[i] from score(i,j)
                Tile::gatv2_accum_grad_al(&grada_sh[base_f], &gradli_sh[base_f], grad_e_fwd, lv, rv, av, negative_slope);

                // Reverse: grad_r[i] from score(j,i) = a^T . LeakyReLU(l[j] + r[i])
                vec_t ljv  = Tile::load(lj_base, v);
                vec_t riv  = Tile::load(ri_sh, v);
                vec_t ghjv = Tile::load(ghj_base, v);
                Tile::gatv2_accum_grad_r(&gradri_sh[base_f], alpha_rev, ghjv, grad_e_rev, ljv, riv, av, negative_slope);
            }
        }
    }

    __syncthreads();

// Write grad_l (cuda_t), grad_a (float32), grad_r (cuda_t)
#pragma unroll
    for (int t = 0; t < TILES_PER_THREAD; ++t) {
        int v = lane + kWarpSize * t;
        if (v < TILES) {
            int base_f = v * Tile::TW;
            Tile::write_typed(grad_l_base, v, &gradli_sh[base_f]);
            Tile::write_float(grad_a_base, v, &grada_sh[base_f]);
            Tile::write_typed(grad_r_base, v, &gradri_sh[base_f]);
        }
    }
}
