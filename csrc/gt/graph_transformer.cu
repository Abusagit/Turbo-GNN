#include "common.cuh"

template<int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
GraphAttentionForward_CSR_MH_v2_D(
    const int N,
    const int H,
    const cuda_t* __restrict__ Q,
    const cuda_t* __restrict__ K,
    const cuda_t* __restrict__ V,
    const int64_t stride_q_n, const int64_t stride_q_h,
    const int64_t stride_k_n, const int64_t stride_k_h,
    const int64_t stride_v_n, const int64_t stride_v_h,
    const index_t* __restrict__ row_ptr,
    const index_t* __restrict__ col_idx,
    const index_t* __restrict__ node_indices,  // node indirection: node_i = node_indices[blockIdx.x]
    cuda_t* __restrict__ O,
    const int64_t stride_o_n, const int64_t stride_o_h,
    float* __restrict__ logsumexp,
    const float scale
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be multiple of 32 for this fast path");

    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;
    constexpr int VEC_D = D_CONST / EPV;
    constexpr int TILES = (VEC_D + kWarpSize - 1) / kWarpSize;
    constexpr int ACCS_PER_LANE = TILES * EPV;

    const int node_i  = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h  = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;

    if (node_i >= N || head_h >= H) {
        return;
    }

    const index_t edge_start    = row_ptr[node_i];
    const index_t edge_end      = row_ptr[node_i + 1];
    const int num_neighbors = static_cast<int>(edge_end - edge_start);

    // Shared memory layout (unchanged):
    // k_shared[D_CONST] as cuda_t
    // warp_out[WARPS_PER_BLOCK * D_CONST] as float
    // warp_max[WARPS_PER_BLOCK] as float
    // warp_sum[WARPS_PER_BLOCK] as float
    extern __shared__ char sh_raw[];
    cuda_t* k_shared = reinterpret_cast<cuda_t*>(sh_raw);
    float*  warp_out = reinterpret_cast<float*>(sh_raw + D_CONST * sizeof(cuda_t));
    float*  warp_max = warp_out + WARPS_PER_BLOCK * D_CONST;
    float*  warp_sum = warp_max + WARPS_PER_BLOCK;

    float* my_out = warp_out + warp_id * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            cuda_t* out_base = O + node_i * stride_o_n + head_h * stride_o_h;
            for (int vi = lane_id; vi < VEC_D; vi += kWarpSize) {
                Tile::write_zero(out_base, vi);
            }
            if (lane_id == 0) {
                logsumexp[node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    // cooperative load of K_i via 128-bit transactions (unchanged)
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_K_LOADS = D_CONST / ELEMS_PER_F4;
        const cuda_t* k_base = K + node_i * stride_k_n + head_h * stride_k_h;
        const float4* k_src = reinterpret_cast<const float4*>(k_base);
        float4* k_sh = reinterpret_cast<float4*>(k_shared);
        for (int i = threadIdx.x; i < NUM_K_LOADS; i += WARPS_PER_BLOCK * kWarpSize) {
            k_sh[i] = k_src[i];
        }
    }
    __syncthreads();

    OnlineSoftmaxState softmax_state;

    float o_acc[ACCS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ACCS_PER_LANE; ++i) {
        o_acc[i] = 0.0f;
    }

    // neighbor loop
    for (int e = warp_id; e < num_neighbors; e += WARPS_PER_BLOCK) {
        const index_t j = __ldg(&col_idx[edge_start + e]);

        const cuda_t* q_base = Q + j * stride_q_n + head_h * stride_q_h;
        const cuda_t* v_base = V + j * stride_v_n + head_h * stride_v_h;

        // Q·K dot product (uses improved dot_product with native mul)
        float s_partial = 0.0f;
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < VEC_D) {
                auto kv = Tile::load(k_shared, vi);
                auto qv = Tile::load(q_base, vi);
                s_partial += Tile::dot_product(kv, qv);
            }
        }

        const float score = warp_reduce_sum(s_partial) * scale;
        const float correction = softmax_state.update(score);
        const float w = __expf(score - softmax_state.max_val);

        // V accumulation (keeps fmaf via weighted_accum)
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < VEC_D) {
                #pragma unroll
                for (int ep = 0; ep < EPV; ++ep)
                    o_acc[t * EPV + ep] *= correction;
                auto vv = Tile::load(v_base, vi);
                Tile::weighted_accum(&o_acc[t * EPV], w, vv);
            }
        }
    }

    // write per-warp results to float32 shared
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int vi = lane_id + t * kWarpSize;
        if (vi < VEC_D) {
            Tile::write_float(my_out, vi, &o_acc[t * EPV]);
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
                warp_sum[w] = __expf(warp_max[w] - global_max); // scale_w
            }

            inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
            logsumexp[node_i * H + head_h] = (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY;
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

        // cross-warp output write (uses write_typed for vec2 stores)
        cuda_t* out_base = O + node_i * stride_o_n + head_h * stride_o_h;
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int vi = lane_id + t * kWarpSize;
            if (vi < VEC_D) {
                float combined[EPV];
                #pragma unroll
                for (int ep = 0; ep < EPV; ++ep) {
                    combined[ep] = 0.0f;
                    int d_idx = vi * EPV + ep;
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

// ===================================================
// ================== BACKWARD =======================
// ===================================================

// D[i,h] = sum_d dO[i,h,d] * O[i,h,d]
template<int D_CONST, typename cuda_t>
__global__ void __launch_bounds__(kWarpSize)
compute_D_mh_kernel_D(
    const cuda_t* __restrict__ dO,   // [N, H, D]
    const cuda_t* __restrict__ O_in, // [N, H, D]
    float* __restrict__ D_out,       // [N, H]
    int64_t N,
    int64_t H,
    int64_t stride_do_n,
    int64_t stride_do_h,
    int64_t stride_o_n,
    int64_t stride_o_h
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;
    constexpr int D_VEC = D_CONST / EPV;

    const int node_i = blockIdx.x;
    const int head_h = blockIdx.y;
    const int lane   = threadIdx.x;   // 0..31

    if (node_i >= (int)N || head_h >= (int)H) {
        return;
    }

    const cuda_t* dO_base = dO   + node_i * stride_do_n + head_h * stride_do_h;
    const cuda_t* O_base  = O_in + node_i * stride_o_n  + head_h * stride_o_h;

    float sum = 0.0f;

    #pragma unroll
    for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
        auto dO_v = Tile::load(dO_base, fv);
        auto O_v  = Tile::load(O_base, fv);
        sum += Tile::dot_product(dO_v, O_v);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        D_out[node_i * H + head_h] = sum;
    }
}


// Q, K, V, dO are [N, H, D] with contiguous D (stride(2)==1), D % 4 == 0
// Q, K, V may be non-contiguous in N,H dims (e.g. from split/view).
// logsumexp and Delta are [N, H].
// dQ, dK, dV are cuda_t output (contiguous); internal accumulation in float32
template<int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
graph_attn_backward_csrT_kernel_D(
    int64_t N,
    int64_t H,
    const index_t* __restrict__ row_ptr_T,   // [N+1], CSR^T row pointers
    const index_t* __restrict__ col_idx_T,   // [E],   CSR^T col indices
    const index_t* __restrict__ node_indices, // node indirection
    const cuda_t* __restrict__ Q,        // [N, H, D]
    const cuda_t* __restrict__ K,        // [N, H, D]
    const cuda_t* __restrict__ V,        // [N, H, D]
    int64_t stride_q_n, int64_t stride_q_h,
    int64_t stride_k_n, int64_t stride_k_h,
    int64_t stride_v_n, int64_t stride_v_h,
    const cuda_t* __restrict__ dO,       // [N, H, D]
    const float* __restrict__ logsumexp, // [N, H]
    const float* __restrict__ Delta,     // [N, H]
    float scale,
    cuda_t* __restrict__ dQ,             // [N, H, D] (contiguous)
    float* __restrict__ dK,              // [N, H, D] (contiguous, float32 for atomicAdd)
    cuda_t* __restrict__ dV              // [N, H, D] (contiguous)
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;
    constexpr int D_VEC = D_CONST / EPV;

    const int node_j = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    if (node_j >= N || head_h >= H) {
        return;
    }

    index_t edge_start    = row_ptr_T[node_j];
    index_t edge_end      = row_ptr_T[node_j + 1];
    int num_incoming  = static_cast<int>(edge_end - edge_start);

    // Contiguous offset for output dQ, dV (freshly allocated, always contiguous)
    const size_t out_jh = (node_j * H + head_h) * D_CONST;

    // nothing to do if this node has no incoming edges — all warps write zeros and return
    if (num_incoming == 0) {
        if (warp_id == 0) {
            for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
                Tile::write_zero(dQ + out_jh, fv);
                Tile::write_zero(dV + out_jh, fv);
            }
        }
        return;
    }

    // Shared memory layout:
    // qj_shared: D_CONST * sizeof(cuda_t)                        -- read-only, 1 copy
    // vj_shared: D_CONST * sizeof(cuda_t)                        -- read-only, 1 copy
    // warp_gq:   WARPS_PER_BLOCK * D_CONST * sizeof(float)       -- per-warp dQ accumulators
    // warp_gv:   WARPS_PER_BLOCK * D_CONST * sizeof(float)       -- per-warp dV accumulators
    extern __shared__ char sh_raw[];
    cuda_t* qj_shared = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* vj_shared = qj_shared + D_CONST;
    float*  warp_gq   = reinterpret_cast<float*>(sh_raw + 2 * D_CONST * sizeof(cuda_t));
    float*  warp_gv   = warp_gq + WARPS_PER_BLOCK * D_CONST;

    // Per-warp accumulator pointers
    float* my_gq = warp_gq + warp_id * D_CONST;
    float* my_gv = warp_gv + warp_id * D_CONST;

    // Cooperative load of qj, vj using all threads across all warps
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS = D_CONST / ELEMS_PER_F4;
        const float4* qj_src = reinterpret_cast<const float4*>(Q + node_j * stride_q_n + head_h * stride_q_h);
        const float4* vj_src = reinterpret_cast<const float4*>(V + node_j * stride_v_n + head_h * stride_v_h);
        float4* qj_sh_f4 = reinterpret_cast<float4*>(qj_shared);
        float4* vj_sh_f4 = reinterpret_cast<float4*>(vj_shared);
        for (int i = threadIdx.x; i < NUM_LOADS; i += WARPS_PER_BLOCK * kWarpSize) {
            qj_sh_f4[i] = qj_src[i];
            vj_sh_f4[i] = vj_src[i];
        }
    }

    // Zero per-warp float32 gradient accumulators
    {
        constexpr int NUM_F4 = D_CONST / 4;
        float4* my_gq_f4 = reinterpret_cast<float4*>(my_gq);
        float4* my_gv_f4 = reinterpret_cast<float4*>(my_gv);
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            my_gq_f4[i] = {0.f, 0.f, 0.f, 0.f};
            my_gv_f4[i] = {0.f, 0.f, 0.f, 0.f};
        }
    }
    __syncthreads();

    // Warp-strided edge loop
    for (int e = warp_id; e < num_incoming; e += WARPS_PER_BLOCK) {
        index_t node_i = 0;
        if (lane == 0) {
            node_i = __ldg(&col_idx_T[edge_start + e]);
        }
        node_i = __shfl_sync(FULL_WARP_MASK, node_i, 0);

        if (node_i >= N) continue;

        const cuda_t* ki_base  = K  + node_i * stride_k_n + head_h * stride_k_h;
        const size_t out_ih = static_cast<size_t>(node_i) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        const cuda_t* dOi_base = dO + out_ih;

        // 1) dot(k_i, q_j) and dP_ij = <dO_i, v_j>
        float dot_kq = 0.0f;
        float dP_ij  = 0.0f;

        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            auto ki  = Tile::load(ki_base, fv);
            auto qj  = Tile::load(qj_shared, fv);
            auto vj  = Tile::load(vj_shared, fv);
            auto dOi = Tile::load(dOi_base, fv);

            dot_kq += Tile::dot_product(ki, qj);
            dP_ij  += Tile::dot_product(dOi, vj);
        }

        dot_kq = warp_reduce_sum(dot_kq);
        dP_ij  = warp_reduce_sum(dP_ij);

        const float score = dot_kq * scale;

        float L_i = 0.0f, Delta_i = 0.0f;
        if (lane == 0) {
            const size_t idx_ih = static_cast<size_t>(node_i) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_i     = __ldg(&logsumexp[idx_ih]);
            Delta_i = __ldg(&Delta[idx_ih]);
        }
        L_i     = __shfl_sync(FULL_WARP_MASK, L_i, 0);
        Delta_i = __shfl_sync(FULL_WARP_MASK, Delta_i, 0);

        const float alpha     = __expf(score - L_i);
        const float dS        = alpha * (dP_ij - Delta_i);
        const float dS_scaled = dS * scale;

        // 2) accumulate dV_j, dQ_j in per-warp float32 shared; atomicAdd dK_i
        float* dK_i_base = dK + out_ih;

        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            int base_f = fv * EPV;
            auto ki  = Tile::load(ki_base, fv);
            auto dOi = Tile::load(dOi_base, fv);
            auto qj  = Tile::load(qj_shared, fv);

            Tile::weighted_accum(&my_gv[base_f], alpha, dOi);
            Tile::weighted_accum(&my_gq[base_f], dS_scaled, ki);
            Tile::atomic_add_scaled_f32(dK_i_base, base_f, dS_scaled, qj);
        }
    }

    // 3) Cross-warp reduction: warp 0 sums all per-warp accumulators and writes output
    __syncthreads();

    if (warp_id == 0) {
        cuda_t* dQ_base = dQ + out_jh;
        cuda_t* dV_base = dV + out_jh;

        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            int base_f = fv * EPV;
            float gq_sum[EPV];
            float gv_sum[EPV];
            #pragma unroll
            for (int ep = 0; ep < EPV; ++ep) {
                gq_sum[ep] = 0.f;
                gv_sum[ep] = 0.f;
            }
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                #pragma unroll
                for (int ep = 0; ep < EPV; ++ep) {
                    gq_sum[ep] += warp_gq[w * D_CONST + base_f + ep];
                    gv_sum[ep] += warp_gv[w * D_CONST + base_f + ep];
                }
            }
            Tile::write_typed(dQ_base, fv, gq_sum);
            Tile::write_typed(dV_base, fv, gv_sum);
        }
    }
}


// =============================================================================
// Undirected backward kernel: uses forward CSR, zero atomics.
// For each dst node d, iterates over src neighbors s. Computes:
//   Forward direction: dK[d] (local)
//   Reverse direction: dQ[d], dV[d] (local, exploiting symmetric adjacency)
// =============================================================================
template<int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kWarpSize)
graph_attn_backward_fwd_csr_undirected_kernel_D(
    int64_t N,
    int64_t H,
    const index_t* __restrict__ row_ptr,     // [N+1], forward CSR row pointers
    const index_t* __restrict__ col_idx,     // [E],   forward CSR col indices
    const cuda_t* __restrict__ Q,            // [N, H, D]
    const cuda_t* __restrict__ K,            // [N, H, D]
    const cuda_t* __restrict__ V,            // [N, H, D]
    int64_t stride_q_n, int64_t stride_q_h,
    int64_t stride_k_n, int64_t stride_k_h,
    int64_t stride_v_n, int64_t stride_v_h,
    const cuda_t* __restrict__ dO,           // [N, H, D] (contiguous)
    const float* __restrict__ logsumexp,     // [N, H]
    const float* __restrict__ Delta,         // [N, H]
    float scale,
    cuda_t* __restrict__ dQ,                 // [N, H, D] (contiguous)
    cuda_t* __restrict__ dK,                 // [N, H, D] (contiguous, cuda_t — no atomics)
    cuda_t* __restrict__ dV                  // [N, H, D] (contiguous)
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;
    constexpr int D_VEC = D_CONST / EPV;

    int node_d = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x; // 0..31

    if (node_d >= N || head_h >= H) {
        return;
    }

    index_t edge_start = row_ptr[node_d];
    index_t edge_end   = row_ptr[node_d + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    const size_t out_dh = (node_d * H + head_h) * D_CONST;

    // Handle isolated nodes: write zeros
    if (num_neighbors == 0) {
        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            Tile::write_zero(dQ + out_dh, fv);
            Tile::write_zero(dK + out_dh, fv);
            Tile::write_zero(dV + out_dh, fv);
        }
        return;
    }

    // Shared memory layout:
    //   kd_shared:  D_CONST * sizeof(cuda_t)   -- K[d]
    //   qd_shared:  D_CONST * sizeof(cuda_t)   -- Q[d]
    //   vd_shared:  D_CONST * sizeof(cuda_t)   -- V[d]
    //   gk_shared:  D_CONST * sizeof(float)    -- float32 accumulator for dK[d]
    //   gq_shared:  D_CONST * sizeof(float)    -- float32 accumulator for dQ[d]
    //   gv_shared:  D_CONST * sizeof(float)    -- float32 accumulator for dV[d]
    extern __shared__ char sh_raw[];
    cuda_t* kd_shared = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* qd_shared = kd_shared + D_CONST;
    cuda_t* vd_shared = qd_shared + D_CONST;
    float*  gk_shared = reinterpret_cast<float*>(sh_raw + 3 * D_CONST * sizeof(cuda_t));
    float*  gq_shared = gk_shared + D_CONST;
    float*  gv_shared = gq_shared + D_CONST;

    // Load K[d], Q[d], V[d] via 128-bit transactions
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS = D_CONST / ELEMS_PER_F4;
        const float4* kd_src = reinterpret_cast<const float4*>(K + node_d * stride_k_n + head_h * stride_k_h);
        const float4* qd_src = reinterpret_cast<const float4*>(Q + node_d * stride_q_n + head_h * stride_q_h);
        const float4* vd_src = reinterpret_cast<const float4*>(V + node_d * stride_v_n + head_h * stride_v_h);
        float4* kd_sh_f4 = reinterpret_cast<float4*>(kd_shared);
        float4* qd_sh_f4 = reinterpret_cast<float4*>(qd_shared);
        float4* vd_sh_f4 = reinterpret_cast<float4*>(vd_shared);
        for (int i = lane; i < NUM_LOADS; i += kWarpSize) {
            kd_sh_f4[i] = kd_src[i];
            qd_sh_f4[i] = qd_src[i];
            vd_sh_f4[i] = vd_src[i];
        }
    }

    // Zero float32 gradient accumulators
    {
        constexpr int NUM_F4 = D_CONST / 4;
        float4* gk_f4 = reinterpret_cast<float4*>(gk_shared);
        float4* gq_f4 = reinterpret_cast<float4*>(gq_shared);
        float4* gv_f4 = reinterpret_cast<float4*>(gv_shared);
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            gk_f4[i] = {0.f, 0.f, 0.f, 0.f};
            gq_f4[i] = {0.f, 0.f, 0.f, 0.f};
            gv_f4[i] = {0.f, 0.f, 0.f, 0.f};
        }
    }
    __syncwarp(FULL_WARP_MASK);

    // Row scalars
    float L_d = 0.0f, Delta_d = 0.0f;
    if (lane == 0) {
        const size_t idx_dh = static_cast<size_t>(node_d) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
        L_d     = __ldg(&logsumexp[idx_dh]);
        Delta_d = __ldg(&Delta[idx_dh]);
    }
    L_d     = __shfl_sync(FULL_WARP_MASK, L_d, 0);
    Delta_d = __shfl_sync(FULL_WARP_MASK, Delta_d, 0);

    // dO[d] base pointer (contiguous)
    const cuda_t* dOd_base = dO + out_dh;

    for (int e = 0; e < num_neighbors; ++e) {
        index_t node_s = 0;
        if (lane == 0) {
            node_s = __ldg(&col_idx[edge_start + e]);
        }
        node_s = __shfl_sync(FULL_WARP_MASK, node_s, 0);

        if (node_s >= N) continue;

        // Column node pointers (strided)
        const cuda_t* qs_base  = Q  + node_s * stride_q_n + head_h * stride_q_h;
        const cuda_t* ks_base  = K  + node_s * stride_k_n + head_h * stride_k_h;
        const cuda_t* vs_base  = V  + node_s * stride_v_n + head_h * stride_v_h;
        // dO[s] is contiguous
        const size_t out_sh = static_cast<size_t>(node_s) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        const cuda_t* dOs_base = dO + out_sh;

        // 1) Compute dot products for both directions
        float dot_kd_qs = 0.0f;  // K[d] . Q[s]  -> forward score
        float dP_fwd    = 0.0f;  // dO[d] . V[s]  -> forward dP
        float dot_qd_ks = 0.0f;  // Q[d] . K[s]  -> reverse score
        float dP_rev    = 0.0f;  // V[d] . dO[s]  -> reverse dP

        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            auto kd  = Tile::load(kd_shared, fv);
            auto qd  = Tile::load(qd_shared, fv);
            auto vd  = Tile::load(vd_shared, fv);
            auto dOd = Tile::load(dOd_base, fv);
            auto qs  = Tile::load(qs_base, fv);
            auto ks  = Tile::load(ks_base, fv);
            auto vs  = Tile::load(vs_base, fv);
            auto dOs = Tile::load(dOs_base, fv);

            dot_kd_qs += Tile::dot_product(kd, qs);
            dP_fwd    += Tile::dot_product(dOd, vs);
            dot_qd_ks += Tile::dot_product(qd, ks);
            dP_rev    += Tile::dot_product(vd, dOs);
        }

        dot_kd_qs = warp_reduce_sum(dot_kd_qs);
        dP_fwd    = warp_reduce_sum(dP_fwd);
        dot_qd_ks = warp_reduce_sum(dot_qd_ks);
        dP_rev    = warp_reduce_sum(dP_rev);

        // 2) Load L[s] and Delta[s] for reverse direction
        float L_s = 0.0f, Delta_s = 0.0f;
        if (lane == 0) {
            const size_t idx_sh = static_cast<size_t>(node_s) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_s     = __ldg(&logsumexp[idx_sh]);
            Delta_s = __ldg(&Delta[idx_sh]);
        }
        L_s     = __shfl_sync(FULL_WARP_MASK, L_s, 0);
        Delta_s = __shfl_sync(FULL_WARP_MASK, Delta_s, 0);

        // 3) Forward direction: dK[d] += dS_fwd * Q[s]
        const float score_fwd    = dot_kd_qs * scale;
        const float alpha_fwd    = __expf(score_fwd - L_d);
        const float dS_fwd       = alpha_fwd * (dP_fwd - Delta_d);
        const float dS_fwd_scaled = dS_fwd * scale;

        // 4) Reverse direction: dQ[d] += dS_rev * K[s], dV[d] += alpha_rev * dO[s]
        const float score_rev    = dot_qd_ks * scale;
        const float alpha_rev    = __expf(score_rev - L_s);
        const float dS_rev       = alpha_rev * (dP_rev - Delta_s);
        const float dS_rev_scaled = dS_rev * scale;

        // 5) Accumulate all three gradients in shared float32
        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            int base_f = fv * EPV;
            auto qs  = Tile::load(qs_base, fv);
            auto ks  = Tile::load(ks_base, fv);
            auto dOs = Tile::load(dOs_base, fv);

            Tile::weighted_accum(&gk_shared[base_f], dS_fwd_scaled, qs);   // dK[d] += dS_fwd * Q[s]
            Tile::weighted_accum(&gq_shared[base_f], dS_rev_scaled, ks);   // dQ[d] += dS_rev * K[s]
            Tile::weighted_accum(&gv_shared[base_f], alpha_rev, dOs);      // dV[d] += P_rev * dO[s]
        }
    }

    // Write all three gradients: convert float32 accumulators to cuda_t
    cuda_t* dK_base = dK + out_dh;
    cuda_t* dQ_base = dQ + out_dh;
    cuda_t* dV_base = dV + out_dh;

    for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
        int base_f = fv * EPV;
        Tile::write_typed(dK_base, fv, &gk_shared[base_f]);
        Tile::write_typed(dQ_base, fv, &gq_shared[base_f]);
        Tile::write_typed(dV_base, fv, &gv_shared[base_f]);
    }
}


std::tuple<torch::Tensor, torch::Tensor>
graph_attention_forward_csr_mh_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block
) {

    at::cuda::CUDAGuard device_guard(Q.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(Q.device().index());

    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "CSR indices must be CUDA");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be [N, H, D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V sizes must match");

    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(),
                "Q, K, V must have the same dtype");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 || Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16,
                "Q must be float32, float16, or bfloat16");

    auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype),
                "row_ptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype,
                "col_idx must have same dtype as row_ptr");

    const int N = Q.size(0);
    const int H = Q.size(1);
    const int D = Q.size(2);

    TORCH_CHECK(D % 4 == 0, "D must be divisible by 4");
    TORCH_CHECK(D <= 256, "D > 256 not supported");

    auto q_strides = Q.strides();
    auto k_strides = K.strides();
    auto v_strides = V.strides();

    TORCH_CHECK(q_strides[2] == 1 && k_strides[2] == 1 && v_strides[2] == 1,
                "Feature dim must be contiguous");

    // O matches input dtype, logsumexp always float32
    torch::Tensor O = torch::empty({N, H, D}, torch::TensorOptions().dtype(Q.dtype()).device(Q.device()));
    torch::Tensor lse = torch::empty({N, H}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));

    auto o_strides = O.strides();

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "GT forward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    // Lambda to launch the kernel for a bucket of nodes with a given warp count
    auto launch_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
        if (num_nodes_bucket == 0) return;

        std::visit([&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t = typename decltype(typeInfo)::CudaType;
            constexpr int DC = decltype(d_c)::value;
            constexpr int W = decltype(warp_c)::value;

            auto* Q_ptr = reinterpret_cast<const cuda_t*>(Q.data_ptr<torch_t>());
            auto* K_ptr = reinterpret_cast<const cuda_t*>(K.data_ptr<torch_t>());
            auto* V_ptr = reinterpret_cast<const cuda_t*>(V.data_ptr<torch_t>());
            auto* O_ptr = reinterpret_cast<cuda_t*>(O.data_ptr<torch_t>());

            size_t shmem = DC * sizeof(cuda_t)
                         + W * DC * sizeof(float)
                         + 2 * W * sizeof(float);

            dim3 blocks(num_nodes_bucket, H);
            dim3 threads(W * kWarpSize);

            GraphAttentionForward_CSR_MH_v2_D<W, DC, cuda_t, index_t><<<blocks, threads, shmem, stream>>>(
                N, H,
                Q_ptr, K_ptr, V_ptr,
                q_strides[0], q_strides[1],
                k_strides[0], k_strides[1],
                v_strides[0], v_strides[1],
                index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
                index_ptr<index_t>(node_indices),
                O_ptr,
                o_strides[0], o_strides[1],
                lse.data_ptr<float>(),
                scale
            );
        }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
           MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
           MakeIntVariant<32, 64, 128, 256>(D),
           warp_variant);
    };

    // Light nodes
    launch_bucket(light_nodes, light_nodes.numel(),
                  MakeIntVariant<1, 2, 4>(light_warps_per_block));

    // Heavy nodes
    launch_bucket(heavy_nodes, heavy_nodes.numel(),
                  MakeIntVariant<8, 16, 32>(heavy_warps_per_block));

    CUDA_KERNEL_CHECK();

    return std::make_tuple(O, lse);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
graph_attention_backward_csr_mh_cuda(
    torch::Tensor row_ptr,     // [N+1], forward CSR
    torch::Tensor col_idx,     // [E],   forward CSR
    torch::Tensor row_ptr_T,   // [N+1], CSR^T (backward)
    torch::Tensor col_idx_T,   // [E],   CSR^T (backward)
    torch::Tensor Q,           // [N, H, D]
    torch::Tensor K,           // [N, H, D]
    torch::Tensor V,           // [N, H, D]
    torch::Tensor O,           // [N, H, D] (forward output)
    torch::Tensor dO,          // [N, H, D]
    torch::Tensor logsumexp,   // [N, H],   float32
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed
) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(),
                "Forward CSR indices must be CUDA");
    TORCH_CHECK(row_ptr_T.is_cuda() && col_idx_T.is_cuda(),
                "CSR^T indices must be CUDA");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() &&
                O.is_cuda() && dO.is_cuda() && logsumexp.is_cuda(),
                "Q, K, V, O, dO, logsumexp must be CUDA");

    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3 &&
                O.dim() == 3 && dO.dim() == 3,
                "Q, K, V, O, dO must be [N, H, D]");
    TORCH_CHECK(Q.sizes() == K.sizes() &&
                Q.sizes() == V.sizes() &&
                Q.sizes() == O.sizes() &&
                Q.sizes() == dO.sizes(),
                "Q, K, V, O, dO sizes must match [N, H, D]");

    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype() &&
                Q.dtype() == O.dtype() && Q.dtype() == dO.dtype(),
                "Q, K, V, O, dO must have the same dtype");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 || Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16,
                "Q must be float32, float16, or bfloat16");

    auto idx_dtype = row_ptr_T.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype),
                "row_ptr_T must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx_T.scalar_type() == idx_dtype,
                "col_idx_T must have same dtype as row_ptr_T");

    TORCH_CHECK(logsumexp.dtype() == torch::kFloat32,
                "logsumexp must be float32");
    TORCH_CHECK(logsumexp.dim() == 2,
                "logsumexp must be [N, H]");

    const int64_t N = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t D = Q.size(2);

    TORCH_CHECK(row_ptr_T.dim() == 1 && row_ptr_T.size(0) == N + 1,
                "row_ptr_T must be [N+1]");
    TORCH_CHECK(col_idx_T.dim() == 1,
                "col_idx_T must be [E]");
    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.size(0) == N + 1,
                "row_ptr must be [N+1]");
    TORCH_CHECK(col_idx.dim() == 1,
                "col_idx must be [E]");

    TORCH_CHECK(logsumexp.size(0) == N && logsumexp.size(1) == H,
                "logsumexp must be [N, H]");

    TORCH_CHECK(D % 4 == 0, "D must be divisible by 4");
    TORCH_CHECK(D <= 256, "D > 256 not supported");

    auto q_strides = Q.strides();
    auto k_strides = K.strides();
    auto v_strides = V.strides();
    auto o_strides = O.strides();

    const int64_t stride_q_d = q_strides[2];
    const int64_t stride_k_d = k_strides[2];
    const int64_t stride_v_d = v_strides[2];
    const int64_t stride_o_d = o_strides[2];

    TORCH_CHECK(stride_q_d == 1 && stride_k_d == 1 && stride_v_d == 1 && stride_o_d == 1,
            "feature dim (D) must be contiguous (stride(2) == 1) for Q, K, V, O");

    TORCH_CHECK(O.is_contiguous(),  "O must be contiguous [N, H, D]");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous [N, H, D]");
    TORCH_CHECK(logsumexp.is_contiguous(),
                "logsumexp must be contiguous [N, H]");
    TORCH_CHECK(row_ptr_T.is_contiguous() && col_idx_T.is_contiguous(),
                "CSR^T arrays must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous() && col_idx.is_contiguous(),
                "Forward CSR arrays must be contiguous");

    auto input_dtype = Q.dtype();
    auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto typed_options = torch::TensorOptions().dtype(input_dtype).device(Q.device());

    // Delta[i,h] = <O[i,h,:], dO[i,h,:]>
    torch::Tensor Delta = torch::empty({N, H}, f32_options);
    auto do_strides = dO.strides();

    const int64_t stride_do_n = do_strides[0];
    const int64_t stride_do_h = do_strides[1];
    const int64_t stride_o_n  = o_strides[0];
    const int64_t stride_o_h  = o_strides[1];

    TORCH_CHECK(do_strides[2] == 1 && o_strides[2] == 1,
                "dO and O feature dim (D) must be contiguous (stride(2) == 1)");

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "GT backward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    // Launch compute_D for ALL nodes (always 1 warp, no bucketing)
    {
        dim3 blocks_D(N, H);
        dim3 threads_D(kWarpSize);
        std::visit([&](auto typeInfo, auto d_c) {
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t = typename decltype(typeInfo)::CudaType;
            constexpr int DC = decltype(d_c)::value;

            auto cuda_stream = at::cuda::getDefaultCUDAStream();
            auto* dO_ptr = reinterpret_cast<const cuda_t*>(dO.data_ptr<torch_t>());
            auto* O_ptr  = reinterpret_cast<const cuda_t*>(O.data_ptr<torch_t>());

            compute_D_mh_kernel_D<DC, cuda_t><<<blocks_D, threads_D, 0, cuda_stream>>>(
                dO_ptr, O_ptr, Delta.data_ptr<float>(),
                N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
            );
        }, MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
           MakeIntVariant<32, 64, 128, 256>((int)D));
    }

    torch::Tensor dQ = torch::empty({N, H, D}, typed_options);
    torch::Tensor dV = torch::empty({N, H, D}, typed_options);
    // Directed: dK in float32 for atomicAdd; undirected: dK in input dtype (no atomics)
    torch::Tensor dK_f32;
    torch::Tensor dK_typed;
    if (is_directed) {
        dK_f32 = torch::zeros({N, H, D}, f32_options);
    } else {
        dK_typed = torch::empty({N, H, D}, typed_options);
    }

    if (is_directed) {
        // Directed path: warp-parallel bucketed backward using CSR^T
        auto launch_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
            if (num_nodes_bucket == 0) return;

            std::visit([&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                using index_t = typename decltype(idxInfo)::Type;
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;
                constexpr int W = decltype(warp_c)::value;

                auto cuda_stream = at::cuda::getDefaultCUDAStream();

                auto* Q_ptr  = reinterpret_cast<const cuda_t*>(Q.data_ptr<torch_t>());
                auto* K_ptr  = reinterpret_cast<const cuda_t*>(K.data_ptr<torch_t>());
                auto* V_ptr  = reinterpret_cast<const cuda_t*>(V.data_ptr<torch_t>());
                auto* dO_ptr = reinterpret_cast<const cuda_t*>(dO.data_ptr<torch_t>());
                auto* dQ_ptr = reinterpret_cast<cuda_t*>(dQ.data_ptr<torch_t>());
                auto* dV_ptr = reinterpret_cast<cuda_t*>(dV.data_ptr<torch_t>());
                auto* dK_ptr = dK_f32.data_ptr<float>();

                // qj + vj (read-only) + W * (gq + gv) per-warp accumulators
                size_t shmem_bwd = 2 * DC * sizeof(cuda_t) + W * 2 * DC * sizeof(float);

                dim3 blocks(num_nodes_bucket, H);
                dim3 threads(W * kWarpSize);

                graph_attn_backward_csrT_kernel_D<W, DC, cuda_t, index_t><<<blocks, threads, shmem_bwd, cuda_stream>>>(
                    N, H,
                    index_ptr<index_t>(row_ptr_T), index_ptr<index_t>(col_idx_T),
                    index_ptr<index_t>(node_indices),
                    Q_ptr, K_ptr, V_ptr,
                    q_strides[0], q_strides[1],
                    k_strides[0], k_strides[1],
                    v_strides[0], v_strides[1],
                    dO_ptr,
                    logsumexp.data_ptr<float>(),
                    Delta.data_ptr<float>(),
                    scale,
                    dQ_ptr, dK_ptr, dV_ptr
                );
            }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
               MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
               MakeIntVariant<32, 64, 128, 256>((int)D),
               warp_variant);
        };

        // Light nodes
        launch_bucket(light_nodes, light_nodes.numel(),
                      MakeIntVariant<1, 2, 4>(light_warps_per_block));

        // Heavy nodes
        launch_bucket(heavy_nodes, heavy_nodes.numel(),
                      MakeIntVariant<8, 16, 32>(heavy_warps_per_block));
    } else {
        // Undirected path: forward CSR, no atomics, no bucketing
        std::visit([&](auto idxInfo, auto typeInfo, auto d_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t = typename decltype(typeInfo)::CudaType;
            constexpr int DC = decltype(d_c)::value;

            auto cuda_stream = at::cuda::getDefaultCUDAStream();

            auto* Q_ptr  = reinterpret_cast<const cuda_t*>(Q.data_ptr<torch_t>());
            auto* K_ptr  = reinterpret_cast<const cuda_t*>(K.data_ptr<torch_t>());
            auto* V_ptr  = reinterpret_cast<const cuda_t*>(V.data_ptr<torch_t>());
            auto* dO_ptr = reinterpret_cast<const cuda_t*>(dO.data_ptr<torch_t>());
            auto* dQ_ptr = reinterpret_cast<cuda_t*>(dQ.data_ptr<torch_t>());
            auto* dV_ptr = reinterpret_cast<cuda_t*>(dV.data_ptr<torch_t>());
            auto* dK_ptr = reinterpret_cast<cuda_t*>(dK_typed.data_ptr<torch_t>());

            // 3 cuda_t vectors (K,Q,V) + 3 float accumulators (dK,dQ,dV)
            size_t shmem_bwd = 3 * DC * sizeof(cuda_t) + 3 * DC * sizeof(float);

            dim3 blocks_bwd(N, H);
            dim3 threads_bwd(kWarpSize);

            graph_attn_backward_fwd_csr_undirected_kernel_D<DC, cuda_t, index_t><<<blocks_bwd, threads_bwd, shmem_bwd, cuda_stream>>>(
                N, H,
                index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
                Q_ptr, K_ptr, V_ptr,
                q_strides[0], q_strides[1],
                k_strides[0], k_strides[1],
                v_strides[0], v_strides[1],
                dO_ptr,
                logsumexp.data_ptr<float>(),
                Delta.data_ptr<float>(),
                scale,
                dQ_ptr, dK_ptr, dV_ptr
            );
        }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
           MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
           MakeIntVariant<32, 64, 128, 256>((int)D));
    }

    CUDA_KERNEL_CHECK();

    // Convert float32 dK accumulator back to input dtype for directed path
    torch::Tensor dK = is_directed ? dK_f32.to(input_dtype) : dK_typed;

    return std::make_tuple(dQ, dK, dV);
}
