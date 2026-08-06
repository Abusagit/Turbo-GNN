#include <cstdint>

#include "common.cuh"

// ===================================================
// ================== BACKWARD =======================
// ===================================================

// D[i,h] = sum_d dO[i,h,d] * O[i,h,d]
template <int D_CONST, typename cuda_t>
__global__ void __launch_bounds__(kWarpSize) compute_D_mh_kernel_D(
    cuda_t const *const __restrict__ dO,    // [N, H, D]
    cuda_t const *const __restrict__ O_in,  // [N, H, D]
    float *const __restrict__ D_out,        // [N, H]
    int64_t N,
    int64_t H,
    int64_t stride_do_n,
    int64_t stride_do_h,
    int64_t stride_o_n,
    int64_t stride_o_h
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;                                                  // Accumulatores used by one thread

    using Tile = TileOps<TW, cuda_t>;

    const int node_i = blockIdx.x;
    const int head_h = blockIdx.y;
    const int lane   = threadIdx.x;  // 0..31

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) {
        return;
    }

    const cuda_t *dO_base = dO + node_i * stride_do_n + head_h * stride_do_h;
    const cuda_t *O_base  = O_in + node_i * stride_o_n + head_h * stride_o_h;

    float sum = 0.0f;

#pragma unroll
    for (int fv = lane; fv < TILES; fv += kWarpSize) {
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
template <int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) graph_attn_backward_csrT_kernel_D(
    int64_t N, int64_t H,
    index_t const *const __restrict__ row_ptr_T,     // [N+1], CSR^T row pointers
    index_t const *const __restrict__ col_idx_T,     // [E],   CSR^T col indices
    index_t const *const __restrict__ node_indices,  // node indirection
    cuda_t const *const __restrict__ Q,              // [N, H, D]
    cuda_t const *const __restrict__ K,              // [N, H, D]
    cuda_t const *const __restrict__ V,              // [N, H, D]
    int64_t stride_q_n, int64_t stride_q_h, int64_t stride_k_n, int64_t stride_k_h, int64_t stride_v_n, int64_t stride_v_h,
    cuda_t const *const __restrict__ dO,        // [N, H, D]
    float const *const __restrict__ logsumexp,  // [N, H]
    float const *const __restrict__ Delta,      // [N, H]
    float scale,
    cuda_t *const __restrict__ dQ,  // [N, H, D] (contiguous)
    float *const __restrict__ dK,   // [N, H, D] (contiguous, float32 for atomicAdd)
    cuda_t *const __restrict__ dV   // [N, H, D] (contiguous)
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;                                                  // Accumulatores used by one thread

    using Tile = TileOps<TW, cuda_t>;

    const int node_j  = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h  = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    if (node_j >= N || head_h >= H) [[unlikely]] {
        return;
    }

    index_t edge_start = row_ptr_T[node_j];
    index_t edge_end   = row_ptr_T[node_j + 1];
    int num_incoming   = static_cast<int>(edge_end - edge_start);

    // Contiguous offset for output dQ, dV (freshly allocated, always contiguous)
    const size_t out_jh = (node_j * H + head_h) * D_CONST;

    // nothing to do if this node has no incoming edges — all warps write zeros and return
    if (num_incoming == 0) {
        if (warp_id == 0) {
            for (int fv = lane; fv < TILES; fv += kWarpSize) {
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
    extern __shared__ uint8_t sh_raw[];
    cuda_t *qj_shared = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *vj_shared = qj_shared + D_CONST;
    float *warp_gq    = reinterpret_cast<float *>(sh_raw + 2 * D_CONST * sizeof(cuda_t));
    float *warp_gv    = warp_gq + WARPS_PER_BLOCK * D_CONST;

    // Per-warp accumulator pointers
    float *my_gq = warp_gq + warp_id * D_CONST;
    float *my_gv = warp_gv + warp_id * D_CONST;

    // Cooperative load of qj, vj using all threads across all warps
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        const float4 *qj_src       = reinterpret_cast<const float4 *>(Q + node_j * stride_q_n + head_h * stride_q_h);
        const float4 *vj_src       = reinterpret_cast<const float4 *>(V + node_j * stride_v_n + head_h * stride_v_h);
        float4 *qj_sh_f4           = reinterpret_cast<float4 *>(qj_shared);
        float4 *vj_sh_f4           = reinterpret_cast<float4 *>(vj_shared);
        for (int i = threadIdx.x; i < NUM_LOADS; i += WARPS_PER_BLOCK * kWarpSize) {
            qj_sh_f4[i] = qj_src[i];
            vj_sh_f4[i] = vj_src[i];
        }
    }

    // Zero per-warp float32 gradient accumulators
    {
        constexpr int NUM_F4 = D_CONST / 4;
        float4 *my_gq_f4     = reinterpret_cast<float4 *>(my_gq);
        float4 *my_gv_f4     = reinterpret_cast<float4 *>(my_gv);
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            my_gq_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            my_gv_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
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

        if (node_i >= N) [[unlikely]] {
            continue;
        }

        const cuda_t *ki_base  = K + node_i * stride_k_n + head_h * stride_k_h;
        const size_t out_ih    = static_cast<size_t>(node_i) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        const cuda_t *dOi_base = dO + out_ih;

        // 1) dot(k_i, q_j) and dP_ij = <dO_i, v_j>
        float dot_kq = 0.0f;
        float dP_ij  = 0.0f;

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            auto ki  = Tile::load(ki_base, fv);
            auto qj  = Tile::load(qj_shared, fv);
            auto vj  = Tile::load(vj_shared, fv);
            auto dOi = Tile::load(dOi_base, fv);

            dot_kq += Tile::dot_product(ki, qj);
            dP_ij += Tile::dot_product(dOi, vj);
        }

        dot_kq = warp_reduce_sum(dot_kq);
        dP_ij  = warp_reduce_sum(dP_ij);

        const float score = dot_kq * scale;

        float L_i = 0.0f, Delta_i = 0.0f;
        if (lane == 0) {
            const size_t idx_ih = static_cast<size_t>(node_i) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_i                 = __ldg(&logsumexp[idx_ih]);
            Delta_i             = __ldg(&Delta[idx_ih]);
        }
        L_i     = __shfl_sync(FULL_WARP_MASK, L_i, 0);
        Delta_i = __shfl_sync(FULL_WARP_MASK, Delta_i, 0);

        const float alpha     = __expf(score - L_i);
        const float dS        = alpha * (dP_ij - Delta_i);
        const float dS_scaled = dS * scale;

        // 2) accumulate dV_j, dQ_j in per-warp float32 shared; atomicAdd dK_i
        float *dK_i_base = dK + out_ih;

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            int base_f = fv * TW;
            auto ki    = Tile::load(ki_base, fv);
            auto dOi   = Tile::load(dOi_base, fv);
            auto qj    = Tile::load(qj_shared, fv);

            Tile::weighted_accum(&my_gv[base_f], alpha, dOi);
            Tile::weighted_accum(&my_gq[base_f], dS_scaled, ki);
            Tile::atomic_add_scaled_f32(dK_i_base, base_f, dS_scaled, qj);
        }
    }

    // 3) Cross-warp reduction: warp 0 sums all per-warp accumulators and writes output
    __syncthreads();

    if (warp_id == 0) {
        cuda_t *dQ_base = dQ + out_jh;
        cuda_t *dV_base = dV + out_jh;

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            int base_f = fv * TW;
            float gq_sum[TW];
            float gv_sum[TW];
#pragma unroll
            for (int ep = 0; ep < TW; ++ep) {
                gq_sum[ep] = 0.0f;
                gv_sum[ep] = 0.0f;
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
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
template <int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kWarpSize) graph_attn_backward_fwd_csr_undirected_kernel_D(
    int64_t N, int64_t H,
    index_t const *const __restrict__ row_ptr,  // [N+1], forward CSR row pointers
    index_t const *const __restrict__ col_idx,  // [E],   forward CSR col indices
    cuda_t const *const __restrict__ Q,         // [N, H, D]
    cuda_t const *const __restrict__ K,         // [N, H, D]
    cuda_t const *const __restrict__ V,         // [N, H, D]
    int64_t stride_q_n, int64_t stride_q_h, int64_t stride_k_n, int64_t stride_k_h, int64_t stride_v_n, int64_t stride_v_h,
    cuda_t const *const __restrict__ dO,        // [N, H, D] (contiguous)
    float const *const __restrict__ logsumexp,  // [N, H]
    float const *const __restrict__ Delta,      // [N, H]
    float scale,
    cuda_t *const __restrict__ dQ,  // [N, H, D] (contiguous)
    cuda_t *const __restrict__ dK,  // [N, H, D] (contiguous, cuda_t — no atomics)
    cuda_t *const __restrict__ dV   // [N, H, D] (contiguous)
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;                                                  // Accumulatores used by one thread

    using Tile = TileOps<TW, cuda_t>;

    int node_d = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x;  // 0..31

    if (node_d >= N || head_h >= H) {
        return;
    }

    index_t edge_start = row_ptr[node_d];
    index_t edge_end   = row_ptr[node_d + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    const size_t out_dh = (node_d * H + head_h) * D_CONST;

    // Handle isolated nodes: write zeros
    if (num_neighbors == 0) {
        for (int fv = lane; fv < TILES; fv += kWarpSize) {
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
    extern __shared__ uint8_t sh_raw[];
    cuda_t *kd_shared = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *qd_shared = kd_shared + D_CONST;
    cuda_t *vd_shared = qd_shared + D_CONST;
    float *gk_shared  = reinterpret_cast<float *>(sh_raw + 3 * D_CONST * sizeof(cuda_t));
    float *gq_shared  = gk_shared + D_CONST;
    float *gv_shared  = gq_shared + D_CONST;

    // Load K[d], Q[d], V[d] via 128-bit transactions
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        const float4 *kd_src       = reinterpret_cast<float4 const *>(K + node_d * stride_k_n + head_h * stride_k_h);
        const float4 *qd_src       = reinterpret_cast<float4 const *>(Q + node_d * stride_q_n + head_h * stride_q_h);
        const float4 *vd_src       = reinterpret_cast<float4 const *>(V + node_d * stride_v_n + head_h * stride_v_h);
        float4 *kd_sh_f4           = reinterpret_cast<float4 *>(kd_shared);
        float4 *qd_sh_f4           = reinterpret_cast<float4 *>(qd_shared);
        float4 *vd_sh_f4           = reinterpret_cast<float4 *>(vd_shared);
        for (int i = lane; i < NUM_LOADS; i += kWarpSize) {
            kd_sh_f4[i] = kd_src[i];
            qd_sh_f4[i] = qd_src[i];
            vd_sh_f4[i] = vd_src[i];
        }
    }

    // Zero float32 gradient accumulators
    {
        constexpr int NUM_F4 = D_CONST / 4;
        float4 *gk_f4        = reinterpret_cast<float4 *>(gk_shared);
        float4 *gq_f4        = reinterpret_cast<float4 *>(gq_shared);
        float4 *gv_f4        = reinterpret_cast<float4 *>(gv_shared);
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            gk_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            gq_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            gv_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    __syncwarp(FULL_WARP_MASK);

    // Row scalars
    float L_d = 0.0f, Delta_d = 0.0f;
    if (lane == 0) {
        const size_t idx_dh = static_cast<size_t>(node_d) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
        L_d                 = __ldg(&logsumexp[idx_dh]);
        Delta_d             = __ldg(&Delta[idx_dh]);
    }
    L_d     = __shfl_sync(FULL_WARP_MASK, L_d, 0);
    Delta_d = __shfl_sync(FULL_WARP_MASK, Delta_d, 0);

    // dO[d] base pointer (contiguous)
    const cuda_t *dOd_base = dO + out_dh;

    for (int e = 0; e < num_neighbors; ++e) {
        index_t node_s = 0;
        if (lane == 0) {
            node_s = __ldg(&col_idx[edge_start + e]);
        }
        node_s = __shfl_sync(FULL_WARP_MASK, node_s, 0);

        if (node_s >= N) [[unlikely]] {
            continue;
        }

        // Column node pointers (strided)
        const cuda_t *qs_base = Q + node_s * stride_q_n + head_h * stride_q_h;
        const cuda_t *ks_base = K + node_s * stride_k_n + head_h * stride_k_h;
        const cuda_t *vs_base = V + node_s * stride_v_n + head_h * stride_v_h;
        // dO[s] is contiguous
        const size_t out_sh    = static_cast<size_t>(node_s) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        const cuda_t *dOs_base = dO + out_sh;

        // 1) Compute dot products for both directions
        float dot_kd_qs = 0.0f;  // K[d] . Q[s]  -> forward score
        float dP_fwd    = 0.0f;  // dO[d] . V[s]  -> forward dP
        float dot_qd_ks = 0.0f;  // Q[d] . K[s]  -> reverse score
        float dP_rev    = 0.0f;  // V[d] . dO[s]  -> reverse dP

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            auto kd  = Tile::load(kd_shared, fv);
            auto qd  = Tile::load(qd_shared, fv);
            auto vd  = Tile::load(vd_shared, fv);
            auto dOd = Tile::load(dOd_base, fv);
            auto qs  = Tile::load(qs_base, fv);
            auto ks  = Tile::load(ks_base, fv);
            auto vs  = Tile::load(vs_base, fv);
            auto dOs = Tile::load(dOs_base, fv);

            dot_kd_qs += Tile::dot_product(kd, qs);
            dP_fwd += Tile::dot_product(dOd, vs);
            dot_qd_ks += Tile::dot_product(qd, ks);
            dP_rev += Tile::dot_product(vd, dOs);
        }

        dot_kd_qs = warp_reduce_sum(dot_kd_qs);
        dP_fwd    = warp_reduce_sum(dP_fwd);
        dot_qd_ks = warp_reduce_sum(dot_qd_ks);
        dP_rev    = warp_reduce_sum(dP_rev);

        // 2) Load L[s] and Delta[s] for reverse direction
        float L_s = 0.0f, Delta_s = 0.0f;
        if (lane == 0) {
            const size_t idx_sh = static_cast<size_t>(node_s) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_s                 = __ldg(&logsumexp[idx_sh]);
            Delta_s             = __ldg(&Delta[idx_sh]);
        }
        L_s     = __shfl_sync(FULL_WARP_MASK, L_s, 0);
        Delta_s = __shfl_sync(FULL_WARP_MASK, Delta_s, 0);

        // 3) Forward direction: dK[d] += dS_fwd * Q[s]
        const float score_fwd     = dot_kd_qs * scale;
        const float alpha_fwd     = __expf(score_fwd - L_d);
        const float dS_fwd        = alpha_fwd * (dP_fwd - Delta_d);
        const float dS_fwd_scaled = dS_fwd * scale;

        // 4) Reverse direction: dQ[d] += dS_rev * K[s], dV[d] += alpha_rev * dO[s]
        const float score_rev     = dot_qd_ks * scale;
        const float alpha_rev     = __expf(score_rev - L_s);
        const float dS_rev        = alpha_rev * (dP_rev - Delta_s);
        const float dS_rev_scaled = dS_rev * scale;

        // 5) Accumulate all three gradients in shared float32
        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            int base_f = fv * TW;
            auto qs    = Tile::load(qs_base, fv);
            auto ks    = Tile::load(ks_base, fv);
            auto dOs   = Tile::load(dOs_base, fv);

            Tile::weighted_accum(&gk_shared[base_f], dS_fwd_scaled, qs);  // dK[d] += dS_fwd * Q[s]
            Tile::weighted_accum(&gq_shared[base_f], dS_rev_scaled, ks);  // dQ[d] += dS_rev * K[s]
            Tile::weighted_accum(&gv_shared[base_f], alpha_rev, dOs);     // dV[d] += P_rev * dO[s]
        }
    }

    // Write all three gradients: convert float32 accumulators to cuda_t
    cuda_t *dK_base = dK + out_dh;
    cuda_t *dQ_base = dQ + out_dh;
    cuda_t *dV_base = dV + out_dh;

    for (int fv = lane; fv < TILES; fv += kWarpSize) {
        int base_f = fv * TW;
        Tile::write_typed(dK_base, fv, &gk_shared[base_f]);
        Tile::write_typed(dQ_base, fv, &gq_shared[base_f]);
        Tile::write_typed(dV_base, fv, &gv_shared[base_f]);
    }
}