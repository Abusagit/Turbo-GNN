#include <cstdint>

#include "common.cuh"

// ===================================================
// ================== BACKWARD =======================
// ===================================================

// D[i,h] = sum_d dO[i,h,d] * O[i,h,d]
template <turbo_gnn::sched::ScheduleKind SK, int D_CONST, FloatingNum cuda_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kWarpSize) compute_D_mh_kernel_D(
    turbo_gnn::sched::SchedulerParams<int32_t> sched_params,
    cuda_t const *const __restrict__ dO,    // [N, H, D]
    cuda_t const *const __restrict__ O_in,  // [N, H, D]
    accum_t *const __restrict__ D_out,      // [N, H]
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

    using Tile = TileOps<TW, cuda_t, accum_t>;

    const int head_h = blockIdx.y;
    const int lane   = threadIdx.x;  // 0..31

    // Body in a lambda: its `return`s become per-node `continue` semantics.
    auto process_node = [&](const int node_i) {

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) [[unlikely]] {
        return;
    }

    cuda_t const *const dO_base = dO + node_i * stride_do_n + head_h * stride_do_h;
    cuda_t const *const O_base  = O_in + node_i * stride_o_n + head_h * stride_o_h;

    accum_t sum{};

#pragma unroll
    for (int fv = lane; fv < TILES; fv += kWarpSize) {
        const typename Tile::vec_t dO_v = Tile::read(dO_base, fv);
        const typename Tile::vec_t O_v  = Tile::read(O_base, fv);
        dO_v.dot_product_(&sum, O_v);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        D_out[node_i * H + head_h] = sum;
    }
    };  // process_node

    using Sched = turbo_gnn::sched::NodeScheduler<SK, int32_t, /*SyncBlock=*/false>;
    __shared__ typename Sched::SharedStorage sched_smem;
    Sched sched(sched_params, sched_smem);
    for (auto work = sched.first(); sched.valid(work); work = sched.next(work)) {
        process_node(static_cast<int>(sched.node(work)));
    }
}


// Q, K, V, dO are [N, H, D] with contiguous D (stride(2)==1), D % 4 == 0
// Q, K, V may be non-contiguous in N,H dims (e.g. from split/view).
// logsumexp and Delta are [N, H].
// dQ, dK, dV are cuda_t output (contiguous); internal accumulation in float32
template <
    turbo_gnn::sched::ScheduleKind SK, int WARPS_PER_BLOCK, int D_CONST, FloatingNum cuda_t, typename index_t,
    FloatingNum accum_t = float>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) graph_attn_backward_csrT_kernel_D(
    int64_t N, int64_t H,
    index_t const *const __restrict__ row_ptr_T,     // [N+1], CSR^T row pointers
    index_t const *const __restrict__ col_idx_T,     // [E],   CSR^T col indices
    turbo_gnn::sched::SchedulerParams<index_t> sched_params,
    cuda_t const *const __restrict__ Q,              // [N, H, D]
    cuda_t const *const __restrict__ K,              // [N, H, D]
    cuda_t const *const __restrict__ V,              // [N, H, D]
    int64_t stride_q_n, int64_t stride_q_h, int64_t stride_k_n, int64_t stride_k_h, int64_t stride_v_n, int64_t stride_v_h,
    cuda_t const *const __restrict__ dO,          // [N, H, D]
    accum_t const *const __restrict__ logsumexp,  // [N, H]
    accum_t const *const __restrict__ Delta,      // [N, H]
    accum_t scale,
    cuda_t *const __restrict__ dQ,   // [N, H, D] (contiguous)
    accum_t *const __restrict__ dK,  // [N, H, D] (contiguous, float32 for atomicAdd)
    cuda_t *const __restrict__ dV    // [N, H, D] (contiguous)
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    const int head_h  = blockIdx.y;

    // Body in a lambda: its `return`s become per-node `continue` semantics.
    auto process_node = [&](const int node_j) {
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
    // warp_gq:   WARPS_PER_BLOCK * D_CONST * sizeof(accum_t)       -- per-warp dQ accumulators
    // warp_gv:   WARPS_PER_BLOCK * D_CONST * sizeof(accum_t)       -- per-warp dV accumulators
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *qj_shared = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *vj_shared = qj_shared + D_CONST;
    accum_t *warp_gq  = reinterpret_cast<accum_t *>(sh_raw + 2 * D_CONST * sizeof(cuda_t));
    accum_t *warp_gv  = warp_gq + WARPS_PER_BLOCK * D_CONST;

    // Per-warp accumulator pointers
    accum_t *my_gq = warp_gq + warp_id * D_CONST;
    accum_t *my_gv = warp_gv + warp_id * D_CONST;

    // Cooperative load of qj, vj using all threads across all warps
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        float4 const *const qj_src = reinterpret_cast<const float4 *>(Q + node_j * stride_q_n + head_h * stride_q_h);
        float4 const *const vj_src = reinterpret_cast<const float4 *>(V + node_j * stride_v_n + head_h * stride_v_h);
        float4 *const qj_sh_f4     = reinterpret_cast<float4 *>(qj_shared);
        float4 *const vj_sh_f4     = reinterpret_cast<float4 *>(vj_shared);
        for (int i = threadIdx.x; i < NUM_LOADS; i += WARPS_PER_BLOCK * kWarpSize) {
            qj_sh_f4[i] = qj_src[i];
            vj_sh_f4[i] = vj_src[i];
        }
    }

    // Zero per-warp float32 gradient accumulators
    {
        constexpr int NUM_F4   = D_CONST / 4;
        float4 *const my_gq_f4 = reinterpret_cast<float4 *>(my_gq);
        float4 *const my_gv_f4 = reinterpret_cast<float4 *>(my_gv);
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

        cuda_t const *ki_base  = K + node_i * stride_k_n + head_h * stride_k_h;
        const size_t out_ih    = static_cast<size_t>(node_i) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        cuda_t const *dOi_base = dO + out_ih;

        // 1) dot(k_i, q_j) and dP_ij = <dO_i, v_j>
        accum_t dot_kq{};
        accum_t dP_ij{};

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            const typename Tile::vec_t ki  = Tile::read(ki_base, fv);
            const typename Tile::vec_t qj  = Tile::read(qj_shared, fv);
            const typename Tile::vec_t vj  = Tile::read(vj_shared, fv);
            const typename Tile::vec_t dOi = Tile::read(dOi_base, fv);

            ki.dot_product_(&dot_kq, qj);
            dOi.dot_product_(&dP_ij, vj);
        }

        dot_kq = warp_reduce_sum(dot_kq);
        dP_ij  = warp_reduce_sum(dP_ij);

        const accum_t score = dot_kq * scale;

        accum_t L_i{}, Delta_i{};
        if (lane == 0) {
            const size_t idx_ih = static_cast<size_t>(node_i) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_i                 = __ldg(&logsumexp[idx_ih]);
            Delta_i             = __ldg(&Delta[idx_ih]);
        }
        L_i     = __shfl_sync(FULL_WARP_MASK, L_i, 0);
        Delta_i = __shfl_sync(FULL_WARP_MASK, Delta_i, 0);

        const accum_t alpha     = AccumOps::exp(score - L_i);
        const accum_t dS        = alpha * (dP_ij - Delta_i);
        const accum_t dS_scaled = dS * scale;

        // 2) accumulate dV_j, dQ_j in per-warp float32 shared; atomicAdd dK_i
        accum_t *const dK_i_base = dK + out_ih;

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            int base_f                     = fv * TW;
            const typename Tile::vec_t ki  = Tile::read(ki_base, fv);
            const typename Tile::vec_t dOi = Tile::read(dOi_base, fv);
            const typename Tile::vec_t qj  = Tile::read(qj_shared, fv);

            dOi.weighted_accum_(&my_gv[base_f], alpha);
            ki.weighted_accum_(&my_gq[base_f], dS_scaled);
            Tile::atomic_add_scaled_f32(dK_i_base, base_f, dS_scaled, qj);
        }
    }

    // 3) Cross-warp reduction: warp 0 sums all per-warp accumulators and writes output
    __syncthreads();

    if (warp_id == 0) {
        cuda_t *const dQ_base = dQ + out_jh;
        cuda_t *const dV_base = dV + out_jh;

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            int base_f         = fv * TW;
            accum_t gq_sum[TW] = {0};
            accum_t gv_sum[TW] = {0};
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    gq_sum[ep] += warp_gq[w * D_CONST + base_f + ep];
                    gv_sum[ep] += warp_gv[w * D_CONST + base_f + ep];
                }
            }
            Tile::write_convert_from_accum(&dQ_base[fv * TW], gq_sum);
            Tile::write_convert_from_accum(&dV_base[fv * TW], gv_sum);
        }
    }
    };  // process_node

    using Sched = turbo_gnn::sched::NodeScheduler<SK, index_t, /*SyncBlock=*/true>;
    __shared__ typename Sched::SharedStorage sched_smem;
    Sched sched(sched_params, sched_smem);
    for (auto work = sched.first(); sched.valid(work); work = sched.next(work)) {
        process_node(static_cast<int>(sched.node(work)));
    }
}


// =============================================================================
// Undirected backward kernel: uses forward CSR, zero atomics.
// For each dst node d, iterates over src neighbors s. Computes:
//   Forward direction: dK[d] (local)
//   Reverse direction: dQ[d], dV[d] (local, exploiting symmetric adjacency)
// =============================================================================
template <turbo_gnn::sched::ScheduleKind SK, int D_CONST, typename cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kWarpSize) graph_attn_backward_fwd_csr_undirected_kernel_D(
    turbo_gnn::sched::SchedulerParams<index_t> sched_params,
    int64_t N, int64_t H,
    index_t const *const __restrict__ row_ptr,  // [N+1], forward CSR row pointers
    index_t const *const __restrict__ col_idx,  // [E],   forward CSR col indices
    cuda_t const *const __restrict__ Q,         // [N, H, D]
    cuda_t const *const __restrict__ K,         // [N, H, D]
    cuda_t const *const __restrict__ V,         // [N, H, D]
    int64_t stride_q_n, int64_t stride_q_h, int64_t stride_k_n, int64_t stride_k_h, int64_t stride_v_n, int64_t stride_v_h,
    cuda_t const *const __restrict__ dO,          // [N, H, D] (contiguous)
    accum_t const *const __restrict__ logsumexp,  // [N, H]
    accum_t const *const __restrict__ Delta,      // [N, H]
    accum_t scale,
    cuda_t *const __restrict__ dQ,  // [N, H, D] (contiguous)
    cuda_t *const __restrict__ dK,  // [N, H, D] (contiguous, cuda_t — no atomics)
    cuda_t *const __restrict__ dV   // [N, H, D] (contiguous)
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    const int head_h = blockIdx.y;

    // Body in a lambda: its `return`s become per-node `continue` semantics.
    auto process_node = [&](const int node_d) {
    const int lane   = threadIdx.x;  // 0..31

    if (node_d >= N || head_h >= H) [[unlikely]] {
        return;
    }

    const index_t edge_start = row_ptr[node_d];
    const index_t edge_end   = row_ptr[node_d + 1];
    const int num_neighbors  = static_cast<int>(edge_end - edge_start);

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
    //   gk_shared:  D_CONST * sizeof(accum_t)  -- float32 accumulator for dK[d]
    //   gq_shared:  D_CONST * sizeof(accum_t)  -- float32 accumulator for dQ[d]
    //   gv_shared:  D_CONST * sizeof(accum_t)  -- float32 accumulator for dV[d]
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *const kd_shared  = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *const qd_shared  = kd_shared + D_CONST;
    cuda_t *const vd_shared  = qd_shared + D_CONST;
    accum_t *const gk_shared = reinterpret_cast<accum_t *>(sh_raw + 3 * D_CONST * sizeof(cuda_t));
    accum_t *const gq_shared = gk_shared + D_CONST;
    accum_t *const gv_shared = gq_shared + D_CONST;

    // Load K[d], Q[d], V[d] via 128-bit transactions
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        float4 const *const kd_src = reinterpret_cast<float4 const *>(K + node_d * stride_k_n + head_h * stride_k_h);
        float4 const *const qd_src = reinterpret_cast<float4 const *>(Q + node_d * stride_q_n + head_h * stride_q_h);
        float4 const *const vd_src = reinterpret_cast<float4 const *>(V + node_d * stride_v_n + head_h * stride_v_h);
        float4 *const kd_sh_f4     = reinterpret_cast<float4 *>(kd_shared);
        float4 *const qd_sh_f4     = reinterpret_cast<float4 *>(qd_shared);
        float4 *const vd_sh_f4     = reinterpret_cast<float4 *>(vd_shared);
#pragma unroll_k 4
        for (size_t i = lane; i < NUM_LOADS; i += kWarpSize) {
            kd_sh_f4[i] = kd_src[i];
            qd_sh_f4[i] = qd_src[i];
            vd_sh_f4[i] = vd_src[i];
        }
    }

    // Zero float32 gradient accumulators
    {
        constexpr int NUM_F4 = D_CONST / 4;
        float4 *const gk_f4  = reinterpret_cast<float4 *>(gk_shared);
        float4 *const gq_f4  = reinterpret_cast<float4 *>(gq_shared);
        float4 *const gv_f4  = reinterpret_cast<float4 *>(gv_shared);
#pragma unroll_k 4
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            gk_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            gq_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            gv_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    __syncwarp(FULL_WARP_MASK);

    // Row scalars
    accum_t L_d{}, Delta_d{};
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
        cuda_t const *const qs_base = Q + node_s * stride_q_n + head_h * stride_q_h;
        cuda_t const *const ks_base = K + node_s * stride_k_n + head_h * stride_k_h;
        cuda_t const *const vs_base = V + node_s * stride_v_n + head_h * stride_v_h;
        // dO[s] is contiguous
        const size_t out_sh          = static_cast<size_t>(node_s) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        cuda_t const *const dOs_base = dO + out_sh;

        // 1) Compute dot products for both directions
        accum_t dot_kd_qs{};  // K[d] . Q[s]  -> forward score
        accum_t dP_fwd{};     // dO[d] . V[s]  -> forward dP
        accum_t dot_qd_ks{};  // Q[d] . K[s]  -> reverse score
        accum_t dP_rev{};     // V[d] . dO[s]  -> reverse dP

        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            const typename Tile::vec_t kd  = Tile::read(kd_shared, fv);
            const typename Tile::vec_t qd  = Tile::read(qd_shared, fv);
            const typename Tile::vec_t vd  = Tile::read(vd_shared, fv);
            const typename Tile::vec_t dOd = Tile::read(dOd_base, fv);
            const typename Tile::vec_t qs  = Tile::read(qs_base, fv);
            const typename Tile::vec_t ks  = Tile::read(ks_base, fv);
            const typename Tile::vec_t vs  = Tile::read(vs_base, fv);
            const typename Tile::vec_t dOs = Tile::read(dOs_base, fv);

            kd.dot_product_(&dot_kd_qs, qs);
            dOd.dot_product_(&dP_fwd, vs);
            qd.dot_product_(&dot_qd_ks, ks);
            vd.dot_product_(&dP_rev, dOs);
        }

        dot_kd_qs = warp_reduce_sum(dot_kd_qs);
        dP_fwd    = warp_reduce_sum(dP_fwd);
        dot_qd_ks = warp_reduce_sum(dot_qd_ks);
        dP_rev    = warp_reduce_sum(dP_rev);

        // 2) Load L[s] and Delta[s] for reverse direction
        accum_t L_s{}, Delta_s{};
        if (lane == 0) {
            const size_t idx_sh = static_cast<size_t>(node_s) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_s                 = __ldg(&logsumexp[idx_sh]);
            Delta_s             = __ldg(&Delta[idx_sh]);
        }
        L_s     = __shfl_sync(FULL_WARP_MASK, L_s, 0);
        Delta_s = __shfl_sync(FULL_WARP_MASK, Delta_s, 0);

        // 3) Forward direction: dK[d] += dS_fwd * Q[s]
        const accum_t score_fwd     = dot_kd_qs * scale;
        const accum_t alpha_fwd     = AccumOps::exp(score_fwd - L_d);
        const accum_t dS_fwd        = alpha_fwd * (dP_fwd - Delta_d);
        const accum_t dS_fwd_scaled = dS_fwd * scale;

        // 4) Reverse direction: dQ[d] += dS_rev * K[s], dV[d] += alpha_rev * dO[s]
        const accum_t score_rev     = dot_qd_ks * scale;
        const accum_t alpha_rev     = AccumOps::exp(score_rev - L_s);
        const accum_t dS_rev        = alpha_rev * (dP_rev - Delta_s);
        const accum_t dS_rev_scaled = dS_rev * scale;

        // 5) Accumulate all three gradients in shared float32
        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            int base_f                     = fv * TW;
            const typename Tile::vec_t qs  = Tile::read(qs_base, fv);
            const typename Tile::vec_t ks  = Tile::read(ks_base, fv);
            const typename Tile::vec_t dOs = Tile::read(dOs_base, fv);

            qs.weighted_accum_(&gk_shared[base_f], dS_fwd_scaled); // dK[d] += dS_fwd * Q[s]
            ks.weighted_accum_(&gq_shared[base_f], dS_rev_scaled); // dQ[d] += dS_rev * K[s]
            dOs.weighted_accum_(&gv_shared[base_f], alpha_rev); // dV[d] += P_rev * dO[s]
        }
    }

    // Write all three gradients: convert float32 accumulators to cuda_t
    cuda_t *const dK_base = dK + out_dh;
    cuda_t *const dQ_base = dQ + out_dh;
    cuda_t *const dV_base = dV + out_dh;

    for (int fv = lane; fv < TILES; fv += kWarpSize) {
        int base_f = fv * TW;
        Tile::write_convert_from_accum(&dK_base[fv * TW], &gk_shared[base_f]);
        Tile::write_convert_from_accum(&dQ_base[fv * TW], &gq_shared[base_f]);
        Tile::write_convert_from_accum(&dV_base[fv * TW], &gv_shared[base_f]);
    }
    };  // process_node

    using Sched = turbo_gnn::sched::NodeScheduler<SK, index_t, /*SyncBlock=*/true>;
    __shared__ typename Sched::SharedStorage sched_smem;
    Sched sched(sched_params, sched_smem);
    for (auto work = sched.first(); sched.valid(work); work = sched.next(work)) {
        process_node(static_cast<int>(sched.node(work)));
    }
}

// ================================================================================================
// Split-K heavy path for the directed backward.
//
// One block per fixed-size slice of one destination node's incoming edge list, mirroring the
// forward split in csrc/gt/gt_forward.cu. The backward is simpler to split than the forward:
// `alpha` is recomputed from the saved logsumexp rather than tracked online, so a slice's dQ and
// dV contributions are plain sums over its own edges and the merge is elementwise addition --
// no rescaling, no max to reconcile.
//
// dK needs nothing at all. It is already scattered with atomicAdd to arbitrary source nodes, so
// splitting the destination's edge list across blocks changes only which block issues which
// atomic, which the accumulation is indifferent to.
// ================================================================================================

template <int WARPS_PER_BLOCK, int D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize) graph_attn_backward_csrT_slice_kernel_D(
    int64_t N, int64_t H,
    index_t const *const __restrict__ row_ptr_T,
    index_t const *const __restrict__ col_idx_T,
    index_t const *const __restrict__ heavy_nodes,
    int const *const __restrict__ chunk_node,
    int const *const __restrict__ chunk_start,
    int slice_size, int num_slices,
    cuda_t const *const __restrict__ Q,
    cuda_t const *const __restrict__ K,
    cuda_t const *const __restrict__ V,
    int64_t stride_q_n, int64_t stride_q_h, int64_t stride_k_n, int64_t stride_k_h, int64_t stride_v_n, int64_t stride_v_h,
    cuda_t const *const __restrict__ dO,
    accum_t const *const __restrict__ logsumexp,
    accum_t const *const __restrict__ Delta,
    accum_t scale,
    accum_t *const __restrict__ part_gq,
    accum_t *const __restrict__ part_gv,
    accum_t *const __restrict__ dK
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;
    constexpr int TW    = TW_SELECTOR::value;
    constexpr int TILES = (D_CONST + TW - 1) / TW;

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    const int slice_id = blockIdx.x;
    const int head_h   = blockIdx.y;
    if (slice_id >= num_slices || head_h >= H) [[unlikely]] {
        return;
    }

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    const int slot   = chunk_node[slice_id];
    const int node_j = static_cast<int>(heavy_nodes[slot]);
    if (node_j >= N) [[unlikely]] {
        return;
    }

    const index_t edge_start = row_ptr_T[node_j];
    const int num_incoming   = static_cast<int>(row_ptr_T[node_j + 1] - edge_start);

    const int local_start = chunk_start[slice_id];
    const int local_end   = min(local_start + slice_size, num_incoming);

    const size_t part_off = (static_cast<size_t>(slice_id) * H + head_h) * D_CONST;

    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *qj_shared = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *vj_shared = qj_shared + D_CONST;
    accum_t *warp_gq  = reinterpret_cast<accum_t *>(sh_raw + 2 * D_CONST * sizeof(cuda_t));
    accum_t *warp_gv  = warp_gq + WARPS_PER_BLOCK * D_CONST;
    accum_t *my_gq    = warp_gq + warp_id * D_CONST;
    accum_t *my_gv    = warp_gv + warp_id * D_CONST;

    if (local_start >= local_end) [[unlikely]] {
        if (warp_id == 0) {
            for (int f = lane; f < D_CONST; f += kWarpSize) {
                part_gq[part_off + f] = accum_t{};
                part_gv[part_off + f] = accum_t{};
            }
        }
        return;
    }

    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        float4 const *const qj_src = reinterpret_cast<const float4 *>(Q + node_j * stride_q_n + head_h * stride_q_h);
        float4 const *const vj_src = reinterpret_cast<const float4 *>(V + node_j * stride_v_n + head_h * stride_v_h);
        float4 *const qj_sh_f4     = reinterpret_cast<float4 *>(qj_shared);
        float4 *const vj_sh_f4     = reinterpret_cast<float4 *>(vj_shared);
        for (int i = threadIdx.x; i < NUM_LOADS; i += WARPS_PER_BLOCK * kWarpSize) {
            qj_sh_f4[i] = qj_src[i];
            vj_sh_f4[i] = vj_src[i];
        }
    }
    {
        constexpr int NUM_F4   = D_CONST / 4;
        float4 *const my_gq_f4 = reinterpret_cast<float4 *>(my_gq);
        float4 *const my_gv_f4 = reinterpret_cast<float4 *>(my_gv);
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            my_gq_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            my_gv_f4[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    __syncthreads();

    // Warps stride over this slice's incoming edges only.
    for (int e = local_start + warp_id; e < local_end; e += WARPS_PER_BLOCK) {
        index_t node_i = 0;
        if (lane == 0) {
            node_i = __ldg(&col_idx_T[edge_start + e]);
        }
        node_i = __shfl_sync(FULL_WARP_MASK, node_i, 0);
        if (node_i >= N) [[unlikely]] {
            continue;
        }

        cuda_t const *ki_base  = K + node_i * stride_k_n + head_h * stride_k_h;
        const size_t out_ih    = static_cast<size_t>(node_i) * H * D_CONST + static_cast<size_t>(head_h) * D_CONST;
        cuda_t const *dOi_base = dO + out_ih;

        accum_t dot_kq{};
        accum_t dP_ij{};
        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            const typename Tile::vec_t ki  = Tile::read(ki_base, fv);
            const typename Tile::vec_t qj  = Tile::read(qj_shared, fv);
            const typename Tile::vec_t vj  = Tile::read(vj_shared, fv);
            const typename Tile::vec_t dOi = Tile::read(dOi_base, fv);
            ki.dot_product_(&dot_kq, qj);
            dOi.dot_product_(&dP_ij, vj);
        }
        dot_kq = warp_reduce_sum(dot_kq);
        dP_ij  = warp_reduce_sum(dP_ij);

        const accum_t score = dot_kq * scale;

        accum_t L_i{}, Delta_i{};
        if (lane == 0) {
            const size_t idx_ih = static_cast<size_t>(node_i) * static_cast<size_t>(H) + static_cast<size_t>(head_h);
            L_i                 = __ldg(&logsumexp[idx_ih]);
            Delta_i             = __ldg(&Delta[idx_ih]);
        }
        L_i     = __shfl_sync(FULL_WARP_MASK, L_i, 0);
        Delta_i = __shfl_sync(FULL_WARP_MASK, Delta_i, 0);

        const accum_t alpha     = AccumOps::exp(score - L_i);
        const accum_t dS        = alpha * (dP_ij - Delta_i);
        const accum_t dS_scaled = dS * scale;

        accum_t *const dK_i_base = dK + out_ih;
        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            const int base_f               = fv * TW;
            const typename Tile::vec_t ki  = Tile::read(ki_base, fv);
            const typename Tile::vec_t dOi = Tile::read(dOi_base, fv);
            const typename Tile::vec_t qj  = Tile::read(qj_shared, fv);

            dOi.weighted_accum_(&my_gv[base_f], alpha);
            ki.weighted_accum_(&my_gq[base_f], dS_scaled);
            Tile::atomic_add_scaled_f32(dK_i_base, base_f, dS_scaled, qj);
        }
    }

    __syncthreads();

    // Cross-warp sum into this slice's partial. Plain addition -- no softmax state to reconcile.
    if (warp_id == 0) {
        for (int f = lane; f < D_CONST; f += kWarpSize) {
            accum_t gq{}, gv{};
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                gq += warp_gq[w * D_CONST + f];
                gv += warp_gv[w * D_CONST + f];
            }
            part_gq[part_off + f] = gq;
            part_gv[part_off + f] = gv;
        }
    }
}

/// Sum every slice's dQ/dV partials into one heavy node's gradient rows.
/// Grid (num_heavy, H), one warp per block.
template <int D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kWarpSize) graph_attn_backward_merge_slices_D(
    int64_t H,
    index_t const *const __restrict__ row_ptr_T,
    index_t const *const __restrict__ heavy_nodes,
    int const *const __restrict__ node_chunk_offset,
    accum_t const *const __restrict__ part_gq,
    accum_t const *const __restrict__ part_gv,
    cuda_t *const __restrict__ dQ,
    cuda_t *const __restrict__ dV,
    int num_heavy
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;
    constexpr int TW    = TW_SELECTOR::value;
    constexpr int TILES = (D_CONST + TW - 1) / TW;
    using Tile          = TileOps<TW, cuda_t, accum_t>;

    const int slot   = blockIdx.x;
    const int head_h = blockIdx.y;
    if (slot >= num_heavy || head_h >= H) [[unlikely]] {
        return;
    }

    const int lane      = threadIdx.x;
    const int node_j    = static_cast<int>(heavy_nodes[slot]);
    const size_t out_jh = (static_cast<size_t>(node_j) * H + head_h) * D_CONST;

    // Isolated nodes take the same path as the in-place kernel: zeroed dQ and dV rows.
    if (row_ptr_T[node_j + 1] == row_ptr_T[node_j]) [[unlikely]] {
        for (int fv = lane; fv < TILES; fv += kWarpSize) {
            Tile::write_zero(dQ + out_jh, fv);
            Tile::write_zero(dV + out_jh, fv);
        }
        return;
    }

    const int lo = node_chunk_offset[slot];
    const int hi = node_chunk_offset[slot + 1];

    for (int fv = lane; fv < TILES; fv += kWarpSize) {
        accum_t gq[TW];
        accum_t gv[TW];
#pragma unroll
        for (int ep = 0; ep < TW; ++ep) {
            gq[ep] = accum_t{};
            gv[ep] = accum_t{};
        }
        for (int s = lo; s < hi; ++s) {
            const size_t off = (static_cast<size_t>(s) * H + head_h) * D_CONST + fv * TW;
#pragma unroll
            for (int ep = 0; ep < TW; ++ep) {
                gq[ep] += part_gq[off + ep];
                gv[ep] += part_gv[off + ep];
            }
        }
        Tile::write_convert_from_accum(&dQ[out_jh + fv * TW], gq);
        Tile::write_convert_from_accum(&dV[out_jh + fv * TW], gv);
    }
}
