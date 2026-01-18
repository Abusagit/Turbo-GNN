#include <cstddef>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xffffffff
#endif

#ifndef kWarpSize
constexpr int kWarpSize = 32;
#endif

#define DISPATCH_D(DVAL) do { \
    constexpr int D_CONST = (DVAL); \
    size_t shmem = ((1 + kWarpsPerBlock) * D_CONST + (2 * kWarpsPerBlock)) * sizeof(float); \
    GraphAttentionForward_CSR_MH_v2_D<D_CONST><<<blocks, threads, shmem, stream>>>( \
        N, H, \
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
        q_strides[0], q_strides[1], \
        k_strides[0], k_strides[1], \
        v_strides[0], v_strides[1], \
        row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), \
        O.data_ptr<float>(), \
        o_strides[0], o_strides[1], \
        lse.data_ptr<float>(), \
        scale \
    ); \
} while(0)

constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

template<int D_CONST>
__global__ void GraphAttentionForward_CSR_MH_v2_D(
    const int N,
    const int H,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const int64_t stride_q_n, const int64_t stride_q_h,
    const int64_t stride_k_n, const int64_t stride_k_h,
    const int64_t stride_v_n, const int64_t stride_v_h,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ O,
    const int64_t stride_o_n, const int64_t stride_o_h,
    float* __restrict__ logsumexp,
    const float scale
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be multiple of 32 for this fast path");

    constexpr int TILES = (D_CONST + kWarpSize - 1) / kWarpSize; // 1..8

    const int node_i  = blockIdx.x;
    const int head_h  = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;

    if (node_i >= N || head_h >= H) {
        return;
    }

    const int edge_start    = row_ptr[node_i];
    const int edge_end      = row_ptr[node_i + 1];
    const int num_neighbors = edge_end - edge_start;

    // shared memory layout:
    // k_shared[D_CONST]
    // warp_out[kWarpsPerBlock][D_CONST]
    // warp_max[kWarpsPerBlock]
    // warp_sum[kWarpsPerBlock] (also reused to store exp scale_w during combine)
    extern __shared__ float shared[];
    float* k_shared = shared;
    float* warp_out = shared + D_CONST;
    float* warp_max = shared + D_CONST + kWarpsPerBlock * D_CONST;
    float* warp_sum = warp_max + kWarpsPerBlock;

    float* my_out = warp_out + warp_id * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            float* out_base = O + node_i * stride_o_n + head_h * stride_o_h;
            for (int d = lane_id; d < D_CONST; d += kWarpSize) {
                out_base[d] = 0.0f;
            }
            if (lane_id == 0) {
                logsumexp[node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    // cooperative load of K_i into shared memory
    {
        const float* k_base = K + node_i * stride_k_n + head_h * stride_k_h;
        for (int d = threadIdx.x; d < D_CONST; d += kWarpsPerBlock * kWarpSize) {
            k_shared[d] = __ldg(&k_base[d]);
        }
    }
    __syncthreads();

    // per-warp online softmax state
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // accumulator sized to what D needs
    float o_acc_s[TILES];
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
        o_acc_s[t] = 0.0f;
    }

    // neighbor loop
    for (int e = warp_id; e < num_neighbors; e += kWarpsPerBlock) {
        const int j = __ldg(&col_idx[edge_start + e]);

        const float* q_base = Q + j * stride_q_n + head_h * stride_q_h;
        const float* v_base = V + j * stride_v_n + head_h * stride_v_h;

        float s_partial = 0.0f;
        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int d = lane_id + t * kWarpSize;
            s_partial = fmaf(k_shared[d], __ldg(&q_base[d]), s_partial);
        }

        const float score = warp_reduce_sum(s_partial) * scale;

        const float old_max = max_val;
        max_val = fmaxf(max_val, score);
        const float correction = __expf(old_max - max_val);

        const float exp_term = __expf(score - max_val);
        sum_exp = fmaf(sum_exp, correction, exp_term);
        const float w = exp_term;

        #pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int d = lane_id + t * kWarpSize;
            o_acc_s[t] *= correction;
            o_acc_s[t] = fmaf(w, __ldg(&v_base[d]), o_acc_s[t]);
        }
    }

    // write per-warp results (only TILES worth)
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int d = lane_id + t * kWarpSize;
        my_out[d] = o_acc_s[t];
    }

    if (lane_id == 0) {
        warp_max[warp_id] = max_val;
        warp_sum[warp_id] = sum_exp;
    }
    __syncthreads();

    // cross-warp reduction for warp 0
    if (warp_id == 0) {

        float global_max = -FLT_MAX;
        float global_sum = 0.0f;
        float inv_sum    = 0.0f;

        if (lane_id == 0) {
            #pragma unroll
            for (int w = 0; w < kWarpsPerBlock; ++w) {
                global_max = fmaxf(global_max, warp_max[w]);
            }
            #pragma unroll
            for (int w = 0; w < kWarpsPerBlock; ++w) {
                global_sum = fmaf(warp_sum[w], __expf(warp_max[w] - global_max), global_sum);
            }
            #pragma unroll
            for (int w = 0; w < kWarpsPerBlock; ++w) {
                warp_sum[w] = __expf(warp_max[w] - global_max); // scale_w
            }

            inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
            logsumexp[node_i * H + head_h] = (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY;
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

        float* out_base = O + node_i * stride_o_n + head_h * stride_o_h;
        for (int d = lane_id; d < D_CONST; d += kWarpSize) {
            float combined = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarpsPerBlock; ++w) {
                combined = fmaf(warp_sum[w], warp_out[w * D_CONST + d], combined);
            }
            out_base[d] = combined * inv_sum;
        }
    }
}

__global__ void GraphAttentionForward_CSR_MH_v2(
    const int N,
    const int H,
    const int D,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const int64_t stride_q_n, const int64_t stride_q_h,
    const int64_t stride_k_n, const int64_t stride_k_h,
    const int64_t stride_v_n, const int64_t stride_v_h,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ O,
    const int64_t stride_o_n, const int64_t stride_o_h,
    float* __restrict__ logsumexp,
    const float scale
) {
    const int node_i  = (int)blockIdx.x;
    const int head_h  = (int)blockIdx.y;
    const int warp_id = (int)(threadIdx.x / kWarpSize);
    const int lane_id = (int)(threadIdx.x % kWarpSize);

    if (node_i >= N || head_h >= H) return;

    const int edge_start    = row_ptr[node_i];
    const int edge_end      = row_ptr[node_i + 1];
    const int num_neighbors = edge_end - edge_start;

    // Shared memory layout:
    // k_shared[D]
    // warp_out[kWarpsPerBlock][D]
    // warp_max[kWarpsPerBlock]
    // warp_sum[kWarpsPerBlock]   (also reused to store exp scale_w during combine)
    extern __shared__ float shared[];
    float* k_shared = shared;
    float* warp_out = shared + D;
    float* warp_max = shared + D + kWarpsPerBlock * D;
    float* warp_sum = warp_max + kWarpsPerBlock;

    float* my_out = warp_out + warp_id * D;

    // Handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            float* out_base = O + (int64_t)node_i * stride_o_n + (int64_t)head_h * stride_o_h;
            for (int d = lane_id; d < D; d += kWarpSize) {
                out_base[d] = 0.0f;
            }
            if (lane_id == 0) {
                logsumexp[(int64_t)node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    // Cooperative load of K_i into shared memory
    {
        const float* k_base = K + (int64_t)node_i * stride_k_n + (int64_t)head_h * stride_k_h;
        const int total_threads = kWarpsPerBlock * kWarpSize;
        const int tid = (int)threadIdx.x;
        for (int d = tid; d < D; d += total_threads) {
            k_shared[d] = __ldg(&k_base[d]);
        }
    }
    __syncthreads();


    // Per-warp online softmax state (registers)
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // Per-warp output accumulator in registers (warp-striped scalars)
    // Lane accumulates dims: lane_id + 0*32, lane_id + 1*32, ... up to < D
    float o_acc_s[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) o_acc_s[t] = 0.0f;

    // Each warp processes neighbors: warp_id, warp_id + kWarpsPerBlock, ...
    for (int e = warp_id; e < num_neighbors; e += kWarpsPerBlock) {
        const int j = __ldg(&col_idx[edge_start + e]);

        const float* q_base = Q + (int64_t)j * stride_q_n + (int64_t)head_h * stride_q_h;
        const float* v_base = V + (int64_t)j * stride_v_n + (int64_t)head_h * stride_v_h;

        // Dot(K_i, Q_j) with warp-striped, coalesced scalar LDG loads
        float s_partial = 0.0f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = lane_id + t * kWarpSize;
            if (d < D) {
                const float kk = k_shared[d];        // shared
                const float qq = __ldg(&q_base[d]);  // coalesced across lanes
                s_partial += kk * qq;
            }
        }

        const float score = warp_reduce_sum(s_partial) * scale;

        // Online softmax update
        const float old_max = max_val;
        max_val = fmaxf(max_val, score);
        const float correction = __expf(old_max - max_val);

        // Update sum_exp
        sum_exp = sum_exp * correction + __expf(score - max_val);

        // Unnormalized weight for this neighbor
        const float w = __expf(score - max_val);

        // Fused rescale + add weighted V_j into warp-striped scalar accum
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = lane_id + t * kWarpSize;
            if (d < D) {
                o_acc_s[t] = o_acc_s[t] * correction + w * __ldg(&v_base[d]);
            }
        }
    }

    // Write per-warp results to shared memory (scalars)
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int d = lane_id + t * kWarpSize;
        if (d < D) my_out[d] = o_acc_s[t];
    }
    if (lane_id == 0) {
        warp_max[warp_id] = max_val;
        warp_sum[warp_id] = sum_exp;
    }
    __syncthreads();

    // Cross-warp reduction (warp 0 only)
    if (warp_id == 0) {
        // Compute global max
        float global_max = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            global_max = fmaxf(global_max, warp_max[w]);
        }

        // Compute global sum with corrections
        float global_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            global_sum += warp_sum[w] * __expf(warp_max[w] - global_max);
        }

        // Precompute scale_w once and reuse (store into warp_sum as scratch)
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            warp_sum[w] = __expf(warp_max[w] - global_max);
        }

        const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

        // Combine and normalize outputs (scalars, warp-striped => coalesced stores)
        float* out_base = O + (int64_t)node_i * stride_o_n + (int64_t)head_h * stride_o_h;

        for (int d = lane_id; d < D; d += kWarpSize) {
            float combined = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarpsPerBlock; ++w) {
                combined += warp_sum[w] * warp_out[w * D + d];
            }
            out_base[d] = combined * inv_sum;
        }

        if (lane_id == 0) {
            logsumexp[(int64_t)node_i * H + head_h] =
                (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY;
        }
    }
}

// ===================================================
// ================== BACKWARD =======================
// ===================================================

// D[i,h] = sum_d dO[i,h,d] * O[i,h,d]
template<int D_CONST>
__global__ void compute_D_mh_kernel_D(
    const float* __restrict__ dO,   // [N, H, D]
    const float* __restrict__ O,    // [N, H, D]
    float* __restrict__ D_out,      // [N, H]
    int64_t N,
    int64_t H,
    int64_t stride_do_n,
    int64_t stride_do_h,
    int64_t stride_o_n,
    int64_t stride_o_h
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");
    const int node_i = blockIdx.x;
    const int head_h = blockIdx.y;
    const int lane   = threadIdx.x;   // 0..31

    if (node_i >= (int)N || head_h >= (int)H) {
        return;
    }

    const float* dO_base = dO + node_i * stride_do_n + head_h * stride_do_h;
    const float* O_base  = O  + node_i * stride_o_n  + head_h * stride_o_h;

    constexpr int NF4 = D_CONST / 4;
    const float4* dO4 = reinterpret_cast<const float4*>(dO_base);
    const float4* O4  = reinterpret_cast<const float4*>(O_base);

    float sum = 0.0f;
    // lanes cover idx4 = lane, lane+32,... (for D<=256 -> at most 2 iters)
    #pragma unroll
    for (int idx4 = lane; idx4 < NF4; idx4 += 32) {
        float4 a = dO4[idx4];
        float4 b = O4[idx4];
        sum = fmaf(a.x, b.x, sum);
        sum = fmaf(a.y, b.y, sum);
        sum = fmaf(a.z, b.z, sum);
        sum = fmaf(a.w, b.w, sum);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        D_out[node_i * H + head_h] = sum;
    }
}


// Q, K, V, dO, dQ, dK, dV are [N, H, D] with contiguous D, D % 4 == 0
// logsumexp and Delta are [N, H] (row-wise log-sum-exp and Delta = <O,dO>).
template<int D_CONST>
__global__ void graph_attn_backward_csrT_kernel_D(
    int64_t N,
    int64_t H,
    const int* __restrict__ row_ptr_T,   // [N+1], CSR^T row pointers: rows = source nodes j
    const int* __restrict__ col_idx_T,   // [E],   CSR^T col indices: dest nodes i
    const float* __restrict__ Q,         // [N, H, D]
    const float* __restrict__ K,         // [N, H, D]
    const float* __restrict__ V,         // [N, H, D]
    const float* __restrict__ dO,        // [N, H, D]
    const float* __restrict__ logsumexp, // [N, H]
    const float* __restrict__ Delta,     // [N, H]
    float scale,
    float* __restrict__ dQ,              // [N, H, D]
    float* __restrict__ dK,              // [N, H, D]
    float* __restrict__ dV               // [N, H, D]
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");

    int node_j = blockIdx.x; // source node index j
    int head_h = blockIdx.y; // head index h
    int lane   = threadIdx.x; // 0..31

    if (node_j >= N || head_h >= H) {
        return;
    }

    int edge_start    = row_ptr_T[node_j];
    int edge_end      = row_ptr_T[node_j + 1];
    int num_incoming  = edge_end - edge_start;

    // nothing to do if this node has no incoming edges
    if (num_incoming == 0) {
        return;
    }

    constexpr int NF4 = D_CONST / 4;

    extern __shared__ float shared[];
    float* qj_shared      = shared;                 // D
    float* vj_shared      = qj_shared + D_CONST;    // D
    float* gq_shared      = vj_shared + D_CONST;    // D
    float* gv_shared      = gq_shared + D_CONST;    // D

    float4* qj_sh4 = reinterpret_cast<float4*>(qj_shared);
    float4* vj_sh4 = reinterpret_cast<float4*>(vj_shared);
    float4* gq_sh4 = reinterpret_cast<float4*>(gq_shared);
    float4* gv_sh4 = reinterpret_cast<float4*>(gv_shared);

    const size_t base_jh = (node_j * H + head_h) * D_CONST;

    const float4* qj4 = reinterpret_cast<const float4*>(Q + base_jh);
    const float4* vj4 = reinterpret_cast<const float4*>(V + base_jh);

    // load q_j, v_j into shared and zero grad buffers (vector)
    #pragma unroll
    for (int idx4 = lane; idx4 < NF4; idx4 += 32) {
        qj_sh4[idx4] = qj4[idx4];
        vj_sh4[idx4] = vj4[idx4];
        gq_sh4[idx4] = make_float4(0.f, 0.f, 0.f, 0.f);
        gv_sh4[idx4] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // single-warp block, but keep a warp sync to be conservative
    __syncwarp(FULL_WARP_MASK);

    // Iterate incoming edges (i -> j)
    for (int e = 0; e < num_incoming; ++e) {
        // node_i is warp-uniform: load once and broadcast
        int node_i = 0;
        if (lane == 0) {
            node_i = __ldg(&col_idx_T[edge_start + e]);
        }
        node_i = __shfl_sync(FULL_WARP_MASK, node_i, 0);

        // If your CSR^T is guaranteed valid, you can delete this check entirely.
        if ((unsigned)node_i >= (unsigned)N) continue;

        const size_t base_ih = ((size_t)node_i * (size_t)H + (size_t)head_h) * (size_t)D_CONST;

        const float4* ki4  = reinterpret_cast<const float4*>(K  + base_ih);
        const float4* dOi4 = reinterpret_cast<const float4*>(dO + base_ih);

        // 1) dot(k_i, q_j) and dP_ij = <dO_i, v_j>
        float dot_kq = 0.0f;
        float dP_ij  = 0.0f;

        #pragma unroll
        for (int idx4 = lane; idx4 < NF4; idx4 += 32) {
            const float4 ki = ki4[idx4];
            const float4 qj = qj_sh4[idx4];
            const float4 vj = vj_sh4[idx4];
            const float4 dOiv = dOi4[idx4];

            dot_kq = fmaf(ki.x, qj.x, dot_kq);
            dot_kq = fmaf(ki.y, qj.y, dot_kq);
            dot_kq = fmaf(ki.z, qj.z, dot_kq);
            dot_kq = fmaf(ki.w, qj.w, dot_kq);

            dP_ij  = fmaf(dOiv.x, vj.x, dP_ij);
            dP_ij  = fmaf(dOiv.y, vj.y, dP_ij);
            dP_ij  = fmaf(dOiv.z, vj.z, dP_ij);
            dP_ij  = fmaf(dOiv.w, vj.w, dP_ij);
        }

        // all lanes now see the same scalars
        dot_kq = warp_reduce_sum(dot_kq);
        dP_ij  = warp_reduce_sum(dP_ij);

        const float score = dot_kq * scale;

        // L_i and Delta_i are warp-uniform: lane0 load + broadcast
        float L_i = 0.0f, Delta_i = 0.0f;
        if (lane == 0) {
            const size_t idx_ih = (size_t)node_i * (size_t)H + (size_t)head_h;
            L_i     = __ldg(&logsumexp[idx_ih]);
            Delta_i = __ldg(&Delta[idx_ih]);
        }
        L_i     = __shfl_sync(FULL_WARP_MASK, L_i, 0);
        Delta_i = __shfl_sync(FULL_WARP_MASK, Delta_i, 0);

        const float alpha     = __expf(score - L_i);
        const float dS        = alpha * (dP_ij - Delta_i);
        const float dS_scaled = dS * scale;

        // 2) accumulate dV_j, dQ_j in shared, atomic-add dK_i
        float* dK_i_base = dK + base_ih;

        #pragma unroll
        for (int idx4 = lane; idx4 < NF4; idx4 += kWarpSize) {
            const float4 ki   = ki4[idx4];
            const float4 dOiv = dOi4[idx4];
            const float4 qj   = qj_sh4[idx4];

            // dV_j += alpha * dO_i
            float4 gv = gv_sh4[idx4];
            gv.x = fmaf(alpha, dOiv.x, gv.x);
            gv.y = fmaf(alpha, dOiv.y, gv.y);
            gv.z = fmaf(alpha, dOiv.z, gv.z);
            gv.w = fmaf(alpha, dOiv.w, gv.w);
            gv_sh4[idx4] = gv;

            // dQ_j += dS_scaled * k_i
            float4 gq = gq_sh4[idx4];
            gq.x = fmaf(dS_scaled, ki.x, gq.x);
            gq.y = fmaf(dS_scaled, ki.y, gq.y);
            gq.z = fmaf(dS_scaled, ki.z, gq.z);
            gq.w = fmaf(dS_scaled, ki.w, gq.w);
            gq_sh4[idx4] = gq;

            // dK_i += dS_scaled * q_j (atomics)
            const int base_feat = idx4 * 4;
            atomicAdd(&dK_i_base[base_feat + 0], dS_scaled * qj.x);
            atomicAdd(&dK_i_base[base_feat + 1], dS_scaled * qj.y);
            atomicAdd(&dK_i_base[base_feat + 2], dS_scaled * qj.z);
            atomicAdd(&dK_i_base[base_feat + 3], dS_scaled * qj.w);
        }
    }

    // 3) write dQ_j and dV_j back to global
    float4* dQ4 = reinterpret_cast<float4*>(dQ + base_jh);
    float4* dV4 = reinterpret_cast<float4*>(dV + base_jh);

    #pragma unroll
    for (int idx4 = lane; idx4 < NF4; idx4 += kWarpSize) {
        dQ4[idx4] = gq_sh4[idx4];
        dV4[idx4] = gv_sh4[idx4];
    }
}


std::tuple<torch::Tensor, torch::Tensor>
graph_attention_forward_csr_mh_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {

    at::cuda::CUDAGuard device_guard(Q.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(Q.device().index());

    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "CSR indices must be CUDA");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be [N, H, D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V sizes must match");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q, K, V must be float32");
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32 && col_idx.dtype() == torch::kInt32,
                "row_ptr, col_idx must be int32");

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

    auto options = Q.options();
    torch::Tensor O = torch::empty({N, H, D}, options);
    torch::Tensor lse = torch::empty({N, H}, options);

    auto o_strides = O.strides();

    dim3 blocks(N, H);
    dim3 threads(kWarpsPerBlock * kWarpSize);  // 128 threads

    switch (D) {

        case 32:  DISPATCH_D(32);   break;
        case 64:  DISPATCH_D(64);   break;
        case 128: DISPATCH_D(128);  break;
        case 256: DISPATCH_D(256);  break;

        default: {
            // fallback to runtime-D kernel
            size_t shmem = D * sizeof(float) * (1 + kWarpsPerBlock) + (2 * kWarpsPerBlock) * sizeof(float);

            GraphAttentionForward_CSR_MH_v2<<<blocks, threads, shmem, stream>>>(
                N, H, D,
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                q_strides[0], q_strides[1],
                k_strides[0], k_strides[1],
                v_strides[0], v_strides[1],
                row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(),
                O.data_ptr<float>(),
                o_strides[0], o_strides[1],
                lse.data_ptr<float>(),
                scale
            );
            break;
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "`graph_attention_forward_csr_mh_cuda` kernel failed: ", cudaGetErrorString(err));

    return std::make_tuple(O, lse);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
graph_attention_backward_csr_mh_cuda(
    torch::Tensor row_ptr_T,   // [N+1], int32, CSR^T
    torch::Tensor col_idx_T,   // [E],   int32, CSR^T
    torch::Tensor Q,           // [N, H, D], float32
    torch::Tensor K,           // [N, H, D], float32
    torch::Tensor V,           // [N, H, D], float32
    torch::Tensor O,           // [N, H, D], float32 (forward output)
    torch::Tensor dO,          // [N, H, D], float32
    torch::Tensor logsumexp,   // [N, H],   float32
    float scale
) {
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

    TORCH_CHECK(Q.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32 &&
                O.dtype() == torch::kFloat32 &&
                dO.dtype() == torch::kFloat32,
                "Q, K, V, O, dO must be float32");

    TORCH_CHECK(row_ptr_T.dtype() == torch::kInt32 &&
                col_idx_T.dtype() == torch::kInt32,
                "row_ptr_T, col_idx_T must be int32");

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

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

    // ---------------------------
    // Allocate outputs
    // ---------------------------
    torch::Tensor dQ = torch::zeros_like(Q);
    torch::Tensor dK = torch::zeros_like(K);
    torch::Tensor dV = torch::zeros_like(V);

    // Delta[i,h] = <O[i,h,:], dO[i,h,:]>
    torch::Tensor Delta = torch::empty({N, H}, options);
    auto do_strides = dO.strides();

    const int64_t stride_do_n = do_strides[0];
    const int64_t stride_do_h = do_strides[1];
    const int64_t stride_o_n  = o_strides[0];
    const int64_t stride_o_h  = o_strides[1];

    TORCH_CHECK(do_strides[2] == 1 && o_strides[2] == 1,
                "dO and O feature dim (D) must be contiguous (stride(2) == 1)");

    dim3 blocks_D(N, H);
    dim3 threads_D(kWarpSize);

    dim3 blocks_bwd(N, H);
    dim3 threads_bwd(kWarpSize);

    auto shmem_bwd_bytes = [&](int Dval) -> size_t {
        return (size_t)4 * (size_t)Dval * sizeof(float);
    };

    switch ((int)D) {
        case 32:
            compute_D_mh_kernel_D<32><<<blocks_D, threads_D, 0, at::cuda::getDefaultCUDAStream()>>>(
                dO.data_ptr<float>(), O.data_ptr<float>(), Delta.data_ptr<float>(),
                N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
            );

            graph_attn_backward_csrT_kernel_D<32><<<blocks_bwd, threads_bwd, shmem_bwd_bytes(32), at::cuda::getDefaultCUDAStream()>>>(
                N, H,
                row_ptr_T.data_ptr<int>(), col_idx_T.data_ptr<int>(),
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                dO.data_ptr<float>(),
                logsumexp.data_ptr<float>(),
                Delta.data_ptr<float>(),
                scale,
                dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>()
            );
            break;

        case 64:
            compute_D_mh_kernel_D<64><<<blocks_D, threads_D, 0, at::cuda::getDefaultCUDAStream()>>>(
                dO.data_ptr<float>(), O.data_ptr<float>(), Delta.data_ptr<float>(),
                N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
            );

            graph_attn_backward_csrT_kernel_D<64><<<blocks_bwd, threads_bwd, shmem_bwd_bytes(64), at::cuda::getDefaultCUDAStream()>>>(
                N, H,
                row_ptr_T.data_ptr<int>(), col_idx_T.data_ptr<int>(),
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                dO.data_ptr<float>(),
                logsumexp.data_ptr<float>(),
                Delta.data_ptr<float>(),
                scale,
                dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>()
            );
            break;

        case 128:
            compute_D_mh_kernel_D<128><<<blocks_D, threads_D, 0, at::cuda::getDefaultCUDAStream()>>>(
                dO.data_ptr<float>(), O.data_ptr<float>(), Delta.data_ptr<float>(),
                N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
            );

            graph_attn_backward_csrT_kernel_D<128><<<blocks_bwd, threads_bwd, shmem_bwd_bytes(128), at::cuda::getDefaultCUDAStream()>>>(
                N, H,
                row_ptr_T.data_ptr<int>(), col_idx_T.data_ptr<int>(),
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                dO.data_ptr<float>(),
                logsumexp.data_ptr<float>(),
                Delta.data_ptr<float>(),
                scale,
                dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>()
            );
            break;

        case 256:
            compute_D_mh_kernel_D<256><<<blocks_D, threads_D, 0, at::cuda::getDefaultCUDAStream()>>>(
                dO.data_ptr<float>(), O.data_ptr<float>(), Delta.data_ptr<float>(),
                N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
            );

            graph_attn_backward_csrT_kernel_D<256><<<blocks_bwd, threads_bwd, shmem_bwd_bytes(256), at::cuda::getDefaultCUDAStream()>>>(
                N, H,
                row_ptr_T.data_ptr<int>(), col_idx_T.data_ptr<int>(),
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                dO.data_ptr<float>(),
                logsumexp.data_ptr<float>(),
                Delta.data_ptr<float>(),
                scale,
                dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>()
            );
            break;

        default:
            TORCH_CHECK(false, "Unsupported D: ", D, " (supported: 32, 64, 128, 256)");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "`graph_attention_backward_csr_mh_cuda` backward kernels failed: ", cudaGetErrorString(err));
    return std::make_tuple(dQ, dK, dV);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gt_forward_csr_mh",
        &graph_attention_forward_csr_mh_cuda,
        "Graph Transformer forward (CSR, multi-head, FlashAttn2-style) - returns (O, logsumexp)",
        py::arg("row_ptr"),
        py::arg("col_idx"),
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("scale")
    );

    m.def(
        "gt_backward_csr_mh",
        &graph_attention_backward_csr_mh_cuda,
        "Graph Transformer backward (CSR^T, multi-head, FlashAttn2-style) - returns (dQ, dK, dV)",
        py::arg("row_ptr_T"),
        py::arg("col_idx_T"),
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("O"),
        py::arg("dO"),
        py::arg("logsumexp"),
        py::arg("scale")
    );
}
