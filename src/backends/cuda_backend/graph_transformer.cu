#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>

#define FULL_WARP_MASK 0xffffffff


constexpr int THREADS_MID = 128;

constexpr int WARPS_PER_BLOCK_MID    = 16;
constexpr int WARPS_PER_BLOCK_HUGE   = 32;

constexpr int THREADS_PER_BLOCK_MID  = WARPS_PER_BLOCK_MID  * 32;
constexpr int THREADS_PER_BLOCK_HUGE = WARPS_PER_BLOCK_HUGE * 32;

constexpr int TILE_D_HUGE            = 16; // threshold to classify "huge"


__device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
}

__device__ __forceinline__ float dot_vec4(
    const float* __restrict__ k_ptr,
    const float* __restrict__ q_ptr,
    int d
) {
    int d4 = d / 4;
    const float4* __restrict__ k4 = reinterpret_cast<const float4*>(k_ptr);
    const float4* __restrict__ q4 = reinterpret_cast<const float4*>(q_ptr);

    float accum = 0.0f;
    #pragma unroll
    for (int i = 0; i < d4; ++i) {
        float4 kk = k4[i];
        float4 qq = q4[i];
        accum += (kk.x * qq.x) + (kk.y * qq.y) + (kk.z * qq.z) + (kk.w * qq.w);
    }

    // tail if d % 4 != 0
    #pragma unroll
    for (int f = d4 * 4; f < d; ++f) {
        accum += k_ptr[f] * q_ptr[f];
    }
    return accum;
}

// ============================================================================
// Block-level reduction primitive (sum) - optimized for power-of-2 sizes
// ============================================================================
template<int THREADS_PER_BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / 32;
    __shared__ float warp_sums[NUM_WARPS];

    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(FULL_WARP_MASK, val, offset);
    }

    // First lane of each warp writes to shared
    if (lane == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    // First warp reduces across warp sums
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(FULL_WARP_MASK, val, offset);
        }

        // Broadcast result via shared memory
        if (lane == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    return warp_sums[0];
}

// ============================================================================
// feature-parallel kernel for mid-degree nodes
// key difference from huge-node kernel: optimized for moderate degree (10-1000)
// 1 block = 1 node, all threads = parallelize features
// ============================================================================
template<int THREADS_PER_BLOCK>
__global__ void gt_kernel_forward_mid_nodes_feature_parallel(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const int* __restrict__ mid_nodes,
    int num_mid_degree_nodes,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int d,
    float scale
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_mid_degree_nodes) return;

    const int dst = mid_nodes[block_idx];
    const int row_start = edge_ptr[dst];
    const int row_end = edge_ptr[dst + 1];
    const int degree = row_end - row_start;

    if (degree == 0) return;

    // shared memory layout:
    // [0..d-1]:           K_dst (destination node's key vector)
    // [d..2*d-1]:         O_acc (output accumulator)
    // [2*d..2*d+NUM_WARPS-1]: warp reduction workspace
    extern __shared__ float smem[];
    float* K_dst = smem;
    float* O_acc = smem + d;

    const int tid = threadIdx.x;

    // ========================================================================
    // 1: load K[dst] and initialize O accumulator
    // ========================================================================
    #pragma unroll 4
    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        K_dst[f] = K[dst * d + f];
        O_acc[f] = 0.0f;
    }
    __syncthreads();

    // ========================================================================
    // 2: streaming softmax over neighbors
    // ========================================================================
    float m_i = -INFINITY;  // running max
    float l_i = 0.0f;       // running denominator

    // sequential loop over neighbors (all threads move together)
    for (int eid = row_start; eid < row_end; eid++) {
        const int nbr = edge_idx[eid];
        const float* Q_nbr = Q + nbr * d;
        const float* V_nbr = V + nbr * d;

        // --------------------------------------------------------------------
        // 2a: compute attention score via dot product
        // each thread computes partial dot over its features
        // --------------------------------------------------------------------
        float partial_dot = 0.0f;

        // vectorized loads: use float4 when d is multiple of 4 and aligned
        const bool use_vec4 = (d % 4 == 0) &&
                              (((uintptr_t)Q_nbr & 15) == 0) &&
                              (((uintptr_t)K_dst & 15) == 0);

        if (use_vec4) {
            const float4* Q_nbr_vec4 = reinterpret_cast<const float4*>(Q_nbr);
            const float4* K_dst_vec4 = reinterpret_cast<const float4*>(K_dst);
            const int d4 = d / 4;

            #pragma unroll 4
            for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                float4 q4 = Q_nbr_vec4[f4];
                float4 k4 = K_dst_vec4[f4];
                partial_dot += (q4.x * k4.x) + (q4.y * k4.y) +
                               (q4.z * k4.z) + (q4.w * k4.w);
            }
        } else {
            // scalar fallback
            #pragma unroll 4
            for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                partial_dot += K_dst[f] * Q_nbr[f];
            }
        }

        // block-wide reduction to get full dot product ---> CAN BE SLOW
        float S_ij = block_reduce_sum<THREADS_PER_BLOCK>(partial_dot) * scale;

        // --------------------------------------------------------------------
        // 2b: online softmax update (FlashAttention-2 style)
        // --------------------------------------------------------------------
        float m_i_new = fmaxf(m_i, S_ij);
        float alpha = expf(m_i - m_i_new);      // rescale old accumulator
        float exp_weight = expf(S_ij - m_i_new); // current neighbor weight

        float l_i_new = alpha * l_i + exp_weight;

        // --------------------------------------------------------------------
        // 2c: update output accumulator O
        // O_new = alpha * O_old + exp_weight * V_nbr
        // --------------------------------------------------------------------
        if (use_vec4 && (((uintptr_t)V_nbr & 15) == 0)) {
            const float4* V_nbr_vec4 = reinterpret_cast<const float4*>(V_nbr);
            float4* O_acc_vec4 = reinterpret_cast<float4*>(O_acc);
            const int d4 = d / 4;

            #pragma unroll 4
            for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                float4 v4 = V_nbr_vec4[f4];
                float4 o4 = O_acc_vec4[f4];

                o4.x = alpha * o4.x + exp_weight * v4.x;
                o4.y = alpha * o4.y + exp_weight * v4.y;
                o4.z = alpha * o4.z + exp_weight * v4.z;
                o4.w = alpha * o4.w + exp_weight * v4.w;

                O_acc_vec4[f4] = o4;
            }
        } else {
            // scalar update
            #pragma unroll 4
            for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                O_acc[f] = alpha * O_acc[f] + exp_weight * V_nbr[f];
            }
        }

        // update running statistics
        m_i = m_i_new;
        l_i = l_i_new;


    }

    // ========================================================================
    // 3: final normalization and write to global memory
    // ========================================================================
    float inv_l_i = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

    // coalesced writes with vectorization
    if ((d % 4 == 0) && (((uintptr_t)out & 15) == 0)) {
        float4* O_acc_vec4 = reinterpret_cast<float4*>(O_acc);
        float4* out_vec4 = reinterpret_cast<float4*>(out + dst * d);
        const int d4 = d / 4;

        for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
            float4 o4 = O_acc_vec4[f4];
            o4.x *= inv_l_i;
            o4.y *= inv_l_i;
            o4.z *= inv_l_i;
            o4.w *= inv_l_i;
            out_vec4[f4] = o4;
        }
    } else {
        for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
            out[dst * d + f] = O_acc[f] * inv_l_i;
        }
    }
}

// single-pass streaming softmax for huge nodes
template<int WARPS_PER_BLOCK, int TILE_D>
__global__ void gt_kernel_forward_huge_nodes(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const int* __restrict__ huge_nodes,
    int num_huge,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int d,
    float scale
) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int block_node = blockIdx.x;
    if (block_node >= num_huge) {
        return;
    }

    const int dst = huge_nodes[block_node];
    const int row_start = edge_ptr[dst];
    const int row_end   = edge_ptr[dst + 1];

    // shared memory layout:
    // [0..d-1]: k_dst (cached K[dst])
    // [d..d+WARPS*TILE_D-1]: warp_partial (per-warp feature accumulators)
    // [d+WARPS*TILE_D..d+WARPS*TILE_D+WARPS-1]: warp_max (per-warp running max)
    // [d+WARPS*TILE_D+WARPS..]: warp_denom (per-warp running denominator)
    extern __shared__ float shmem[];
    float* k_dst        = shmem;
    float* warp_partial = k_dst + d;
    float* warp_max     = warp_partial + WARPS_PER_BLOCK * TILE_D;
    float* warp_denom   = warp_max + WARPS_PER_BLOCK;

    // load K[dst] into shared (all threads cooperate)
    for (int f = threadIdx.x; f < d; f += blockDim.x) {
        k_dst[f] = K[dst * d + f];
    }
    __syncthreads();

    // each warp handles a slice of features [feat_base, feat_base+TILE_D)
    const int feat_base  = warp_id * TILE_D;
    const int feat_limit = (feat_base + TILE_D <= d) ? TILE_D : (d - feat_base);

    if (feat_base >= d) {
        // this warp has no features to process
        return;
    }

    // per-lane register accumulators for this warp's feature slice
    float acc_feat[TILE_D];
    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        acc_feat[j] = 0.f;
    }

    // per-lane running statistics for streaming softmax
    float running_max = -INFINITY;
    float running_sum = 0.f;

    // ============================================================
    // single-pass streaming softmax over neighbors
    // process neighbors in tiles (warp-strided iteration)
    // ============================================================

    constexpr int TILE_SIZE = 32;
    for (int tile_start = row_start + warp_id * TILE_SIZE;
         tile_start < row_end;
         tile_start += WARPS_PER_BLOCK * TILE_SIZE) {

        // each lane handles one neighbor in this tile
        const int eid = tile_start + lane;
        const int nbr = (eid < row_end) ? edge_idx[eid] : -1;

        // compute attention score for this lane's neighbor
        float score = -INFINITY;
        if (nbr >= 0) {
            float dot = dot_vec4(k_dst, Q + nbr * d, d);
            score = dot * scale;
        }

        // warp-level max of this tile
        float tile_max = warp_reduce_max(score);

        // update running statistics (streaming softmax trick)
        // see: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018)
        float old_max = running_max;
        float new_max = fmaxf(running_max, tile_max);
        float rescale_factor = __expf(old_max - new_max);

        // rescale previous accumulator and denominator
        #pragma unroll
        for (int j = 0; j < TILE_D; ++j) {
            acc_feat[j] *= rescale_factor;
        }
        running_sum *= rescale_factor;

        // compute this neighbor's softmax weight (unnormalized)
        float w = (nbr >= 0) ? __expf(score - new_max) : 0.f;
        running_sum += w;
        running_max = new_max;

        // accumulate weighted values into per-lane registers
        // vectorized load for V when possible
        if (nbr >= 0 && feat_limit > 0) {
            const float* v_ptr = V + nbr * d + feat_base;

            // vectorized path: load 4 floats at once
            int j = 0;
            if (feat_limit >= 4 && (((size_t)v_ptr) & 15) == 0) {  // Check alignment
                const float4* v4_ptr = reinterpret_cast<const float4*>(v_ptr);
                for (; j + 4 <= feat_limit; j += 4) {
                    float4 v4 = v4_ptr[j / 4];
                    acc_feat[j + 0] += w * v4.x;
                    acc_feat[j + 1] += w * v4.y;
                    acc_feat[j + 2] += w * v4.z;
                    acc_feat[j + 3] += w * v4.w;
                }
            }

            // scalar tail (or full scalar path if not aligned)
            for (; j < feat_limit; ++j) {
                acc_feat[j] += w * v_ptr[j];
            }
        }
    }

    // ============================================================
    // reduce accumulators across lanes within each warp
    // ============================================================

    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        float vj = (j < feat_limit) ? acc_feat[j] : 0.f;
        vj = warp_reduce_sum(vj);
        if (lane == 0) {
            warp_partial[warp_id * TILE_D + j] = vj;
        }
    }

    // store per-warp statistics
    float warp_sum = warp_reduce_sum(running_sum);
    float warp_max_val = warp_reduce_max(running_max);

    if (lane == 0) {
        warp_max[warp_id] = warp_max_val;
        warp_denom[warp_id] = warp_sum;
    }
    __syncthreads();

    // ============================================================
    // final reduction and normalization (warp 0 only)
    // all 32 lanes in warp 0 cooperate for coalesced writes
    // ============================================================

    if (warp_id == 0) {
        // find global max across all warps
        float global_max = (lane < WARPS_PER_BLOCK) ? warp_max[lane] : -INFINITY;
        global_max = warp_reduce_max(global_max);

        // accumulate denominator with max rescaling
        float global_denom = 0.f;
        if (lane < WARPS_PER_BLOCK) {
            float rescale = __expf(warp_max[lane] - global_max);
            global_denom = warp_denom[lane] * rescale;
        }
        global_denom = warp_reduce_sum(global_denom);

        // write output (all lanes participate for coalescing)
        if (global_denom > 0.f) {
            float inv_denom = 1.f / global_denom;

            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                int base = w * TILE_D;
                if (base >= d) break;

                float warp_rescale = __expf(warp_max[w] - global_max);
                int limit = (base + TILE_D <= d) ? TILE_D : (d - base);

                // coalesced strided writes across lanes
                for (int j = lane; j < limit; j += 32) {
                    float val = warp_partial[w * TILE_D + j] * warp_rescale * inv_denom;
                    out[dst * d + (base + j)] = val;
                }
            }
        }
    }
}

template<int WARPS_PER_BLOCK>
__global__ void gt_kernel_forward_mid_nodes(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const int* __restrict__ mid_nodes,
    int num_mid_degree_nodes,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int d,
    float scale,
    int stride_d
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    int mid_i         = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (mid_i >= num_mid_degree_nodes) {
        return;
    }

    int node_idx      = mid_nodes[mid_i];
    int row_start     = edge_ptr[node_idx];
    int row_end       = edge_ptr[node_idx + 1];

    extern __shared__ float smem[];
    float* warp_base  = smem + warp_id * (2 * stride_d);
    float* k_s        = warp_base;
    float* o_s        = warp_base + stride_d;

    // load key vector for the current node into shared memory
    // initialize output accumulator in the shared memory as well
    for (int f = lane; f < d; f += 32) {
        k_s[f] = K[node_idx * d + f];
        o_s[f] = 0.0f;
    }
    __syncwarp();

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // process neighbors -- each warp processes one neighbor at a time
    for (int base_edge = row_start; base_edge < row_end; base_edge += 32) {
        int eid = base_edge + lane;
        int nbr = (eid < row_end) ? edge_idx[eid] : -1;

        float score = -INFINITY;
        if (nbr >= 0) {
            float dot = dot_vec4(k_s, Q + nbr * d, d);
            score = dot * scale;
        }

        float tile_max = warp_reduce_max(score);
        float new_max = fmaxf(running_max, tile_max);
        float rescale_prev = __expf(running_max - new_max);
        running_sum *= rescale_prev;

        float w_exp = (nbr >= 0) ? __expf(score - new_max) : 0.f;
        running_sum += w_exp;
        running_max = new_max;

        for (int f = lane; f < d; f += 32) {
            o_s[f] *= rescale_prev;
        }
        __syncwarp();

        #pragma unroll
        for (int f0 = 0; f0 < d; f0 += 4) {
            // each lane loads same 4 features from its neighbor
            float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
            int base_offset = nbr * d + f0;
            if (nbr >= 0) {
                if (f0 < d) v0 = V[base_offset];
                if (f0 + 1 < d) v1 = V[base_offset + 1];
                if (f0 + 2 < d) v2 = V[base_offset + 2];
                if (f0 + 3 < d) v3 = V[base_offset + 3];
            }
            // reduce across neighbors (lanes) for each feature
            float a0 = warp_reduce_sum(w_exp * v0);
            float a1 = warp_reduce_sum(w_exp * v1);
            float a2 = warp_reduce_sum(w_exp * v2);
            float a3 = warp_reduce_sum(w_exp * v3);
            if (lane == 0) {
                if (f0 < d) o_s[f0] += a0;
                if (f0 + 1 < d) o_s[f0 + 1] += a1;
                if (f0 + 2 < d) o_s[f0 + 2] += a2;
                if (f0 + 3 < d) o_s[f0 + 3] += a3;
            }
        }
        __syncwarp();
    }

    float denom = warp_reduce_sum(running_sum);

    if (lane == 0) {
        float inv = 1.f / denom;
        for (int f = 0; f < d; ++f) {
            out[node_idx * d + f] = o_s[f] * inv;
        }
    }
}

// ============================================================================
// optimized kernel: process NBR_TILE neighbors per reduction
// warp loads consecutive features & computes partial dot products
// ============================================================================
template<int THREADS_PER_BLOCK, int NBR_TILE = 8>
__global__ void gt_kernel_forward_mid_nodes_feature_parallel_tiled(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const int* __restrict__ mid_nodes,
    int num_mid_degree_nodes,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int d,
    float scale
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_mid_degree_nodes) {
        return;
    }

    const int dst = mid_nodes[block_idx];
    const int row_start = edge_ptr[dst];
    const int row_end = edge_ptr[dst + 1];
    const int degree = row_end - row_start;

    if (degree == 0) {
        return;
    }

    extern __shared__ float smem[];
    float* K_dst = smem;
    float* O_acc = smem + d;

    const int tid = threadIdx.x;

    // load K[dst] and initialize O
    #pragma unroll
    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        K_dst[f] = K[dst * d + f];
        O_acc[f] = 0.0f;
    }
    __syncthreads();

    // streaming softmax statistics
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // process neighbors in tiles to amortize block reduction cost (optimal tile size can vary)
    for (int tile_start = row_start; tile_start < row_end; tile_start += NBR_TILE) {
        const int tile_end = min(tile_start + NBR_TILE, row_end);
        const int tile_size = tile_end - tile_start;

        // --------------------------------------------------------------------
        // 1: Compute attention scores for all neighbors in tile
        // each thread computes partial dots for NBR_TILE neighbors
        // --------------------------------------------------------------------
        float partial_dots[NBR_TILE];
        int nbrs[NBR_TILE];

        #pragma unroll
        for (int i = 0; i < NBR_TILE; i++) {
            const int eid = tile_start + i;
            nbrs[i] = (eid < row_end) ? edge_idx[eid] : -1;
            partial_dots[i] = 0.0f;
        }

        // partial dot products (vectorized when possible)
        const bool use_vec4 = (d % 4 == 0) && (((uintptr_t)K_dst & 15) == 0);

        if (use_vec4) {
            const float4* K_dst_vec4 = reinterpret_cast<const float4*>(K_dst);
            const int d4 = d / 4;

            #pragma unroll
            for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                float4 k4 = K_dst_vec4[f4];

                #pragma unroll
                for (int i = 0; i < NBR_TILE; i++) {
                    if (nbrs[i] >= 0) {
                        const float4* Q_vec4 = reinterpret_cast<const float4*>(Q + nbrs[i] * d);
                        float4 q4 = Q_vec4[f4];
                        partial_dots[i] += (q4.x * k4.x) + (q4.y * k4.y) +
                                          (q4.z * k4.z) + (q4.w * k4.w);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                float k_val = K_dst[f];
                #pragma unroll
                for (int i = 0; i < NBR_TILE; i++) {
                    if (nbrs[i] >= 0) {
                        partial_dots[i] += k_val * Q[nbrs[i] * d + f];
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // 2: block reductions for all neighbors in tile (amortized!)
        // --------------------------------------------------------------------
        float scores[NBR_TILE];
        #pragma unroll
        for (int i = 0; i < NBR_TILE; i++) {
            scores[i] = block_reduce_sum<THREADS_PER_BLOCK>(partial_dots[i]) * scale;
        }

        // --------------------------------------------------------------------
        // 3: update online softmax for entire tile
        // --------------------------------------------------------------------
        float tile_max = m_i;
        #pragma unroll
        for (int i = 0; i < tile_size; i++) {
            tile_max = fmaxf(tile_max, scores[i]);
        }

        float alpha = expf(m_i - tile_max);
        float new_sum = alpha * l_i;

        float exp_weights[NBR_TILE];
        #pragma unroll
        for (int i = 0; i < tile_size; i++) {
            exp_weights[i] = expf(scores[i] - tile_max);
            new_sum += exp_weights[i];
        }

        // --------------------------------------------------------------------
        // 4: update output accumulator with entire tile
        // --------------------------------------------------------------------
        if (use_vec4) {
            float4* O_acc_vec4 = reinterpret_cast<float4*>(O_acc);
            const int d4 = d / 4;

            #pragma unroll 2
            for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                float4 o4 = O_acc_vec4[f4];

                // scale old accumulator
                o4.x *= alpha;
                o4.y *= alpha;
                o4.z *= alpha;
                o4.w *= alpha;

                // add contributions from all neighbors in tile
                #pragma unroll
                for (int i = 0; i < tile_size; i++) {
                    if (nbrs[i] >= 0) {
                        const float4* V_vec4 = reinterpret_cast<const float4*>(V + nbrs[i] * d);
                        float4 v4 = V_vec4[f4];
                        float w = exp_weights[i];

                        o4.x += w * v4.x;
                        o4.y += w * v4.y;
                        o4.z += w * v4.z;
                        o4.w += w * v4.w;
                    }
                }

                O_acc_vec4[f4] = o4;
            }
        } else {
            #pragma unroll 2
            for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                float o_val = O_acc[f] * alpha;

                #pragma unroll
                for (int i = 0; i < tile_size; i++) {
                    if (nbrs[i] >= 0) {
                        o_val += exp_weights[i] * V[nbrs[i] * d + f];
                    }
                }

                O_acc[f] = o_val;
            }
        }

        // update statistics
        m_i = tile_max;
        l_i = new_sum;
    }

    // final normalization and write the result
    float inv_l_i = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        out[dst * d + f] = O_acc[f] * inv_l_i;
    }
}

torch::Tensor graph_attention_forward_buckets_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor edge_idx,
    torch::Tensor mid_nodes,
    torch::Tensor huge_nodes,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(edge_ptr.is_cuda(), "edge_ptr must be CUDA int32");
    TORCH_CHECK(edge_idx.is_cuda(), "edge_idx must be CUDA int32");
    TORCH_CHECK(mid_nodes.is_cuda() && huge_nodes.is_cuda(), "node lists must be CUDA");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32,
                "currently FP32 only");
    TORCH_CHECK(edge_ptr.dtype() == torch::kInt32 &&
                edge_idx.dtype() == torch::kInt32 &&
                mid_nodes.dtype() == torch::kInt32 &&
                huge_nodes.dtype() == torch::kInt32,
                "indices must be int32");

    int  num_nodes = Q.size(0);
    int  d         = Q.size(1);
    auto out       = torch::zeros_like(V);

    float scale  = 1.0f / std::sqrt((float)d);
    int num_mid  = mid_nodes.size(0);
    int num_huge = huge_nodes.size(0);

    // if (num_mid > 0) {
    //     bool use_feature_parallel = (d >= 128);

    //     if (use_feature_parallel) {
    //         // feature-parallel with neighbor tiling
    //         constexpr int THREADS = 64;
    //         constexpr int NBR_TILE = 4;  // Process 8 neighbors per reduction
    //         constexpr int NUM_WARPS = THREADS / 32;
    //         size_t smem = (2 * d + NUM_WARPS) * sizeof(float);

    //         gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
    //             <<<num_mid, THREADS, smem>>>(
    //                 edge_ptr.data_ptr<int>(),
    //                 edge_idx.data_ptr<int>(),
    //                 mid_nodes.data_ptr<int>(),
    //                 num_mid,
    //                 Q.data_ptr<float>(),
    //                 K.data_ptr<float>(),
    //                 V.data_ptr<float>(),
    //                 out.data_ptr<float>(),
    //                 d,
    //                 scale
    //         );
    //     } else {
    //         //  original warp-parallel kernel for d < 128
    //         int blocks_mid = (num_mid + WARPS_PER_BLOCK_MID - 1) / WARPS_PER_BLOCK_MID;
    //         int stride_d = ((d + 31) / 32) * 32;
    //         size_t smem_mid = WARPS_PER_BLOCK_MID * (2 * stride_d) * sizeof(float);

    //         gt_kernel_forward_mid_nodes<WARPS_PER_BLOCK_MID>
    //             <<<blocks_mid, THREADS_PER_BLOCK_MID, smem_mid>>>(
    //                 edge_ptr.data_ptr<int>(),
    //                 edge_idx.data_ptr<int>(),
    //                 mid_nodes.data_ptr<int>(),
    //                 num_mid,
    //                 Q.data_ptr<float>(),
    //                 K.data_ptr<float>(),
    //                 V.data_ptr<float>(),
    //                 out.data_ptr<float>(),
    //                 d,
    //                 scale,
    //                 stride_d
    //         );
    //     }


    //     // int blocks_mid = (num_mid + WARPS_PER_BLOCK_MID - 1) / WARPS_PER_BLOCK_MID;
    //     // // round d to the nearest multiple of 32 to avoid shared memory bank conflicts
    //     // int stride_d = ((d + 31) / 32) * 32;
    //     // size_t smem_mid = WARPS_PER_BLOCK_MID * (2 * stride_d) * sizeof(float);

    //     // gt_kernel_forward_mid_nodes<WARPS_PER_BLOCK_MID>
    //     //     <<<blocks_mid, THREADS_PER_BLOCK_MID, smem_mid>>>(
    //     //         edge_ptr.data_ptr<int>(),
    //     //         edge_idx.data_ptr<int>(),
    //     //         mid_nodes.data_ptr<int>(),
    //     //         num_mid,
    //     //         Q.data_ptr<float>(),
    //     //         K.data_ptr<float>(),
    //     //         V.data_ptr<float>(),
    //     //         out.data_ptr<float>(),
    //     //         d,
    //     //         scale,
    //     //         stride_d
    //     // );
    // }

    if (num_mid > 0) {
        if (d >= 256) {
            // THREADS=64, NBR_TILE=4 for large d
            constexpr int THREADS = 64;
            constexpr int NBR_TILE = 4;
            constexpr int NUM_WARPS = 2;
            size_t smem = (2 * d + NUM_WARPS) * sizeof(float);

            gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
                <<<num_mid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    out.data_ptr<float>(),
                    d,
                    scale
            );

        } else if (d >= 128) {
            // THREADS=32, NBR_TILE=1 for medium d
            constexpr int THREADS = 32;
            constexpr int NBR_TILE = 1;
            constexpr int NUM_WARPS = 1;
            size_t smem = (2 * d + NUM_WARPS) * sizeof(float);

            gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
                <<<num_mid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    out.data_ptr<float>(),
                    d,
                    scale
            );

        } else {
            // THREADS=32, NBR_TILE=1 for medium d
            constexpr int THREADS = 32;
            constexpr int NBR_TILE = 1;
            constexpr int NUM_WARPS = 1;
            size_t smem = (2 * d + NUM_WARPS) * sizeof(float);

            gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
                <<<num_mid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    out.data_ptr<float>(),
                    d,
                    scale
            );

            // // original warp-parallel for small d
            // int blocks_mid = (num_mid + WARPS_PER_BLOCK_MID - 1) / WARPS_PER_BLOCK_MID;
            // int stride_d = ((d + 31) / 32) * 32;
            // size_t smem_mid = WARPS_PER_BLOCK_MID * (2 * stride_d) * sizeof(float);

            // gt_kernel_forward_mid_nodes<WARPS_PER_BLOCK_MID>
            //     <<<blocks_mid, THREADS_PER_BLOCK_MID, smem_mid>>>(
            //         edge_ptr.data_ptr<int>(),
            //         edge_idx.data_ptr<int>(),
            //         mid_nodes.data_ptr<int>(),
            //         num_mid,
            //         Q.data_ptr<float>(),
            //         K.data_ptr<float>(),
            //         V.data_ptr<float>(),
            //         out.data_ptr<float>(),
            //         d,
            //         scale,
            //         stride_d
            // );
        }
    }

    // streaming softmax
    if (num_huge > 0) {

        // Use more warps for better occupancy when huge_nodes is small
        TORCH_CHECK(d <= WARPS_PER_BLOCK_HUGE * TILE_D_HUGE,
                    "d=", d, " too large for WARPS=", WARPS_PER_BLOCK_HUGE,
                    " TILE_D=", TILE_D_HUGE, " (max ", WARPS_PER_BLOCK_HUGE * TILE_D_HUGE, ")");

        // Shared memory for streaming kernel:
        // k_dst[d] + warp_partial[WARPS*TILE_D] + warp_max[WARPS] + warp_denom[WARPS]
        size_t smem_huge = (
            d +
            WARPS_PER_BLOCK_HUGE * TILE_D_HUGE +
            2 * WARPS_PER_BLOCK_HUGE
        ) * sizeof(float);

        gt_kernel_forward_huge_nodes<WARPS_PER_BLOCK_HUGE, TILE_D_HUGE>
            <<<num_huge, THREADS_PER_BLOCK_HUGE, smem_huge>>>(
                edge_ptr.data_ptr<int>(),
                edge_idx.data_ptr<int>(),
                huge_nodes.data_ptr<int>(),
                num_huge,
                Q.data_ptr<float>(),
                K.data_ptr<float>(),
                V.data_ptr<float>(),
                out.data_ptr<float>(),
                d,
                scale
        );

    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_buckets",
        &graph_attention_forward_buckets_cuda,
        "Graph Attention Forward (CUDA, prebucketed)",
        py::arg("edge_ptr"),
        py::arg("edge_indices"),
        py::arg("mid_nodes"),
        py::arg("huge_nodes"),
        py::arg("Q"),
        py::arg("K"),
        py::arg("V")
    );
}
