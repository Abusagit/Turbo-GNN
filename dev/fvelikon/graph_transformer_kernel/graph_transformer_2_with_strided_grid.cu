#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>

#define FULL_WARP_MASK 0xffffffff

constexpr int WARPS_PER_BLOCK_HUGE   = 8;
constexpr int THREADS_PER_BLOCK_HUGE = WARPS_PER_BLOCK_HUGE * 32;
constexpr int TILE_D_HUGE            = 32;


// utilities for warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
}

// calculate dot-product using vectorized 128-bit loads
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

    val = warp_reduce_sum(val);

    // first lane of each warp writes to shared
    if (lane == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    // first warp reduces across warp sums
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);

        // broadcast result via shared memory
        if (lane == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    return warp_sums[0];
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
    float* __restrict__ logsumexp,
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

    if (threadIdx.x < WARPS_PER_BLOCK) {
        warp_max[threadIdx.x] = -INFINITY;
        warp_denom[threadIdx.x] = 0.0f;
    }

    // load K[dst] into shared (all threads cooperate)
    for (int f = threadIdx.x; f < d; f += blockDim.x) {
        k_dst[f] = K[dst * d + f];
    }
    __syncthreads();

    // each warp handles a slice of features [feat_base, feat_base+TILE_D)
    const int feat_base  = warp_id * TILE_D;
    const int feat_limit = (feat_base + TILE_D <= d) ? TILE_D : (d - feat_base);

    const bool warp_is_active = (feat_base < d);

    // per-lane register accumulators for this warp's feature slice
    float acc_feat[TILE_D];
    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        acc_feat[j] = 0.f;
    }

    // per-lane running statistics for streaming softmax
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // ============================================================
    // single-pass streaming softmax over neighbors
    // process neighbors in tiles (warp-strided iteration)
    // ============================================================

    if (warp_is_active) {
        constexpr int TILE_SIZE = 32;
        // for (int tile_start = row_start; tile_start < row_end; tile_start += WARPS_PER_BLOCK * TILE_SIZE) {
        for (int tile_start = row_start; tile_start < row_end; tile_start += TILE_SIZE) {
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
            float w = (nbr >= 0) ? __expf(score - new_max) : 0.0f;
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
    }

    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        float vj = (j < feat_limit && warp_is_active) ? acc_feat[j] : 0.0f;
        vj = warp_reduce_sum(vj);
        if (lane == 0) {
            warp_partial[warp_id * TILE_D + j] = vj;
        }
    }

    // store per-warp statistics
    float warp_sum = warp_reduce_sum(warp_is_active ? running_sum : 0.0f);
    float warp_max_val = warp_reduce_max(warp_is_active ? running_max : -INFINITY);

    if (lane == 0) {
        warp_max[warp_id] = warp_max_val;
        warp_denom[warp_id] = warp_sum;
    }
    __syncthreads();

    // final reduction and logsumexp computation (warp 0 only)
    if (warp_id == 0) {
        float global_max = warp_max[0];
        float global_denom = warp_denom[0];

        if (lane == 0) {
            if (global_denom > 0.0f) {
                logsumexp[dst] = global_max + logf(global_denom);
            } else {
                logsumexp[dst] = -INFINITY;  // Prevent log(0)
            }
        }

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
    float* __restrict__ logsumexp,
    int d,
    float scale
) {
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* K_dst = smem;
    float* O_acc = smem + d;

    // grid-strided loop over mid-degree modes
    for (int block_idx = blockIdx.x; block_idx < num_mid_degree_nodes; block_idx += gridDim.x){
        const int dst = mid_nodes[block_idx];
        const int row_start = edge_ptr[dst];
        const int row_end   = edge_ptr[dst + 1];
        const int degree    = row_end - row_start;

        if (degree == 0) {
            continue;
        }

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
        // process neighbors in tiles
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

                for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                    float4 k4 = K_dst_vec4[f4];

                    for (int i = 0; i < NBR_TILE; i++) {
                        if (nbrs[i] >= 0) { // check if neighbor exists (b.c. there can not be multiple of NBR_TILE  neighbors)
                            const float4* Q_vec4 = reinterpret_cast<const float4*>(Q + nbrs[i] * d);
                            float4 q4 = Q_vec4[f4];

                            partial_dots[i] += (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                    float k_val = K_dst[f];
                    #pragma unroll
                    for (int i = 0; i < NBR_TILE; i++) {
                        if (nbrs[i] >= 0) { // check if neighbor exists (b.c. there can not be multiple of NBR_TILE  neighbors)
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

                #pragma unroll
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
                #pragma unroll
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

        float inv_l_i = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

        for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
            out[dst * d + f] = O_acc[f] * inv_l_i;
        }

        if (tid == 0) {
            if (l_i > 0.0f) {
                logsumexp[dst] = m_i + logf(l_i);
            } else {
                logsumexp[dst] = -INFINITY;  // prevent log(0)
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// BACKWARD PASS STARTS HERE
// Kernel: Compute D = rowsum(dO * O)
// ============================================================================

__global__ void compute_D_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    int num_nodes,
    int d
) {
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    __shared__ float warp_sums[32];

    // Grid-stride loop over nodes
    for (int node_idx = blockIdx.x; node_idx < num_nodes; node_idx += gridDim.x) {
        const float* dO_node = dO + node_idx * d;
        const float* O_node  = O  + node_idx * d;

        float sum = 0.0f;

        // vectorized path if aligned
        if (d % 4 == 0 && (((uintptr_t)dO_node & 15) == 0) && (((uintptr_t)O_node  & 15) == 0)) {
            const float4* dO_vec4 = reinterpret_cast<const float4*>(dO_node);
            const float4* O_vec4  = reinterpret_cast<const float4*>(O_node);
            const int d4 = d / 4;

            for (int f4 = threadIdx.x; f4 < d4; f4 += blockDim.x) {
                float4 do4 = dO_vec4[f4];
                float4 o4  = O_vec4[f4];
                sum += do4.x * o4.x + do4.y * o4.y +
                       do4.z * o4.z + do4.w * o4.w;
            }
        } else {
            for (int f = threadIdx.x; f < d; f += blockDim.x) {
                sum += dO_node[f] * O_node[f];
            }
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            warp_sums[wid] = sum;
        }
        __syncthreads();

        if (wid == 0) {
            float block_sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
            block_sum = warp_reduce_sum(block_sum);
            if (lane == 0) {
                D[node_idx] = block_sum;
            }
        }
        __syncthreads();
    }

    // const int node_idx = blockIdx.x;
    // if (node_idx >= num_nodes) return;

    // const float* dO_node = dO + node_idx * d;
    // const float* O_node = O + node_idx * d;

    // float sum = 0.0f;

    // // check whether address is aligned and we can use vectorized loads:
    // if (d % 4 == 0 && (((uintptr_t)dO_node & 15) == 0) && (((uintptr_t)O_node & 15) == 0)) {
    //     const float4* dO_vec4 = reinterpret_cast<const float4*>(dO_node);
    //     const float4* O_vec4 = reinterpret_cast<const float4*>(O_node);
    //     const int d4 = d / 4;

    //     for (int f4 = threadIdx.x; f4 < d4; f4 += blockDim.x) {
    //         float4 do4 = dO_vec4[f4];
    //         float4 o4 = O_vec4[f4];
    //         sum += do4.x * o4.x + do4.y * o4.y + do4.z * o4.z + do4.w * o4.w;
    //     }
    // } else {
    //     for (int f = threadIdx.x; f < d; f += blockDim.x) {
    //         sum += dO_node[f] * O_node[f];
    //     }
    // }

    // sum = warp_reduce_sum(sum);

    // __shared__ float warp_sums[32];
    // const int lane = threadIdx.x % 32;
    // const int wid = threadIdx.x / 32;

    // if (lane == 0) {
    //     warp_sums[wid] = sum;
    // }
    // __syncthreads();

    // if (wid == 0) {
    //     sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
    //     sum = warp_reduce_sum(sum);
    //     if (lane == 0) {
    //         D[node_idx] = sum;
    //     }
    // }
}

// ============================================================================
// Backward kernel: Mid-degree nodes with warp-cooperative atomics
// ============================================================================

template<int THREADS_PER_BLOCK, int NBR_TILE = 8>
__global__ void gt_kernel_backward_mid_nodes_tiled_warp_atomic(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const int* __restrict__ mid_nodes,
    int num_mid_degree_nodes,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ logsumexp,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int d,
    float scale
) {

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;

    extern __shared__ float smem[];
    float* K_dst  = smem;
    float* dO_dst = smem + d;
    float* dK_acc = smem + 2 * d;

    // Grid-stride loop over mid-degree nodes
    for (int block_idx = blockIdx.x; block_idx < num_mid_degree_nodes; block_idx += gridDim.x) {
        const int dst = mid_nodes[block_idx];
        const int row_start = edge_ptr[dst];
        const int row_end   = edge_ptr[dst + 1];

        if (row_end == row_start) {
            continue;
        }

        #pragma unroll
        for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
            K_dst[f]  = K[dst * d + f];
            dO_dst[f] = dO[dst * d + f];
            dK_acc[f] = 0.0f;
        }
        __syncthreads();

        const float lse_dst = logsumexp[dst];
        const float D_dst   = D[dst];

        for (int tile_start = row_start; tile_start < row_end; tile_start += NBR_TILE) {
            const int tile_end = min(tile_start + NBR_TILE, row_end);
            const int tile_size = tile_end - tile_start;

            // ====================================================================
            // 1. Compute attention weights P
            // ====================================================================
            float partial_dots[NBR_TILE];
            int nbrs[NBR_TILE];

            #pragma unroll
            for (int i = 0; i < NBR_TILE; i++) {
                const int eid = tile_start + i;
                nbrs[i] = (eid < row_end) ? edge_idx[eid] : -1;
                partial_dots[i] = 0.0f;
            }

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
                            partial_dots[i] += (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
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

            float scores[NBR_TILE];
            float P_weights[NBR_TILE];

            #pragma unroll
            for (int i = 0; i < NBR_TILE; i++) {
                scores[i] = block_reduce_sum<THREADS_PER_BLOCK>(partial_dots[i]) * scale;
                P_weights[i] = (nbrs[i] >= 0) ? expf(scores[i] - lse_dst) : 0.0f;
            }

            // ====================================================================
            // 2. Warp-cooperative atomic scatter for dV
            // Each warp handles a partition of features to reduce contention
            // ====================================================================
            constexpr int WARPS = THREADS_PER_BLOCK / 32;

            #pragma unroll
            for (int i = 0; i < tile_size; i++) {
                if (nbrs[i] >= 0) {
                    float p = P_weights[i];
                    float* dV_nbr = dV + nbrs[i] * d;

                    // Each warp processes d/WARPS features
                    int feat_per_warp = (d + WARPS - 1) / WARPS;
                    int feat_start = warp_id * feat_per_warp;
                    int feat_end = min(feat_start + feat_per_warp, d);

                    // Warp-strided access within partition
                    for (int f = feat_start + lane; f < feat_end; f += 32) {
                        atomicAdd(&dV_nbr[f], p * dO_dst[f]);
                    }
                }
            }

            // ====================================================================
            // 3. Compute dP and dS
            // ====================================================================
            float dP_vals[NBR_TILE];

            #pragma unroll
            for (int i = 0; i < NBR_TILE; i++) {
                dP_vals[i] = 0.0f;
            }

            if (use_vec4) {
                const float4* dO_dst_vec4 = reinterpret_cast<const float4*>(dO_dst);
                const int d4 = d / 4;

                #pragma unroll
                for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                    float4 do4 = dO_dst_vec4[f4];

                    #pragma unroll
                    for (int i = 0; i < tile_size; i++) {
                        if (nbrs[i] >= 0) {
                            const float4* V_vec4 = reinterpret_cast<const float4*>(V + nbrs[i] * d);
                            float4 v4 = V_vec4[f4];
                            dP_vals[i] += (do4.x * v4.x) + (do4.y * v4.y) +
                                        (do4.z * v4.z) + (do4.w * v4.w);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                    float do_val = dO_dst[f];

                    #pragma unroll
                    for (int i = 0; i < tile_size; i++) {
                        if (nbrs[i] >= 0) {
                            dP_vals[i] += do_val * V[nbrs[i] * d + f];
                        }
                    }
                }
            }

            float dS_vals[NBR_TILE];

            #pragma unroll
            for (int i = 0; i < NBR_TILE; i++) {
                float dP_full = block_reduce_sum<THREADS_PER_BLOCK>(dP_vals[i]);
                dS_vals[i] = P_weights[i] * (dP_full - D_dst);
            }

            // ====================================================================
            // 4. Accumulate dK locally (no atomics)
            // ====================================================================
            if (use_vec4) {
                float4* dK_acc_vec4 = reinterpret_cast<float4*>(dK_acc);
                const int d4 = d / 4;

                #pragma unroll
                for (int f4 = tid; f4 < d4; f4 += THREADS_PER_BLOCK) {
                    float4 dk4 = dK_acc_vec4[f4];

                    #pragma unroll
                    for (int i = 0; i < tile_size; i++) {
                        if (nbrs[i] >= 0) {
                            const float4* Q_vec4 = reinterpret_cast<const float4*>(Q + nbrs[i] * d);
                            float4 q4 = Q_vec4[f4];
                            float ds = dS_vals[i] * scale;

                            dk4.x += ds * q4.x;
                            dk4.y += ds * q4.y;
                            dk4.z += ds * q4.z;
                            dk4.w += ds * q4.w;
                        }
                    }

                    dK_acc_vec4[f4] = dk4;
                }
            } else {
                #pragma unroll
                for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
                    float dk_val = dK_acc[f];

                    #pragma unroll
                    for (int i = 0; i < tile_size; i++) {
                        if (nbrs[i] >= 0) {
                            dk_val += dS_vals[i] * scale * Q[nbrs[i] * d + f];
                        }
                    }

                    dK_acc[f] = dk_val;
                }
            }
            // ====================================================================
            // 5. Warp-cooperative atomic scatter for dQ
            // ====================================================================
            #pragma unroll
            for (int i = 0; i < tile_size; i++) {
                if (nbrs[i] >= 0) {
                    float ds_scaled = dS_vals[i] * scale;
                    float* dQ_nbr = dQ + nbrs[i] * d;

                    constexpr int WARPS = THREADS_PER_BLOCK / 32;
                    int feat_per_warp = (d + WARPS - 1) / WARPS;
                    int feat_start = warp_id * feat_per_warp;
                    int feat_end = min(feat_start + feat_per_warp, d);

                    for (int f = feat_start + lane; f < feat_end; f += 32) {
                        atomicAdd(&dQ_nbr[f], ds_scaled * K_dst[f]);
                    }
                }
            }
        }

        // Write dK to global memory (no atomics)
        for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
            dK[dst * d + f] = dK_acc[f];
        }
        __syncthreads();
    }
}

// ============================================================================
// Backward kernel: Huge-degree nodes with warp-cooperative atomics
// ============================================================================

template<int WARPS_PER_BLOCK, int TILE_D>
__global__ void gt_kernel_backward_huge_nodes_warp_atomic(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const int* __restrict__ huge_nodes,
    int num_huge,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ logsumexp,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int d,
    float scale
) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int block_node = blockIdx.x;

    const int dst = huge_nodes[block_node];
    const int row_start = edge_ptr[dst];
    const int row_end = edge_ptr[dst + 1];

    if ((block_node >= num_huge) || (row_end == row_start)) return;

    extern __shared__ float shmem[];
    float* k_dst = shmem;
    float* do_dst = k_dst + d;
    float* dk_partial = do_dst + d;

    for (int f = threadIdx.x; f < d; f += blockDim.x) {
        k_dst[f] = K[dst * d + f];
        do_dst[f] = dO[dst * d + f];
    }
    __syncthreads();

    const int feat_base = warp_id * TILE_D;
    const int feat_limit = (feat_base + TILE_D <= d) ? feat_base + TILE_D : (d - feat_base);

    const bool warp_is_active = (feat_base < d);

    float dk_feat[TILE_D];

    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        dk_feat[j] = 0.0f;
    }

    const float lse_dst = logsumexp[dst];
    const float D_dst = D[dst];
    if (!isfinite(lse_dst)) {
        goto finalize_output;
    }

    if (warp_is_active) {
        constexpr int TILE_SIZE = 32;

        for (int tile_start = row_start; tile_start < row_end; tile_start += TILE_SIZE) {
            // ====================================================================
            // Step 1: Each lane computes its neighbor's attention weight and gradient
            // ====================================================================
            const int eid = tile_start + lane;
            const int nbr = (eid < row_end) ? edge_idx[eid] : -1;

            float P_weight = 0.0f;
            float dS_scaled = 0.0f;

            if (nbr >= 0) {
                // Compute attention weight P
                float score = dot_vec4(k_dst, Q + nbr * d, d) * scale;
                P_weight = expf(score - lse_dst);

                // Compute gradient dS = P * (dP - D)
                float dP = dot_vec4(do_dst, V + nbr * d, d);
                dS_scaled = P_weight * (dP - D_dst) * scale;
            }

            // ====================================================================
            // Step 2: Warp-cooperative scatter to ALL neighbors in this tile
            // Use shuffle to broadcast each neighbor's data to all lanes
            // ====================================================================
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                if (tile_start + i >= row_end) break;

                // Broadcast neighbor i's data to all lanes
                int target_nbr = __shfl_sync(0xffffffff, nbr, i);
                if (target_nbr < 0) continue;

                float target_P = __shfl_sync(0xffffffff, P_weight, i);
                float target_dS = __shfl_sync(0xffffffff, dS_scaled, i);

                // Each lane writes ONE feature for this neighbor
                if (lane < TILE_D && (feat_base + lane) < d) {
                    atomicAdd(&dV[target_nbr * d + feat_base + lane],
                            target_P * do_dst[feat_base + lane]);

                    atomicAdd(&dQ[target_nbr * d + feat_base + lane],
                            target_dS * k_dst[feat_base + lane]);
                }
            }

            // ====================================================================
            // Step 3: Accumulate dK locally (no atomics needed)
            // Each lane accumulates for its own neighbor
            // ====================================================================
            if (nbr >= 0) {
                const float* q_ptr = Q + nbr * d + feat_base;

                // Calculate actual number of features this warp handles
                int feat_count = (feat_base + TILE_D <= d) ? TILE_D : (d - feat_base);

                // Vectorized path
                int j = 0;
                if (feat_count >= 4 && (((size_t)q_ptr) & 15) == 0) {
                    const float4* q4_ptr = reinterpret_cast<const float4*>(q_ptr);
                    for (; j + 4 <= feat_count; j += 4) {
                        float4 q4 = q4_ptr[j / 4];
                        dk_feat[j + 0] += dS_scaled * q4.x;
                        dk_feat[j + 1] += dS_scaled * q4.y;
                        dk_feat[j + 2] += dS_scaled * q4.z;
                        dk_feat[j + 3] += dS_scaled * q4.w;
                    }
                }

                // Scalar tail
                for (; j < feat_count; ++j) {
                    dk_feat[j] += dS_scaled * q_ptr[j];
                }
            }
        }
    }

    finalize_output:
    // Reduce dK within warp and write to shared
    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        float dk_j = (j < feat_limit && warp_is_active) ? dk_feat[j] : 0.0f;
        dk_j = warp_reduce_sum(dk_j);

        if (lane == 0 && (feat_base + j) < d) {
            dk_partial[warp_id * TILE_D + j] = dk_j;
        }
    }
    __syncthreads();

    // Warp 0: write final dK results
    if (warp_id == 0) {
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            int base = w * TILE_D;
            if (base >= d) break;

            int limit = (base + TILE_D <= d) ? TILE_D : (d - base);

            for (int j = lane; j < limit; j += 32) {
                dK[dst * d + (base + j)] = dk_partial[w * TILE_D + j];
            }
        }
    }
}

// ============================================================================
// Main backward wrapper function
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
graph_attention_backward_buckets_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor edge_idx,
    torch::Tensor mid_nodes,
    torch::Tensor huge_nodes,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor logsumexp
) {
    TORCH_CHECK(dO.is_cuda() && O.is_cuda(), "gradients must be CUDA");

    int num_nodes = Q.size(0);
    int d = Q.size(1);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    auto D = torch::zeros({num_nodes}, Q.options());

    float scale = 1.0f / std::sqrt((float)d);

    // ========================================================================
    // Step 1: Compute D = rowsum(dO ⊙ O)
    // ========================================================================
    {
        constexpr int THREADS = 128;
        const int max_blocks = 4096;
        int grid = (num_nodes < max_blocks) ? num_nodes : max_blocks;
        compute_D_kernel<<<grid, THREADS>>>(
            dO.data_ptr<float>(),
            O.data_ptr<float>(),
            D.data_ptr<float>(),
            num_nodes,
            d
        );
    }

    cudaDeviceSynchronize();
    cudaError_t err1 = cudaGetLastError();
    TORCH_CHECK(err1 == cudaSuccess, "Backward CUDA kernel failed on `compute_D_kernel`: ", cudaGetErrorString(err1));

    // ========================================================================
    // Step 2: Mid-degree backward with warp-cooperative atomics
    // ========================================================================
    int num_mid = mid_nodes.size(0);
    if (num_mid > 0) {
        if (d >= 256) {
            constexpr int THREADS = 64;
            constexpr int NBR_TILE = 4;
            size_t smem = 3 * d * sizeof(float);
            const int max_blocks = 4096;
            int grid = (num_mid < max_blocks) ? num_mid : max_blocks;


            gt_kernel_backward_mid_nodes_tiled_warp_atomic<THREADS, NBR_TILE>
                <<<grid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    dO.data_ptr<float>(),
                    logsumexp.data_ptr<float>(),
                    D.data_ptr<float>(),
                    dQ.data_ptr<float>(),
                    dK.data_ptr<float>(),
                    dV.data_ptr<float>(),
                    d,
                    scale
            );
        } else if (d >= 128) {
            constexpr int THREADS = 32;
            constexpr int NBR_TILE = 2;
            size_t smem = 3 * d * sizeof(float);
            const int max_blocks = 4096;
            int grid = (num_mid < max_blocks) ? num_mid : max_blocks;


            gt_kernel_backward_mid_nodes_tiled_warp_atomic<THREADS, NBR_TILE>
                <<<grid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    dO.data_ptr<float>(),
                    logsumexp.data_ptr<float>(),
                    D.data_ptr<float>(),
                    dQ.data_ptr<float>(),
                    dK.data_ptr<float>(),
                    dV.data_ptr<float>(),
                    d,
                    scale
            );
        } else {
            constexpr int THREADS = 32;
            constexpr int NBR_TILE = 1;
            size_t smem = 3 * d * sizeof(float);
            const int max_blocks = 4096;
            int grid = (num_mid < max_blocks) ? num_mid : max_blocks;


            gt_kernel_backward_mid_nodes_tiled_warp_atomic<THREADS, NBR_TILE>
                <<<grid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    dO.data_ptr<float>(),
                    logsumexp.data_ptr<float>(),
                    D.data_ptr<float>(),
                    dQ.data_ptr<float>(),
                    dK.data_ptr<float>(),
                    dV.data_ptr<float>(),
                    d,
                    scale
            );
        }
    }

    // ========================================================================
    // Step 3: Huge-degree backward with warp-cooperative atomics
    // ========================================================================
    int num_huge = huge_nodes.size(0);
    if (num_huge > 0) {
        constexpr int WARPS = 8;
        constexpr int TILE_D = 32;

        TORCH_CHECK(d <= WARPS * TILE_D,
                    "d=", d, " too large for WARPS=", WARPS,
                    " TILE_D=", TILE_D, " (max ", WARPS * TILE_D, ")");

        size_t smem_huge = (
            2 * d +              // k_dst, do_dst
            WARPS * TILE_D       // dk_partial
        ) * sizeof(float);

        gt_kernel_backward_huge_nodes_warp_atomic<WARPS, TILE_D>
            <<<num_huge, WARPS * 32, smem_huge>>>(
                edge_ptr.data_ptr<int>(),
                edge_idx.data_ptr<int>(),
                huge_nodes.data_ptr<int>(),
                num_huge,
                Q.data_ptr<float>(),
                K.data_ptr<float>(),
                V.data_ptr<float>(),
                dO.data_ptr<float>(),
                logsumexp.data_ptr<float>(),
                D.data_ptr<float>(),
                dQ.data_ptr<float>(),
                dK.data_ptr<float>(),
                dV.data_ptr<float>(),
                d,
                scale
        );
    }
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    TORCH_CHECK(err2 == cudaSuccess, "Backward CUDA kernel failed: ", cudaGetErrorString(err2));

    return std::make_tuple(dQ, dK, dV);
}


std::tuple<torch::Tensor, torch::Tensor> graph_attention_forward_buckets_cuda(
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
    auto logsumexp = torch::full({num_nodes}, -INFINITY, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));

    float scale  = 1.0f / std::sqrt((float)d);
    int num_mid  = mid_nodes.size(0);
    int num_huge = huge_nodes.size(0);

    if (num_mid > 0) {
        if (d >= 256) {
            // THREADS=64, NBR_TILE=4 for large d
            constexpr int THREADS = 64;
            constexpr int NBR_TILE = 4;
            constexpr int NUM_WARPS = 2;
            size_t smem = (2 * d + NUM_WARPS) * sizeof(float);
            const int max_blocks = 4096;
            int grid = (num_mid < max_blocks) ? num_mid : max_blocks;


            gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
                <<<grid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    out.data_ptr<float>(),
                    logsumexp.data_ptr<float>(),
                    d,
                    scale
            );

        } else if (d >= 128) {
            // THREADS=32, NBR_TILE=1 for medium d
            constexpr int THREADS = 32;
            constexpr int NBR_TILE = 1;
            constexpr int NUM_WARPS = 1;
            size_t smem = (2 * d + NUM_WARPS) * sizeof(float);
            const int max_blocks = 4096;
            int grid = (num_mid < max_blocks) ? num_mid : max_blocks;


            gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
                <<<grid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    out.data_ptr<float>(),
                    logsumexp.data_ptr<float>(),
                    d,
                    scale
            );

        } else {
            // THREADS=32, NBR_TILE=1 for medium d
            constexpr int THREADS = 32;
            constexpr int NBR_TILE = 1;
            constexpr int NUM_WARPS = 1;
            size_t smem = (2 * d + NUM_WARPS) * sizeof(float);
            const int max_blocks = 4096;
            int grid = (num_mid < max_blocks) ? num_mid : max_blocks;


            gt_kernel_forward_mid_nodes_feature_parallel_tiled<THREADS, NBR_TILE>
                <<<grid, THREADS, smem>>>(
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    mid_nodes.data_ptr<int>(),
                    num_mid,
                    Q.data_ptr<float>(),
                    K.data_ptr<float>(),
                    V.data_ptr<float>(),
                    out.data_ptr<float>(),
                    logsumexp.data_ptr<float>(),
                    d,
                    scale
            );
        }
    }

    if (num_huge > 0) {
        // use more warps for better occupancy when huge_nodes is small
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
                logsumexp.data_ptr<float>(),
                d,
                scale
        );
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return std::make_tuple(out, logsumexp);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_buckets",
        &graph_attention_forward_buckets_cuda,
        "Graph Attention Forward - returns (out, logsumexp)",
        py::arg("edge_ptr"),
        py::arg("edge_indices"),
        py::arg("mid_nodes"),
        py::arg("huge_nodes"),
        py::arg("Q"),
        py::arg("K"),
        py::arg("V")
    );

    m.def(
        "backward_buckets",
        &graph_attention_backward_buckets_cuda,
        "Graph Attention Backward - returns (dQ, dK, dV)",
        py::arg("edge_ptr"),
        py::arg("edge_indices"),
        py::arg("mid_nodes"),
        py::arg("huge_nodes"),
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("O"),
        py::arg("dO"),
        py::arg("logsumexp")
    );
}
