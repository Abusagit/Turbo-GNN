#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define FULL_WARP_MASK 0xffffffff
constexpr int kMaxThreadsInWarp = 32;

// =============================================================================
// GATv2 Kernel with CSR Graph Format
// =============================================================================


__device__ __forceinline__ float leaky_relu_elementwise(float x, float negative_slope) {
    return  (x > 0.0f) ? x : negative_slope * x;
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
}


struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;

    __device__ __forceinline__ OnlineSoftmaxState() : max_val(-FLT_MAX), sum_exp(0.0f) {}

    __device__ __forceinline__ float update(float logit) {
        float old_max = max_val;
        max_val = fmaxf(max_val, logit);

        // correction factor for previous sum when max changes
        float correction = __expf(old_max - max_val);
        sum_exp = sum_exp * correction + __expf(logit - max_val);
        return correction;
    }

    __device__ __forceinline__ float get_alpha(float logit) const {
        return __expf(logit - max_val) / sum_exp;
    }
};


__global__ void GATv2Kernel_CSR(
    size_t N,
    size_t z,
    const float* __restrict__ d_l,          // [N, z] - left features
    const float* __restrict__ d_r,          // [N, z] - right features
    const int* __restrict__ d_row_ptr,      // [N+1] - CSR row pointers
    const int* __restrict__ d_col_idx,      // [E] - CSR column indices (neighbor IDs)
    const float* __restrict__ d_attn_vec,   // [z] - attention vector
    float* __restrict__ d_h_out,            // [N, z] - output node features
    float* __restrict__ d_logits_out,       // [E] - attention weights per edge -- can be used for backward
    float* __restrict__ d_logsumexp_out,        // [N] -- logsumexp values (used for backward pass)
    float negative_slope
) {
    // shared memory layout:
    // [0, z): l_i cached
    extern __shared__ float shared[];
    float* l_shared = shared;

    int node_i = blockIdx.x;
    int lane_id = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= N) {
        return;
    }

    // neighbor range from CSR
    int edge_start = d_row_ptr[node_i];
    int edge_end = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;

    // skip isolated nodes (no neighbors)
    if (num_neighbors == 0) {
        return;
    }

    const float4* attn_ptr = reinterpret_cast<const float4*>(d_attn_vec);
    int num_float4 = z / 4;

    // ==========================================
    // 0: Load l_i into shared memory
    // ==========================================
    {
        const float4* l_ptr = reinterpret_cast<const float4*>(d_l + node_i * z);
        float4* l_shared_f4 = reinterpret_cast<float4*>(l_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            l_shared_f4[i] = l_ptr[i];
        }
    }
    __syncthreads();


    // ==========================================
    // 1: compute attention logits for each neighbor
    // e_ij = a^T @ LeakyReLU(l_i + r_j)
    // ONLINE SOFTMAX
    // ==========================================
    const float4* l_f4 = reinterpret_cast<const float4*>(l_shared);
    OnlineSoftmaxState softmax_state;


    // ==========================================
    // 2: compute attention weights and aggregate
    // h_i = sum_j a_ij * r_j
    // ==========================================

    // register accumulators for output
    int float4_per_thread = (num_float4 + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;  // Ceiling division
    float4 h_acc[8];  // Support up to z=1024 NOTE TODO THIS IS A WORKAROUND!!!!!!!
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        h_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // accumulate over all neighbors
    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = d_col_idx[edge_start + k];
        const float4* r_ptr = reinterpret_cast<const float4*>(d_r + neighbor_j * z);

        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_ptr[i];
            float4 a_val = attn_ptr[i];

            float4 sum;
            sum.x = l_val.x + r_val.x;
            sum.y = l_val.y + r_val.y;
            sum.z = l_val.z + r_val.z;
            sum.w = l_val.w + r_val.w;

            sum.x = leaky_relu_elementwise(sum.x, negative_slope);
            sum.y = leaky_relu_elementwise(sum.y, negative_slope);
            sum.z = leaky_relu_elementwise(sum.z, negative_slope);
            sum.w = leaky_relu_elementwise(sum.w, negative_slope);
            dot += sum.x * a_val.x + sum.y * a_val.y + sum.z * a_val.z + sum.w * a_val.w;
        }

        dot = warp_reduce_sum(dot);
        // save old max before update
        float old_max = softmax_state.max_val;
        // update softmax statistics and compute rescaling factor
        float rescale = softmax_state.update(dot);

        // rescale previously accumulated values
        #pragma unroll
        for (int i = 0; i < float4_per_thread; ++i) {  // NOTE currently thys loop has at most 8 iterations
                                                    // TODO support dimension > 1024, maybe use more blocks?
            h_acc[i].x *= rescale;
            h_acc[i].y *= rescale;
            h_acc[i].z *= rescale;
            h_acc[i].w *= rescale;
        }

        // compute unnormalized contribution for this neighbor
        float contribution = __expf(dot - softmax_state.max_val);

        // add weighted contribution from this neighbor
        for (int i = 0; i < float4_per_thread; ++i) {
            int idx = lane_id + i * kMaxThreadsInWarp;
            if (idx < num_float4) {
                float4 r_val = r_ptr[idx];

                h_acc[i].x += contribution * r_val.x;
                h_acc[i].y += contribution * r_val.y;
                h_acc[i].z += contribution * r_val.z;
                h_acc[i].w += contribution * r_val.w;
            }
        }
    }

    // write logsumexp value:
    if (lane_id == 0){
        float L_i = softmax_state.max_val + __logf(softmax_state.sum_exp);
        d_logsumexp_out[node_i] = L_i;
    }

    float4* h_out_ptr = reinterpret_cast<float4*>(d_h_out + node_i * z);

    for (int i = 0; i < float4_per_thread; ++i) {
        int idx = lane_id + i * kMaxThreadsInWarp;
        if (idx < num_float4) {
            h_out_ptr[idx] = h_acc[i];
        }
    }
}


// =============================================================================
// Launcher for forward pass
// =============================================================================

void GATv2Forward_CSR(
    size_t N, size_t z,
    const float* d_l,
    const float* d_r,
    const int* d_row_ptr,
    const int* d_col_idx,
    const float* d_attn_vec,
    float* d_h_out,
    float* d_logits_out,
    float* d_logsumexp_out,
    float negative_slope,
    int max_neighbors,
    cudaStream_t stream = 0
) {
    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N);

    // shared memory: z floats for l_i + max_neighbors floats for logits
    size_t shared_mem_size = z * sizeof(float);

    GATv2Kernel_CSR<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, z, d_l, d_r, d_row_ptr, d_col_idx, d_attn_vec,
        d_h_out, d_logits_out,d_logsumexp_out, negative_slope
    );
}


// FlashAttention2-like logsumexp trick
__device__ __forceinline__ float recompute_alpha(
    float e_ij,          // logit
    float L_i            // saved log-sum-exp
) {
    return __expf(e_ij - L_i);
}


__global__ void PrecomputeG(
    size_t N, size_t z,
    const float* __restrict__ grad_h,
    const float* __restrict__ d_r,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_l,
    const float* __restrict__ d_attn_vec,
    const float* __restrict__ d_logsumexp,
    float* __restrict__ d_G,
    float negative_slope
) {
    extern __shared__ float shared[];
    float* l_shared = shared;
    float* grad_h_shared = l_shared + z;

    int node_i = blockIdx.x;
    int lane_id = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= N) return;

    int edge_start = d_row_ptr[node_i];
    int edge_end = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;

    if (num_neighbors == 0) {
        if (lane_id == 0) d_G[node_i] = 0.0f;
        return;
    }

    const float4* attn_ptr = reinterpret_cast<const float4*>(d_attn_vec);
    int num_float4 = z / 4;

    // load l_i and grad_h_i into shared memory
    {
        const float4* l_ptr = reinterpret_cast<const float4*>(d_l + node_i * z);
        const float4* gh_ptr = reinterpret_cast<const float4*>(grad_h + node_i * z);
        float4* l_shared_f4 = reinterpret_cast<float4*>(l_shared);
        float4* gh_shared_f4 = reinterpret_cast<float4*>(grad_h_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            l_shared_f4[i] = l_ptr[i];
            gh_shared_f4[i] = gh_ptr[i];
        }
    }
    __syncthreads();

    const float4* l_f4 = reinterpret_cast<const float4*>(l_shared);
    const float4* grad_h_f4 = reinterpret_cast<const float4*>(grad_h_shared);
    float L_i = d_logsumexp[node_i];

    // compute G_i = sum_j alpha_{ij} · (grad_h_i · r_j)
    float G_i = 0.0f;

    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = d_col_idx[edge_start + k];
        const float4* r_ptr = reinterpret_cast<const float4*>(d_r + neighbor_j * z);

        // recompute e_ij = a^T @ LeakyReLU(l_i + r_j) :((((
        float e_ij = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_ptr[i];
            float4 a_val = attn_ptr[i];

            e_ij += leaky_relu_elementwise(l_val.x + r_val.x, negative_slope) * a_val.x
                 +  leaky_relu_elementwise(l_val.y + r_val.y, negative_slope) * a_val.y
                 +  leaky_relu_elementwise(l_val.z + r_val.z, negative_slope) * a_val.z
                 +  leaky_relu_elementwise(l_val.w + r_val.w, negative_slope) * a_val.w;
        }

        e_ij = warp_reduce_sum(e_ij);

        float alpha_ij = recompute_alpha(e_ij, L_i);

        // compute grad_h_i * r_j
        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 gh = grad_h_f4[i];
            float4 r = r_ptr[i];
            dot += gh.x * r.x + gh.y * r.y + gh.z * r.z + gh.w * r.w;
        }
        dot = warp_reduce_sum(dot);

        G_i += alpha_ij * dot;
    }

    if (lane_id == 0) {
        d_G[node_i] = G_i;
    }
}


__global__ void GATv2Backward_L(
    size_t N, size_t z,
    const float* __restrict__ grad_h,
    const float* __restrict__ d_l,
    const float* __restrict__ d_r,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_attn_vec,
    const float* __restrict__ d_logsumexp,
    const float* __restrict__ d_G,
    float* __restrict__ grad_l,
    float negative_slope
) {
    extern __shared__ float shared[];
    float* l_shared = shared;
    float* grad_h_shared = l_shared + z;

    int node_i = blockIdx.x;
    int lane_id = threadIdx.x;

    if (node_i >= N) return;

    int edge_start = d_row_ptr[node_i];
    int edge_end = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;

    if (num_neighbors == 0) {
        return;
    }

    const float4* attn_ptr = reinterpret_cast<const float4*>(d_attn_vec);
    int num_float4 = z / 4;

    {
        const float4* l_ptr = reinterpret_cast<const float4*>(d_l + node_i * z);
        const float4* gh_ptr = reinterpret_cast<const float4*>(grad_h + node_i * z);
        float4* l_shared_f4 = reinterpret_cast<float4*>(l_shared);
        float4* gh_shared_f4 = reinterpret_cast<float4*>(grad_h_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            l_shared_f4[i] = l_ptr[i];
            gh_shared_f4[i] = gh_ptr[i];
        }
    }
    __syncthreads();

    const float4* l_f4 = reinterpret_cast<const float4*>(l_shared);
    const float4* grad_h_f4 = reinterpret_cast<const float4*>(grad_h_shared);
    float L_i = d_logsumexp[node_i];
    float G_i = d_G[node_i];

    int float4_per_thread = (num_float4 + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;
    float4 grad_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        grad_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = d_col_idx[edge_start + k];
        const float4* r_ptr = reinterpret_cast<const float4*>(d_r + neighbor_j * z);

        float e_ij = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_ptr[i];
            float4 a_val = attn_ptr[i];

            float4 z;
            z.x = l_val.x + r_val.x;
            z.y = l_val.y + r_val.y;
            z.z = l_val.z + r_val.z;
            z.w = l_val.w + r_val.w;

            z.x = leaky_relu_elementwise(z.x, negative_slope);
            z.y = leaky_relu_elementwise(z.y, negative_slope);
            z.z = leaky_relu_elementwise(z.z, negative_slope);
            z.w = leaky_relu_elementwise(z.w, negative_slope);

            e_ij += z.x * a_val.x + z.y * a_val.y + z.z * a_val.z + z.w * a_val.w;
        }
        e_ij = warp_reduce_sum(e_ij);

        float alpha_ij = recompute_alpha(e_ij, L_i);

        // grad_h_i * r_j
        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 gh = grad_h_f4[i];
            float4 r = r_ptr[i];
            dot += gh.x * r.x + gh.y * r.y + gh.z * r.z + gh.w * r.w;
        }
        dot = warp_reduce_sum(dot);

        // grad_e_ij = α_{ij} * (grad_h_i * r_j - G_i)
        float grad_e_ij = alpha_ij * (dot - G_i);

        for (int i = 0; i < float4_per_thread; ++i) {
            int idx = lane_id + i * kMaxThreadsInWarp;
            if (idx < num_float4) {
                float4 l_val = l_f4[idx];
                float4 r_val = r_ptr[idx];
                float4 a_val = attn_ptr[idx];

                float4 z;
                z.x = l_val.x + r_val.x;
                z.y = l_val.y + r_val.y;
                z.z = l_val.z + r_val.z;
                z.w = l_val.w + r_val.w;

                float4 sigma_prime;
                sigma_prime.x = (z.x > 0.0f) ? 1.0f : negative_slope;
                sigma_prime.y = (z.y > 0.0f) ? 1.0f : negative_slope;
                sigma_prime.z = (z.z > 0.0f) ? 1.0f : negative_slope;
                sigma_prime.w = (z.w > 0.0f) ? 1.0f : negative_slope;

                grad_acc[i].x += grad_e_ij * a_val.x * sigma_prime.x;
                grad_acc[i].y += grad_e_ij * a_val.y * sigma_prime.y;
                grad_acc[i].z += grad_e_ij * a_val.z * sigma_prime.z;
                grad_acc[i].w += grad_e_ij * a_val.w * sigma_prime.w;
            }
        }
    }

    float4* grad_l_ptr = reinterpret_cast<float4*>(grad_l + node_i * z);
    for (int i = 0; i < float4_per_thread; ++i) {
        int idx = lane_id + i * kMaxThreadsInWarp;
        if (idx < num_float4) {
            grad_l_ptr[idx] = grad_acc[i];
        }
    }
}


__global__ void GATv2Backward_R(
    size_t N, size_t z,
    const float* __restrict__ grad_h,
    const float* __restrict__ d_l,
    const float* __restrict__ d_r,
    const int* __restrict__ d_row_ptr_T,    // Transposed graph
    const int* __restrict__ d_col_idx_T,    // Source nodes (incoming edges)
    const float* __restrict__ d_attn_vec,
    const float* __restrict__ d_logsumexp,
    const float* __restrict__ d_G,          // G_i indexed by source node
    float* __restrict__ grad_r,
    float negative_slope
) {
    extern __shared__ float shared[];
    float* r_shared = shared;

    int node_j = blockIdx.x;  // Current node receiving gradients
    int lane_id = threadIdx.x;

    if (node_j >= N) return;

    int edge_start = d_row_ptr_T[node_j];
    int edge_end = d_row_ptr_T[node_j + 1];
    int num_incoming = edge_end - edge_start;

    const float4* attn_ptr = reinterpret_cast<const float4*>(d_attn_vec);
    int num_float4 = z / 4;

    // Load r_j into shared memory
    {
        const float4* r_ptr = reinterpret_cast<const float4*>(d_r + node_j * z);
        float4* r_shared_f4 = reinterpret_cast<float4*>(r_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            r_shared_f4[i] = r_ptr[i];
        }
    }
    __syncthreads();

    const float4* r_f4 = reinterpret_cast<const float4*>(r_shared);

    // init gradient accumulator
    int float4_per_thread = (num_float4 + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;
    float4 grad_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        grad_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // process each incoming edge (i -> j)
    for (int k = 0; k < num_incoming; ++k) {
        int node_i = d_col_idx_T[edge_start + k];  // source node

        const float4* l_ptr = reinterpret_cast<const float4*>(d_l + node_i * z);
        const float4* grad_h_ptr = reinterpret_cast<const float4*>(grad_h + node_i * z);

        float L_i = d_logsumexp[node_i];
        float G_i = d_G[node_i];

        // recompute e_ij = a^T @ LeakyReLU(l_i + r_j)
        float e_ij = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_ptr[i];
            float4 r_val = r_f4[i];
            float4 a_val = attn_ptr[i];

            float4 z;
            z.x = l_val.x + r_val.x;
            z.y = l_val.y + r_val.y;
            z.z = l_val.z + r_val.z;
            z.w = l_val.w + r_val.w;

            z.x = leaky_relu_elementwise(z.x, negative_slope);
            z.y = leaky_relu_elementwise(z.y, negative_slope);
            z.z = leaky_relu_elementwise(z.z, negative_slope);
            z.w = leaky_relu_elementwise(z.w, negative_slope);

            e_ij += z.x * a_val.x + z.y * a_val.y + z.z * a_val.z + z.w * a_val.w;
        }
        e_ij = warp_reduce_sum(e_ij);

        // recompute α_{ij}
        float alpha_ij = __expf(e_ij - L_i);

        // compute grad_h_i * r_j
        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 gh = grad_h_ptr[i];
            float4 r = r_f4[i];
            dot += gh.x * r.x + gh.y * r.y + gh.z * r.z + gh.w * r.w;
        }
        dot = warp_reduce_sum(dot);

        // grad_e_ij = α_{ij} * (grad_h_i · r_j - G_i)
        float grad_e_ij = alpha_ij * (dot - G_i);

        // Accumulate both gradient paths
        for (int i = 0; i < float4_per_thread; ++i) {
            int idx = lane_id + i * kMaxThreadsInWarp;
            if (idx < num_float4) {
                float4 l_val = l_ptr[idx];
                float4 r_val = r_f4[idx];
                float4 a_val = attn_ptr[idx];
                float4 gh = grad_h_ptr[idx];

                float4 z;
                z.x = l_val.x + r_val.x;
                z.y = l_val.y + r_val.y;
                z.z = l_val.z + r_val.z;
                z.w = l_val.w + r_val.w;

                float4 sigma_prime;
                sigma_prime.x = (z.x > 0.0f) ? 1.0f : negative_slope;
                sigma_prime.y = (z.y > 0.0f) ? 1.0f : negative_slope;
                sigma_prime.z = (z.z > 0.0f) ? 1.0f : negative_slope;
                sigma_prime.w = (z.w > 0.0f) ? 1.0f : negative_slope;

                // 1: attention path
                float4 grad_attn;
                grad_attn.x = grad_e_ij * a_val.x * sigma_prime.x;
                grad_attn.y = grad_e_ij * a_val.y * sigma_prime.y;
                grad_attn.z = grad_e_ij * a_val.z * sigma_prime.z;
                grad_attn.w = grad_e_ij * a_val.w * sigma_prime.w;

                // 2: direct aggregation
                float4 grad_direct;
                grad_direct.x = alpha_ij * gh.x;
                grad_direct.y = alpha_ij * gh.y;
                grad_direct.z = alpha_ij * gh.z;
                grad_direct.w = alpha_ij * gh.w;

                grad_acc[i].x += grad_attn.x + grad_direct.x;
                grad_acc[i].y += grad_attn.y + grad_direct.y;
                grad_acc[i].z += grad_attn.z + grad_direct.z;
                grad_acc[i].w += grad_attn.w + grad_direct.w;
            }
        }
    }

    float4* grad_r_ptr = reinterpret_cast<float4*>(grad_r + node_j * z);
    for (int i = 0; i < float4_per_thread; ++i) {
        int idx = lane_id + i * kMaxThreadsInWarp;
        if (idx < num_float4) {
            grad_r_ptr[idx] = grad_acc[i];
        }
    }
}

__global__ void GATv2Backward_A(
    size_t N, size_t z,
    const float* __restrict__ grad_h,
    const float* __restrict__ d_l,
    const float* __restrict__ d_r,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_attn_vec,    // Need this!
    const float* __restrict__ d_logsumexp,
    const float* __restrict__ d_G,
    float* __restrict__ grad_a,              // [z] - OUTPUT
    float negative_slope
) {
    extern __shared__ float shared[];
    float* l_shared = shared;
    float* grad_h_shared = l_shared + z;
    float* grad_a_local = grad_h_shared + z;  // [z] - local accumulator

    int node_i = blockIdx.x;
    int lane_id = threadIdx.x;

    int num_float4 = z / 4;

    // Initialize local grad_a accumulator
    for (int i = lane_id; i < z; i += kMaxThreadsInWarp) {
        grad_a_local[i] = 0.0f;
    }
    __syncthreads();


    int edge_start = d_row_ptr[node_i];
    int edge_end = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;


    const float4* attn_ptr = reinterpret_cast<const float4*>(d_attn_vec);

    // Load l_i and grad_h_i
    {
        const float4* l_ptr = reinterpret_cast<const float4*>(d_l + node_i * z);
        const float4* gh_ptr = reinterpret_cast<const float4*>(grad_h + node_i * z);
        float4* l_shared_f4 = reinterpret_cast<float4*>(l_shared);
        float4* gh_shared_f4 = reinterpret_cast<float4*>(grad_h_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            l_shared_f4[i] = l_ptr[i];
            gh_shared_f4[i] = gh_ptr[i];
        }
    }
    __syncthreads();

    const float4* l_f4 = reinterpret_cast<const float4*>(l_shared);
    const float4* grad_h_f4 = reinterpret_cast<const float4*>(grad_h_shared);
    float L_i = d_logsumexp[node_i];
    float G_i = d_G[node_i];

    // grad_a = Σ_{all edges} grad_e_ij · s_ij
    // where grad_e_ij = α_{ij}(grad_h_i · r_j - G_i)

    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = d_col_idx[edge_start + k];
        const float4* r_ptr = reinterpret_cast<const float4*>(d_r + neighbor_j * z);

        // Compute e_ij and s_ij = LeakyReLU(l_i + r_j)
        float e_ij = 0.0f;

        // First pass: compute e_ij
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_ptr[i];
            float4 a_val = attn_ptr[i];

            float4 z;
            z.x = l_val.x + r_val.x;
            z.y = l_val.y + r_val.y;
            z.z = l_val.z + r_val.z;
            z.w = l_val.w + r_val.w;

            float4 s;
            s.x = leaky_relu_elementwise(z.x, negative_slope);
            s.y = leaky_relu_elementwise(z.y, negative_slope);
            s.z = leaky_relu_elementwise(z.z, negative_slope);
            s.w = leaky_relu_elementwise(z.w, negative_slope);

            e_ij += s.x * a_val.x + s.y * a_val.y + s.z * a_val.z + s.w * a_val.w;
        }
        e_ij = warp_reduce_sum(e_ij);

        // Recompute α_{ij}
        float alpha_ij = recompute_alpha(e_ij, L_i);

        // Compute grad_h_i * r_j
        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 gh = grad_h_f4[i];
            float4 r = r_ptr[i];
            dot += gh.x * r.x + gh.y * r.y + gh.z * r.z + gh.w * r.w;
        }
        dot = warp_reduce_sum(dot);

        // grad_e_ij
        float grad_e_ij = alpha_ij * (dot - G_i);

        // Second pass: accumulate grad_e_ij · s_ij to grad_a_local
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_ptr[i];

            float4 z;
            z.x = l_val.x + r_val.x;
            z.y = l_val.y + r_val.y;
            z.z = l_val.z + r_val.z;
            z.w = l_val.w + r_val.w;

            float4 s;
            s.x = leaky_relu_elementwise(z.x, negative_slope);
            s.y = leaky_relu_elementwise(z.y, negative_slope);
            s.z = leaky_relu_elementwise(z.z, negative_slope);
            s.w = leaky_relu_elementwise(z.w, negative_slope);


            float* grad_a_ptr = grad_a_local + i * 4;
            atomicAdd(&grad_a_ptr[0], grad_e_ij * s.x);
            atomicAdd(&grad_a_ptr[1], grad_e_ij * s.y);
            atomicAdd(&grad_a_ptr[2], grad_e_ij * s.z);
            atomicAdd(&grad_a_ptr[3], grad_e_ij * s.w);
        }
    }

    __syncthreads();

    for (int i = lane_id; i < z; i += kMaxThreadsInWarp) {
        atomicAdd(&grad_a[i], grad_a_local[i]);
    }
}


// =============================================================================
// Launcher for backward pass
// =============================================================================

void GATv2Backward_CSR(
    size_t N, size_t z,
    const float* grad_h,
    const float* d_l,
    const float* d_r,
    const int* d_row_ptr,
    const int* d_col_idx,
    const int* d_row_ptr_T,
    const int* d_col_idx_T,
    const float* d_attn_vec,
    const float* d_logsumexp,
    float* grad_l,
    float* grad_r,
    float* grad_a,
    float negative_slope,
    cudaStream_t stream = 0
) {
    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N);

    // Step 1: Precompute G_i
    float* d_G;
    CUDA_CHECK(cudaMalloc(&d_G, N * sizeof(float)));

    size_t shared_G = 2 * z * sizeof(float);  // l_i + grad_h_i
    PrecomputeG<<<nBlocks, nThreads, shared_G, stream>>>(
        N, z, grad_h, d_r, d_row_ptr, d_col_idx, d_l, d_attn_vec,
        d_logsumexp, d_G, negative_slope
    );

    // Step 2: Compute grad_l
    size_t shared_L = 2 * z * sizeof(float);
    GATv2Backward_L<<<nBlocks, nThreads, shared_L, stream>>>(
        N, z, grad_h, d_l, d_r, d_row_ptr, d_col_idx, d_attn_vec,
        d_logsumexp, d_G, grad_l, negative_slope
    );

    // Step 3: Compute grad_r (uses transposed graph)
    size_t shared_R = z * sizeof(float);  // r_j only
    GATv2Backward_R<<<nBlocks, nThreads, shared_R, stream>>>(
        N, z, grad_h, d_l, d_r, d_row_ptr_T, d_col_idx_T, d_attn_vec,
        d_logsumexp, d_G, grad_r, negative_slope
    );

    // Step 4: Compute grad_a
    size_t shared_A = 3 * z * sizeof(float);  // l_i + grad_h_i + grad_a_local
    GATv2Backward_A<<<nBlocks, nThreads, shared_A, stream>>>(
        N, z, grad_h, d_l, d_r, d_row_ptr, d_col_idx, d_attn_vec,
        d_logsumexp, d_G, grad_a, negative_slope
    );

    CUDA_CHECK(cudaFree(d_G));
}
