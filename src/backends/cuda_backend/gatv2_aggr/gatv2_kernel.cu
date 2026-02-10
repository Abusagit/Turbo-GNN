#include <cstddef>
#include <torch/extension.h>
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

__device__ __forceinline__ float leaky_relu_der_elementwise(float x, float negative_slope) {
    return (x > 0.0f) ? 1.0f : negative_slope;
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

__device__ __forceinline__ int warp_bcast_i32(int v0) {
    return __shfl_sync(FULL_WARP_MASK, v0, 0);
}

__device__ __forceinline__ float warp_bcast_f32(float v0) {
    return __shfl_sync(FULL_WARP_MASK, v0, 0);
}

__device__ __forceinline__ float dot_product_f4(float4 a, float4 b) {
    float acc = 0.f;
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    acc = fmaf(a.w, b.w, acc);
    return acc;
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
    size_t H,
    size_t D,
    const float* __restrict__ d_l,          // [N, H, D] - left features
    const float* __restrict__ d_r,          // [N, H, D] - right features

    int64_t stride_l_n,
    int64_t stride_l_h,
    int64_t stride_r_n,
    int64_t stride_r_h,

    const int* __restrict__ d_row_ptr,      // [N+1] - CSR row pointers
    const int* __restrict__ d_col_idx,      // [E] - CSR column indices (neighbor IDs)
    const float* __restrict__ d_attn_vec,   // [H, D] - attention vector
    float* __restrict__ d_h_out,            // [N, H, D] - output node features
    float* __restrict__ d_logsumexp_out,    // [N, H] -- logsumexp values (used for backward pass)
    float negative_slope
) {
    // shared memory layout:
    // [0, z): l_i cached
    extern __shared__ float shared[];
    float* l_shared = shared;

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;

    int lane_id = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= N || head_h >= H) {
        return;
    }

    // neighbor range from CSR
    int edge_start = d_row_ptr[node_i];
    int edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;

    float* h_out_base = d_h_out + ((node_i * H + head_h) * D);

    // skip isolated nodes (no neighbors)
    if (num_neighbors == 0) {
        float4* h_out_f4 = reinterpret_cast<float4*>(h_out_base);
        int num_float4 = D / 4;

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            h_out_f4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if (lane_id == 0) {
            d_logsumexp_out[node_i * H + head_h] = -INFINITY;
        }
        return;
    }

    const float* l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const float* a_base = d_attn_vec + head_h * D;
    const float4* a_f4 = reinterpret_cast<const float4*>(a_base);


    int num_float4 = D / 4;
    // ==========================================
    // 0: Load l_i into shared memory
    // ==========================================

    {
        const float4* l_src4 = reinterpret_cast<const float4*>(l_base);
        float4* l_shared4    = reinterpret_cast<float4*>(l_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            // because stride_l_d == 1, contiguous access is valid
            l_shared4[i] = l_src4[i];
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

        const float* r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;
        const float4* r_f4 = reinterpret_cast<const float4*>(r_base);

        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_f4[i];
            float4 a_val = a_f4[i];

            float4 sum;
            sum.x = leaky_relu_elementwise(l_val.x + r_val.x, negative_slope);
            sum.y = leaky_relu_elementwise(l_val.y + r_val.y, negative_slope);
            sum.z = leaky_relu_elementwise(l_val.z + r_val.z, negative_slope);
            sum.w = leaky_relu_elementwise(l_val.w + r_val.w, negative_slope);

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
            h_acc[i].x *= rescale;
            h_acc[i].y *= rescale;
            h_acc[i].z *= rescale;
            h_acc[i].w *= rescale;
        }

        float contribution = __expf(dot - softmax_state.max_val);

        for (int i = 0; i < float4_per_thread; ++i) {
            int idx4 = lane_id + i * kMaxThreadsInWarp;
            if (idx4 < num_float4) {
                float4 r_val = r_f4[idx4];
                h_acc[i].x += contribution * r_val.x;
                h_acc[i].y += contribution * r_val.y;
                h_acc[i].z += contribution * r_val.z;
                h_acc[i].w += contribution * r_val.w;
            }
        }
    }

    // write logsumexp value:
    if (lane_id == 0){
        d_logsumexp_out[node_i * H + head_h] = softmax_state.max_val + __logf(softmax_state.sum_exp);
    }

    float inv_sum = 1.0f / softmax_state.sum_exp;

    float4* h_out_f4 = reinterpret_cast<float4*>(h_out_base);
    #pragma unroll
    for (int i = 0; i < float4_per_thread; ++i) {
        int idx4 = lane_id + i * kMaxThreadsInWarp;
        if (idx4 < num_float4) {
            float4 v = h_acc[i];
            v.x *= inv_sum;
            v.y *= inv_sum;
            v.z *= inv_sum;
            v.w *= inv_sum;
            h_out_f4[idx4] = v;
        }
    }
}


// =============================================================================
// scalar (non-float4) templated forward for D in {32,64}
// =============================================================================
template<int D_CONST>
__global__ void GATv2Kernel_CSR_Scalar_D(
    size_t N,
    size_t H,
    size_t D,
    const float* __restrict__ d_l,          // [N, H, D]
    const float* __restrict__ d_r,          // [N, H, D]
    int64_t stride_l_n,
    int64_t stride_l_h,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_attn_vec,   // [H, D]
    float* __restrict__ d_h_out,            // [N, H, D]
    float* __restrict__ d_logsumexp_out,    // [N, H]
    float negative_slope
) {

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= N || head_h >= H) return;

    int edge_start = d_row_ptr[node_i];
    int edge_end = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;
    if (num_neighbors == 0) return;

    const float* l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const float* a_base = d_attn_vec + head_h * D_CONST;
    float* h_out_base   = d_h_out + ((node_i * H + head_h) * D_CONST);

    // shared: cache l_i only
    extern __shared__ float sh[];
    float* l_sh = sh; // [D_CONST]

    for (int f = lane; f < D_CONST; f += kMaxThreadsInWarp) {
        l_sh[f] = __ldg(&l_base[f]);
    }
    __syncthreads();

    constexpr int kPerLane = (D_CONST + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp; // 1,2,4,8 for 32/64/128/256
    float h_acc[kPerLane];
    #pragma unroll
    for (int t = 0; t < kPerLane; ++t) h_acc[t] = 0.f;

    OnlineSoftmaxState softmax_state;

    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = __ldg(&d_col_idx[edge_start + k]);

        const float* r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        // dot lane partial
        float dot_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < kPerLane; ++t) {
            int f = lane + kMaxThreadsInWarp * t;
            if (f < D_CONST) {
                float s = leaky_relu_elementwise(l_sh[f] + __ldg(&r_base[f]), negative_slope);
                float a = __ldg(&a_base[f]);
                dot_lane = fmaf(s, a, dot_lane);
            }
        }
        float dot = warp_reduce_sum(dot_lane);

        float rescale = softmax_state.update(dot);
        #pragma unroll
        for (int t = 0; t < kPerLane; ++t) h_acc[t] *= rescale;

        float contrib = __expf(dot - softmax_state.max_val);
        #pragma unroll
        for (int t = 0; t < kPerLane; ++t) {
            int f = lane + kMaxThreadsInWarp * t;
            if (f < D_CONST) {
                h_acc[t] = fmaf(contrib, __ldg(&r_base[f]), h_acc[t]);
            }
        }
    }

    if (lane == 0) {
        d_logsumexp_out[(int64_t)node_i * (int64_t)H + head_h] =
            softmax_state.max_val + __logf(softmax_state.sum_exp);
    }

    float inv_sum = 1.f / softmax_state.sum_exp;
    #pragma unroll
    for (int t = 0; t < kPerLane; ++t) {
        int f = lane + kMaxThreadsInWarp * t;
        if (f < D_CONST) h_out_base[f] = h_acc[t] * inv_sum;
    }
}



template<int D_CONST>
__global__ void GATv2Kernel_CSR_D(
    size_t N,
    size_t H,
    size_t D, // kept only so the call site stays identical
    const float* __restrict__ d_l,
    const float* __restrict__ d_r,
    int64_t stride_l_n,
    int64_t stride_l_h,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_attn_vec,
    float* __restrict__ d_h_out,
    float* __restrict__ d_logsumexp_out,
    float negative_slope
) {
    static_assert(D_CONST % 4 == 0, "D_CONST must be divisible by 4");
    static_assert(D_CONST <= 1024, "D_CONST must be <= 1024");

    constexpr int num_float4 = D_CONST / 4;
    constexpr int float4_per_thread = (num_float4 + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp; // 1..8

    extern __shared__ float shared[];
    float* l_shared = shared;

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane_id = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= (int)N || head_h >= (int)H) return;

    int edge_start = d_row_ptr[node_i];
    int edge_end   = d_row_ptr[node_i + 1];

    int num_neighbors = edge_end - edge_start;

    float* h_out_base = d_h_out + ((node_i * H + head_h) * D_CONST);
    // skip isolated nodes (no neighbors)
    if (num_neighbors == 0) {
        float4* h_out_f4 = reinterpret_cast<float4*>(h_out_base);
        int num_float4 = D_CONST / 4;

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            h_out_f4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if (lane_id == 0) {
            d_logsumexp_out[node_i * H + head_h] = -INFINITY;
        }
        return;
    }

    const float* l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const float* a_base = d_attn_vec + head_h * D_CONST;

    const float4* a_f4 = reinterpret_cast<const float4*>(a_base);

    {
        const float4* l_src4 = reinterpret_cast<const float4*>(l_base);
        float4* l_sh4        = reinterpret_cast<float4*>(l_shared);
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            l_sh4[i] = l_src4[i];
        }
    }
    __syncthreads();

    const float4* l_f4 = reinterpret_cast<const float4*>(l_shared);
    OnlineSoftmaxState softmax_state;

    float4 h_acc[float4_per_thread];
    #pragma unroll
    for (int i = 0; i < float4_per_thread; ++i) {
        h_acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = 0;
        if (lane_id == 0) {
            neighbor_j = __ldg(&d_col_idx[edge_start + k]);
        }
        neighbor_j = warp_bcast_i32(neighbor_j);

        const float* r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;
        const float4* r_f4 = reinterpret_cast<const float4*>(r_base);

        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_f4[i];
            float4 a_val = a_f4[i];

            float4 sum;
            sum.x = leaky_relu_elementwise(l_val.x + r_val.x, negative_slope);
            sum.y = leaky_relu_elementwise(l_val.y + r_val.y, negative_slope);
            sum.z = leaky_relu_elementwise(l_val.z + r_val.z, negative_slope);
            sum.w = leaky_relu_elementwise(l_val.w + r_val.w, negative_slope);

            // FMA chain (dot += sum * a)
            dot = fmaf(sum.x, a_val.x, dot);
            dot = fmaf(sum.y, a_val.y, dot);
            dot = fmaf(sum.z, a_val.z, dot);
            dot = fmaf(sum.w, a_val.w, dot);
        }

        dot = warp_reduce_sum(dot);

        float rescale = softmax_state.update(dot);

        // rescale previous accumulator
        #pragma unroll
        for (int i = 0; i < float4_per_thread; ++i) {
            h_acc[i].x *= rescale;
            h_acc[i].y *= rescale;
            h_acc[i].z *= rescale;
            h_acc[i].w *= rescale;
        }

        float contrib = __expf(dot - softmax_state.max_val);

        #pragma unroll
        for (int i = 0; i < float4_per_thread; ++i) {
            int idx4 = lane_id + i * kMaxThreadsInWarp;
            if (idx4 < num_float4) {
                float4 r_val = r_f4[idx4];
                h_acc[i].x = fmaf(contrib, r_val.x, h_acc[i].x);
                h_acc[i].y = fmaf(contrib, r_val.y, h_acc[i].y);
                h_acc[i].z = fmaf(contrib, r_val.z, h_acc[i].z);
                h_acc[i].w = fmaf(contrib, r_val.w, h_acc[i].w);
            }
        }
    }

    // logsumexp
    if (lane_id == 0) {
        d_logsumexp_out[node_i * H + head_h] = softmax_state.max_val + __logf(softmax_state.sum_exp);
    }

    float inv_sum = 1.0f / softmax_state.sum_exp;

    float4* h_out_f4 = reinterpret_cast<float4*>(h_out_base);
    #pragma unroll
    for (int i = 0; i < float4_per_thread; ++i) {
        int idx4 = lane_id + i * kMaxThreadsInWarp;
        if (idx4 < num_float4) {
            float4 v = h_acc[i];
            v.x *= inv_sum;
            v.y *= inv_sum;
            v.z *= inv_sum;
            v.w *= inv_sum;
            h_out_f4[idx4] = v;
        }
    }
}


// FlashAttention2-like logsumexp trick
__device__ __forceinline__ float recompute_alpha(
    float e_ij,          // logit
    float L_i            // saved log-sum-exp
) {
    return __expf(e_ij - L_i);
}


__global__ void GATv2Backward_AL(
    // inputs
    size_t N, size_t H, size_t D,
    const float* __restrict__ grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,


    const float* __restrict__ d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,


    const float* __restrict__ d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,

    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,  // [N, H]
    float negative_slope,

    // outputs
    float* __restrict__ grad_a,  // [N, H, D]
    float* __restrict__ grad_l,  // [N, H, D]
    float* __restrict__ d_G      // [N, H]
) {
    int node_i = blockIdx.x;    // 0..N-1
    int head_h = blockIdx.y;  // 0..H-1
    int lane_id = threadIdx.x;

    if (node_i >= N || head_h >= H) {
        return;
    }

    int edge_start = 0, edge_end = 0;
    if (lane_id == 0) {
        edge_start = __ldg(&d_row_ptr[node_i]);
        edge_end   = __ldg(&d_row_ptr[node_i + 1]);
    }
    edge_start = warp_bcast_i32(edge_start);
    edge_end   = warp_bcast_i32(edge_end);

    float L_i = 0.f;
    if (lane_id == 0) {
        L_i = __ldg(&d_logsumexp[node_i * H + head_h]);
    }
    L_i = warp_bcast_f32(L_i);

    int num_neighbors = edge_end - edge_start;

    int num_float4 = D / 4;

    extern __shared__ float shared[];
    float* li_shared          = shared;
    float* grad_hi_shared     = li_shared + D;
    float* grad_a_shared      = grad_hi_shared + D;
    float* grad_li_shared     = grad_a_shared + D;

    float4* li_shared_f4      = reinterpret_cast<float4*>(li_shared);
    float4* grad_hi_shared_f4 = reinterpret_cast<float4*>(grad_hi_shared);
    float4* grad_a_shared_f4  = reinterpret_cast<float4*>(grad_a_shared);
    float4* grad_li_shared_f4 = reinterpret_cast<float4*>(grad_li_shared);


    const float* li_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const float* grad_hi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;

    const float* a_base = d_attn_vec + head_h * D;
    const float4* a_f4 = reinterpret_cast<const float4*>(a_base);

    for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
        grad_a_shared_f4[f_idx_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        grad_li_shared_f4[f_idx_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (num_neighbors > 0) {
        // Copy l_{i, h} and grad_h_{i, h} to shared memory
        const float4* li_f4      = reinterpret_cast<const float4*>(li_base);
        const float4* grad_hi_f4 = reinterpret_cast<const float4*>(grad_hi_base);

        // feature dim contiguous, so float4 loads are valid
        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            li_shared_f4[f_idx_f4]      = li_f4[f_idx_f4];
            grad_hi_shared_f4[f_idx_f4] = grad_hi_f4[f_idx_f4];
        }
    }
    __syncthreads();

    // First pass: compute G_{i, h} = sum_j alpha_ij * <grad_h_{i, h}, r_{j, h}>
    float G_i_h = 0.0f;

    for (int neighbor_idx = 0; neighbor_idx < num_neighbors; ++neighbor_idx) {
        int neighbor_j = 0;
        if (lane_id == 0) {
            neighbor_j = __ldg(&d_col_idx[edge_start + neighbor_idx]);
        }
        neighbor_j = warp_bcast_i32(neighbor_j);


        const float* rj_base = d_r + neighbor_j * stride_r_n + head_h    * stride_r_h;
        const float4* rj_f4 = reinterpret_cast<const float4*>(rj_base);

        float e_ij_cum = 0.0f;
        float p_ij_cum = 0.0f;

        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            float4 li_val      = li_shared_f4[f_idx_f4];
            float4 rj_val      = rj_f4[f_idx_f4];
            float4 a_val       = a_f4[f_idx_f4];
            float4 grad_hi_val = grad_hi_shared_f4[f_idx_f4];

            float4 t_ij_val = make_float4(
                leaky_relu_elementwise(li_val.x + rj_val.x, negative_slope),
                leaky_relu_elementwise(li_val.y + rj_val.y, negative_slope),
                leaky_relu_elementwise(li_val.z + rj_val.z, negative_slope),
                leaky_relu_elementwise(li_val.w + rj_val.w, negative_slope)
            );

            e_ij_cum += dot_product_f4(t_ij_val, a_val);

            p_ij_cum += dot_product_f4(grad_hi_val, rj_val);
        }

        float e_ij = warp_reduce_sum(e_ij_cum);
        float alpha_ij = recompute_alpha(e_ij, L_i);
        float p_ij = warp_reduce_sum(p_ij_cum);

        // G_i_h += alpha_ij * p_ij;
        if (lane_id == 0) {
            G_i_h = fmaf(alpha_ij, p_ij, G_i_h);
        }

    }
    G_i_h = warp_bcast_f32(G_i_h);

    // Second pass: compute gradients using G_{i, h}
    for (int neighbor_idx = 0; neighbor_idx < num_neighbors; ++neighbor_idx) {
        // int neighbor_j = d_col_idx[edge_start + neighbor_idx];
        int neighbor_j = 0;
        if (lane_id == 0) {
            neighbor_j = __ldg(&d_col_idx[edge_start + neighbor_idx]);
        }
        neighbor_j = warp_bcast_i32(neighbor_j);


        const float* rj_base = d_r + neighbor_j * stride_r_n + head_h    * stride_r_h;
        const float4* rj_f4 = reinterpret_cast<const float4*>(rj_base);

        float e_ij_cum = 0.0f;
        float p_ij_cum = 0.0f;

        // Recompute e_ij and p_ij
        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            float4 li_val      = li_shared_f4[f_idx_f4];
            float4 rj_val      = rj_f4[f_idx_f4];
            float4 a_val       = a_f4[f_idx_f4];
            float4 grad_hi_val = grad_hi_shared_f4[f_idx_f4];

            float4 t_ij_val = make_float4(
                leaky_relu_elementwise(li_val.x + rj_val.x, negative_slope),
                leaky_relu_elementwise(li_val.y + rj_val.y, negative_slope),
                leaky_relu_elementwise(li_val.z + rj_val.z, negative_slope),
                leaky_relu_elementwise(li_val.w + rj_val.w, negative_slope)
            );

            e_ij_cum += dot_product_f4(t_ij_val, a_val);

            p_ij_cum += dot_product_f4(grad_hi_val, rj_val);
        }

        // float e_ij = warp_reduce_sum(e_ij_cum);
        // float alpha_ij = recompute_alpha(e_ij, L_i);
        // float p_ij = warp_reduce_sum(p_ij_cum);
        // float grad_e_ij = alpha_ij * (p_ij - G_i_h);
        float e_ij = warp_reduce_sum(e_ij_cum);
        float p_ij = warp_reduce_sum(p_ij_cum);

        float alpha_ij = 0.f, grad_e_ij = 0.f;
        if (lane_id == 0) {
            alpha_ij  = recompute_alpha(e_ij, L_i);
            grad_e_ij = alpha_ij * (p_ij - G_i_h);
        }
        alpha_ij  = warp_bcast_f32(alpha_ij);
        grad_e_ij = warp_bcast_f32(grad_e_ij);

        // Accumulate gradients
        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            float4 li_val = li_shared_f4[f_idx_f4];
            float4 rj_val = rj_f4[f_idx_f4];
            float4 a_val  = a_f4[f_idx_f4];

            float4 edge_ij = make_float4(
                li_val.x + rj_val.x,
                li_val.y + rj_val.y,
                li_val.z + rj_val.z,
                li_val.w + rj_val.w
            );

            float4 t_ij_der_val = make_float4(
                leaky_relu_der_elementwise(edge_ij.x, negative_slope),
                leaky_relu_der_elementwise(edge_ij.y, negative_slope),
                leaky_relu_der_elementwise(edge_ij.z, negative_slope),
                leaky_relu_der_elementwise(edge_ij.w, negative_slope)
            );

            float4 t_ij_val = make_float4(
                t_ij_der_val.x * edge_ij.x,
                t_ij_der_val.y * edge_ij.y,
                t_ij_der_val.z * edge_ij.z,
                t_ij_der_val.w * edge_ij.w
            );

            // Accumulate grad_a
            float4 grad_a_shared_val = grad_a_shared_f4[f_idx_f4];
            grad_a_shared_val.x = fmaf(grad_e_ij, t_ij_val.x, grad_a_shared_val.x);
            grad_a_shared_val.y = fmaf(grad_e_ij, t_ij_val.y, grad_a_shared_val.y);
            grad_a_shared_val.z = fmaf(grad_e_ij, t_ij_val.z, grad_a_shared_val.z);
            grad_a_shared_val.w = fmaf(grad_e_ij, t_ij_val.w, grad_a_shared_val.w);

            grad_a_shared_f4[f_idx_f4] = grad_a_shared_val;

            // Accumulate grad_li
            float4 grad_li_shared_val = grad_li_shared_f4[f_idx_f4];
            grad_li_shared_val.x = fmaf(grad_e_ij * t_ij_der_val.x, a_val.x, grad_li_shared_val.x);
            grad_li_shared_val.y = fmaf(grad_e_ij * t_ij_der_val.y, a_val.y, grad_li_shared_val.y);
            grad_li_shared_val.z = fmaf(grad_e_ij * t_ij_der_val.z, a_val.z, grad_li_shared_val.z);
            grad_li_shared_val.w = fmaf(grad_e_ij * t_ij_der_val.w, a_val.w, grad_li_shared_val.w);
            grad_li_shared_f4[f_idx_f4] = grad_li_shared_val;
        }
    }

    __syncthreads();
    // write G_{i, h} to global memory (needed by R kernel)
    if (lane_id == 0) {
        d_G[node_i * H + head_h] = G_i_h;
    }

    // write grad_a and grad_l to global memory
    float4* grad_l_node_f4 = reinterpret_cast<float4*>(grad_l + ((node_i * H + head_h) * D));
    for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
        grad_l_node_f4[f_idx_f4] = grad_li_shared_f4[f_idx_f4];
    }

    float4* grad_a_node_f4 = reinterpret_cast<float4*>(grad_a + ((node_i * H + head_h) * D));
    for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
        grad_a_node_f4[f_idx_f4] = grad_a_shared_f4[f_idx_f4];
    }
}


__global__ void GATv2Backward_R(
    // inputs
    size_t N, size_t H, size_t D,

    const float* __restrict__ grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,

    const float* __restrict__ d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,

    const float* __restrict__ d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,

    const int* __restrict__ d_row_ptr_T,    // transposed graph
    const int* __restrict__ d_col_idx_T,    // source nodes (incoming edges)
    const float* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,  // [N, H]
    const float* __restrict__ d_G,          // [N, H]
    float negative_slope,

    // outputs
    float* __restrict__ grad_r              // [N, H, D]
) {
    int node_j = blockIdx.x; // destination node
    int head_h = blockIdx.y; // head

    int lane_id = threadIdx.x;

    if (node_j >= N || head_h >= H) {
        return;
    }


    int edge_start = d_row_ptr_T[node_j];
    int edge_end   = d_row_ptr_T[node_j + 1];
    int num_incoming = edge_end - edge_start;

    int num_float4 = D / 4;

    extern __shared__ float shared[];
    float* rj_shared      = shared;
    float* grad_rj_shared = rj_shared + D;

    float4* rj_shared_f4      = reinterpret_cast<float4*>(rj_shared);
    float4* grad_rj_shared_f4 = reinterpret_cast<float4*>(grad_rj_shared);


    const float* rj_base = d_r + node_j * stride_r_n + head_h * stride_r_h;
    const float* a_base = d_attn_vec + head_h * D;
    const float4* a_f4  = reinterpret_cast<const float4*>(a_base);

    // init output values
    for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
        grad_rj_shared_f4[f_idx_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (num_incoming > 0) {
        // copy r_j to shared memory
        const float4* rj_f4 = reinterpret_cast<const float4*>(rj_base);

        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            rj_shared_f4[f_idx_f4] = rj_f4[f_idx_f4];
        }
    }
    __syncthreads();

    // Process each incoming edge (i -> j)
    for (int incoming_idx = 0; incoming_idx < num_incoming; ++incoming_idx) {
        // int node_i = d_col_idx_T[edge_start + incoming_idx];
        int node_i = 0;
        if (lane_id == 0) node_i = __ldg(&d_col_idx_T[edge_start + incoming_idx]);
        node_i = warp_bcast_i32(node_i);


        const float* li_base      = d_l + node_i * stride_l_n + head_h * stride_l_h;
        const float* grad_hi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;

        const float4* li_f4       = reinterpret_cast<const float4*>(li_base);
        const float4* grad_hi_f4  = reinterpret_cast<const float4*>(grad_hi_base);

        // float L_i_h = d_logsumexp[node_i * H + head_h];
        // float G_i_h = d_G[node_i * H + head_h];
        float L_i_h = 0.f, G_i_h = 0.f;
        if (lane_id == 0) {
            L_i_h = __ldg(&d_logsumexp[(int64_t)node_i * H + head_h]);
            G_i_h = __ldg(&d_G[(int64_t)node_i * H + head_h]);
        }
        L_i_h = warp_bcast_f32(L_i_h);
        G_i_h = warp_bcast_f32(G_i_h);



        float e_ij_cum = 0.0f;

        float p_ij_cum = 0.0f;

        // compute e_ij and p_ij
        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            float4 li_val      = li_f4[f_idx_f4];
            float4 rj_val      = rj_shared_f4[f_idx_f4];
            float4 a_val       = a_f4[f_idx_f4];
            float4 grad_hi_val = grad_hi_f4[f_idx_f4];

            float4 t_ij_val = make_float4(
                leaky_relu_elementwise(li_val.x + rj_val.x, negative_slope),
                leaky_relu_elementwise(li_val.y + rj_val.y, negative_slope),
                leaky_relu_elementwise(li_val.z + rj_val.z, negative_slope),
                leaky_relu_elementwise(li_val.w + rj_val.w, negative_slope)
            );

            e_ij_cum += dot_product_f4(t_ij_val, a_val);

            p_ij_cum += dot_product_f4(grad_hi_val, rj_val);
        }

        float e_ij = warp_reduce_sum(e_ij_cum);
        float alpha_ij = recompute_alpha(e_ij, L_i_h);
        float p_ij = warp_reduce_sum(p_ij_cum);
        float grad_e_ij = alpha_ij * (p_ij - G_i_h);

        // accumulate both gradient paths
        for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
            float4 li_val      = li_f4[f_idx_f4];
            float4 rj_val      = rj_shared_f4[f_idx_f4];
            float4 a_val       = a_f4[f_idx_f4];
            float4 grad_hi_val = grad_hi_f4[f_idx_f4];

            float4 edge_ij = make_float4(
                li_val.x + rj_val.x,
                li_val.y + rj_val.y,
                li_val.z + rj_val.z,
                li_val.w + rj_val.w
            );

            float4 t_ij_der_val = make_float4(
                leaky_relu_der_elementwise(edge_ij.x, negative_slope),
                leaky_relu_der_elementwise(edge_ij.y, negative_slope),
                leaky_relu_der_elementwise(edge_ij.z, negative_slope),
                leaky_relu_der_elementwise(edge_ij.w, negative_slope)
            );

            // // 1: attention path
            // float4 grad_attn = make_float4(
            //     grad_e_ij * a_val.x * t_ij_der_val.x,
            //     grad_e_ij * a_val.y * t_ij_der_val.y,
            //     grad_e_ij * a_val.z * t_ij_der_val.z,
            //     grad_e_ij * a_val.w * t_ij_der_val.w
            // );

            // // 2: direct aggregation
            // float4 grad_direct = make_float4(
            //     alpha_ij * grad_hi_val.x,
            //     alpha_ij * grad_hi_val.y,
            //     alpha_ij * grad_hi_val.z,
            //     alpha_ij * grad_hi_val.w
            // );

            // Accumulate both paths
            float4 grad_rj_shared_val = grad_rj_shared_f4[f_idx_f4];

            // direct path: alpha * grad_hi
            grad_rj_shared_val.x = fmaf(alpha_ij, grad_hi_val.x, grad_rj_shared_val.x);
            grad_rj_shared_val.y = fmaf(alpha_ij, grad_hi_val.y, grad_rj_shared_val.y);
            grad_rj_shared_val.z = fmaf(alpha_ij, grad_hi_val.z, grad_rj_shared_val.z);
            grad_rj_shared_val.w = fmaf(alpha_ij, grad_hi_val.w, grad_rj_shared_val.w);

            // attention path: grad_e * a * t_der
            grad_rj_shared_val.x = fmaf(grad_e_ij * t_ij_der_val.x, a_val.x, grad_rj_shared_val.x);
            grad_rj_shared_val.y = fmaf(grad_e_ij * t_ij_der_val.y, a_val.y, grad_rj_shared_val.y);
            grad_rj_shared_val.z = fmaf(grad_e_ij * t_ij_der_val.z, a_val.z, grad_rj_shared_val.z);
            grad_rj_shared_val.w = fmaf(grad_e_ij * t_ij_der_val.w, a_val.w, grad_rj_shared_val.w);

            grad_rj_shared_f4[f_idx_f4] = grad_rj_shared_val;

        }
    }

    __syncthreads();
    // Write grad_r[j, h, :] to global memory
    float* grad_r_base = grad_r + ((node_j * H + head_h) * D);
    float4* grad_r_node_f4 = reinterpret_cast<float4*>(grad_r_base);
    for (int f_idx_f4 = lane_id; f_idx_f4 < num_float4; f_idx_f4 += kMaxThreadsInWarp) {
        grad_r_node_f4[f_idx_f4] = grad_rj_shared_f4[f_idx_f4];
    }
}

// =============================================================================
// Scalar (non-float4) templated backward for D in {32,64}
// =============================================================================
template<int D_CONST>
__global__ void GATv2Backward_AL_Scalar_D(
    size_t N, size_t H, size_t D,
    const float* __restrict__ grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,
    const float* __restrict__ d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,
    const float* __restrict__ d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,  // [N, H]
    float negative_slope,
    float* __restrict__ grad_a,  // [N, H, D]
    float* __restrict__ grad_l,  // [N, H, D]
    float* __restrict__ d_G      // [N, H]
) {
    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= N || head_h >= H) return;

    int edge_start = d_row_ptr[node_i];
    int edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;

    if (num_neighbors == 0) return;

    float L_i = d_logsumexp[node_i * H + head_h];

    const float* li_base      = d_l    + node_i * stride_l_n  + head_h * stride_l_h;
    const float* ghi_base     = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const float* a_base       = d_attn_vec + head_h * D_CONST;

    // shared: li, grad_hi, grad_a_acc, grad_li_acc
    extern __shared__ float sh[];
    float* li_sh     = sh;                  // [D_CONST]
    float* ghi_sh    = li_sh + D_CONST;     // [D_CONST]
    float* grada_sh  = ghi_sh + D_CONST;    // [D_CONST]
    float* gradli_sh = grada_sh + D_CONST;  // [D_CONST]

    for (int f = lane; f < D_CONST; f += kMaxThreadsInWarp) {
        li_sh[f]     = __ldg(&li_base[f]);
        ghi_sh[f]    = __ldg(&ghi_base[f]);
        grada_sh[f]  = 0.f;
        gradli_sh[f] = 0.f;
    }
    __syncthreads();

    constexpr int kPerLane = (D_CONST + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;

    // Pass 1: G_i_h = sum_j alpha_ij * <grad_hi, rj>
    float G_i_h = 0.f;
    for (int k = 0; k < num_neighbors; ++k) {

        int neighbor_j = d_col_idx[edge_start + k];

        const float* rj_base = d_r + (int64_t)neighbor_j * stride_r_n + (int64_t)head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < kPerLane; ++t) {
            int f = lane + 32 * t;
            if (f < D_CONST) {
                float rj = __ldg(&rj_base[f]);
                float tval = leaky_relu_elementwise(li_sh[f] + rj, negative_slope);
                float a = __ldg(&a_base[f]); // NO reg cache
                e_lane = fmaf(tval, a, e_lane);
                p_lane = fmaf(ghi_sh[f], rj, p_lane);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        if (lane == 0) {
            float alpha = recompute_alpha(e_ij, L_i);
            G_i_h = fmaf(alpha, p_ij, G_i_h);
        }
        G_i_h = warp_bcast_f32(G_i_h);
    }

    // Pass 2: accumulate grad_a and grad_l
    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = __ldg(&d_col_idx[edge_start + k]);

        const float* rj_base = d_r + (int64_t)neighbor_j * stride_r_n + (int64_t)head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < kPerLane; ++t) {
            int f = lane + 32 * t;
            if (f < D_CONST) {
                float rj = __ldg(&rj_base[f]);
                float tval = leaky_relu_elementwise(li_sh[f] + rj, negative_slope);
                float a = __ldg(&a_base[f]); // NO reg cache
                e_lane = fmaf(tval, a, e_lane);
                p_lane = fmaf(ghi_sh[f], rj, p_lane);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha = 0.f, grad_e = 0.f;
        if (lane == 0) {
            alpha  = recompute_alpha(e_ij, L_i);
            grad_e = alpha * (p_ij - G_i_h);
        }
        alpha  = warp_bcast_f32(alpha);
        grad_e = warp_bcast_f32(grad_e);

        for (int f = lane; f < D_CONST; f += 32) {
            float rj   = __ldg(&rj_base[f]);
            float edge = li_sh[f] + rj;
            float tder = leaky_relu_der_elementwise(edge, negative_slope);

            // Keep EXACT original algebra from your float4 AL kernel:
            // grad_a uses t_ij_val = t_der * edge
            float t_ij_val = tder * edge;
            grada_sh[f] = fmaf(grad_e, t_ij_val, grada_sh[f]);

            // grad_li uses grad_e * t_der * a[f]
            float a = __ldg(&a_base[f]); // NO reg cache
            gradli_sh[f] = fmaf(grad_e * tder, a, gradli_sh[f]);
        }
    }

    __syncthreads();
    if (lane == 0) d_G[node_i * H + head_h] = G_i_h;

    float* grad_l_base = grad_l + ((node_i * H + head_h) * D_CONST);
    float* grad_a_base = grad_a + ((node_i * H + head_h) * D_CONST);
    for (int f = lane; f < D_CONST; f += kMaxThreadsInWarp) {
        grad_l_base[f] = gradli_sh[f];
        grad_a_base[f] = grada_sh[f];
    }
}

template<int D_CONST>
__global__ void GATv2Backward_R_Scalar_D(
    size_t N, size_t H, size_t D,
    const float* __restrict__ grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,
    const float* __restrict__ d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,
    const float* __restrict__ d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const int* __restrict__ d_row_ptr_T,
    const int* __restrict__ d_col_idx_T,
    const float* __restrict__ d_attn_vec,
    const float* __restrict__ d_logsumexp,
    const float* __restrict__ d_G,
    float negative_slope,
    float* __restrict__ grad_r
) {
    int node_j = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kMaxThreadsInWarp;

    if (node_j >= N || head_h >= H) return;


    int edge_start = __ldg(&d_row_ptr_T[node_j]);
    int edge_end   = __ldg(&d_row_ptr_T[node_j + 1]);
    int num_incoming = edge_end - edge_start;
    if (num_incoming == 0) return;

    const float* rj_base = d_r + node_j * stride_r_n + head_h * stride_r_h;
    const float* a_base  = d_attn_vec + head_h * D_CONST;

    extern __shared__ float sh[];
    float* rj_sh     = sh;              // [D_CONST]
    float* grad_r_sh = rj_sh + D_CONST; // [D_CONST]

    for (int f = lane; f < D_CONST; f += 32) {
        rj_sh[f]     = __ldg(&rj_base[f]);
        grad_r_sh[f] = 0.f;
    }
    __syncthreads();

    constexpr int kPerLane = (D_CONST + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;

    for (int idx = 0; idx < num_incoming; ++idx) {
        int node_i = 0;
        if (lane == 0) node_i = __ldg(&d_col_idx_T[edge_start + idx]);
        node_i = warp_bcast_i32(node_i);

        const float* li_base      = d_l + node_i * stride_l_n + head_h * stride_l_h;
        const float* ghi_base     = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;

        float L_i_h = 0.f, G_i_h = 0.f;
        if (lane == 0) {
            L_i_h = __ldg(&d_logsumexp[node_i * H + head_h]);
            G_i_h = __ldg(&d_G[node_i * H + head_h]);
        }
        L_i_h = warp_bcast_f32(L_i_h);
        G_i_h = warp_bcast_f32(G_i_h);

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < kPerLane; ++t) {
            int f = lane + kMaxThreadsInWarp * t;
            if (f < D_CONST) {
                float li  = __ldg(&li_base[f]);
                float rj  = rj_sh[f];
                float ghi = __ldg(&ghi_base[f]);
                float tval = leaky_relu_elementwise(li + rj, negative_slope);
                float a = __ldg(&a_base[f]); // NO reg cache
                e_lane = fmaf(tval, a, e_lane);
                p_lane = fmaf(ghi, rj, p_lane);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha = recompute_alpha(e_ij, L_i_h);
        float grad_e = alpha * (p_ij - G_i_h);

        for (int f = lane; f < D_CONST; f += 32) {
            float li  = __ldg(&li_base[f]);
            float rj  = rj_sh[f];
            float ghi = __ldg(&ghi_base[f]);

            float edge = li + rj;
            float tder = leaky_relu_der_elementwise(edge, negative_slope);
            float a = __ldg(&a_base[f]); // NO reg cache

            float acc = grad_r_sh[f];
            // direct path: alpha * grad_hi
            acc = fmaf(alpha, ghi, acc);
            // attention path: grad_e * t_der * a
            acc = fmaf(grad_e * tder, a, acc);
            grad_r_sh[f] = acc;
        }
    }

    __syncthreads();
    float* grad_r_base = grad_r + (((int64_t)node_j * (int64_t)H + head_h) * (int64_t)D_CONST);
    for (int f = lane; f < D_CONST; f += kMaxThreadsInWarp) {
        grad_r_base[f] = grad_r_sh[f];
    }
}


template<int grad_A_reduce_row_chunk_size>
__global__ void ReduceGradAKernel(
    size_t N, size_t H, size_t D,

    const float* __restrict__ grad_a,        // [N, H, D]
    float* __restrict__ d_grad_a_reduced_out // [H, D]
){

    //head inbex
    int head_h = blockIdx.z; // 0..H-1

    // define feature chunk and node chunk to reduce
    int row_chunk_start     = grad_A_reduce_row_chunk_size * blockIdx.x;
    int feature_chunk_start = blockDim.y * blockIdx.y;


    // define thread-specific indices and feature locations
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int fx = feature_chunk_start + tx;


    // define shared memory chunk and accumulatur
    __shared__ float tile_reduce[kMaxThreadsInWarp][kMaxThreadsInWarp + 1];
    __shared__ float result_accum[kMaxThreadsInWarp];

    float accum = 0.0f;

    // looped logic across row chunks:
    const int row_chunk_end = min((int)N, (int)(row_chunk_start + grad_A_reduce_row_chunk_size));
    for (int base_row_offset = row_chunk_start; base_row_offset < row_chunk_end; base_row_offset += blockDim.y){

        int row_to_load = base_row_offset + ty; // node index
        if (row_to_load < (int)N && fx < (int)D && head_h < (int)H){
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
    if (tx == 0){
        result_accum[ty] = accum;
    }

    __syncthreads();
    // now  threads with ty==0 and tx selecting feature within chunk
    // write out the final reduced result

    if (ty == 0 && fx < (int)D && head_h < (int)H) {
        // output layout: [H, D] contiguous
        size_t out_idx = (size_t)head_h * D + (size_t)fx;
        atomicAdd(d_grad_a_reduced_out + out_idx, result_accum[tx]); // TODO for multihead attention, we will need more dimensions here
    }
}

// =============================================================================
// Launcher for backward pass
// =============================================================================

template<int grad_A_reduce_row_chunk_size>
void GATv2Backward_CSR(
    // inputs
    size_t N, size_t H, size_t D,

    const float* grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,

    const float* d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,

    const float* d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,

    const int* d_row_ptr,
    const int* d_col_idx,
    const int* d_row_ptr_T,
    const int* d_col_idx_T,
    const float* d_attn_vec,
    const float* d_logsumexp,  // [N, H]
    float negative_slope,
    cudaStream_t stream,

    // outputs
    float* grad_l,             // [N, H, D]
    float* grad_r,             // [N, H, D]
    float* grad_a,             // [N, H, D]
    float* d_grad_a_reduced    // [H, D]
) {
    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N, H);

    // G has shape [N, H]
    float* d_G;
    CUDA_CHECK(cudaMalloc(&d_G, N * H * sizeof(float)));
    // 1: AL kernel - computes grad_a, grad_l, G
    if (D == 32) {
        size_t sh = 4 * 32 * sizeof(float);
        GATv2Backward_AL_Scalar_D<32><<<nBlocks, nThreads, sh, stream>>>(
            N,H,D, grad_h, stride_gh_n,stride_gh_h, d_l, stride_l_n,stride_l_h, d_r, stride_r_n,stride_r_h,
            d_row_ptr,d_col_idx, d_attn_vec, d_logsumexp, negative_slope, grad_a, grad_l, d_G);
    } else if (D == 64) {
        size_t sh = 4 * 64 * sizeof(float);
        GATv2Backward_AL_Scalar_D<64><<<nBlocks, nThreads, sh, stream>>>(
            N,H,D, grad_h, stride_gh_n,stride_gh_h, d_l, stride_l_n,stride_l_h, d_r, stride_r_n,stride_r_h,
            d_row_ptr,d_col_idx, d_attn_vec, d_logsumexp, negative_slope, grad_a, grad_l, d_G);
    } else {
        // fallback to existing float4 kernel for any other D
        size_t shared_AL = 4 * D * sizeof(float); // li, grad_hi, grad_a, grad_li
        GATv2Backward_AL<<<nBlocks, nThreads, shared_AL, stream>>>(
            N, H, D,
            grad_h,
            stride_gh_n, stride_gh_h,
            d_l,
            stride_l_n, stride_l_h,
            d_r,
            stride_r_n, stride_r_h,
            d_row_ptr,
            d_col_idx,
            d_attn_vec,
            d_logsumexp,
            negative_slope,
            grad_a,
            grad_l,
            d_G
        );
    }

    // 2: R kernel - computes grad_r
    if (D == 32) {
        size_t sh = 2 * 32 * sizeof(float);
        GATv2Backward_R_Scalar_D<32><<<nBlocks, nThreads, sh, stream>>>(
            N,H,D, grad_h, stride_gh_n,stride_gh_h, d_l, stride_l_n,stride_l_h, d_r, stride_r_n,stride_r_h,
            d_row_ptr_T,d_col_idx_T, d_attn_vec, d_logsumexp, d_G, negative_slope, grad_r);
    } else if (D == 64) {
        size_t sh = 2 * 64 * sizeof(float);
        GATv2Backward_R_Scalar_D<64><<<nBlocks, nThreads, sh, stream>>>(
            N,H,D, grad_h, stride_gh_n,stride_gh_h, d_l, stride_l_n,stride_l_h, d_r, stride_r_n,stride_r_h,
            d_row_ptr_T,d_col_idx_T, d_attn_vec, d_logsumexp, d_G, negative_slope, grad_r);
    } else {
        size_t shared_R = 2 * D * sizeof(float); // r_j, grad_r_j
        GATv2Backward_R<<<nBlocks, nThreads, shared_R, stream>>>(
            N, H, D,
            grad_h,
            stride_gh_n, stride_gh_h,
            d_l,
            stride_l_n, stride_l_h,
            d_r,
            stride_r_n, stride_r_h,
            d_row_ptr_T,
            d_col_idx_T,
            d_attn_vec,
            d_logsumexp,
            d_G,
            negative_slope,
            grad_r
        );
    }

    // Here we need to sum-reduce grad_a [N, H, D] over N into [H, D]

    size_t shmem_gradA_reduce_size = (kMaxThreadsInWarp * (kMaxThreadsInWarp + 2)) * sizeof(float); // deal with shmem bank conflicts
    dim3 grad_A_reduce_gridDim(
        (N + grad_A_reduce_row_chunk_size - 1) / grad_A_reduce_row_chunk_size,
        (D + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp,
        H
    );
    dim3 grad_A_reduce_blockDim(kMaxThreadsInWarp, kMaxThreadsInWarp);

    ReduceGradAKernel<grad_A_reduce_row_chunk_size><<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(
        N, H, D, grad_a, d_grad_a_reduced
    );

    CUDA_CHECK(cudaFree(d_G));
}


std::vector<torch::Tensor> gatv2_forward_cuda(
    torch::Tensor l,              // [N, H, D] - left features
    torch::Tensor r,              // [N, H, D] - right features
    torch::Tensor row_ptr,        // [N+1] - CSR row pointers
    torch::Tensor col_idx,        // [E] - CSR column indices
    torch::Tensor attn_vec,       // [H, D] - contihuous attention vector
    float negative_slope
) {

    TORCH_CHECK(l.is_cuda() && r.is_cuda(), "l, r must be CUDA");
    TORCH_CHECK(l.dim() == 3 && r.dim() == 3, "l, r must be [N, H, D]");
    TORCH_CHECK(l.sizes() == r.sizes(), "l, r sizes must match");

    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(attn_vec.is_cuda(), "attn_vec must be a CUDA tensor");

    TORCH_CHECK(l.size(0) == r.size(0), "l and r must have same number of nodes");
    TORCH_CHECK(l.size(1) == r.size(1), "l and r must have same head dimension");
    TORCH_CHECK(l.size(2) == r.size(2), "l and r must have same feature dimension");
    TORCH_CHECK(l.size(2) == attn_vec.size(1), "attn_vec dimension must match features");

    TORCH_CHECK(row_ptr.dtype() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.dtype() == torch::kInt32, "col_idx must be int32");


    const int64_t N = l.size(0);
    const int64_t H = l.size(1);
    const int64_t D = l.size(2);

    TORCH_CHECK(attn_vec.dim() == 2, "attn_vec must be [H, D]");
    TORCH_CHECK(attn_vec.size(0) == H, "attn_vec H mismatch");
    TORCH_CHECK(attn_vec.size(1) == D, "attn_vec D mismatch");
    TORCH_CHECK(D % 4 == 0, "head_dim (D) must be divisible by 4");

    TORCH_CHECK(attn_vec.is_contiguous(), "attn_vec must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");

    auto l_strides = l.strides();
    auto r_strides = r.strides();

    int64_t stride_l_n = l_strides[0];
    int64_t stride_l_h = l_strides[1];
    int64_t stride_l_d = l_strides[2];


    int64_t stride_r_n = r_strides[0];
    int64_t stride_r_h = r_strides[1];
    int64_t stride_r_d = r_strides[2];

    TORCH_CHECK(stride_l_d == 1 && stride_r_d == 1, "Feature dim (D) must be contiguous (stride_d == 1)");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());

    torch::Tensor h_out     = torch::empty({N, H, D}, options);
    torch::Tensor logsumexp = torch::empty({N, H}, options);

    const float* d_l = l.data_ptr<float>();
    const float* d_r = r.data_ptr<float>();
    const int*   d_row_ptr = row_ptr.data_ptr<int>();
    const int*   d_col_idx = col_idx.data_ptr<int>();
    const float* d_attn_vec = attn_vec.data_ptr<float>();
    float*       d_h_out = h_out.data_ptr<float>();
    float*       d_logsumexp = logsumexp.data_ptr<float>();

    // get CUDA stream from PyTorch
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = 0;

    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N, H);

    // shared memory: z floats for l_i + max_neighbors floats for logits
    size_t shared_mem_size = D * sizeof(float);

    switch ((int)D) {
        case 32:  GATv2Kernel_CSR_Scalar_D<32><<<nBlocks, nThreads, 32*sizeof(float),  stream>>>(N,H,D, d_l,d_r, stride_l_n,stride_l_h, stride_r_n,stride_r_h, d_row_ptr,d_col_idx, d_attn_vec, d_h_out, d_logsumexp, negative_slope); break;
        case 64:  GATv2Kernel_CSR_Scalar_D<64><<<nBlocks, nThreads, 32*sizeof(float),  stream>>>(N,H,D, d_l,d_r, stride_l_n,stride_l_h, stride_r_n,stride_r_h, d_row_ptr,d_col_idx, d_attn_vec, d_h_out, d_logsumexp, negative_slope); break;
        case 128: GATv2Kernel_CSR_D<128><<<nBlocks, nThreads, shared_mem_size, stream>>>(N,H,D, d_l,d_r, stride_l_n,stride_l_h, stride_r_n,stride_r_h, d_row_ptr,d_col_idx, d_attn_vec, d_h_out, d_logsumexp, negative_slope); break;
        case 256: GATv2Kernel_CSR_D<256><<<nBlocks, nThreads, shared_mem_size, stream>>>(N,H,D, d_l,d_r, stride_l_n,stride_l_h, stride_r_n,stride_r_h, d_row_ptr,d_col_idx, d_attn_vec, d_h_out, d_logsumexp, negative_slope); break;
        default:
            // fallback to your existing runtime-D kernel
            GATv2Kernel_CSR<<<nBlocks, nThreads, shared_mem_size, stream>>>(N,H,D, d_l,d_r, stride_l_n,stride_l_h, stride_r_n,stride_r_h, d_row_ptr,d_col_idx, d_attn_vec, d_h_out, d_logsumexp, negative_slope);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return {h_out, logsumexp};
}


std::vector<torch::Tensor> gatv2_backward_cuda(
    torch::Tensor grad_h,         // [N, H, D] - gradient from output
    torch::Tensor l,              // [N, H, D] - left features (saved)
    torch::Tensor r,              // [N, H, D] - right features (saved)
    torch::Tensor row_ptr,        // [N+1] - CSR row pointers
    torch::Tensor col_idx,        // [E] - CSR column indices
    torch::Tensor row_ptr_T,      // [N+1] - CSR^T row pointers
    torch::Tensor col_idx_T,      // [E] - CSR^T column indices
    torch::Tensor attn_vec,       // [H, D] - attention vector (saved)
    torch::Tensor logsumexp,      // [N, H] - logsumexp (saved)
    float negative_slope,
    int grad_A_reduce_row_chunk_size
) {
    TORCH_CHECK(grad_h.is_cuda(), "grad_h must be a CUDA tensor");
    TORCH_CHECK(l.is_cuda(), "l must be a CUDA tensor");
    TORCH_CHECK(r.is_cuda(), "r must be a CUDA tensor");
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(row_ptr_T.is_cuda(), "row_ptr_T must be a CUDA tensor");
    TORCH_CHECK(col_idx_T.is_cuda(), "col_idx_T must be a CUDA tensor");
    TORCH_CHECK(attn_vec.is_cuda(), "attn_vec must be a CUDA tensor");
    TORCH_CHECK(logsumexp.is_cuda(), "logsumexp must be a CUDA tensor");

    TORCH_CHECK(grad_h.dtype() == torch::kFloat32, "grad_h must be float32");
    TORCH_CHECK(l.dtype() == torch::kFloat32, "l must be float32");
    TORCH_CHECK(r.dtype() == torch::kFloat32, "r must be float32");
    TORCH_CHECK(attn_vec.dtype() == torch::kFloat32, "attn_vec must be float32");
    TORCH_CHECK(logsumexp.dtype() == torch::kFloat32, "logsumexp must be float32");
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.dtype() == torch::kInt32, "col_idx must be int32");
    TORCH_CHECK(row_ptr_T.dtype() == torch::kInt32, "row_ptr_T must be int32");
    TORCH_CHECK(col_idx_T.dtype() == torch::kInt32, "col_idx_T must be int32");


    TORCH_CHECK(attn_vec.is_contiguous(), "attn_vec must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");
    TORCH_CHECK(row_ptr_T.is_contiguous(), "row_ptr_T must be contiguous");
    TORCH_CHECK(col_idx_T.is_contiguous(), "col_idx_T must be contiguous");

    TORCH_CHECK(grad_h.dim() == 3 && l.dim() == 3 && r.dim() == 3, "grad_h, l, r must be [N, H, D]");
    TORCH_CHECK(grad_h.sizes() == l.sizes() && l.sizes() == r.sizes(), "grad_h, l, r sizes must match");
    TORCH_CHECK(attn_vec.dim() == 2, "attn_vec must be [H, D]");
    TORCH_CHECK(logsumexp.dim() == 2, "logsumexp must be [N, H]");

    const int64_t N = l.size(0);
    const int64_t H = l.size(1);
    const int64_t D = l.size(2);

    TORCH_CHECK(attn_vec.size(0) == H && attn_vec.size(1) == D,

                "attn_vec must be [H, D]");

    TORCH_CHECK(logsumexp.size(0) == N && logsumexp.size(1) == H,

                "logsumexp must be [N, H]");

    TORCH_CHECK(D % 4 == 0, "head_dim (D) must be divisible by 4");

    auto gh_strides = grad_h.strides();
    auto l_strides  = l.strides();
    auto r_strides  = r.strides();

    int64_t stride_gh_n = gh_strides[0];
    int64_t stride_gh_h = gh_strides[1];
    int64_t stride_gh_d = gh_strides[2];

    int64_t stride_l_n = l_strides[0];
    int64_t stride_l_h = l_strides[1];
    int64_t stride_l_d = l_strides[2];

    int64_t stride_r_n = r_strides[0];
    int64_t stride_r_h = r_strides[1];
    int64_t stride_r_d = r_strides[2];

    TORCH_CHECK(stride_gh_d == 1 && stride_l_d == 1 && stride_r_d == 1, "For now, feature dim (D) must be contiguous (stride_d == 1) for grad_h, l, r");


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());

    torch::Tensor grad_l = torch::empty({N, H, D}, options);
    torch::Tensor grad_r = torch::empty({N, H, D}, options);
    torch::Tensor grad_a = torch::empty({N, H, D}, options);      // only internal
    torch::Tensor grad_a_reduced = torch::zeros({H, D}, options); // returned

    const float* d_grad_h    = grad_h.data_ptr<float>();
    const float* d_l         = l.data_ptr<float>();
    const float* d_r         = r.data_ptr<float>();
    const int* d_row_ptr     = row_ptr.data_ptr<int>();
    const int* d_col_idx     = col_idx.data_ptr<int>();
    const int* d_row_ptr_T   = row_ptr_T.data_ptr<int>();
    const int* d_col_idx_T   = col_idx_T.data_ptr<int>();
    const float* d_attn_vec  = attn_vec.data_ptr<float>();
    const float* d_logsumexp = logsumexp.data_ptr<float>();
    float* d_grad_l          = grad_l.data_ptr<float>();
    float* d_grad_r          = grad_r.data_ptr<float>();
    float* d_grad_a          = grad_a.data_ptr<float>();
    float* d_grad_a_reduced  = grad_a_reduced.data_ptr<float>();

    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();



    cudaStream_t stream = 0;
    switch (grad_A_reduce_row_chunk_size) {
        case 32:
            GATv2Backward_CSR<32>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;

        case 64:
            GATv2Backward_CSR<64>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;

        case 128:
            GATv2Backward_CSR<128>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;

        case 256:
            GATv2Backward_CSR<256>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;

        case 512:
            GATv2Backward_CSR<512>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;

        case 1024:
            GATv2Backward_CSR<1024>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;

        case 2048:
            GATv2Backward_CSR<2048>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
            break;
        default:
            GATv2Backward_CSR<512>(
                N, H, D,
                d_grad_h,
                stride_gh_n, stride_gh_h,
                d_l,
                stride_l_n, stride_l_h,
                d_r,
                stride_r_n, stride_r_h,
                d_row_ptr, d_col_idx,
                d_row_ptr_T, d_col_idx_T,
                d_attn_vec,
                d_logsumexp,
                negative_slope,
                stream,
                d_grad_l, d_grad_r, d_grad_a, d_grad_a_reduced
            );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return {grad_l, grad_r, grad_a_reduced};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gatv2_forward_cuda, "GATv2 forward pass (CUDA)",
          py::arg("l"),
          py::arg("r"),
          py::arg("row_ptr"),
          py::arg("col_idx"),
          py::arg("attn_vec"),
          py::arg("negative_slope") = 0.2f);

    m.def("backward", &gatv2_backward_cuda, "GATv2 backward pass (CUDA)",
          py::arg("grad_h"),
          py::arg("l"),
          py::arg("r"),
          py::arg("row_ptr"),
          py::arg("col_idx"),
          py::arg("row_ptr_T"),
          py::arg("col_idx_T"),
          py::arg("attn_vec"),
          py::arg("logsumexp"),
          py::arg("negative_slope") = 0.2f,
          py::arg("grad_A_reduce_row_chunk_size") = 512);
}
