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


struct OnlineSoftmaxStateGT {
    float max_val;
    float sum_exp;

    __device__ __forceinline__ OnlineSoftmaxStateGT()
        : max_val(-FLT_MAX), sum_exp(0.0f) {}

    __device__ __forceinline__ float update(float logit) {
        float old_max = max_val;
        max_val = fmaxf(max_val, logit);
        float correction = __expf(old_max - max_val);
        sum_exp = sum_exp * correction + __expf(logit - max_val);
        return correction;
    }

    __device__ __forceinline__ float logsumexp() const {
        return (sum_exp > 0.0f)
            ? max_val + logf(sum_exp)
            : -INFINITY;
    }
};


// Multi-head Graph Transformer forward on CSR graph.
// Q, K, V, O: [N, H, D] float32
// row_ptr: [N+1], col_idx: [E]
// logsumexp: [N, H]
__global__ void GraphAttentionForward_CSR_MH(
    size_t N,
    size_t H,
    size_t D,
    const float* __restrict__ Q,   // [N, H, D]
    const float* __restrict__ K,   // [N, H, D]
    const float* __restrict__ V,   // [N, H, D]
    int64_t stride_q_n,
    int64_t stride_q_h,
    int64_t stride_k_n,
    int64_t stride_k_h,
    int64_t stride_v_n,
    int64_t stride_v_h,
    const int* __restrict__ row_ptr,   // [N+1]
    const int* __restrict__ col_idx,   // [E]
    float* __restrict__ O,             // [N, H, D]
    int64_t stride_o_n,
    int64_t stride_o_h,
    float* __restrict__ logsumexp,     // [N, H]
    float scale
) {
    const int node_i = blockIdx.x;   // 0..N-1
    const int head_h = blockIdx.y;   // 0..H-1
    const int lane_id = threadIdx.x; // 0..31

    if (node_i >= (int)N || head_h >= (int)H) {
        return;
    }

    const int warp_size = kMaxThreadsInWarp;

    const int edge_start = row_ptr[node_i];
    const int edge_end   = row_ptr[node_i + 1];
    const int num_neighbors = edge_end - edge_start;

    // output pointer for this (i, h)
    float* out_base = O + node_i * stride_o_n + head_h * stride_o_h;
    float4* out_f4  = reinterpret_cast<float4*>(out_base);

    const int num_float4 = D / 4;
    const int float4_per_thread = (num_float4 + warp_size - 1) / warp_size;

    // handle isolated nodes early
    if (num_neighbors == 0) {
        // zero output
        for (int t = 0; t < float4_per_thread; ++t) {
            int idx4 = lane_id + t * warp_size;
            if (idx4 < num_float4) {
                out_f4[idx4] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        if (lane_id == 0) {
            logsumexp[node_i * H + head_h] = -INFINITY;
        }
        return;
    }

    // shared memory: K_{i,h} cached
    extern __shared__ float shared[];
    float* k_shared = shared;

    // load K_{i,h,:} into shared
    {
        const float* k_base   = K + node_i * stride_k_n + head_h * stride_k_h;
        const float4* k_src4  = reinterpret_cast<const float4*>(k_base);
        float4* k_dst4        = reinterpret_cast<float4*>(k_shared);

        for (int idx4 = lane_id; idx4 < num_float4; idx4 += warp_size) {
            k_dst4[idx4] = k_src4[idx4];
        }
    }
    __syncthreads();

    const float4* k_i_f4 = reinterpret_cast<const float4*>(k_shared);

    // online softmax and numerator accumulator
    OnlineSoftmaxStateGT softmax_state;

    float4 o_acc[8];  // NOTE supports D <= 1024
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        o_acc[t] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // loop over neighbors j of i
    for (int e = 0; e < num_neighbors; ++e) {
        const int eid = edge_start + e;
        const int j   = col_idx[eid];

        const float* q_base = Q + j * stride_q_n + head_h * stride_q_h;
        const float* v_base = V + j * stride_v_n + head_h * stride_v_h;
        const float4* q_f4  = reinterpret_cast<const float4*>(q_base);
        const float4* v_f4  = reinterpret_cast<const float4*>(v_base);

        // S_{ij,h} = scale * (K_{i,h} · Q_{j,h})
        float s_partial = 0.0f;
        for (int idx4 = lane_id; idx4 < num_float4; idx4 += warp_size) {
            float4 kk = k_i_f4[idx4];
            float4 qq = q_f4[idx4];
            s_partial += kk.x * qq.x +
                         kk.y * qq.y +
                         kk.z * qq.z +
                         kk.w * qq.w;
        }
        float score = warp_reduce_sum(s_partial) * scale;

        // online softmax update and rescale previous accumulator
        float correction = softmax_state.update(score);

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= float4_per_thread) break;
            o_acc[t].x *= correction;
            o_acc[t].y *= correction;
            o_acc[t].z *= correction;
            o_acc[t].w *= correction;
        }

        // add contribution for this neighbor
        float w = __expf(score - softmax_state.max_val);  // unnormalized weight

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= float4_per_thread) break;
            int idx4 = lane_id + t * warp_size;
            if (idx4 < num_float4) {
                float4 vv = v_f4[idx4];
                o_acc[t].x += w * vv.x;
                o_acc[t].y += w * vv.y;
                o_acc[t].z += w * vv.z;
                o_acc[t].w += w * vv.w;
            }
        }
    }

    // normalize accumulator and write output + logsumexp
    if (softmax_state.sum_exp > 0.0f) {
        float inv_sum = 1.0f / softmax_state.sum_exp;

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= float4_per_thread) break;
            int idx4 = lane_id + t * warp_size;
            if (idx4 < num_float4) {
                float4 val = o_acc[t];
                val.x *= inv_sum;
                val.y *= inv_sum;
                val.z *= inv_sum;
                val.w *= inv_sum;
                out_f4[idx4] = val;
            }
        }

        if (lane_id == 0) {
            logsumexp[node_i * H + head_h] = softmax_state.logsumexp();
        }
    } else {
        // shouldn't happen if num_neighbors > 0, but guard anyway
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= float4_per_thread) break;
            int idx4 = lane_id + t * warp_size;
            if (idx4 < num_float4) {
                out_f4[idx4] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        if (lane_id == 0) {
            logsumexp[node_i * H + head_h] = -INFINITY;
        }
    }
}



// ===================================================
// ================== BACKWARD =======================
// ===================================================


// D[i,h] = sum_d dO[i,h,d] * O[i,h,d]
__global__ void compute_D_mh_kernel(
    const float* __restrict__ dO,   // [N, H, D]
    const float* __restrict__ O,    // [N, H, D]
    float* __restrict__ D_out,      // [N, H]
    size_t N,
    size_t H,
    size_t D,
    int64_t stride_do_n,
    int64_t stride_do_h,
    int64_t stride_o_n,
    int64_t stride_o_h
) {
    const int node_i = blockIdx.x;
    const int head_h = blockIdx.y;
    const int lane   = threadIdx.x;   // 0..31

    if (node_i >= (int)N || head_h >= (int)H) {
        return;
    }

    const float* dO_base = dO + node_i * stride_do_n + head_h * stride_do_h;
    const float* O_base  = O  + node_i * stride_o_n  + head_h * stride_o_h;

    float sum = 0.0f;
    for (int d_idx = lane; d_idx < (int)D; d_idx += kMaxThreadsInWarp) {
        sum += dO_base[d_idx] * O_base[d_idx];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        D_out[node_i * H + head_h] = sum;
    }
}


// Q, K, V, dO, dQ, dK, dV are [N, H, D] with contiguous D, D % 4 == 0
// logsumexp and Delta are [N, H] (row-wise log-sum-exp and Delta = <O,dO>).
__global__ void graph_attn_backward_csrT_kernel(
    int64_t N,
    int64_t H,
    int64_t D,
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
    int node_j = blockIdx.x; // source node index j
    int head_h = blockIdx.y; // head index h
    int lane   = threadIdx.x; // 0..31

    if (node_j >= N || head_h >= H) {
        return;
    }

    int edge_start    = row_ptr_T[node_j];
    int edge_end      = row_ptr_T[node_j + 1];
    int num_incoming  = edge_end - edge_start;
    int num_float4    = D / 4;

    // nothing to do if this node has no incoming edges
    if (num_incoming == 0) {
        return;
    }

    extern __shared__ float shared[];
    float* qj_shared       = shared;           // [D]
    float* vj_shared       = qj_shared + D;    // [D]
    float* grad_qj_shared  = vj_shared + D;    // [D]
    float* grad_vj_shared  = grad_qj_shared + D; // [D]

    // base offset for (node_j, head_h) in [N, H, D] layout
    size_t base_jh = ((size_t)node_j * (size_t)H + (size_t)head_h) * (size_t)D;

    const float* qj_base = Q + base_jh;
    const float* vj_base = V + base_jh;
    float*       dQ_base = dQ + base_jh;
    float*       dV_base = dV + base_jh;

    float4* qj_shared_f4      = reinterpret_cast<float4*>(qj_shared);
    float4* vj_shared_f4      = reinterpret_cast<float4*>(vj_shared);
    float4* grad_qj_shared_f4 = reinterpret_cast<float4*>(grad_qj_shared);
    float4* grad_vj_shared_f4 = reinterpret_cast<float4*>(grad_vj_shared);

    const float4* qj_f4 = reinterpret_cast<const float4*>(qj_base);
    const float4* vj_f4 = reinterpret_cast<const float4*>(vj_base);

    // load q_j and v_j into shared and zero grad buffers
    for (int f_idx4 = lane; f_idx4 < num_float4; f_idx4 += kMaxThreadsInWarp) {
        qj_shared_f4[f_idx4]      = qj_f4[f_idx4];
        vj_shared_f4[f_idx4]      = vj_f4[f_idx4];
        grad_qj_shared_f4[f_idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        grad_vj_shared_f4[f_idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // iterate over all incoming edges (i -> j) using CSR^T
    for (int e = 0; e < num_incoming; ++e) {
        int node_i = col_idx_T[edge_start + e]; // destination node i
        if (node_i < 0 || node_i >= N) {
            continue; // safety
        }

        size_t base_ih = ((size_t)node_i * (size_t)H + (size_t)head_h) * (size_t)D;

        const float* ki_base  = K  + base_ih;
        const float* dOi_base = dO + base_ih;

        const float4* ki_f4   = reinterpret_cast<const float4*>(ki_base);
        const float4* dOi_f4  = reinterpret_cast<const float4*>(dOi_base);

        // 1) compute dot(k_i, q_j) and dP_ij = <dO_i, v_j>
        float dot_kq = 0.0f;
        float dP_ij  = 0.0f;

        for (int f_idx4 = lane; f_idx4 < num_float4; f_idx4 += kMaxThreadsInWarp) {
            float4 ki_val  = ki_f4[f_idx4];
            float4 qj_val  = qj_shared_f4[f_idx4];
            float4 vj_val  = vj_shared_f4[f_idx4];
            float4 dOi_val = dOi_f4[f_idx4];

            dot_kq += ki_val.x * qj_val.x + ki_val.y * qj_val.y +
                      ki_val.z * qj_val.z + ki_val.w * qj_val.w;

            dP_ij  += dOi_val.x * vj_val.x + dOi_val.y * vj_val.y +
                      dOi_val.z * vj_val.z + dOi_val.w * vj_val.w;
        }

        dot_kq = warp_reduce_sum(dot_kq);
        dP_ij  = warp_reduce_sum(dP_ij);

        // all lanes now see the same scalars
        float score = dot_kq * scale;

        size_t idx_ih = (size_t)node_i * (size_t)H + (size_t)head_h;
        float L_i     = logsumexp[idx_ih];
        float Delta_i = Delta[idx_ih];

        // alpha_ij and dE_ij
        float alpha    = __expf(score - L_i);               // exp(e_ij - logsumexp_i)
        float dS       = alpha * (dP_ij - Delta_i);         // dE_ij
        float dS_scaled = dS * scale;                       // chain rule for e_ij = scale * <k,q>

        // 2) accumulate dV_j and dQ_j locally, and atomic-add dK_i
        float4* qj_sh_f4 = qj_shared_f4;

        for (int f_idx4 = lane; f_idx4 < num_float4; f_idx4 += kMaxThreadsInWarp) {
            float4 ki_val  = ki_f4[f_idx4];
            float4 dOi_val = dOi_f4[f_idx4];
            float4 qj_val  = qj_sh_f4[f_idx4];

            // dV_j += alpha * dO_i
            float4 grad_vj = grad_vj_shared_f4[f_idx4];
            grad_vj.x += alpha * dOi_val.x;
            grad_vj.y += alpha * dOi_val.y;
            grad_vj.z += alpha * dOi_val.z;
            grad_vj.w += alpha * dOi_val.w;
            grad_vj_shared_f4[f_idx4] = grad_vj;

            // dQ_j += dS_scaled * k_i
            float4 grad_qj = grad_qj_shared_f4[f_idx4];
            grad_qj.x += dS_scaled * ki_val.x;
            grad_qj.y += dS_scaled * ki_val.y;
            grad_qj.z += dS_scaled * ki_val.z;
            grad_qj.w += dS_scaled * ki_val.w;
            grad_qj_shared_f4[f_idx4] = grad_qj;

            // dK_i += dS_scaled * q_j   (many j contribute to same i -> atomics)
            float* dK_i = dK + base_ih;
            int base_feat = f_idx4 * 4;

            atomicAdd(&dK_i[base_feat + 0], dS_scaled * qj_val.x);
            atomicAdd(&dK_i[base_feat + 1], dS_scaled * qj_val.y);
            atomicAdd(&dK_i[base_feat + 2], dS_scaled * qj_val.z);
            atomicAdd(&dK_i[base_feat + 3], dS_scaled * qj_val.w);
        }
    }

    __syncthreads();

    // 3) write dQ_j and dV_j back to global memory
    float4* dQ_f4 = reinterpret_cast<float4*>(dQ_base);
    float4* dV_f4 = reinterpret_cast<float4*>(dV_base);

    for (int f_idx4 = lane; f_idx4 < num_float4; f_idx4 += kMaxThreadsInWarp) {
        dQ_f4[f_idx4] = grad_qj_shared_f4[f_idx4];
        dV_f4[f_idx4] = grad_vj_shared_f4[f_idx4];
    }
}



std::tuple<torch::Tensor, torch::Tensor>
graph_attention_forward_csr_mh_cuda(
    torch::Tensor row_ptr,   // [N+1], int32
    torch::Tensor col_idx,   // [E],   int32
    torch::Tensor Q,         // [N, H, D], float32
    torch::Tensor K,         // [N, H, D], float32
    torch::Tensor V,          // [N, H, D], float32
    float scale
) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "CSR indices must be CUDA");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA");

    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3,
                "Q, K, V must be [N, H, D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V sizes must match");

    TORCH_CHECK(Q.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32,
                "Q, K, V must be float32");

    TORCH_CHECK(row_ptr.dtype() == torch::kInt32 &&
                col_idx.dtype() == torch::kInt32,
                "row_ptr, col_idx must be int32");

    const int64_t N = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t D = Q.size(2);

    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.size(0) == N + 1,
                "row_ptr must be [N+1]");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be [E]");

    TORCH_CHECK(D % 4 == 0, "head_dim (D) must be divisible by 4");
    TORCH_CHECK(D <= 1024, "D > 1024 not supported in this kernel (increase float4 accum)");

    auto q_strides = Q.strides();
    auto k_strides = K.strides();
    auto v_strides = V.strides();

    const int64_t stride_q_n = q_strides[0];
    const int64_t stride_q_h = q_strides[1];
    const int64_t stride_q_d = q_strides[2];

    const int64_t stride_k_n = k_strides[0];
    const int64_t stride_k_h = k_strides[1];
    const int64_t stride_k_d = k_strides[2];

    const int64_t stride_v_n = v_strides[0];
    const int64_t stride_v_h = v_strides[1];
    const int64_t stride_v_d = v_strides[2];

    TORCH_CHECK(stride_q_d == 1 && stride_k_d == 1 && stride_v_d == 1,
                "feature dim (D) must be contiguous (stride(2) == 1)");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

    torch::Tensor O         = torch::empty({N, H, D}, options);
    torch::Tensor logsumexp = torch::empty({N, H}, options);

    auto o_strides = O.strides();
    const int64_t stride_o_n = o_strides[0];
    const int64_t stride_o_h = o_strides[1];
    const int64_t stride_o_d = o_strides[2];

    TORCH_CHECK(stride_o_d == 1,
                "output feature dim (D) must be contiguous (stride(2) == 1)");

    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    const int*   d_row_ptr = row_ptr.data_ptr<int>();
    const int*   d_col_idx = col_idx.data_ptr<int>();
    float*       d_O       = O.data_ptr<float>();
    float*       d_L       = logsumexp.data_ptr<float>();

    dim3 blocks(N, H);
    dim3 threads(kMaxThreadsInWarp);
    size_t shmem = D * sizeof(float);  // K_{i,h} cached

    GraphAttentionForward_CSR_MH<<<blocks, threads, shmem>>>(
        N, H, D,
        d_Q, d_K, d_V,
        stride_q_n, stride_q_h,
        stride_k_n, stride_k_h,
        stride_v_n, stride_v_h,
        d_row_ptr,
        d_col_idx,
        d_O,
        stride_o_n, stride_o_h,
        d_L,
        scale
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "GraphAttentionForward_CSR_MH failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(O, logsumexp);
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

    TORCH_CHECK(D % 4 == 0, "head_dim (D) must be divisible by 4");
    TORCH_CHECK(D <= 1024, "D > 1024 not supported in this kernel (increase float4 accum)");

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

    {
        auto do_strides = dO.strides();
        auto o_strides  = O.strides();

        const int64_t stride_do_n = do_strides[0];
        const int64_t stride_do_h = do_strides[1];
        const int64_t stride_o_n  = o_strides[0];
        const int64_t stride_o_h  = o_strides[1];

        TORCH_CHECK(do_strides[2] == 1 && o_strides[2] == 1,
                    "dO and O feature dim (D) must be contiguous (stride(2) == 1)");

        dim3 blocks_D(N, H);
        dim3 threads_D(kMaxThreadsInWarp);

        compute_D_mh_kernel<<<blocks_D, threads_D>>>(
            dO.data_ptr<float>(),
            O.data_ptr<float>(),
            Delta.data_ptr<float>(),
            static_cast<size_t>(N),
            static_cast<size_t>(H),
            static_cast<size_t>(D),
            stride_do_n, stride_do_h,
            stride_o_n,  stride_o_h
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "compute_D_mh_kernel failed: ",
                    cudaGetErrorString(err));
    }

    // main backward kernel
    {
        dim3 blocks_bwd(N, H);
        dim3 threads_bwd(kMaxThreadsInWarp);
        size_t shmem_bwd = 4 * D * sizeof(float); // q_j, v_j, grad_q_j, grad_v_j

        graph_attn_backward_csrT_kernel<<<blocks_bwd, threads_bwd, shmem_bwd>>>(
            N, H, D,
            row_ptr_T.data_ptr<int>(),
            col_idx_T.data_ptr<int>(),
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            dO.data_ptr<float>(),
            logsumexp.data_ptr<float>(),
            Delta.data_ptr<float>(),
            scale,
            dQ.data_ptr<float>(),
            dK.data_ptr<float>(),
            dV.data_ptr<float>()
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "graph_attn_backward_csrT_kernel failed: ",
                    cudaGetErrorString(err));
    }

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
