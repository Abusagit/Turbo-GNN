#include "../common.cuh"

constexpr int kWarpsPerBlock = 4;

template<int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kWarpsPerBlock * kWarpSize)
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

    const int node_i  = blockIdx.x;
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
    // warp_out[kWarpsPerBlock * D_CONST] as float
    // warp_max[kWarpsPerBlock] as float
    // warp_sum[kWarpsPerBlock] as float
    extern __shared__ char sh_raw[];
    cuda_t* k_shared = reinterpret_cast<cuda_t*>(sh_raw);
    float*  warp_out = reinterpret_cast<float*>(sh_raw + D_CONST * sizeof(cuda_t));
    float*  warp_max = warp_out + kWarpsPerBlock * D_CONST;
    float*  warp_sum = warp_max + kWarpsPerBlock;

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
        for (int i = threadIdx.x; i < NUM_K_LOADS; i += kWarpsPerBlock * kWarpSize) {
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
    for (int e = warp_id; e < num_neighbors; e += kWarpsPerBlock) {
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
                    for (int w = 0; w < kWarpsPerBlock; ++w) {
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
template<int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kWarpSize)
graph_attn_backward_csrT_kernel_D(
    int64_t N,
    int64_t H,
    const index_t* __restrict__ row_ptr_T,   // [N+1], CSR^T row pointers
    const index_t* __restrict__ col_idx_T,   // [E],   CSR^T col indices
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

    int node_j = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x; // 0..31

    if (node_j >= N || head_h >= H) {
        return;
    }

    index_t edge_start    = row_ptr_T[node_j];
    index_t edge_end      = row_ptr_T[node_j + 1];
    int num_incoming  = static_cast<int>(edge_end - edge_start);

    // Contiguous offset for output dQ, dV (freshly allocated, always contiguous)
    const size_t out_jh = (node_j * H + head_h) * D_CONST;

    // nothing to do if this node has no incoming edges
    if (num_incoming == 0) {
        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            Tile::write_zero(dQ + out_jh, fv);
            Tile::write_zero(dV + out_jh, fv);
        }
        return;
    }

    // Shared memory layout (unified):
    // qj_shared: D_CONST * sizeof(cuda_t)
    // vj_shared: D_CONST * sizeof(cuda_t)
    // gq_shared: D_CONST * sizeof(float)
    // gv_shared: D_CONST * sizeof(float)
    extern __shared__ char sh_raw[];
    cuda_t* qj_shared = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* vj_shared = qj_shared + D_CONST;
    float*  gq_shared = reinterpret_cast<float*>(sh_raw + 2 * D_CONST * sizeof(cuda_t));
    float*  gv_shared = gq_shared + D_CONST;

    // Load qj, vj via 128-bit transactions (stride-aware)
    {
        constexpr int ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);
        constexpr int NUM_LOADS = D_CONST / ELEMS_PER_F4;
        const float4* qj_src = reinterpret_cast<const float4*>(Q + node_j * stride_q_n + head_h * stride_q_h);
        const float4* vj_src = reinterpret_cast<const float4*>(V + node_j * stride_v_n + head_h * stride_v_h);
        float4* qj_sh_f4 = reinterpret_cast<float4*>(qj_shared);
        float4* vj_sh_f4 = reinterpret_cast<float4*>(vj_shared);
        for (int i = lane; i < NUM_LOADS; i += kWarpSize) {
            qj_sh_f4[i] = qj_src[i];
            vj_sh_f4[i] = vj_src[i];
        }
    }

    // Zero float32 gradient accumulators
    {
        constexpr int NUM_F4 = D_CONST / 4;
        float4* gq_f4 = reinterpret_cast<float4*>(gq_shared);
        float4* gv_f4 = reinterpret_cast<float4*>(gv_shared);
        for (int i = lane; i < NUM_F4; i += kWarpSize) {
            gq_f4[i] = {0.f, 0.f, 0.f, 0.f};
            gv_f4[i] = {0.f, 0.f, 0.f, 0.f};
        }
    }
    __syncwarp(FULL_WARP_MASK);

    for (int e = 0; e < num_incoming; ++e) {
        index_t node_i = 0;
        if (lane == 0) {
            node_i = __ldg(&col_idx_T[edge_start + e]);
        }
        node_i = __shfl_sync(FULL_WARP_MASK, node_i, 0);

        if (node_i >= N) continue;

        const cuda_t* ki_base  = K  + node_i * stride_k_n + head_h * stride_k_h;
        // dO is contiguous (checked by TORCH_CHECK)
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

        // 2) accumulate dV_j, dQ_j in float32 shared; atomicAdd dK_i (float32, contiguous)
        float* dK_i_base = dK + out_ih;

        for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
            int base_f = fv * EPV;
            auto ki  = Tile::load(ki_base, fv);
            auto dOi = Tile::load(dOi_base, fv);
            auto qj  = Tile::load(qj_shared, fv);

            Tile::weighted_accum(&gv_shared[base_f], alpha, dOi);
            Tile::weighted_accum(&gq_shared[base_f], dS_scaled, ki);
            Tile::atomic_add_scaled_f32(dK_i_base, base_f, dS_scaled, qj);
        }
    }

    // 3) write dQ_j and dV_j: convert float32 accumulators to cuda_t (contiguous output)
    cuda_t* dQ_base = dQ + out_jh;
    cuda_t* dV_base = dV + out_jh;

    for (int fv = lane; fv < D_VEC; fv += kWarpSize) {
        int base_f = fv * EPV;
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
    float scale
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

    dim3 blocks(N, H);
    dim3 threads(kWarpsPerBlock * kWarpSize);  // 128 threads

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "GT forward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    std::visit([&](auto idxInfo, auto typeInfo, auto d_c) {
        using index_t = typename decltype(idxInfo)::Type;
        using torch_t = typename decltype(typeInfo)::TorchType;
        using cuda_t = typename decltype(typeInfo)::CudaType;
        constexpr int DC = decltype(d_c)::value;

        auto* Q_ptr = reinterpret_cast<const cuda_t*>(Q.data_ptr<torch_t>());
        auto* K_ptr = reinterpret_cast<const cuda_t*>(K.data_ptr<torch_t>());
        auto* V_ptr = reinterpret_cast<const cuda_t*>(V.data_ptr<torch_t>());
        auto* O_ptr = reinterpret_cast<cuda_t*>(O.data_ptr<torch_t>());

        // k_shared: DC * sizeof(cuda_t)
        // warp_out: kWarpsPerBlock * DC * sizeof(float)
        // warp_max + warp_sum: 2 * kWarpsPerBlock * sizeof(float)
        size_t shmem = DC * sizeof(cuda_t)
                     + kWarpsPerBlock * DC * sizeof(float)
                     + 2 * kWarpsPerBlock * sizeof(float);

        GraphAttentionForward_CSR_MH_v2_D<DC, cuda_t, index_t><<<blocks, threads, shmem, stream>>>(
            N, H,
            Q_ptr, K_ptr, V_ptr,
            q_strides[0], q_strides[1],
            k_strides[0], k_strides[1],
            v_strides[0], v_strides[1],
            index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
            O_ptr,
            o_strides[0], o_strides[1],
            lse.data_ptr<float>(),
            scale
        );
    }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
       MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
       MakeIntVariant<32, 64, 128, 256>(D));

    CUDA_KERNEL_CHECK();

    return std::make_tuple(O, lse);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
graph_attention_backward_csr_mh_cuda(
    torch::Tensor row_ptr_T,   // [N+1], int32, CSR^T
    torch::Tensor col_idx_T,   // [E],   int32, CSR^T
    torch::Tensor Q,           // [N, H, D]
    torch::Tensor K,           // [N, H, D]
    torch::Tensor V,           // [N, H, D]
    torch::Tensor O,           // [N, H, D] (forward output)
    torch::Tensor dO,          // [N, H, D]
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

    auto input_dtype = Q.dtype();
    auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto typed_options = torch::TensorOptions().dtype(input_dtype).device(Q.device());

    // dQ, dV match input dtype; dK accumulates in float32 to avoid bf16/fp16 atomicAdd contention
    torch::Tensor dQ = torch::empty({N, H, D}, typed_options);
    torch::Tensor dV = torch::empty({N, H, D}, typed_options);
    torch::Tensor dK_f32 = torch::zeros({N, H, D}, f32_options);

    // Delta[i,h] = <O[i,h,:], dO[i,h,:]>
    torch::Tensor Delta = torch::empty({N, H}, f32_options);
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

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "GT backward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    std::visit([&](auto idxInfo, auto typeInfo, auto d_c) {
        using index_t = typename decltype(idxInfo)::Type;
        using torch_t = typename decltype(typeInfo)::TorchType;
        using cuda_t = typename decltype(typeInfo)::CudaType;
        constexpr int DC = decltype(d_c)::value;

        auto cuda_stream = at::cuda::getDefaultCUDAStream();

        auto* dO_ptr = reinterpret_cast<const cuda_t*>(dO.data_ptr<torch_t>());
        auto* O_ptr  = reinterpret_cast<const cuda_t*>(O.data_ptr<torch_t>());
        auto* Q_ptr  = reinterpret_cast<const cuda_t*>(Q.data_ptr<torch_t>());
        auto* K_ptr  = reinterpret_cast<const cuda_t*>(K.data_ptr<torch_t>());
        auto* V_ptr  = reinterpret_cast<const cuda_t*>(V.data_ptr<torch_t>());
        auto* dQ_ptr = reinterpret_cast<cuda_t*>(dQ.data_ptr<torch_t>());
        auto* dV_ptr = reinterpret_cast<cuda_t*>(dV.data_ptr<torch_t>());

        compute_D_mh_kernel_D<DC, cuda_t><<<blocks_D, threads_D, 0, cuda_stream>>>(
            dO_ptr, O_ptr, Delta.data_ptr<float>(),
            N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
        );

        // qj + vj as cuda_t, gq + gv as float
        size_t shmem_bwd = 2 * DC * sizeof(cuda_t) + 2 * DC * sizeof(float);

        auto* dK_ptr = dK_f32.data_ptr<float>();

        graph_attn_backward_csrT_kernel_D<DC, cuda_t, index_t><<<blocks_bwd, threads_bwd, shmem_bwd, cuda_stream>>>(
            N, H,
            index_ptr<index_t>(row_ptr_T), index_ptr<index_t>(col_idx_T),
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

    CUDA_KERNEL_CHECK();

    // Convert float32 dK accumulator back to input dtype
    torch::Tensor dK = dK_f32.to(input_dtype);

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
