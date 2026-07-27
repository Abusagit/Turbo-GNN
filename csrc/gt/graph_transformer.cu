#include <cstdint>

#include "common.cuh"

#include "gt/gt_forward.cu"
#include "gt/gt_backward.cu"

// ===================================================
// ================== BACKWARD =======================
// ===================================================

std::tuple<torch::Tensor, torch::Tensor> graph_attention_forward_csr_mh_cuda(
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

    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "Q, K, V must have the same dtype");
    TORCH_CHECK(
        Q.dtype() == torch::kFloat32 || Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16, "Q must be float32, float16, or bfloat16"
    );

    auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");

    const int N = Q.size(0);
    const int H = Q.size(1);
    const int D = Q.size(2);

    TORCH_CHECK(D % 4 == 0, "D must be divisible by 4");
    TORCH_CHECK(D <= 256, "D > 256 not supported");

    auto q_strides = Q.strides();
    auto k_strides = K.strides();
    auto v_strides = V.strides();

    TORCH_CHECK(q_strides[2] == 1 && k_strides[2] == 1 && v_strides[2] == 1, "Feature dim must be contiguous");

    // O matches input dtype, logsumexp always float32
    torch::Tensor O   = torch::empty({N, H, D}, torch::TensorOptions().dtype(Q.dtype()).device(Q.device()));
    torch::Tensor lse = torch::empty({N, H}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));

    auto o_strides = O.strides();

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GT forward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    // Lambda to launch the kernel for a bucket of nodes with a given warp count
    auto launch_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
        if (num_nodes_bucket == 0) return;

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                using index_t    = typename decltype(idxInfo)::Type;
                using torch_t    = typename decltype(typeInfo)::TorchType;
                using cuda_t     = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;
                constexpr int W  = decltype(warp_c)::value;

                auto *Q_ptr = reinterpret_cast<const cuda_t *>(Q.data_ptr<torch_t>());
                auto *K_ptr = reinterpret_cast<const cuda_t *>(K.data_ptr<torch_t>());
                auto *V_ptr = reinterpret_cast<const cuda_t *>(V.data_ptr<torch_t>());
                auto *O_ptr = reinterpret_cast<cuda_t *>(O.data_ptr<torch_t>());

                size_t shmem = DC * sizeof(cuda_t) + W * DC * sizeof(float) + 2 * W * sizeof(float);

                dim3 blocks(num_nodes_bucket, H);
                dim3 threads(W * kWarpSize);
                // dim3 threads(kWarpSize, d_c / kWarpSize, W);

                GraphAttentionForward_CSR_MH_v2_D<W, DC, cuda_t, index_t><<<blocks, threads, shmem, stream>>>(
                    N, H, Q_ptr, K_ptr, V_ptr, q_strides[0], q_strides[1], k_strides[0], k_strides[1], v_strides[0], v_strides[1],
                    index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx), index_ptr<index_t>(node_indices), O_ptr, o_strides[0],
                    o_strides[1], lse.data_ptr<float>(), scale
                );
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype), MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
            MakeIntVariant<32, 64, 128, 256>(D), warp_variant
        );
    };

    // Light nodes
    launch_bucket(light_nodes, light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block));

    // Heavy nodes
    launch_bucket(heavy_nodes, heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block));

    CUDA_KERNEL_CHECK();

    return std::make_tuple(O, lse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_attention_backward_csr_mh_cuda(
    torch::Tensor row_ptr,    // [N+1], forward CSR
    torch::Tensor col_idx,    // [E],   forward CSR
    torch::Tensor row_ptr_T,  // [N+1], CSR^T (backward)
    torch::Tensor col_idx_T,  // [E],   CSR^T (backward)
    torch::Tensor Q,          // [N, H, D]
    torch::Tensor K,          // [N, H, D]
    torch::Tensor V,          // [N, H, D]
    torch::Tensor O,          // [N, H, D] (forward output)
    torch::Tensor dO,         // [N, H, D]
    torch::Tensor logsumexp,  // [N, H],   float32
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed
) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "Forward CSR indices must be CUDA");
    TORCH_CHECK(row_ptr_T.is_cuda() && col_idx_T.is_cuda(), "CSR^T indices must be CUDA");
    TORCH_CHECK(
        Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda() && dO.is_cuda() && logsumexp.is_cuda(),
        "Q, K, V, O, dO, logsumexp must be CUDA"
    );

    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3 && O.dim() == 3 && dO.dim() == 3, "Q, K, V, O, dO must be [N, H, D]");
    TORCH_CHECK(
        Q.sizes() == K.sizes() && Q.sizes() == V.sizes() && Q.sizes() == O.sizes() && Q.sizes() == dO.sizes(),
        "Q, K, V, O, dO sizes must match [N, H, D]"
    );

    TORCH_CHECK(
        Q.dtype() == K.dtype() && Q.dtype() == V.dtype() && Q.dtype() == O.dtype() && Q.dtype() == dO.dtype(),
        "Q, K, V, O, dO must have the same dtype"
    );
    TORCH_CHECK(
        Q.dtype() == torch::kFloat32 || Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16, "Q must be float32, float16, or bfloat16"
    );

    auto idx_dtype = row_ptr_T.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr_T must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx_T.scalar_type() == idx_dtype, "col_idx_T must have same dtype as row_ptr_T");

    TORCH_CHECK(logsumexp.dtype() == torch::kFloat32, "logsumexp must be float32");
    TORCH_CHECK(logsumexp.dim() == 2, "logsumexp must be [N, H]");

    const int64_t N = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t D = Q.size(2);

    TORCH_CHECK(row_ptr_T.dim() == 1 && row_ptr_T.size(0) == N + 1, "row_ptr_T must be [N+1]");
    TORCH_CHECK(col_idx_T.dim() == 1, "col_idx_T must be [E]");
    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.size(0) == N + 1, "row_ptr must be [N+1]");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be [E]");

    TORCH_CHECK(logsumexp.size(0) == N && logsumexp.size(1) == H, "logsumexp must be [N, H]");

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

    TORCH_CHECK(
        stride_q_d == 1 && stride_k_d == 1 && stride_v_d == 1 && stride_o_d == 1,
        "feature dim (D) must be contiguous (stride(2) == 1) for Q, K, V, O"
    );

    TORCH_CHECK(O.is_contiguous(), "O must be contiguous [N, H, D]");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous [N, H, D]");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous [N, H]");
    TORCH_CHECK(row_ptr_T.is_contiguous() && col_idx_T.is_contiguous(), "CSR^T arrays must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous() && col_idx.is_contiguous(), "Forward CSR arrays must be contiguous");

    auto input_dtype   = Q.dtype();
    auto f32_options   = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto typed_options = torch::TensorOptions().dtype(input_dtype).device(Q.device());

    // Delta[i,h] = <O[i,h,:], dO[i,h,:]>
    torch::Tensor Delta = torch::empty({N, H}, f32_options);
    auto do_strides     = dO.strides();

    const int64_t stride_do_n = do_strides[0];
    const int64_t stride_do_h = do_strides[1];
    const int64_t stride_o_n  = o_strides[0];
    const int64_t stride_o_h  = o_strides[1];

    TORCH_CHECK(do_strides[2] == 1 && o_strides[2] == 1, "dO and O feature dim (D) must be contiguous (stride(2) == 1)");

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GT backward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    // Launch compute_D for ALL nodes (always 1 warp, no bucketing)
    {
        dim3 blocks_D(N, H);
        dim3 threads_D(kWarpSize);
        std::visit(
            [&](auto typeInfo, auto d_c) {
                using torch_t    = typename decltype(typeInfo)::TorchType;
                using cuda_t     = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;

                auto cuda_stream = at::cuda::getDefaultCUDAStream();
                auto *dO_ptr     = reinterpret_cast<const cuda_t *>(dO.data_ptr<torch_t>());
                auto *O_ptr      = reinterpret_cast<const cuda_t *>(O.data_ptr<torch_t>());

                compute_D_mh_kernel_D<DC, cuda_t><<<blocks_D, threads_D, 0, cuda_stream>>>(
                    dO_ptr, O_ptr, Delta.data_ptr<float>(), N, H, stride_do_n, stride_do_h, stride_o_n, stride_o_h
                );
            },
            MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()), MakeIntVariant<32, 64, 128, 256>((int)D)
        );
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

            std::visit(
                [&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                    using index_t    = typename decltype(idxInfo)::Type;
                    using torch_t    = typename decltype(typeInfo)::TorchType;
                    using cuda_t     = typename decltype(typeInfo)::CudaType;
                    constexpr int DC = decltype(d_c)::value;
                    constexpr int W  = decltype(warp_c)::value;

                    auto cuda_stream = at::cuda::getDefaultCUDAStream();

                    auto *Q_ptr  = reinterpret_cast<const cuda_t *>(Q.data_ptr<torch_t>());
                    auto *K_ptr  = reinterpret_cast<const cuda_t *>(K.data_ptr<torch_t>());
                    auto *V_ptr  = reinterpret_cast<const cuda_t *>(V.data_ptr<torch_t>());
                    auto *dO_ptr = reinterpret_cast<const cuda_t *>(dO.data_ptr<torch_t>());
                    auto *dQ_ptr = reinterpret_cast<cuda_t *>(dQ.data_ptr<torch_t>());
                    auto *dV_ptr = reinterpret_cast<cuda_t *>(dV.data_ptr<torch_t>());
                    auto *dK_ptr = dK_f32.data_ptr<float>();

                    // qj + vj (read-only) + W * (gq + gv) per-warp accumulators
                    size_t shmem_bwd = 2 * DC * sizeof(cuda_t) + W * 2 * DC * sizeof(float);

                    dim3 blocks(num_nodes_bucket, H);
                    dim3 threads(W * kWarpSize);

                    graph_attn_backward_csrT_kernel_D<W, DC, cuda_t, index_t><<<blocks, threads, shmem_bwd, cuda_stream>>>(
                        N, H, index_ptr<index_t>(row_ptr_T), index_ptr<index_t>(col_idx_T), index_ptr<index_t>(node_indices), Q_ptr, K_ptr,
                        V_ptr, q_strides[0], q_strides[1], k_strides[0], k_strides[1], v_strides[0], v_strides[1], dO_ptr,
                        logsumexp.data_ptr<float>(), Delta.data_ptr<float>(), scale, dQ_ptr, dK_ptr, dV_ptr
                    );
                },
                MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
                MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()), MakeIntVariant<32, 64, 128, 256>((int)D), warp_variant
            );
        };

        // Light nodes
        launch_bucket(light_nodes, light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block));

        // Heavy nodes
        launch_bucket(heavy_nodes, heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block));
    } else {
        // Undirected path: forward CSR, no atomics, no bucketing
        std::visit(
            [&](auto idxInfo, auto typeInfo, auto d_c) {
                using index_t    = typename decltype(idxInfo)::Type;
                using torch_t    = typename decltype(typeInfo)::TorchType;
                using cuda_t     = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;

                auto cuda_stream = at::cuda::getDefaultCUDAStream();

                auto *Q_ptr  = reinterpret_cast<const cuda_t *>(Q.data_ptr<torch_t>());
                auto *K_ptr  = reinterpret_cast<const cuda_t *>(K.data_ptr<torch_t>());
                auto *V_ptr  = reinterpret_cast<const cuda_t *>(V.data_ptr<torch_t>());
                auto *dO_ptr = reinterpret_cast<const cuda_t *>(dO.data_ptr<torch_t>());
                auto *dQ_ptr = reinterpret_cast<cuda_t *>(dQ.data_ptr<torch_t>());
                auto *dV_ptr = reinterpret_cast<cuda_t *>(dV.data_ptr<torch_t>());
                auto *dK_ptr = reinterpret_cast<cuda_t *>(dK_typed.data_ptr<torch_t>());

                // 3 cuda_t vectors (K,Q,V) + 3 float accumulators (dK,dQ,dV)
                size_t shmem_bwd = 3 * DC * sizeof(cuda_t) + 3 * DC * sizeof(float);

                dim3 blocks_bwd(N, H);
                dim3 threads_bwd(kWarpSize);

                graph_attn_backward_fwd_csr_undirected_kernel_D<DC, cuda_t, index_t><<<blocks_bwd, threads_bwd, shmem_bwd, cuda_stream>>>(
                    N, H, index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx), Q_ptr, K_ptr, V_ptr, q_strides[0], q_strides[1],
                    k_strides[0], k_strides[1], v_strides[0], v_strides[1], dO_ptr, logsumexp.data_ptr<float>(), Delta.data_ptr<float>(), scale,
                    dQ_ptr, dK_ptr, dV_ptr
                );
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype), MakeTypeVariant<float, at::Half, at::BFloat16>(Q.scalar_type()),
            MakeIntVariant<32, 64, 128, 256>((int)D)
        );
    }

    CUDA_KERNEL_CHECK();

    // Convert float32 dK accumulator back to input dtype for directed path
    torch::Tensor dK = is_directed ? dK_f32.to(input_dtype) : dK_typed;

    return std::make_tuple(dQ, dK, dV);
}
