#include "common.cuh"
#include "gatv2/gatv2_backward.cu"
#include "gatv2/gatv2_forward.cu"

// =============================================================================
// Undirected GATv2 backward impl: G kernel + fused ALR kernel + ReduceGradA
// =============================================================================
template <int D_CONST, typename cuda_t, typename index_t>
void GATv2Backward_CSR_Undirected_Impl(
    size_t N, size_t H, size_t D, const cuda_t *grad_h, int64_t stride_gh_n, int64_t stride_gh_h, const cuda_t *d_l, int64_t stride_l_n,
    int64_t stride_l_h, const cuda_t *d_r, int64_t stride_r_n, int64_t stride_r_h, const index_t *d_row_ptr, const index_t *d_col_idx,
    const cuda_t *d_attn_vec, const float *d_logsumexp, float negative_slope, int grad_A_reduce_row_chunk_size, cudaStream_t stream,
    cuda_t *grad_l, cuda_t *grad_r, float *grad_a, float *d_grad_a_reduced, int schedule, int blocks_per_sm, at::Tensor sched_counters, int sched_chunk, int bucket_launch,
    const index_t *light_nodes, int num_light, const index_t *heavy_nodes, int num_heavy, at::Device device,
    const int *chunk_node, const int *chunk_start, const int *node_chunk_offset, int num_slices, int heavy_edge_slice
) {
    namespace sched_ns                   = turbo_gnn::sched;
    const sched_ns::ScheduleKind SK_KIND = sched_ns::schedule_from_int(schedule);

    // 1) Compute G[i,h], bucketed.
    //
    // Both kernels used to run one launch over every node. On an undirected graph that is the
    // whole backward pass -- there was no heavy bucket at all here -- and it ran at ~25% achieved
    // occupancy, because one block per node over a heavy-tailed degree distribution leaves every
    // launch waiting on its longest row. Splitting light from heavy lets each get its own grid,
    // and gives the heavy path something an edge slice can later act on.
    float *d_G;
    CUDA_CHECK(cudaMalloc(&d_G, N * H * sizeof(float)));

    size_t sh_g = 2 * D_CONST * sizeof(cuda_t);  // li + ghi

    auto launch_g = [&](const index_t *nodes, int count, int row, cudaStream_t st) {
        if (count == 0) {
            return;
        }
        const int gx = sched_ns::persistent_grid_x(SK_KIND, count, blocks_per_sm, static_cast<int>(H), sched_chunk);
        at::Tensor offs;
        if (SK_KIND == sched_ns::ScheduleKind::PrecomputedList) {
            offs = sched_ns::default_block_offsets(count, gx, device);
        }
        auto sp = sched_ns::make_params<index_t>(
            SK_KIND, nodes, count, sched_counters, static_cast<int>(H), row, offs, sched_chunk
        );
        std::visit(
            [&](auto sched_c) {
                constexpr auto SK = static_cast<sched_ns::ScheduleKind>(decltype(sched_c)::value);
                GATv2Backward_G_Kernel<SK, D_CONST, cuda_t, index_t><<<dim3(gx, H), dim3(kWarpSize), sh_g, st>>>(
                    sp, N, H, D, grad_h, stride_gh_n, stride_gh_h, d_l, stride_l_n, stride_l_h, d_r, stride_r_n, stride_r_h,
                    d_row_ptr, d_col_idx, d_attn_vec, d_logsumexp, negative_slope, d_G
                );
            },
            MakeIntVariant<0, 1, 2, 3>(schedule)
        );
    };

    // 2) Fused ALR: grad_a, grad_l, grad_r from the forward CSR only.
    size_t sh_alr = 3 * D_CONST * sizeof(cuda_t) + 3 * D_CONST * sizeof(float);

    auto launch_alr = [&](const index_t *nodes, int count, int row, cudaStream_t st) {
        if (count == 0) {
            return;
        }
        const int gx = sched_ns::persistent_grid_x(SK_KIND, count, blocks_per_sm, static_cast<int>(H), sched_chunk);
        at::Tensor offs;
        if (SK_KIND == sched_ns::ScheduleKind::PrecomputedList) {
            offs = sched_ns::default_block_offsets(count, gx, device);
        }
        auto sp = sched_ns::make_params<index_t>(
            SK_KIND, nodes, count, sched_counters, static_cast<int>(H), row, offs, sched_chunk
        );
        std::visit(
            [&](auto sched_c) {
                constexpr auto SK = static_cast<sched_ns::ScheduleKind>(decltype(sched_c)::value);
                GATv2Backward_ALR_Undirected<SK, D_CONST, cuda_t, index_t><<<dim3(gx, H), dim3(kWarpSize), sh_alr, st>>>(
                    sp, N, H, D, grad_h, stride_gh_n, stride_gh_h, d_l, stride_l_n, stride_l_h, d_r, stride_r_n, stride_r_h,
                    d_row_ptr, d_col_idx, d_attn_vec, d_logsumexp, d_G, negative_slope, grad_a, grad_l, grad_r
                );
            },
            MakeIntVariant<0, 1, 2, 3>(schedule)
        );
    };

    // Split-K for the heavy bucket. Once bucketed, the heavy launches are ~81% of this pass at
    // ~7% occupancy: 1,685 blocks cannot fill 108 SMs, and the degree spread inside the bucket
    // means the launch waits on its longest row. One block per fixed-size edge slice fixes both.
    // Partials are plain sums (alpha is recomputed from the saved logsumexp), so the merge adds.
    const bool use_slice = heavy_edge_slice > 0 && num_slices > 0 && num_heavy > 0;
    auto f32 = at::TensorOptions().dtype(at::kFloat).device(device);
    at::Tensor part_G, part_a, part_l, part_r;

    auto launch_g_slice = [&](cudaStream_t st) {
        part_G = at::empty({num_slices, static_cast<int64_t>(H)}, f32);
        GATv2Backward_G_SliceKernel<D_CONST, cuda_t, index_t><<<dim3(num_slices, H), dim3(kWarpSize), sh_g, st>>>(
            N, H, D, grad_h, stride_gh_n, stride_gh_h, d_l, stride_l_n, stride_l_h, d_r, stride_r_n, stride_r_h,
            d_row_ptr, d_col_idx, heavy_nodes, chunk_node, chunk_start, heavy_edge_slice, num_slices,
            d_attn_vec, d_logsumexp, negative_slope, part_G.data_ptr<float>()
        );
        GATv2Backward_G_MergeKernel<cuda_t, index_t><<<dim3(num_heavy, H), dim3(kWarpSize), 0, st>>>(
            H, d_row_ptr, heavy_nodes, node_chunk_offset, part_G.data_ptr<float>(), d_G, num_heavy
        );
    };

    auto launch_alr_slice = [&](cudaStream_t st) {
        part_a = at::empty({num_slices, static_cast<int64_t>(H), static_cast<int64_t>(D_CONST)}, f32);
        part_l = at::empty({num_slices, static_cast<int64_t>(H), static_cast<int64_t>(D_CONST)}, f32);
        part_r = at::empty({num_slices, static_cast<int64_t>(H), static_cast<int64_t>(D_CONST)}, f32);
        GATv2Backward_ALR_SliceKernel<D_CONST, cuda_t, index_t><<<dim3(num_slices, H), dim3(kWarpSize), sh_alr, st>>>(
            N, H, D, grad_h, stride_gh_n, stride_gh_h, d_l, stride_l_n, stride_l_h, d_r, stride_r_n, stride_r_h,
            d_row_ptr, d_col_idx, heavy_nodes, chunk_node, chunk_start, heavy_edge_slice, num_slices,
            d_attn_vec, d_logsumexp, d_G, negative_slope,
            part_a.data_ptr<float>(), part_l.data_ptr<float>(), part_r.data_ptr<float>()
        );
        GATv2Backward_ALR_MergeKernel<D_CONST, cuda_t, index_t><<<dim3(num_heavy, H), dim3(kWarpSize), 0, st>>>(
            H, d_row_ptr, heavy_nodes, node_chunk_offset,
            part_a.data_ptr<float>(), part_l.data_ptr<float>(), part_r.data_ptr<float>(),
            grad_a, grad_l, grad_r, num_heavy
        );
    };

    namespace stream_ns = turbo_gnn::streams;
    const auto launch_mode = stream_ns::bucket_launch_from_int(bucket_launch);

    // ALR reads d_G[neighbor_j] for arbitrary neighbours, not just the ones in its own bucket, so
    // *both* G launches must complete before *either* ALR launch begins. Two separate
    // run_buckets scopes give that: each joins its streams before the next begins.
    {
        stream_ns::BucketStreams g_buckets(launch_mode, device);
        stream_ns::run_buckets(
            g_buckets,
            [&](at::cuda::CUDAStream st) { launch_g(light_nodes, num_light, 0, st); },
            [&](at::cuda::CUDAStream st) {
                if (use_slice) {
                    launch_g_slice(st);
                } else {
                    launch_g(heavy_nodes, num_heavy, 1, st);
                }
            }
        );
    }
    {
        stream_ns::BucketStreams alr_buckets(launch_mode, device);
        stream_ns::run_buckets(
            alr_buckets,
            [&](at::cuda::CUDAStream st) { launch_alr(light_nodes, num_light, 2, st); },
            [&](at::cuda::CUDAStream st) {
                if (use_slice) {
                    launch_alr_slice(st);
                } else {
                    launch_alr(heavy_nodes, num_heavy, 3, st);
                }
            }
        );
    }

    // 3) Reduce grad_a [N, H, D] -> [H, D]
    size_t shmem_gradA_reduce_size = (kWarpSize * (kWarpSize + 2)) * sizeof(float);
    dim3 grad_A_reduce_blockDim(kWarpSize, kWarpSize);

    std::visit(
        [&](auto chunk_c) {
            constexpr int CHUNK = decltype(chunk_c)::value;
            dim3 grad_A_reduce_gridDim((N + CHUNK - 1) / CHUNK, (D + kWarpSize - 1) / kWarpSize, H);
            ReduceGradAKernel<CHUNK, cuda_t>
                <<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(N, H, D, grad_a, d_grad_a_reduced);
        },
        MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(grad_A_reduce_row_chunk_size)
    );

    CUDA_CHECK(cudaFree(d_G));
}

// =============================================================================
// Launcher for backward pass (directed, with light/heavy node dispatch)
// =============================================================================

// Legacy impl kept for reference; actual dispatch is in gatv2_backward_cuda below.
template <int D_CONST, typename cuda_t, typename index_t>
void GATv2Backward_CSR_Impl_UNUSED(
    // inputs
    size_t N, size_t H, size_t D,

    const cuda_t *grad_h, int64_t stride_gh_n, int64_t stride_gh_h,

    const cuda_t *d_l, int64_t stride_l_n, int64_t stride_l_h,

    const cuda_t *d_r, int64_t stride_r_n, int64_t stride_r_h,

    const index_t *d_row_ptr, const index_t *d_col_idx, const index_t *d_row_ptr_T, const index_t *d_col_idx_T, const cuda_t *d_attn_vec,
    const float *d_logsumexp,  // [N, H]
    float negative_slope, int grad_A_reduce_row_chunk_size, cudaStream_t stream,

    // outputs
    cuda_t *grad_l,          // [N, H, D]
    cuda_t *grad_r,          // [N, H, D]
    float *grad_a,           // [N, H, D] always float32
    float *d_grad_a_reduced  // [H, D] output in float32
) {
    dim3 nThreads(kWarpSize);
    dim3 nBlocks(N, H);

    // G has shape [N, H]
    float *d_G;
    CUDA_CHECK(cudaMalloc(&d_G, N * H * sizeof(float)));

    // AL shared: li (cuda_t) + ghi (cuda_t) + grada (float) + gradli (float)
    size_t sh_al = 2 * D_CONST * sizeof(cuda_t) + 2 * D_CONST * sizeof(float);

    // 1: AL kernel - computes grad_a, grad_l, G
    GATv2Backward_AL<D_CONST, cuda_t, index_t><<<nBlocks, nThreads, sh_al, stream>>>(
        N, H, D, grad_h, stride_gh_n, stride_gh_h, d_l, stride_l_n, stride_l_h, d_r, stride_r_n, stride_r_h, d_row_ptr, d_col_idx, d_attn_vec,
        d_logsumexp, negative_slope, grad_a, grad_l, d_G
    );

    // R shared: rj (cuda_t) + gradr (float)
    size_t sh_r = D_CONST * sizeof(cuda_t) + D_CONST * sizeof(float);

    // 2: R kernel - computes grad_r
    GATv2Backward_R<D_CONST, cuda_t, index_t><<<nBlocks, nThreads, sh_r, stream>>>(
        N, H, D, grad_h, stride_gh_n, stride_gh_h, d_l, stride_l_n, stride_l_h, d_r, stride_r_n, stride_r_h, d_row_ptr_T, d_col_idx_T,
        d_attn_vec, d_logsumexp, d_G, negative_slope, grad_r
    );

    // 3: sum-reduce grad_a [N, H, D] over N into [H, D] (always float32)
    size_t shmem_gradA_reduce_size = (kWarpSize * (kWarpSize + 2)) * sizeof(float);
    dim3 grad_A_reduce_blockDim(kWarpSize, kWarpSize);

    std::visit(
        [&](auto chunk_c) {
            constexpr int CHUNK = decltype(chunk_c)::value;
            dim3 grad_A_reduce_gridDim((N + CHUNK - 1) / CHUNK, (D + kWarpSize - 1) / kWarpSize, H);
            ReduceGradAKernel<CHUNK, cuda_t>
                <<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(N, H, D, grad_a, d_grad_a_reduced);
        },
        MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(grad_A_reduce_row_chunk_size)
    );

    CUDA_CHECK(cudaFree(d_G));
}

std::vector<torch::Tensor> gatv2_forward_cuda(
    torch::Tensor l,         // [N, H, D] - left features
    torch::Tensor r,         // [N, H, D] - right features
    torch::Tensor row_ptr,   // [N+1] - CSR row pointers
    torch::Tensor col_idx,   // [E] - CSR column indices
    torch::Tensor attn_vec,  // [H, D] - contiguous attention vector
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice
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

    auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");

    TORCH_CHECK(l.dtype() == r.dtype() && l.dtype() == attn_vec.dtype(), "l, r, and attn_vec must have the same dtype");
    TORCH_CHECK(
        l.dtype() == torch::kFloat32 || l.dtype() == torch::kFloat16 || l.dtype() == torch::kBFloat16, "l must be float32, float16, or bfloat16"
    );

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

    // h_out matches input dtype, logsumexp always float32
    torch::Tensor h_out     = torch::empty({N, H, D}, torch::TensorOptions().dtype(l.dtype()).device(l.device()));
    torch::Tensor logsumexp = torch::empty({N, H}, torch::TensorOptions().dtype(torch::kFloat32).device(l.device()));

    float *d_logsumexp  = logsumexp.data_ptr<float>();
    // The caller's stream, not the legacy default one: launching on stream 0 serialises
    // against every other stream and makes concurrent bucket launches impossible.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(l.device().index());

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GATv2 forward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    namespace sched_ns                   = turbo_gnn::sched;
    const sched_ns::ScheduleKind SK_KIND = sched_ns::schedule_from_int(schedule);
    at::Tensor sched_counters            = sched_ns::make_counters(SK_KIND, static_cast<int>(H), /*num_launches=*/2, l.device());
    int bucket_id                        = 0;

    auto launch_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant,
                             at::cuda::CUDAStream bucket_stream) {
        const int this_bucket = bucket_id++;
        if (num_nodes_bucket == 0) return;
        at::cuda::CUDAStreamGuard guard(bucket_stream);
        cudaStream_t stream = bucket_stream;

        const int gx = sched_ns::persistent_grid_x(SK_KIND, num_nodes_bucket, blocks_per_sm, static_cast<int>(H), sched_chunk);
        at::Tensor offs;
        if (SK_KIND == sched_ns::ScheduleKind::PrecomputedList) {
            offs = sched_ns::degree_balanced_block_offsets(row_ptr, node_indices, num_nodes_bucket, gx);
        }

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c, auto sched_c) {
                using index_t     = typename decltype(idxInfo)::Type;
                using torch_t     = typename decltype(typeInfo)::TorchType;
                using cuda_t      = typename decltype(typeInfo)::CudaType;
                constexpr int DC  = decltype(d_c)::value;
                constexpr int W   = decltype(warp_c)::value;
                constexpr auto SK = static_cast<sched_ns::ScheduleKind>(decltype(sched_c)::value);

                auto *l_ptr     = reinterpret_cast<const cuda_t *>(l.data_ptr<torch_t>());
                auto *r_ptr     = reinterpret_cast<const cuda_t *>(r.data_ptr<torch_t>());
                auto *attn_ptr  = reinterpret_cast<const cuda_t *>(attn_vec.data_ptr<torch_t>());
                auto *h_out_ptr = reinterpret_cast<cuda_t *>(h_out.data_ptr<torch_t>());

                // l_sh + W * D float + 2 * W float
                size_t shmem = DC * sizeof(cuda_t) + W * DC * sizeof(float) + 2 * W * sizeof(float);

                dim3 blocks(gx, H);
                dim3 threads(W * kWarpSize);

                auto sp = sched_ns::make_params<index_t>(
                    SK_KIND, index_ptr<index_t>(node_indices), num_nodes_bucket, sched_counters, static_cast<int>(H), this_bucket, offs, sched_chunk
                );

                GATv2Forward_Kernel<SK, W, DC, cuda_t, index_t><<<blocks, threads, shmem, stream>>>(
                    N, H, DC, l_ptr, r_ptr, stride_l_n, stride_l_h, stride_r_n, stride_r_h, index_ptr<index_t>(row_ptr),
                    index_ptr<index_t>(col_idx), sp, attn_ptr, h_out_ptr, d_logsumexp, negative_slope
                );
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype), MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
            MakeIntVariant<32, 64, 128, 256>((int)D), warp_variant, MakeIntVariant<0, 1, 2, 3>(schedule)
        );
    };

    namespace stream_ns = turbo_gnn::streams;
    stream_ns::BucketStreams buckets(stream_ns::bucket_launch_from_int(bucket_launch), l.device());
    buckets.record_all(l, r, row_ptr, col_idx, attn_vec, h_out, logsumexp, light_nodes, sched_counters);

    // Edge-parallel heavy path; see csrc/gt/graph_transformer.cu for the rationale.
    // heavy_edge_slice == 0 keeps the node-per-block path above.
    torch::Tensor part_o, part_ml;
    auto launch_heavy_split = [&](at::cuda::CUDAStream bucket_stream) {
        const int num_heavy  = static_cast<int>(heavy_nodes.numel());
        const int num_slices = static_cast<int>(chunk_node.numel());
        if (num_heavy == 0 || num_slices == 0) return;
        at::cuda::CUDAStreamGuard guard(bucket_stream);
        cudaStream_t stream = bucket_stream;

        auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());
        part_o   = torch::empty({num_slices, static_cast<int64_t>(H), static_cast<int64_t>(D)}, f32);
        part_ml  = torch::empty({num_slices, static_cast<int64_t>(H), 2}, f32);
        buckets.record_all(part_o, part_ml, chunk_node, chunk_start, node_chunk_offset, heavy_nodes);

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                using index_t    = typename decltype(idxInfo)::Type;
                using torch_t    = typename decltype(typeInfo)::TorchType;
                using cuda_t     = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;
                constexpr int W  = decltype(warp_c)::value;

                auto *l_ptr     = reinterpret_cast<const cuda_t *>(l.data_ptr<torch_t>());
                auto *r_ptr     = reinterpret_cast<const cuda_t *>(r.data_ptr<torch_t>());
                auto *attn_ptr  = reinterpret_cast<const cuda_t *>(attn_vec.data_ptr<torch_t>());
                auto *h_out_ptr = reinterpret_cast<cuda_t *>(h_out.data_ptr<torch_t>());

                size_t shmem = DC * sizeof(cuda_t) + W * DC * sizeof(float) + 2 * W * sizeof(float);

                GATv2ForwardSlice_Kernel<W, DC, cuda_t, index_t><<<dim3(num_slices, H), dim3(W * kWarpSize), shmem, stream>>>(
                    N, H, DC, l_ptr, r_ptr, stride_l_n, stride_l_h, stride_r_n, stride_r_h,
                    index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx), index_ptr<index_t>(heavy_nodes),
                    chunk_node.data_ptr<int>(), chunk_start.data_ptr<int>(), heavy_edge_slice, num_slices,
                    attn_ptr, part_o.data_ptr<float>(), part_ml.data_ptr<float>(), negative_slope
                );

                GATv2MergeSlices_Kernel<DC, cuda_t, index_t><<<dim3(num_heavy, H), kWarpSize, 0, stream>>>(
                    H, index_ptr<index_t>(row_ptr), index_ptr<index_t>(heavy_nodes), node_chunk_offset.data_ptr<int>(),
                    part_o.data_ptr<float>(), part_ml.data_ptr<float>(), h_out_ptr, d_logsumexp, num_heavy
                );
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype), MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
            MakeIntVariant<32, 64, 128, 256>((int)D), MakeIntVariant<8, 16, 32>(heavy_warps_per_block)
        );
    };

    stream_ns::run_buckets(
        buckets,
        [&](at::cuda::CUDAStream st) { launch_bucket(light_nodes, light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block), st); },
        [&](at::cuda::CUDAStream st) {
            if (heavy_edge_slice > 0) {
                launch_heavy_split(st);
            } else {
                launch_bucket(heavy_nodes, heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block), st);
            }
        }
    );

    CUDA_KERNEL_CHECK();

    return {h_out, logsumexp};
}

std::vector<torch::Tensor> gatv2_backward_cuda(
    torch::Tensor grad_h,     // [N, H, D] - gradient from output
    torch::Tensor l,          // [N, H, D] - left features (saved)
    torch::Tensor r,          // [N, H, D] - right features (saved)
    torch::Tensor row_ptr,    // [N+1] - CSR row pointers
    torch::Tensor col_idx,    // [E] - CSR column indices
    torch::Tensor row_ptr_T,  // [N+1] - CSR^T row pointers
    torch::Tensor col_idx_T,  // [E] - CSR^T column indices
    torch::Tensor attn_vec,   // [H, D] - attention vector (saved)
    torch::Tensor logsumexp,  // [N, H] - logsumexp (saved)
    float negative_slope,
    int grad_A_reduce_row_chunk_size,
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    // Edge-slice table for the undirected backward's heavy bucket; 0 keeps node-per-block.
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int backward_heavy_edge_slice
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

    TORCH_CHECK(
        grad_h.dtype() == l.dtype() && l.dtype() == r.dtype() && l.dtype() == attn_vec.dtype(),
        "grad_h, l, r, and attn_vec must have the same dtype"
    );
    TORCH_CHECK(
        l.dtype() == torch::kFloat32 || l.dtype() == torch::kFloat16 || l.dtype() == torch::kBFloat16, "l must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(logsumexp.dtype() == torch::kFloat32, "logsumexp must be float32");
    auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "index tensors must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");
    TORCH_CHECK(row_ptr_T.scalar_type() == idx_dtype, "row_ptr_T must have same dtype as row_ptr");
    TORCH_CHECK(col_idx_T.scalar_type() == idx_dtype, "col_idx_T must have same dtype as row_ptr");

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

    TORCH_CHECK(attn_vec.size(0) == H && attn_vec.size(1) == D, "attn_vec must be [H, D]");
    TORCH_CHECK(logsumexp.size(0) == N && logsumexp.size(1) == H, "logsumexp must be [N, H]");
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

    TORCH_CHECK(
        stride_gh_d == 1 && stride_l_d == 1 && stride_r_d == 1, "For now, feature dim (D) must be contiguous (stride_d == 1) for grad_h, l, r"
    );

    auto input_dtype   = l.dtype();
    auto f32_options   = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());
    auto typed_options = torch::TensorOptions().dtype(input_dtype).device(l.device());

    // grad_l, grad_r: match input dtype
    torch::Tensor grad_l = torch::empty({N, H, D}, typed_options);
    torch::Tensor grad_r = torch::empty({N, H, D}, typed_options);
    // grad_a: always float32 (internal)
    torch::Tensor grad_a = torch::empty({N, H, D}, f32_options);
    // grad_a_reduced: accumulate in float32 to avoid bf16/fp16 atomicAdd contention
    torch::Tensor grad_a_reduced_f32 = torch::zeros({H, D}, f32_options);

    const float *d_logsumexp = logsumexp.data_ptr<float>();
    float *d_grad_a          = grad_a.data_ptr<float>();
    // See the forward entry point: the caller's stream, not the legacy default one.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(l.device().index());

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256, "GATv2 backward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    namespace sched_ns                   = turbo_gnn::sched;
    const sched_ns::ScheduleKind SK_KIND = sched_ns::schedule_from_int(schedule);
    // Rows: AL light/heavy = 0/1, R light/heavy = 2/3 (directed); G/ALR = 0/1 (undirected).
    at::Tensor sched_counters = sched_ns::make_counters(SK_KIND, static_cast<int>(H), /*num_launches=*/4, l.device());

    if (is_directed) {
        // Directed path: warp-parallel bucketed AL + R kernels

        // Allocate G [N, H] for all nodes (both light + heavy AL write into it)
        torch::Tensor G_tensor = torch::empty({N, H}, f32_options);
        float *d_G             = G_tensor.data_ptr<float>();

        // Lambda to launch AL kernel for a bucket
        int launch_al_bucket_id = 0;
        auto launch_al_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant,
                          at::cuda::CUDAStream bucket_stream) {
            const int this_bucket = launch_al_bucket_id++;
            if (num_nodes_bucket == 0) return;
            at::cuda::CUDAStreamGuard guard(bucket_stream);
            cudaStream_t stream = bucket_stream;
            const int gx = sched_ns::persistent_grid_x(SK_KIND, num_nodes_bucket, blocks_per_sm, static_cast<int>(H), sched_chunk);
            at::Tensor offs;
            if (SK_KIND == sched_ns::ScheduleKind::PrecomputedList) {
                offs = sched_ns::degree_balanced_block_offsets(row_ptr, node_indices, num_nodes_bucket, gx);
            }
            std::visit(
                [&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c, auto sched_c) {
                    using index_t     = typename decltype(idxInfo)::Type;
                    using torch_t     = typename decltype(typeInfo)::TorchType;
                    using cuda_t      = typename decltype(typeInfo)::CudaType;
                    constexpr int DC  = decltype(d_c)::value;
                    constexpr int W   = decltype(warp_c)::value;
                    constexpr auto SK = static_cast<sched_ns::ScheduleKind>(decltype(sched_c)::value);

                    auto *grad_h_ptr = reinterpret_cast<const cuda_t *>(grad_h.data_ptr<torch_t>());
                    auto *l_ptr      = reinterpret_cast<const cuda_t *>(l.data_ptr<torch_t>());
                    auto *r_ptr      = reinterpret_cast<const cuda_t *>(r.data_ptr<torch_t>());
                    auto *attn_ptr   = reinterpret_cast<const cuda_t *>(attn_vec.data_ptr<torch_t>());
                    auto *grad_l_ptr = reinterpret_cast<cuda_t *>(grad_l.data_ptr<torch_t>());

                    size_t sh_al = 2 * DC * sizeof(cuda_t) + W * 2 * DC * sizeof(float) + (W + 1) * sizeof(float);

                    dim3 blocks(gx, H);
                    dim3 threads(W * kWarpSize);

                    auto sp = sched_ns::make_params<index_t>(
                        SK_KIND, index_ptr<index_t>(node_indices), num_nodes_bucket, sched_counters, static_cast<int>(H), this_bucket, offs, sched_chunk
                    );

                    GATv2Backward_AL<SK, W, DC, cuda_t, index_t><<<blocks, threads, sh_al, stream>>>(
                        N, H, D, grad_h_ptr, stride_gh_n, stride_gh_h, l_ptr, stride_l_n, stride_l_h, r_ptr, stride_r_n, stride_r_h,
                        index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx), sp, attn_ptr, d_logsumexp,
                        negative_slope, d_grad_a, grad_l_ptr, d_G
                    );
                },
                MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
                MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()), MakeIntVariant<32, 64, 128, 256>((int)D), warp_variant,
                MakeIntVariant<0, 1, 2, 3>(schedule)
            );
        };

        // Lambda to launch R kernel for a bucket
        int launch_r_bucket_id = 2;
        auto launch_r_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant,
                          at::cuda::CUDAStream bucket_stream) {
            const int this_bucket = launch_r_bucket_id++;
            if (num_nodes_bucket == 0) return;
            at::cuda::CUDAStreamGuard guard(bucket_stream);
            cudaStream_t stream = bucket_stream;
            const int gx = sched_ns::persistent_grid_x(SK_KIND, num_nodes_bucket, blocks_per_sm, static_cast<int>(H), sched_chunk);
            at::Tensor offs;
            if (SK_KIND == sched_ns::ScheduleKind::PrecomputedList) {
                offs = sched_ns::degree_balanced_block_offsets(row_ptr_T, node_indices, num_nodes_bucket, gx);
            }
            std::visit(
                [&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c, auto sched_c) {
                    using index_t     = typename decltype(idxInfo)::Type;
                    using torch_t     = typename decltype(typeInfo)::TorchType;
                    using cuda_t      = typename decltype(typeInfo)::CudaType;
                    constexpr int DC  = decltype(d_c)::value;
                    constexpr int W   = decltype(warp_c)::value;
                    constexpr auto SK = static_cast<sched_ns::ScheduleKind>(decltype(sched_c)::value);

                    auto *grad_h_ptr = reinterpret_cast<const cuda_t *>(grad_h.data_ptr<torch_t>());
                    auto *l_ptr      = reinterpret_cast<const cuda_t *>(l.data_ptr<torch_t>());
                    auto *r_ptr      = reinterpret_cast<const cuda_t *>(r.data_ptr<torch_t>());
                    auto *attn_ptr   = reinterpret_cast<const cuda_t *>(attn_vec.data_ptr<torch_t>());
                    auto *grad_r_ptr = reinterpret_cast<cuda_t *>(grad_r.data_ptr<torch_t>());

                    size_t sh_r = DC * sizeof(cuda_t) + W * DC * sizeof(float);

                    dim3 blocks(gx, H);
                    dim3 threads(W * kWarpSize);

                    auto sp = sched_ns::make_params<index_t>(
                        SK_KIND, index_ptr<index_t>(node_indices), num_nodes_bucket, sched_counters, static_cast<int>(H), this_bucket, offs, sched_chunk
                    );

                    GATv2Backward_R<SK, W, DC, cuda_t, index_t><<<blocks, threads, sh_r, stream>>>(
                        N, H, D, grad_h_ptr, stride_gh_n, stride_gh_h, l_ptr, stride_l_n, stride_l_h, r_ptr, stride_r_n, stride_r_h,
                        index_ptr<index_t>(row_ptr_T), index_ptr<index_t>(col_idx_T), sp, attn_ptr, d_logsumexp,
                        d_G, negative_slope, grad_r_ptr
                    );
                },
                MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
                MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()), MakeIntVariant<32, 64, 128, 256>((int)D), warp_variant,
                MakeIntVariant<0, 1, 2, 3>(schedule)
            );
        };

        namespace stream_ns = turbo_gnn::streams;
        const auto bl_mode = stream_ns::bucket_launch_from_int(bucket_launch);

        // 1: AL kernel (forward CSR direction) - light + heavy
        {
            stream_ns::BucketStreams buckets(bl_mode, l.device());
            buckets.record_all(grad_h, l, r, row_ptr, col_idx, attn_vec, logsumexp, fwd_light_nodes, sched_counters);
            stream_ns::run_buckets(
                buckets,
                [&](at::cuda::CUDAStream st) { launch_al_bucket(fwd_light_nodes, fwd_light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block), st); },
                [&](at::cuda::CUDAStream st) { launch_al_bucket(fwd_heavy_nodes, fwd_heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block), st); }
            );
        }

        // 2: R kernel (backward CSR direction) - light + heavy
        // 2: R kernel (transposed CSR direction) - light + heavy
        {
            stream_ns::BucketStreams buckets(bl_mode, l.device());
            buckets.record_all(grad_h, l, r, row_ptr_T, col_idx_T, attn_vec, logsumexp, bwd_light_nodes, sched_counters);
            stream_ns::run_buckets(
                buckets,
                [&](at::cuda::CUDAStream st) { launch_r_bucket(bwd_light_nodes, bwd_light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block), st); },
                [&](at::cuda::CUDAStream st) { launch_r_bucket(bwd_heavy_nodes, bwd_heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block), st); }
            );
        }

        // 3: ReduceGradA
        {
            size_t shmem_gradA_reduce_size = (kWarpSize * (kWarpSize + 2)) * sizeof(float);
            dim3 grad_A_reduce_blockDim(kWarpSize, kWarpSize);

            std::visit(
                [&](auto typeInfo, auto chunk_c) {
                    using cuda_t        = typename decltype(typeInfo)::CudaType;
                    constexpr int CHUNK = decltype(chunk_c)::value;
                    dim3 grad_A_reduce_gridDim((N + CHUNK - 1) / CHUNK, (D + kWarpSize - 1) / kWarpSize, H);
                    ReduceGradAKernel<CHUNK, cuda_t><<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(
                        N, H, D, d_grad_a, grad_a_reduced_f32.data_ptr<float>()
                    );
                },
                MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
                MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(grad_A_reduce_row_chunk_size)
            );
        }
    } else {
        // Undirected path: fused G + ALR kernel (no bucketing, no CSR^T)
        std::visit(
            [&](auto idxInfo, auto typeInfo, auto d_c) {
                using index_t    = typename decltype(idxInfo)::Type;
                using torch_t    = typename decltype(typeInfo)::TorchType;
                using cuda_t     = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;

                auto *grad_h_ptr          = reinterpret_cast<const cuda_t *>(grad_h.data_ptr<torch_t>());
                auto *l_ptr               = reinterpret_cast<const cuda_t *>(l.data_ptr<torch_t>());
                auto *r_ptr               = reinterpret_cast<const cuda_t *>(r.data_ptr<torch_t>());
                auto *attn_ptr            = reinterpret_cast<const cuda_t *>(attn_vec.data_ptr<torch_t>());
                auto *grad_l_ptr          = reinterpret_cast<cuda_t *>(grad_l.data_ptr<torch_t>());
                auto *grad_r_ptr          = reinterpret_cast<cuda_t *>(grad_r.data_ptr<torch_t>());
                float *grad_a_reduced_ptr = grad_a_reduced_f32.data_ptr<float>();

                GATv2Backward_CSR_Undirected_Impl<DC, cuda_t, index_t>(
                    N, H, D, grad_h_ptr, stride_gh_n, stride_gh_h, l_ptr, stride_l_n, stride_l_h, r_ptr, stride_r_n, stride_r_h,
                    index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx), attn_ptr, d_logsumexp, negative_slope,
                    grad_A_reduce_row_chunk_size, stream, grad_l_ptr, grad_r_ptr, d_grad_a, grad_a_reduced_ptr, schedule, blocks_per_sm,
                    sched_counters, sched_chunk, bucket_launch,
                    // The undirected path walks the *forward* CSR, so it buckets on the forward
                    // degree distribution.
                    index_ptr<index_t>(fwd_light_nodes), static_cast<int>(fwd_light_nodes.numel()),
                    index_ptr<index_t>(fwd_heavy_nodes), static_cast<int>(fwd_heavy_nodes.numel()), l.device(),
                    chunk_node.numel() ? chunk_node.data_ptr<int>() : nullptr,
                    chunk_start.numel() ? chunk_start.data_ptr<int>() : nullptr,
                    node_chunk_offset.numel() ? node_chunk_offset.data_ptr<int>() : nullptr,
                    static_cast<int>(chunk_node.numel()), backward_heavy_edge_slice
                );
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype), MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
            MakeIntVariant<32, 64, 128, 256>((int)D)
        );
    }

    CUDA_KERNEL_CHECK();

    torch::Tensor grad_a_reduced = grad_a_reduced_f32.to(input_dtype);

    return {grad_l, grad_r, grad_a_reduced};
}
