#include "reduction/reduction_aggr_kernels.cuh"


template <ReductionOp Op>
void reduction_aggr_forward_partitioned_cuda_impl(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    const at::Tensor& light_nodes,
    const at::Tensor& heavy_nodes,
    int max_degree,
    at::Tensor& out,
    at::Tensor& arg_idx,
    int warps_per_block,
    int edges_per_block_heavy_nodes,
    bool use_2d_kernel,
    int features_per_block,
    int tiles_y,
    int pipeline_stages
) {
    using ROps = ReductionOps<Op>;

    const int d             = X.size(1);
    const int num_out_nodes = out.size(0);

    TORCH_CHECK(edge_ptr.is_cuda(), "edge_ptr must be CUDA");
    TORCH_CHECK(edge_idx.is_cuda(), "edge_idx must be CUDA");
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(light_nodes.is_cuda(), "light_nodes must be CUDA");
    TORCH_CHECK(heavy_nodes.is_cuda(), "heavy_nodes must be CUDA");

    auto idx_dtype = edge_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "index tensors must be int32, int64, uint32, or uint64");
    TORCH_CHECK(edge_idx.scalar_type() == idx_dtype, "edge_idx must have same dtype as edge_ptr");
    TORCH_CHECK(light_nodes.scalar_type() == idx_dtype, "light_nodes must have same dtype as edge_ptr");
    TORCH_CHECK(heavy_nodes.scalar_type() == idx_dtype, "heavy_nodes must have same dtype as edge_ptr");
    TORCH_CHECK(
        X.scalar_type() == at::kFloat || X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16, "X must be float32/float16/bfloat16"
    );
    TORCH_CHECK(out.scalar_type() == X.scalar_type(), "out must have same dtype as X");

    const int num_light = light_nodes.numel();
    if (num_light > 0) {
        std::visit(
            [&](auto idxInfo, auto typeInfo, auto warps_const, auto stages_c) {
                using index_t = typename decltype(idxInfo)::Type;
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t  = typename decltype(typeInfo)::CudaType;

                constexpr int WARPS_PER_BLOCK   = warps_const.value;
                constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;
                constexpr int STAGES            = decltype(stages_c)::value;
                constexpr size_t TW             = VecFloat<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t);

                cuda_t const *X_ptr = reinterpret_cast<const cuda_t *>(X.data_ptr<torch_t>());
                cuda_t *out_ptr     = reinterpret_cast<cuda_t *>(out.data_ptr<torch_t>());

                constexpr size_t TW_L = Vec<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t);
                const size_t d_vec_l  = static_cast<size_t>(d) / TW_L;
                const size_t tile_x   = std::min<size_t>(std::max<size_t>(d_vec_l, 1), THREADS_PER_BLOCK);
                const size_t node_y   = std::max<size_t>(THREADS_PER_BLOCK / tile_x, 1);
                const dim3 threads_l(static_cast<unsigned>(tile_x), static_cast<unsigned>(node_y));
                const unsigned blocks_l = static_cast<unsigned>((num_light + node_y - 1) / node_y);
                // val_dbuf (STAGES == 0 makes this term vanish)
                size_t shmem = THREADS_PER_BLOCK * STAGES * TW * sizeof(cuda_t);

                ensure_dynamic_shmem(
                    reduction_aggr_forward_light_kernel_1d<WARPS_PER_BLOCK, cuda_t, Op, index_t, float, STAGES>, shmem, "reduction_aggr light"
                );

                reduction_aggr_forward_light_kernel_1d<WARPS_PER_BLOCK, cuda_t, Op, index_t, float, STAGES><<<blocks_l, threads_l, shmem>>>(
                    index_ptr<index_t>(light_nodes),
                    index_ptr<index_t>(edge_ptr),
                    index_ptr<index_t>(edge_idx),
                    X_ptr,
                    out_ptr,
                    index_ptr_mut<index_t>(arg_idx),
                    d,
                    static_cast<size_t>(num_light)
                );
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype), MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
            MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block), MakeIntVariant<0, 1>(pipeline_stages)
        );
    }

    const int num_heavy = heavy_nodes.numel();

    if (num_heavy > 0) {
        std::visit(
            [&](auto idxInfo, auto typeInfo) {
                using index_t = typename decltype(idxInfo)::Type;
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t  = typename decltype(typeInfo)::CudaType;

                cuda_t const *X_ptr = reinterpret_cast<const cuda_t *>(X.data_ptr<torch_t>());
                cuda_t *out_ptr     = reinterpret_cast<cuda_t *>(out.data_ptr<torch_t>());

                if constexpr (sizeof(index_t) <= 4 && ROps::TRACKS_ARG) {
                    // 32-bit: user can choose packed atomics or 2D
                    if (use_2d_kernel) {
                        // constexpr size_t TW = (sizeof(cuda_t) <= 2) ? 2 : 1;
                        constexpr size_t TW = VecFloat<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t);

                        dim3 grid(num_heavy);
                        dim3 block(features_per_block, tiles_y);

                        size_t shmem_size = aggr_heavy_shmem_bytes<index_t>((size_t)tiles_y * (size_t)features_per_block * TW);

                        reduction_aggr_forward_heavy_kernel_2d<cuda_t, Op, index_t><<<grid, block, shmem_size>>>(
                            index_ptr<index_t>(heavy_nodes),
                            index_ptr<index_t>(edge_ptr),
                            index_ptr<index_t>(edge_idx),
                            X_ptr,
                            out_ptr,
                            index_ptr_mut<index_t>(arg_idx),
                            d
                        );
                    } else {
                        constexpr uint64_t PACKED_INIT = ROps::PACKED_IDENTITY;

                        auto packed = at::full(
                            {num_heavy, d}, static_cast<int64_t>(PACKED_INIT), at::TensorOptions().dtype(torch::kInt64).device(X.device())
                        );

                        std::visit(
                            [&](auto edges_const, auto warps_const, auto stages_c) {
                                constexpr int EDGES_PER_BLOCK   = edges_const.value;
                                constexpr int WARPS_PER_BLOCK   = warps_const.value;
                                constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;
                                constexpr int STAGES            = decltype(stages_c)::value;
                                constexpr size_t TW             = VecFloat<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t);

                                dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                                // val_dbuf (STAGES == 0 makes this term vanish)
                                size_t shmem = THREADS_PER_BLOCK * STAGES * TW * sizeof(cuda_t);

                                ensure_dynamic_shmem(
                                    reduction_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, WARPS_PER_BLOCK, cuda_t, Op, index_t, float, STAGES>,
                                    shmem, "reduction_aggr heavy"
                                );

                                reduction_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, WARPS_PER_BLOCK, cuda_t, Op, index_t, float, STAGES>
                                    <<<grid, THREADS_PER_BLOCK, shmem>>>(
                                        index_ptr<index_t>(heavy_nodes),
                                        index_ptr<index_t>(edge_ptr),
                                        index_ptr<index_t>(edge_idx),
                                        X_ptr,
                                        reinterpret_cast<uint64_t *>(packed.template data_ptr<int64_t>()),
                                        d
                                    );
                            },
                            MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(edges_per_block_heavy_nodes),
                            MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block), MakeIntVariant<0, 1>(pipeline_stages)
                        );

                        std::visit(
                            [&](auto warps_const) {
                                constexpr int WARPS_PER_BLOCK   = warps_const.value;
                                constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

                                int unpack_blocks = (num_heavy * d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                                unpack_results_kernel<WARPS_PER_BLOCK, cuda_t, index_t><<<unpack_blocks, THREADS_PER_BLOCK>>>(
                                    reinterpret_cast<uint64_t *>(packed.template data_ptr<int64_t>()),
                                    index_ptr<index_t>(heavy_nodes),
                                    out_ptr,
                                    index_ptr_mut<index_t>(arg_idx),
                                    num_heavy,
                                    d
                                );
                            },
                            MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
                        );
                    }
                } else {
                    // Must use 2D: either 64-bit indices (packing doesn't fit),
                    // or an accumulating reducer, which has no order-preserving
                    // (value, index) uint64 packing to atomically fold over.
                    // constexpr size_t TW = (sizeof(cuda_t) <= 2) ? 2 : 1;
                    constexpr size_t TW = VecFloat<1, cuda_t>::max_vec_size_bytes / sizeof(cuda_t);

                    dim3 grid(num_heavy);
                    dim3 block(features_per_block, tiles_y);

                    size_t shmem_size = aggr_heavy_shmem_bytes<index_t>((size_t)tiles_y * (size_t)features_per_block * TW);

                    reduction_aggr_forward_heavy_kernel_2d<cuda_t, Op, index_t><<<grid, block, shmem_size>>>(
                        index_ptr<index_t>(heavy_nodes),
                        index_ptr<index_t>(edge_ptr),
                        index_ptr<index_t>(edge_idx),
                        X_ptr,
                        out_ptr,
                        index_ptr_mut<index_t>(arg_idx),
                        d
                    );
                }
            },
            MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
            MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type())
        );
    }
    CUDA_KERNEL_CHECK();
}

void reduction_aggr_forward_partitioned_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    const at::Tensor& light_nodes,
    const at::Tensor& heavy_nodes,
    int max_degree,
    at::Tensor& out,
    at::Tensor& arg_idx,
    int warps_per_block,
    int edges_per_block_heavy_nodes,
    bool use_2d_kernel,
    int features_per_block,
    int tiles_y,
    const std::string& reduce,
    int pipeline_stages
) {
    if (reduce == "min") {
        reduction_aggr_forward_partitioned_cuda_impl<ReductionOp::MIN>(
            edge_ptr, edge_idx, X, light_nodes, heavy_nodes, max_degree, out, arg_idx, warps_per_block, edges_per_block_heavy_nodes,
            use_2d_kernel, features_per_block, tiles_y, pipeline_stages
        );
    } else if (reduce == "max") {
        reduction_aggr_forward_partitioned_cuda_impl<ReductionOp::MAX>(
            edge_ptr, edge_idx, X, light_nodes, heavy_nodes, max_degree, out, arg_idx, warps_per_block, edges_per_block_heavy_nodes,
            use_2d_kernel, features_per_block, tiles_y, pipeline_stages
        );
    } else if (reduce == "sum") {
        reduction_aggr_forward_partitioned_cuda_impl<ReductionOp::SUM>(
            edge_ptr, edge_idx, X, light_nodes, heavy_nodes, max_degree, out, arg_idx, warps_per_block, edges_per_block_heavy_nodes,
            use_2d_kernel, features_per_block, tiles_y, pipeline_stages
        );
    } else {
        TORCH_CHECK(false, "Unsupported reduce: " + reduce);
    }
}

void reduction_aggr_backward_cuda(const at::Tensor& grad_out, const at::Tensor& arg_idx, at::Tensor& grad_x, int warps_per_block = 8) {
    const int num_nodes = grad_out.size(0);
    const int d         = grad_out.size(1);
    const dim3 blocks(num_nodes);

    auto idx_dtype = arg_idx.scalar_type();

    std::visit(
        [&](auto idxInfo, auto typeInfo, auto warps_const) {
            using index_t                   = typename decltype(idxInfo)::Type;
            using torch_t                   = typename decltype(typeInfo)::TorchType;
            using cuda_t                    = typename decltype(typeInfo)::CudaType;
            constexpr int WARPS_PER_BLOCK   = warps_const.value;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

            cuda_t const *grad_out_ptr = reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>());
            cuda_t *grad_x_ptr         = reinterpret_cast<cuda_t *>(grad_x.data_ptr<torch_t>());

            const dim3 threads(THREADS_PER_BLOCK);

            reduction_aggr_backward_typed<WARPS_PER_BLOCK, cuda_t, index_t>
                <<<blocks, threads>>>(grad_out_ptr, index_ptr<index_t>(arg_idx), grad_x_ptr, num_nodes, d);
        },
        MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
    );

    CUDA_KERNEL_CHECK();
}
