#include "../common.cuh"

template <int WARPS_PER_BLOCK, typename cuda_t, ReductionOp Op>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
reduction_aggr_forward_light_kernel_1d(
    const int* __restrict__ light_nodes_indices,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const cuda_t* __restrict__ X,
    cuda_t* __restrict__ out,
    int* __restrict__ arg_idx,
    int d
) {
    using ROps = ReductionOps<Op>;
    constexpr int VW = (sizeof(cuda_t) <= 2) ? 2 : 1;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;

    int i = blockIdx.x;
    int v = light_nodes_indices[i];

    int row_start = edge_ptr[v];
    int row_end   = edge_ptr[v + 1];

    int tid = threadIdx.x;
    constexpr int BLOCK_DIM = WARPS_PER_BLOCK * kWarpSize;

    int node_stride = v * d;

    cuda_t identity_val = make_cuda_value<cuda_t>(ROps::IDENTITY);
    cuda_t zero_val = make_cuda_value<cuda_t>(0.0f);

    const int d_vec = d / EPV;

    for (int fv = tid; fv < d_vec; fv += BLOCK_DIM) {
        const int base_f = fv * EPV;

        cuda_t best_vals[EPV];
        int best_srcs[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            best_vals[e] = identity_val;
            best_srcs[e] = -1;
        }

        for (int eid = row_start; eid < row_end; ++eid) {
            int src = edge_idx[eid];
            auto val = Tile::load(&X[src * d], fv);
            #pragma unroll
            for (int e = 0; e < EPV; ++e) {
                cuda_t v_e = Tile::extract(val, e);
                if (ROps::is_better(v_e, best_vals[e])) { best_vals[e] = v_e; best_srcs[e] = src; }
            }
        }

        cuda_t result[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            result[e] = (best_srcs[e] != -1) ? best_vals[e] : zero_val;
            arg_idx[node_stride + base_f + e] = best_srcs[e];
        }
        Tile::store_vec(&out[node_stride], fv, Tile::build(result));
    }

    // Scalar tail for d % EPV != 0 (compiles away for EPV=1)
    if (d % EPV != 0 && tid == 0) {
        for (int f = d_vec * EPV; f < d; ++f) {
            cuda_t best_val = identity_val;
            int best_src = -1;
            for (int eid = row_start; eid < row_end; ++eid) {
                int src = edge_idx[eid];
                cuda_t val = X[src * d + f];
                if (ROps::is_better(val, best_val)) { best_val = val; best_src = src; }
            }
            out[node_stride + f] = (best_src != -1) ? best_val : zero_val;
            arg_idx[node_stride + f] = best_src;
        }
    }
}

__device__ __forceinline__ unsigned int float_to_ordered_uint(float x) {
    unsigned int bits = __float_as_uint(x);
    if (bits & 0x80000000u) {
        // negative: invert bits so ordering is preserved
        return ~bits;
    } else {
        // non-negative: set sign bit so they come after all negatives
        return bits | 0x80000000u;
    }
}

__device__ __forceinline__ float ordered_uint_to_float(unsigned int key) {
    unsigned int bits;
    if (key & 0x80000000u) {
        // non-negative branch
        bits = key & 0x7fffffffu;
    } else {
        // negative branch
        bits = ~key;
    }
    return __uint_as_float(bits);
}

// pack float and int into uint64 for atomic updates
__device__ __forceinline__ unsigned long long pack_val_idx(float val, int idx) {
    unsigned int key = float_to_ordered_uint(val);
    return (static_cast<unsigned long long>(key) << 32) |
           static_cast<unsigned int>(idx);
}

// unpack float and int from uint64
__device__ __forceinline__ void unpack_val_idx(
    unsigned long long packed,
    float& val,
    int& idx
) {
    unsigned int key  = static_cast<unsigned int>(packed >> 32);
    unsigned int idxu = static_cast<unsigned int>(packed & 0xFFFFFFFFu);

    val = ordered_uint_to_float(key);
    idx = static_cast<int>(idxu);
}


// 2D kernel: blockIdx.x = node, blockIdx.y = edge chunk
template<int EDGES_PER_BLOCK, int WARPS_PER_BLOCK, typename cuda_t, ReductionOp Op>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
reduction_aggr_forward_heavy_kernel(
    const int* __restrict__ heavy_nodes_indices,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const cuda_t* __restrict__ X,
    unsigned long long* __restrict__ packed,
    int d
) {
    using ROps = ReductionOps<Op>;
    constexpr int VW = (sizeof(cuda_t) <= 2) ? 2 : 1;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;

    int node_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int v = heavy_nodes_indices[node_idx];

    int row_start = edge_ptr[v];
    int row_end = edge_ptr[v + 1];

    int chunk_start = row_start + chunk_idx * EDGES_PER_BLOCK;
    int chunk_end = min(chunk_start + EDGES_PER_BLOCK, row_end);

    // exit for chunks beyond this node's edges
    if (chunk_start >= row_end) {
        return;
    }

    int tid = threadIdx.x;
    constexpr int BLOCK_DIM = WARPS_PER_BLOCK * kWarpSize;
    cuda_t identity_val = make_cuda_value<cuda_t>(ROps::IDENTITY);

    const int d_vec = d / EPV;

    for (int fv = tid; fv < d_vec; fv += BLOCK_DIM) {
        const int base_f = fv * EPV;

        cuda_t best_vals[EPV];
        int best_srcs[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            best_vals[e] = identity_val;
            best_srcs[e] = -1;
        }

        for (int eid = chunk_start; eid < chunk_end; ++eid) {
            int src = edge_idx[eid];
            auto val = Tile::load(&X[src * d], fv);
            #pragma unroll
            for (int e = 0; e < EPV; ++e) {
                cuda_t v_e = Tile::extract(val, e);
                if (ROps::is_better(v_e, best_vals[e])) { best_vals[e] = v_e; best_srcs[e] = src; }
            }
        }

        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            if (best_srcs[e] >= 0) {
                unsigned long long new_val = pack_val_idx(cuda_to_float(best_vals[e]), best_srcs[e]);
                ROps::atomic_reduce(&packed[node_idx * d + base_f + e], new_val);
            }
        }
    }

    // scalar tail for d % EPV != 0
    if (d % EPV != 0 && tid == 0) {
        for (int f = d_vec * EPV; f < d; ++f) {
            cuda_t local_best = identity_val;
            int local_arg = -1;

            for (int eid = chunk_start; eid < chunk_end; ++eid) {
                int src = edge_idx[eid];
                cuda_t val = X[src * d + f];
                if (ROps::is_better(val, local_best)) { local_best = val; local_arg = src; }
            }

            if (local_arg >= 0) {
                unsigned long long new_val = pack_val_idx(cuda_to_float(local_best), local_arg);
                ROps::atomic_reduce(&packed[node_idx * d + f], new_val);
            }
        }
    }
}

// unpack results back to separate arrays
template <int WARPS_PER_BLOCK, typename cuda_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
unpack_results_kernel(
    const unsigned long long* __restrict__ packed,
    const int* __restrict__ nodes,
    cuda_t* __restrict__ out,
    int* __restrict__ arg_idx,
    int num_nodes,
    int d
) {
    constexpr int BLOCK_DIM = WARPS_PER_BLOCK * kWarpSize;
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;

    for (int i = tid; i < num_nodes * d; i += gridDim.x * BLOCK_DIM) {
        int node_idx = i / d;
        int f = i % d;
        int v = nodes[node_idx];

        float val;
        int idx;
        unpack_val_idx(packed[node_idx * d + f], val, idx);

        out[v * d + f] = (idx > -1) ? make_cuda_value<cuda_t>(val) : make_cuda_value<cuda_t>(0.0f);
        arg_idx[v * d + f] = idx;
    }
}

// 2D kernel: blockIdx.x = node, threadIdx.x = feature, threadIdx.y = edge tile
// uses shared memory tree reduction across tiles instead of packed atomicMin/Max
template <typename cuda_t, ReductionOp Op>
__global__ void reduction_aggr_forward_heavy_kernel_2d(
    const int* __restrict__ nodes,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const cuda_t* __restrict__ X,
    cuda_t* __restrict__ out,
    int* __restrict__ arg_idx,
    int d
) {
    using ROps = ReductionOps<Op>;
    constexpr int VW = (sizeof(cuda_t) <= 2) ? 2 : 1;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;

    int i = blockIdx.x;
    int v = nodes[i];

    int row_start = edge_ptr[v];
    int row_end   = edge_ptr[v + 1];
    const int degree = row_end - row_start;

    int fid = threadIdx.x; // feature dimension
    int tid = threadIdx.y; // tile index

    const int F_BLOCK = blockDim.x;
    const int TILES_Y = blockDim.y;
    const int SHMEM_STRIDE = F_BLOCK * EPV;

    extern __shared__ unsigned char shared_mem[];
    float* shmem_val = reinterpret_cast<float*>(shared_mem);
    int* shmem_idx = reinterpret_cast<int*>(shmem_val + (TILES_Y * SHMEM_STRIDE));

    cuda_t identity_val = make_cuda_value<cuda_t>(ROps::IDENTITY);
    cuda_t zero_val = make_cuda_value<cuda_t>(0.0f);

    int tile_size_ceil = (degree + TILES_Y - 1) / TILES_Y;
    int start = row_start + tid * tile_size_ceil;
    int end = min(start + tile_size_ceil, row_end);

    int node_stride = v * d;
    const int d_vec = d / EPV;

    for (int fv = fid; fv < d_vec; fv += F_BLOCK) {
        const int base_f = fv * EPV;

        cuda_t best_vals[EPV];
        int best_srcs[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            best_vals[e] = identity_val;
            best_srcs[e] = -1;
        }

        for (int eid = start; eid < end; ++eid) {
            int src = edge_idx[eid];
            auto val = Tile::load(&X[src * d], fv);
            #pragma unroll
            for (int e = 0; e < EPV; ++e) {
                cuda_t v_e = Tile::extract(val, e);
                if (ROps::is_better(v_e, best_vals[e])) { best_vals[e] = v_e; best_srcs[e] = src; }
            }
        }

        // Write to shmem (convert to float for cross-tile reduction)
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            shmem_val[tid * SHMEM_STRIDE + fid * EPV + e] = cuda_to_float(best_vals[e]);
            shmem_idx[tid * SHMEM_STRIDE + fid * EPV + e] = best_srcs[e];
        }

        __syncthreads();

        // Tree reduction across tiles
        for (int offset = TILES_Y / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                #pragma unroll
                for (int e = 0; e < EPV; ++e) {
                    const int a = tid * SHMEM_STRIDE + fid * EPV + e;
                    const int b = (tid + offset) * SHMEM_STRIDE + fid * EPV + e;

                    const float val_a = shmem_val[a];
                    const int idx_a = shmem_idx[a];
                    const float val_b = shmem_val[b];
                    const int idx_b = shmem_idx[b];

                    if (ROps::is_better_f(val_b, val_a) || (val_b == val_a && idx_b >= 0 && (idx_a < 0 || idx_b < idx_a))) {
                        shmem_val[a] = val_b;
                        shmem_idx[a] = idx_b;
                    }
                }
            }
            __syncthreads();
        }

        // Vectorized final write
        if (tid == 0) {
            cuda_t result[EPV];
            #pragma unroll
            for (int e = 0; e < EPV; ++e) {
                int best_idx = shmem_idx[fid * EPV + e];
                result[e] = (best_idx >= 0) ? make_cuda_value<cuda_t>(shmem_val[fid * EPV + e]) : zero_val;
                arg_idx[node_stride + base_f + e] = best_idx;
            }
            Tile::store_vec(&out[node_stride], fv, Tile::build(result));
        }

        __syncthreads();
    }

    // scalar tail for d % EPV != 0 (compiles away for EPV=1)
    if (d % EPV != 0) {
        const int tail_f = d_vec * EPV;

        // only fid==0 does actual edge scanning; others contribute identity/-1.
        float local_best = ROps::IDENTITY;
        int local_arg = -1;

        if (fid == 0) {
            for (int eid = start; eid < end; ++eid) {
                int src = edge_idx[eid];
                float fval = cuda_to_float(X[src * d + tail_f]);
                if (ROps::is_better_f(fval, local_best)) { local_best = fval; local_arg = src; }
            }
        }

        shmem_val[tid * SHMEM_STRIDE + fid] = local_best;
        shmem_idx[tid * SHMEM_STRIDE + fid] = local_arg;

        __syncthreads();

        for (int offset = TILES_Y / 2; offset > 0; offset /= 2) {
            if (tid < offset && fid == 0) {
                const int a = tid * SHMEM_STRIDE;
                const int b = (tid + offset) * SHMEM_STRIDE;

                const float val_a = shmem_val[a];
                const int idx_a = shmem_idx[a];
                const float val_b = shmem_val[b];
                const int idx_b = shmem_idx[b];

                if (ROps::is_better_f(val_b, val_a) || (val_b == val_a && idx_b >= 0 && (idx_a < 0 || idx_b < idx_a))) {
                    shmem_val[a] = val_b;
                    shmem_idx[a] = idx_b;
                }
            }
            __syncthreads();
        }

        if (tid == 0 && fid == 0) {
            float best_val = shmem_val[0];
            int best_idx = shmem_idx[0];
            out[node_stride + tail_f] = (best_idx >= 0) ? make_cuda_value<cuda_t>(best_val) : zero_val;
            arg_idx[node_stride + tail_f] = best_idx;
        }
    }
}

template <int WARPS_PER_BLOCK, typename cuda_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
reduction_aggr_backward_typed(
    const cuda_t* __restrict__ grad_out,
    const int*   __restrict__ arg_idx,
    cuda_t* __restrict__ grad_x,
    int num_nodes,
    int d
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    int tid = threadIdx.x;
    constexpr int BLOCK_DIM = WARPS_PER_BLOCK * kWarpSize;
    const int base_offset = block_idx * d;

    for (int f = tid; f < d; f += BLOCK_DIM) {
        int src = arg_idx[base_offset + f];
        if (src >= 0) {
            atomicAdd(&grad_x[src * d + f], grad_out[base_offset + f]);
        }
    }
}

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
    int tiles_y
) {
    using ROps = ReductionOps<Op>;

    const int d = X.size(1);
    const int num_out_nodes = out.size(0);

    TORCH_CHECK(edge_ptr.is_cuda(), "edge_ptr must be CUDA");
    TORCH_CHECK(edge_idx.is_cuda(), "edge_idx must be CUDA");
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(light_nodes.is_cuda(), "light_nodes must be CUDA");
    TORCH_CHECK(heavy_nodes.is_cuda(), "heavy_nodes must be CUDA");

    TORCH_CHECK(edge_ptr.dtype() == torch::kInt32, "edge_ptr must be int32");
    TORCH_CHECK(edge_idx.dtype() == torch::kInt32, "edge_idx must be int32");
    TORCH_CHECK(light_nodes.dtype() == torch::kInt32, "light_nodes must be int32");
    TORCH_CHECK(heavy_nodes.dtype() == torch::kInt32, "heavy_nodes must be int32");
    TORCH_CHECK(X.scalar_type() == at::kFloat || X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16, "X must be float32/float16/bfloat16");
    TORCH_CHECK(out.scalar_type() == X.scalar_type(), "out must have same dtype as X");

    const int num_light = light_nodes.numel();
    if (num_light > 0) {
        std::visit([&](auto typeInfo, auto warps_const) {
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t = typename decltype(typeInfo)::CudaType;

            constexpr int WARPS_PER_BLOCK = warps_const.value;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

            auto* X_ptr = reinterpret_cast<const cuda_t*>(X.data_ptr<torch_t>());
            auto* out_ptr = reinterpret_cast<cuda_t*>(out.data_ptr<torch_t>());

            reduction_aggr_forward_light_kernel_1d<WARPS_PER_BLOCK, cuda_t, Op><<<num_light, THREADS_PER_BLOCK>>>(
                light_nodes.data_ptr<int>(),
                edge_ptr.data_ptr<int>(),
                edge_idx.data_ptr<int>(),
                X_ptr,
                out_ptr,
                arg_idx.data_ptr<int>(),
                d
            );
        },
        MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
        MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
        );
    }

    const int num_heavy = heavy_nodes.numel();

    if (num_heavy > 0) {
        if (use_2d_kernel) {
            // 2D kernel path: shared memory tree reduction, no packed buffer needed
            std::visit([&](auto typeInfo) {
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t = typename decltype(typeInfo)::CudaType;

                auto* X_ptr = reinterpret_cast<const cuda_t*>(X.data_ptr<torch_t>());
                auto* out_ptr = reinterpret_cast<cuda_t*>(out.data_ptr<torch_t>());

                constexpr int VW = (sizeof(cuda_t) <= 2) ? 2 : 1;
                constexpr int EPV = TileOps<VW, cuda_t>::ELEM_PER_VEC;

                dim3 grid(num_heavy);
                dim3 block(features_per_block, tiles_y);

                size_t shmem_size = (size_t)tiles_y * (size_t)features_per_block * EPV * (sizeof(float) + sizeof(int));

                reduction_aggr_forward_heavy_kernel_2d<cuda_t, Op><<<grid, block, shmem_size>>>(
                    heavy_nodes.data_ptr<int>(),
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    X_ptr,
                    out_ptr,
                    arg_idx.data_ptr<int>(),
                    d
                );
            },
            MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type())
            );
        } else {
            constexpr unsigned long long PACKED_INIT = ROps::PACKED_IDENTITY;

            auto packed = at::full(
                {num_heavy, d},
                static_cast<int64_t>(PACKED_INIT),
                at::TensorOptions().dtype(torch::kInt64).device(X.device())
            );

            std::visit([&](auto typeInfo, auto edges_const, auto warps_const) {
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t = typename decltype(typeInfo)::CudaType;
                constexpr int EDGES_PER_BLOCK = edges_const.value;
                constexpr int WARPS_PER_BLOCK = warps_const.value;
                constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

                auto* X_ptr = reinterpret_cast<const cuda_t*>(X.data_ptr<torch_t>());

                dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                reduction_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, WARPS_PER_BLOCK, cuda_t, Op><<<grid, THREADS_PER_BLOCK>>>(
                    heavy_nodes.data_ptr<int>(),
                    edge_ptr.data_ptr<int>(),
                    edge_idx.data_ptr<int>(),
                    X_ptr,
                    reinterpret_cast<unsigned long long*>(packed.template data_ptr<int64_t>()),
                    d
                );
            },
            MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
            MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(edges_per_block_heavy_nodes),
            MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
            );

            std::visit([&](auto typeInfo, auto warps_const) {
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t = typename decltype(typeInfo)::CudaType;
                constexpr int WARPS_PER_BLOCK = warps_const.value;
                constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

                auto* out_ptr = reinterpret_cast<cuda_t*>(out.data_ptr<torch_t>());

                int unpack_blocks = (num_heavy * d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                unpack_results_kernel<WARPS_PER_BLOCK, cuda_t><<<unpack_blocks, THREADS_PER_BLOCK>>>(
                    reinterpret_cast<unsigned long long*>(packed.template data_ptr<int64_t>()),
                    heavy_nodes.data_ptr<int>(),
                    out_ptr,
                    arg_idx.data_ptr<int>(),
                    num_heavy,
                    d
                );
            },
            MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
            MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
            );
        }
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
    const std::string& reduce
) {
    if (reduce == "min") {
        reduction_aggr_forward_partitioned_cuda_impl<ReductionOp::MIN>(
            edge_ptr, edge_idx, X, light_nodes, heavy_nodes, max_degree,
            out, arg_idx, warps_per_block, edges_per_block_heavy_nodes,
            use_2d_kernel, features_per_block, tiles_y);
    } else if (reduce == "max") {
        reduction_aggr_forward_partitioned_cuda_impl<ReductionOp::MAX>(
            edge_ptr, edge_idx, X, light_nodes, heavy_nodes, max_degree,
            out, arg_idx, warps_per_block, edges_per_block_heavy_nodes,
            use_2d_kernel, features_per_block, tiles_y);
    } else {
        TORCH_CHECK(false, "Unsupported reduce: " + reduce);
    }
}

void reduction_aggr_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& arg_idx,
    at::Tensor& grad_x,
    int warps_per_block = 8
) {
    const int num_nodes = grad_out.size(0);
    const int d = grad_out.size(1);
    const dim3 blocks(num_nodes);

    std::visit([&](auto typeInfo, auto warps_const) {
        using torch_t = typename decltype(typeInfo)::TorchType;
        using cuda_t = typename decltype(typeInfo)::CudaType;
        constexpr int WARPS_PER_BLOCK = warps_const.value;
        constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

        auto* grad_out_ptr = reinterpret_cast<const cuda_t*>(grad_out.data_ptr<torch_t>());
        auto* grad_x_ptr = reinterpret_cast<cuda_t*>(grad_x.data_ptr<torch_t>());

        const dim3 threads(THREADS_PER_BLOCK);

        reduction_aggr_backward_typed<WARPS_PER_BLOCK, cuda_t><<<blocks, threads>>>(
            grad_out_ptr,
            arg_idx.data_ptr<int>(),
            grad_x_ptr,
            num_nodes,
            d
        );
    },
    MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
    MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
    );

    CUDA_KERNEL_CHECK();
}
