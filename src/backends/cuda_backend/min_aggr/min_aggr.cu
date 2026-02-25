#include "../common.cuh"

template <typename cuda_t>
__global__ void min_aggr_forward_light_kernel_1d(
    const int* __restrict__ light_nodes_indices,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const cuda_t* __restrict__ X,
    cuda_t* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    constexpr int VW = (sizeof(cuda_t) <= 2) ? 2 : 1;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;

    int i = blockIdx.x;
    int v = light_nodes_indices[i];

    int row_start = edge_ptr[v];
    int row_end   = edge_ptr[v + 1];

    int tid = threadIdx.x;

    int node_stride = v * d;

    cuda_t inf_val = make_cuda_value<cuda_t>(INFINITY);
    cuda_t zero_val = make_cuda_value<cuda_t>(0.0f);

    const int d_vec = d / EPV;

    for (int fv = tid; fv < d_vec; fv += blockDim.x) {
        const int base_f = fv * EPV;

        cuda_t best_vals[EPV];
        int best_srcs[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            best_vals[e] = inf_val;
            best_srcs[e] = -1;
        }

        for (int eid = row_start; eid < row_end; ++eid) {
            int src = edge_idx[eid];
            auto val = Tile::load(&X[src * d], fv);
            #pragma unroll
            for (int e = 0; e < EPV; ++e) {
                cuda_t v_e = Tile::extract(val, e);
                if (v_e < best_vals[e]) { best_vals[e] = v_e; best_srcs[e] = src; }
            }
        }

        cuda_t result[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            result[e] = (best_srcs[e] != -1) ? best_vals[e] : zero_val;
            argmin[node_stride + base_f + e] = best_srcs[e];
        }
        Tile::store_vec(&out[node_stride], fv, Tile::build(result));
    }

    // Scalar tail for d % EPV != 0 (compiles away for EPV=1)
    if (d % EPV != 0 && tid == 0) {
        for (int f = d_vec * EPV; f < d; ++f) {
            cuda_t best_val = inf_val;
            int best_src = -1;
            for (int eid = row_start; eid < row_end; ++eid) {
                int src = edge_idx[eid];
                cuda_t val = X[src * d + f];
                if (val < best_val) { best_val = val; best_src = src; }
            }
            out[node_stride + f] = (best_src != -1) ? best_val : zero_val;
            argmin[node_stride + f] = best_src;
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
template<int EDGES_PER_BLOCK, typename cuda_t>
__global__ void min_aggr_forward_heavy_kernel(
    const int* __restrict__ heavy_nodes_indices,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const cuda_t* __restrict__ X,
    unsigned long long* __restrict__ packed,
    int d
) {
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
    cuda_t inf_val = make_cuda_value<cuda_t>(INFINITY);

    const int d_vec = d / EPV;

    for (int fv = tid; fv < d_vec; fv += blockDim.x) {
        const int base_f = fv * EPV;

        cuda_t best_vals[EPV];
        int best_srcs[EPV];
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            best_vals[e] = inf_val;
            best_srcs[e] = -1;
        }

        #pragma unroll
        for (int eid = chunk_start; eid < chunk_end; ++eid) {
            int src = edge_idx[eid];
            auto val = Tile::load(&X[src * d], fv);
            #pragma unroll
            for (int e = 0; e < EPV; ++e) {
                cuda_t v_e = Tile::extract(val, e);
                if (v_e < best_vals[e]) { best_vals[e] = v_e; best_srcs[e] = src; }
            }
        }

        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            if (best_srcs[e] >= 0) {
                unsigned long long new_val = pack_val_idx(cuda_to_float(best_vals[e]), best_srcs[e]);
                atomicMin(&packed[node_idx * d + base_f + e], new_val);
            }
        }
    }

    // scalar tail for d % EPV != 0
    if (d % EPV != 0 && tid == 0) {
        for (int f = d_vec * EPV; f < d; ++f) {
            cuda_t local_min = inf_val;
            int local_arg = -1;

            #pragma unroll
            for (int eid = chunk_start; eid < chunk_end; ++eid) {
                int src = edge_idx[eid];
                cuda_t val = X[src * d + f];
                if (val < local_min) { local_min = val; local_arg = src; }
            }

            if (local_arg >= 0) {
                unsigned long long new_val = pack_val_idx(cuda_to_float(local_min), local_arg);
                atomicMin(&packed[node_idx * d + f], new_val);
            }
        }
    }
}

// unpack results back to separate arrays
template <typename cuda_t>
__global__ void unpack_results_kernel(
    const unsigned long long* __restrict__ packed,
    const int* __restrict__ nodes,
    cuda_t* __restrict__ out,
    int* __restrict__ argmin,
    int num_nodes,
    int d
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num_nodes * d; i += gridDim.x * blockDim.x) {
        int node_idx = i / d;
        int f = i % d;
        int v = nodes[node_idx];

        float val;
        int idx;
        unpack_val_idx(packed[node_idx * d + f], val, idx);

        out[v * d + f] = (idx > -1) ? make_cuda_value<cuda_t>(val) : make_cuda_value<cuda_t>(0.0f);
        argmin[v * d + f] = idx;
    }
}

template <typename cuda_t>
__global__ void min_aggr_backward_typed(
    const cuda_t* __restrict__ grad_out,
    const int*   __restrict__ argmin,
    cuda_t* __restrict__ grad_x,
    int num_nodes,
    int d
) {
    constexpr int VW = (sizeof(cuda_t) <= 2) ? 2 : 1;
    using Tile = TileOps<VW, cuda_t>;
    constexpr int EPV = Tile::ELEM_PER_VEC;

    int block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    int tid = threadIdx.x;
    const int base_offset = block_idx * d;
    const int d_vec = d / EPV;

    for (int fv = tid; fv < d_vec; fv += blockDim.x) {
        const int base_f = fv * EPV;
        auto grad_v = Tile::load(&grad_out[base_offset], fv);
        #pragma unroll
        for (int e = 0; e < EPV; ++e) {
            int src = argmin[base_offset + base_f + e];
            if (src >= 0) {
                atomicAdd(&grad_x[src * d + base_f + e], Tile::extract(grad_v, e));
            }
        }
    }

    // scalar tail for d % EPV != 0
    if (d % EPV != 0 && tid == 0) {
        for (int f = d_vec * EPV; f < d; ++f) {
            int src = argmin[base_offset + f];
            if (src >= 0) {
                atomicAdd(&grad_x[src * d + f], grad_out[base_offset + f]);
            }
        }
    }
}

void min_aggr_forward_partitioned_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    const at::Tensor& light_nodes,
    const at::Tensor& heavy_nodes,
    int max_degree,
    at::Tensor& out,
    at::Tensor& argmin,
    int warps_per_block = 8,
    int edges_per_block_heavy_nodes = 128
) {

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

            min_aggr_forward_light_kernel_1d<cuda_t><<<num_light, THREADS_PER_BLOCK>>>(
                light_nodes.data_ptr<int>(),
                edge_ptr.data_ptr<int>(),
                edge_idx.data_ptr<int>(),
                X_ptr,
                out_ptr,
                argmin.data_ptr<int>(),
                d
            );
        },
        MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
        MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)  // 64*32=2048
        );
    }

    const int num_heavy = heavy_nodes.numel();

    if (num_heavy > 0) {
        constexpr unsigned long long PACKED_INIT = 0xff800000ffffffffULL;

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

            min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, cuda_t><<<grid, THREADS_PER_BLOCK>>>(
                heavy_nodes.data_ptr<int>(),
                edge_ptr.data_ptr<int>(),
                edge_idx.data_ptr<int>(),
                X_ptr,
                reinterpret_cast<unsigned long long*>(packed.data_ptr<int64_t>()),
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
            unpack_results_kernel<cuda_t><<<unpack_blocks, THREADS_PER_BLOCK>>>(
                reinterpret_cast<unsigned long long*>(packed.data_ptr<int64_t>()),
                heavy_nodes.data_ptr<int>(),
                out_ptr,
                argmin.data_ptr<int>(),
                num_heavy,
                d
            );
        },
        MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
        MakeIntVariant<1, 2, 4, 8, 16, 32, 64>(warps_per_block)
        );
    }
    CUDA_KERNEL_CHECK();
}

void min_aggr_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& argmin,
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

        min_aggr_backward_typed<cuda_t><<<blocks, threads>>>(
            grad_out_ptr,
            argmin.data_ptr<int>(),
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
