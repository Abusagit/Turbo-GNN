#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_fp16.h>

#define FULL_WARP_MASK 0xffffffff

constexpr int kWarpSize = 32;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t val);

template <>
__device__ __forceinline__ float to_float<float>(float val) {
    return val;
}

template <>
__device__ __forceinline__ float to_float<double>(double val) {
    return __double2float_rn(val);
}

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half val) {
    return __half2float(__half(val));
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 val) {
    return __bfloat162float(__nv_bfloat16(val));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ double from_float<double>(float v) {
    return static_cast<double>(v);
}

template <>
__device__ __forceinline__ at::Half from_float<at::Half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename scalar_t>
__global__ void min_aggr_forward_light_kernel_1d(
    const int* __restrict__ light_nodes_indices,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    int i = blockIdx.x;
    int v = light_nodes_indices[i];

    int row_start = edge_ptr[v];
    int row_end   = edge_ptr[v + 1];

    int tid = threadIdx.x;

    for (int f = tid; f < d; f += blockDim.x) {
        scalar_t best_val = from_float<scalar_t>(INFINITY);
        int best_src = -1;

        for (int eid = row_start; eid < row_end; ++eid) {
            int src = edge_idx[eid];
            scalar_t val = X[src * d + f];
            if (val < best_val) {
                best_val = val;
                best_src = src;
            }
        }

        out[v * d + f] = best_val;
        argmin[v * d + f] = best_src;
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
template<int EDGES_PER_BLOCK, typename scalar_t>
__global__ void min_aggr_forward_heavy_kernel(
    const int* __restrict__ heavy_nodes_indices,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const scalar_t* __restrict__ X,
    unsigned long long* __restrict__ packed,
    int d
) {
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

    for (int f = tid; f < d; f += blockDim.x) {
        float local_min = INFINITY;
        int local_arg = -1;

        // find local minimum in this chunk
        #pragma unroll
        for (int eid = chunk_start; eid < chunk_end; ++eid) {
            int src = edge_idx[eid];
            float val = to_float(X[src * d + f]);
            if (val < local_min) {
                local_min = val;
                local_arg = src;
            }
        }

        if (local_arg >= 0) {
            unsigned long long* addr = &packed[node_idx * d + f];
            unsigned long long new_val = pack_val_idx(local_min, local_arg);
            atomicMin(addr, new_val);
        }
    }
}

// unpack results back to separate arrays
template <typename scalar_t>
__global__ void unpack_results_kernel(
    const unsigned long long* __restrict__ packed,
    const int* __restrict__ nodes,
    scalar_t* __restrict__ out,
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

        out[v * d + f] = from_float<scalar_t>(val);
        argmin[v * d + f] = idx;
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
    int THREADS_PER_BLOCK;

    if (warps_per_block == 1){
        THREADS_PER_BLOCK = 1 * kWarpSize;
    } else if (warps_per_block == 2) {
        THREADS_PER_BLOCK = 2 * kWarpSize;
    } else if (warps_per_block == 4) {
        THREADS_PER_BLOCK = 4 * kWarpSize;
    } else if (warps_per_block == 8) {
        THREADS_PER_BLOCK = 8 * kWarpSize;
    } else if (warps_per_block == 16) {
        THREADS_PER_BLOCK = 16 * kWarpSize;
    } else if (warps_per_block == 32) {
        THREADS_PER_BLOCK = 32 * kWarpSize;
    } else {
        THREADS_PER_BLOCK = 2048;
    }

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
    TORCH_CHECK(X.scalar_type() == at::kFloat || X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16 || X.scalar_type() == at::kDouble, "X must be float32/float16/bfloat16/float64");
    TORCH_CHECK(out.scalar_type() == X.scalar_type(), "out must have same dtype as X");

    if (light_nodes.numel() > 0) {
        const int num_light = light_nodes.numel();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            X.scalar_type(),
            "min_aggr_forward_light",
            ([&] {
                min_aggr_forward_light_kernel_1d<scalar_t>
                    <<<num_light, THREADS_PER_BLOCK>>>(
                        light_nodes.data_ptr<int>(),
                        edge_ptr.data_ptr<int>(),
                        edge_idx.data_ptr<int>(),
                        X.data_ptr<scalar_t>(),
                        out.data_ptr<scalar_t>(),
                        argmin.data_ptr<int>(),
                        d
                    );
            })
        );
    }

    const int num_heavy = heavy_nodes.numel();

    if (num_heavy > 0) {
        // unsigned long long packing_init_val = pack_val_idx(INFINITY, -1);
        constexpr unsigned long long PACKED_INIT = 0xff800000ffffffffULL;

        // TODO: think about what to do with the doubles
        if (X.scalar_type() == at::kDouble) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                X.scalar_type(),
                "min_aggr_forward_heavy_fallback_double",
                ([&] {
                    min_aggr_forward_light_kernel_1d<scalar_t>
                        <<<num_heavy, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            out.data_ptr<scalar_t>(),
                            argmin.data_ptr<int>(),
                            d
                        );
                })
            );
        } else {
            auto packed = at::full(
                {num_heavy, d},
                static_cast<int64_t>(PACKED_INIT),
                at::TensorOptions().dtype(torch::kInt64).device(X.device())
            );

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                X.scalar_type(),
                "min_aggr_forward_heavy",
                ([&] {
                    if (edges_per_block_heavy_nodes == 32){
                        constexpr int EDGES_PER_BLOCK = 32;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );

                    } else if (edges_per_block_heavy_nodes == 64) {
                        constexpr int EDGES_PER_BLOCK = 64;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );

                    } else if (edges_per_block_heavy_nodes == 128) {
                        constexpr int EDGES_PER_BLOCK = 128;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );
                    } else if (edges_per_block_heavy_nodes == 256) {
                        constexpr int EDGES_PER_BLOCK = 256;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );
                    } else if (edges_per_block_heavy_nodes == 512) {
                        constexpr int EDGES_PER_BLOCK = 512;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );
                    } else if (edges_per_block_heavy_nodes == 1024) {
                        constexpr int EDGES_PER_BLOCK = 1024;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );
                    } else {
                        constexpr int EDGES_PER_BLOCK = 2048;
                        dim3 grid(num_heavy, (max_degree + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK);

                        min_aggr_forward_heavy_kernel<EDGES_PER_BLOCK, scalar_t><<<grid, THREADS_PER_BLOCK>>>(
                            heavy_nodes.data_ptr<int>(),
                            edge_ptr.data_ptr<int>(),
                            edge_idx.data_ptr<int>(),
                            X.data_ptr<scalar_t>(),
                            (unsigned long long*)packed.data_ptr<int64_t>(),
                            d
                        );
                    }

                    int unpack_blocks = (num_heavy * d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                    unpack_results_kernel<scalar_t><<<unpack_blocks, THREADS_PER_BLOCK>>>(
                        (unsigned long long*)packed.data_ptr<int64_t>(),
                        heavy_nodes.data_ptr<int>(),
                        out.data_ptr<scalar_t>(),
                        argmin.data_ptr<int>(),
                        num_heavy,
                        d
                    );
                })
            );
        }
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "min_aggr_forward_partitioned_cuda failed: ", cudaGetErrorString(err));
}

template <typename scalar_t>
__device__ __forceinline__ void atomic_add_wrapper(scalar_t* address, scalar_t val) {
    atomicAdd(address, val);
}

template <>
__device__ __forceinline__ void atomic_add_wrapper<at::Half>(at::Half* address, at::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), *reinterpret_cast<__half*>(&val));
}

template <>
__device__ __forceinline__ void atomic_add_wrapper<at::BFloat16>(at::BFloat16* address, at::BFloat16 val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), *reinterpret_cast<__nv_bfloat16*>(&val));
}

template <typename scalar_t>
__global__ void min_aggr_backward_typed(
    const scalar_t* __restrict__ grad_out,
    const int*   __restrict__ argmin,
    scalar_t* __restrict__ grad_x,
    int num_nodes,
    int d
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    int tid = threadIdx.x;

    #pragma unroll
    for (int f = tid; f < d; f += blockDim.x) {
        int src = argmin[block_idx * d + f];
        if (src < 0) {
            continue;
        }

        scalar_t grad = grad_out[block_idx * d + f];
        atomic_add_wrapper(&grad_x[src * d + f], grad);
    }
}

void min_aggr_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& argmin,
    at::Tensor& grad_x,
    int warps_per_block = 8
) {
    int THREADS_PER_BLOCK;
    const int num_nodes = grad_out.size(0);
    const int d = grad_out.size(1);

    const dim3 blocks(num_nodes);

    if (warps_per_block == 1){
        THREADS_PER_BLOCK = 1 * kWarpSize;
    } else if (warps_per_block == 2) {
        THREADS_PER_BLOCK = 2 * kWarpSize;
    } else if (warps_per_block == 4) {
        THREADS_PER_BLOCK = 4 * kWarpSize;
    } else if (warps_per_block == 8) {
        THREADS_PER_BLOCK = 8 * kWarpSize;
    } else if (warps_per_block == 16) {
        THREADS_PER_BLOCK = 16 * kWarpSize;
    } else if (warps_per_block == 32) {
        THREADS_PER_BLOCK = 32 * kWarpSize;
    } else {
        THREADS_PER_BLOCK = 2048;
    }

    const dim3 threads(THREADS_PER_BLOCK);
    auto st = grad_out.scalar_type();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        st,
        "min_aggr_backward_float_double",
        ([&] {
            min_aggr_backward_typed<scalar_t><<<blocks, threads>>>(
                grad_out.data_ptr<scalar_t>(),
                argmin.data_ptr<int>(),
                grad_x.data_ptr<scalar_t>(),
                num_nodes,
                d
            );
        })
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "backward kernel launch failed: ", cudaGetErrorString(err));
}
