#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>

#define FULL_WARP_MASK 0xffffffff

constexpr int kWarpSize = 32;

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

constexpr int F_TILE = 32;
constexpr int NEI_TILE = 32;

__device__ __forceinline__ void warp_reduce_argmin(float &val, int &src) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        float v2 = __shfl_xor_sync(FULL_WARP_MASK, val, offset);
        int s2 = __shfl_xor_sync(FULL_WARP_MASK, src, offset);
        if (v2 < val) {
            val = v2;
            src = s2;
        }
    }
}

__global__ void min_aggr_forward_light_kernel_1d(
    const int* __restrict__ nodes,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    int i = blockIdx.x;
    int v = nodes[i];

    int row_start = edge_ptr[v];
    int row_end   = edge_ptr[v + 1];

    int tid = threadIdx.x;

    for (int f = tid; f < d; f += blockDim.x) {
        float best_val = INFINITY;
        int best_src = -1;

        for (int eid = row_start; eid < row_end; ++eid) {
            int src = edge_idx[eid];
            float val = X[src * d + f];
            if (val < best_val) {
                best_val = val;
                best_src = src;
            }
        }

        out[v * d + f] = best_val;
        argmin[v * d + f] = best_src;
    }
}

__global__ void min_aggr_forward_heavy_kernel(
    const int* __restrict__ nodes,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    int i = blockIdx.x;
    int v = nodes[i];
    int row_start = edge_ptr[v];
    int row_end = edge_ptr[v + 1];
    int f0 = blockIdx.y * F_TILE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int fx = f0 + tx;
    int fy = f0 + ty;

    // __shared__ float tile_vals[NEI_TILE][F_TILE];
    // __shared__ int   tile_src[NEI_TILE][F_TILE];


    __shared__ float tile_vals[F_TILE][NEI_TILE + 1];
    __shared__ int   tile_src[NEI_TILE];

    __shared__ float out_vals[F_TILE];
    __shared__ int   out_srcs[F_TILE];

    float best_val = INFINITY;
    int best_src = -1;

    for (int base = row_start; base < row_end; base += NEI_TILE) {
        int eid = base + ty;
        float val = INFINITY;
        int src = -1;
        if (eid < row_end && fx < d) {
            src = edge_idx[eid];
            val = X[src * d + fx];
        }

        // Steps to switch to warp-reduce



        // Transposed shared memory layout
        // ++++ !!!!!!!! (0.5) Reduce shared memory banck conflicts similar to matrix tranposition
        tile_vals[tx][ty] = val;
        if (tx == 0) {
            tile_src[ty]  = src;
        }

        // SHMEM layout:
        // Neighbor_i_feature_j                | Neighbor_{i + 1}_feature_j                | ... | Neighbor_{i + NEI_TILE - 1}_feature_j
        // ...
        // Neighbor_i_feature_{j + F_TILE - 1} | Neighbor_{i + 1}_feature_{j + F_TILE - 1} | ... | Neighbor_{i + NEI_TILE - 1}_feature_{j + F_TILE - 1}

        __syncthreads();

        val = tile_vals[ty][tx];
        src = tile_src[tx];

        warp_reduce_argmin(val, src);

        if (tx == 0 && fy < d) {
            if (val < best_val) {
                best_val = val;
                best_src = src;
            }
        }

        __syncthreads();
    }

    if (tx == 0) {
        out_vals[ty] = best_val;
        out_srcs[ty] = best_src;
    }

    __syncthreads();

    if (ty == 0 && fx < d) {
        out[v * d + fx] = out_vals[tx];
        argmin[v * d + fx] = out_srcs[tx];
    }
}



// IDEA: process wedges via chunks in parallel
constexpr int EDGES_PER_BLOCK = 128;
constexpr int MAX_CHUNKS = 512;

// pack float and int into uint64 for atomic updates
__device__ __forceinline__ unsigned long long pack_val_idx(float val, int idx) {
    return ((unsigned long long)__float_as_uint(val) << 32) | (unsigned long long)(unsigned int)idx;
}

// unpack float and int from uint64
__device__ __forceinline__ void unpack_val_idx(unsigned long long packed, float& val, int& idx) {
    val = __uint_as_float((unsigned int)(packed >> 32));
    idx = (int)(unsigned int)(packed & 0xFFFFFFFFULL);
}

// initialize packed buffer for heavy nodes
__global__ void init_packed_kernel(
    unsigned long long* __restrict__ packed,
    const int* __restrict__ nodes,
    int num_nodes,
    int d
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long init_val = pack_val_idx(INFINITY, -1);

    for (int i = tid; i < num_nodes * d; i += gridDim.x * blockDim.x) {
        int node_idx = i / d;
        int f = i % d;
        int v = nodes[node_idx];
        packed[v * d + f] = init_val;
    }
}

// 2D kernel: blockIdx.x = node, blockIdx.y = edge chunk
__global__ void min_aggr_forward_heavy_kernel(
    const int* __restrict__ nodes,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    unsigned long long* __restrict__ packed,
    int d
) {
    int node_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int v = nodes[node_idx];

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
        for (int eid = chunk_start; eid < chunk_end; ++eid) {
            int src = edge_idx[eid];
            float val = X[src * d + f];
            if (val < local_min) {
                local_min = val;
                local_arg = src;
            }
        }

        // atomic update using 64-bit CAS
        if (local_arg >= 0) {
            unsigned long long* addr = &packed[v * d + f];
            unsigned long long new_val = pack_val_idx(local_min, local_arg);
            unsigned long long old = *addr;

            while (new_val < old) {
                unsigned long long assumed = old;
                old = atomicCAS(addr, assumed, new_val);
                if (old == assumed) break;
            }
        }
    }
}

// unpack results back to separate arrays
__global__ void unpack_results_kernel(
    const unsigned long long* __restrict__ packed,
    const int* __restrict__ nodes,
    float* __restrict__ out,
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
        unpack_val_idx(packed[v * d + f], val, idx);

        out[v * d + f] = val;
        argmin[v * d + f] = idx;
    }
}


void min_aggr_forward_partitioned_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    const at::Tensor& light_nodes,
    const at::Tensor& heavy_nodes,
    at::Tensor& out,
    at::Tensor& argmin
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
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");

    if (light_nodes.numel() > 0) {
        const int num_light = light_nodes.numel();
        min_aggr_forward_light_kernel_1d<<<num_light, THREADS_PER_BLOCK>>>(
            light_nodes.data_ptr<int>(),
            edge_ptr.data_ptr<int>(),
            edge_idx.data_ptr<int>(),
            X.data_ptr<float>(),
            out.data_ptr<float>(),
            argmin.data_ptr<int>(),
            d
        );
    }

    if (heavy_nodes.numel() > 0) {
        const int num_heavy = heavy_nodes.numel();

        auto packed = at::empty({num_heavy, d},
            torch::dtype(torch::kInt64).device(X.device()));

        int init_blocks = (num_heavy * d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        init_packed_kernel<<<init_blocks, THREADS_PER_BLOCK>>>(
            (unsigned long long*)packed.data_ptr<int64_t>(),
            heavy_nodes.data_ptr<int>(),
            num_heavy,
            d
        );

        dim3 grid(num_heavy, MAX_CHUNKS);
        min_aggr_forward_heavy_kernel<<<grid, THREADS_PER_BLOCK>>>(
            heavy_nodes.data_ptr<int>(),
            edge_ptr.data_ptr<int>(),
            edge_idx.data_ptr<int>(),
            X.data_ptr<float>(),
            (unsigned long long*)packed.data_ptr<int64_t>(),
            d
        );
        int unpack_blocks = (num_heavy * d + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        unpack_results_kernel<<<unpack_blocks, THREADS_PER_BLOCK>>>(
            (unsigned long long*)packed.data_ptr<int64_t>(),
            heavy_nodes.data_ptr<int>(),
            out.data_ptr<float>(),
            argmin.data_ptr<int>(),
            num_heavy,
            d
        );
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "min_aggr_forward_partitioned_cuda failed: ", cudaGetErrorString(err));
}

// void min_aggr_forward_partitioned_cuda(
//     const at::Tensor& edge_ptr,
//     const at::Tensor& edge_idx,
//     const at::Tensor& X,
//     const at::Tensor& light_nodes,
//     const at::Tensor& heavy_nodes,
//     at::Tensor& out,
//     at::Tensor& argmin
// ) {
//     const int d = X.size(1);
//     const int num_f_blocks = (d + F_TILE - 1) / F_TILE;

//     TORCH_CHECK(edge_ptr.is_cuda(), "edge_ptr must be CUDA");
//     TORCH_CHECK(edge_idx.is_cuda(), "edge_idx must be CUDA");
//     TORCH_CHECK(X.is_cuda(), "X must be CUDA");
//     TORCH_CHECK(light_nodes.is_cuda(), "light_nodes must be CUDA");
//     TORCH_CHECK(heavy_nodes.is_cuda(), "heavy_nodes must be CUDA");

//     TORCH_CHECK(edge_ptr.dtype() == torch::kInt32, "edge_ptr must be int32");
//     TORCH_CHECK(edge_idx.dtype() == torch::kInt32, "edge_idx must be int32");
//     TORCH_CHECK(light_nodes.dtype() == torch::kInt32, "light_nodes must be int32");
//     TORCH_CHECK(heavy_nodes.dtype() == torch::kInt32, "heavy_nodes must be int32");
//     TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");


//     if (light_nodes.numel() > 0) {
//         const int num_light = light_nodes.numel();
//         const dim3 blocks(num_light);
//         min_aggr_forward_light_kernel_1d<<<blocks, THREADS_PER_BLOCK>>>(
//             light_nodes.data_ptr<int>(),
//             edge_ptr.data_ptr<int>(),
//             edge_idx.data_ptr<int>(),
//             X.data_ptr<float>(),
//             out.data_ptr<float>(),
//             argmin.data_ptr<int>(),
//             d
//         );
//     }

//     if (heavy_nodes.numel() > 0) {
//         const dim3 heavy_threads(F_TILE, NEI_TILE);

//         const int num_heavy = heavy_nodes.numel();
//         const dim3 blocks(num_heavy, num_f_blocks);
//         size_t shmem_size = (F_TILE * (NEI_TILE + 2)) * sizeof(float) + (F_TILE + NEI_TILE) * sizeof(int);
//         min_aggr_forward_heavy_kernel<<<blocks, heavy_threads, shmem_size>>>(
//             heavy_nodes.data_ptr<int>(),
//             edge_ptr.data_ptr<int>(),
//             edge_idx.data_ptr<int>(),
//             X.data_ptr<float>(),
//             out.data_ptr<float>(),
//             argmin.data_ptr<int>(),
//             d
//         );
//     }

//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     TORCH_CHECK(err == cudaSuccess, "min_aggr_forward_partitioned_cuda failed: ", cudaGetErrorString(err));
// }

__global__ void min_aggr_backward(
    const float* __restrict__ grad_out,
    const int*   __restrict__ argmin,
    float* __restrict__ grad_x,
    int num_nodes,
    int d
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    int tid = threadIdx.x;

    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        int src = argmin[block_idx * d + f];
        if (src < 0) {
            continue;
        }

        float grad = grad_out[block_idx * d + f];
        atomicAdd(&grad_x[src * d + f], grad);
    }
}

void min_aggr_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& argmin,
    at::Tensor& grad_x
) {
    const int num_nodes = grad_out.size(0);
    const int d = grad_out.size(1);

    const dim3 blocks(num_nodes);
    const dim3 threads(THREADS_PER_BLOCK);
    min_aggr_backward<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        argmin.data_ptr<int>(),
        grad_x.data_ptr<float>(),
        num_nodes,
        d
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "backward kernel launch failed: ", cudaGetErrorString(err));
}
