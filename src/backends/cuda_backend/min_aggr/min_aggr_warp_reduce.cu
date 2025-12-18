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
    const int num_f_blocks = (d + F_TILE - 1) / F_TILE;

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

    // const dim3 threads(F_TILE, NEI_TILE);

    const dim3 heavy_threads(F_TILE, NEI_TILE);
    const int LIGHT_THREADS = 256;

    // std::cout << "light=" << light_nodes.numevl() << " heavy=" << heavy_nodes.numel() << " d=" << d << std::endl;

    if (light_nodes.numel() > 0) {
        const int num_light = light_nodes.numel();
        const dim3 blocks(num_light);
        min_aggr_forward_light_kernel_1d<<<blocks, LIGHT_THREADS>>>(
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
        const dim3 blocks(num_heavy, num_f_blocks);
        min_aggr_forward_heavy_kernel<<<blocks, heavy_threads>>>(
            heavy_nodes.data_ptr<int>(),
            edge_ptr.data_ptr<int>(),
            edge_idx.data_ptr<int>(),
            X.data_ptr<float>(),
            out.data_ptr<float>(),
            argmin.data_ptr<int>(),
            d
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "min_aggr_forward_partitioned_cuda failed: ", cudaGetErrorString(err));
}

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
