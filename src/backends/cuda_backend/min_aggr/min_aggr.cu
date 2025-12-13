#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>


constexpr int THREADS_PER_BLOCK = 64;
constexpr int WARP_SIZE = 32;

__global__ void min_aggr_forward_base(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int num_nodes,
    int d
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_nodes) {
        return;
    }

    int tid = threadIdx.x;
    int row_start = edge_ptr[block_idx];
    int row_end = edge_ptr[block_idx + 1];

    #pragma unroll
    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        float best_val = INFINITY;
        int best_src = -1;
        for (int eid = row_start; eid < row_end; ++eid) {
            int src = edge_idx[eid];
            float v = X[src * d + f];
            if (v < best_val) {
                best_val = v;
                best_src = src;
            }
        }

        out[block_idx * d + f] = best_val;
        argmin[block_idx * d + f] = best_src;
    }
}

constexpr int F_TILE = 32; // признаки по иксу
constexpr int NEI_TILE = 16; // соседи за один заход по игреку

__global__ void min_aggr_forward_upgrade(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int num_nodes,
    int d
) {
    int v = blockIdx.x; // блок на вершину
    if (v >= num_nodes) {
        return;
    }

    int f_block = blockIdx.y; // тайл по фичам
    int f0 = f_block * F_TILE; // первый признак в тайле

    int tx = threadIdx.x;
    int ty = threadIdx.y; // текущий сосед в тайле
    int f = f0 + tx; // текущий признак

    int row_start = edge_ptr[v];
    int row_end = edge_ptr[v + 1];

    __shared__ float tile_vals[NEI_TILE][F_TILE]; // значение f у ty соседа в текущем тайле
    __shared__ int tile_src[NEI_TILE][F_TILE]; // вершина сосед у которого взяли значение

    float best_val = INFINITY;
    int best_src = -1;

    for (int base = row_start; base < row_end; base += NEI_TILE) {
        int eid = base + ty; // номер ребра который щас обрабатываем
        float val = INFINITY;
        int src = -1;
        if (eid < row_end && f < d) {
            src = edge_idx[eid];
            val = X[src * d + f];
        }

        tile_vals[ty][tx] = val;
        tile_src[ty][tx] = src;

        __syncthreads();

        for (int stride = NEI_TILE / 2; stride > 0; stride /= 2) {
            if (ty < stride) {
                float v1 = tile_vals[ty][tx];
                float v2 = tile_vals[ty + stride][tx];
                int s1 = tile_src[ty][tx];
                int s2 = tile_src[ty + stride][tx];
                if (v2 < v1) {
                    tile_vals[ty][tx] = v2;
                    tile_src[ty][tx] = s2;
                }
            }
            __syncthreads();
        }

        if (ty == 0 && f < d) {
            float tile_best_val = tile_vals[0][tx];
            int tile_best_src = tile_src[0][tx];
            if (tile_best_val < best_val) {
                best_val = tile_best_val;
                best_src = tile_best_src;
            }
        }

        __syncthreads();
    }

    if (ty == 0 && f < d) {
        out[v * d + f] = best_val;
        argmin[v * d + f] = best_src;
    }
}

__device__ __forceinline__ void min_aggr_forward_small(
    int v,
    int row_start,
    int row_end,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    int f0 = blockIdx.y * F_TILE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (ty != 0) {
        return;
    }

    int f = f0 + tx;
    if (f >= d) return;

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

__device__ __forceinline__ void min_aggr_forward_large(
    int v,
    int row_start,
    int row_end,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    int f0 = blockIdx.y * F_TILE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int f = f0 + tx;

    __shared__ float tile_vals[NEI_TILE][F_TILE];
    __shared__ int   tile_src[NEI_TILE][F_TILE];

    float best_val = INFINITY;
    int best_src = -1;

    for (int base = row_start; base < row_end; base += NEI_TILE) {
        int eid = base + ty;
        float val = INFINITY;
        int src = -1;
        if (eid < row_end && f < d) {
            src = edge_idx[eid];
            val = X[src * d + f];
        }

        tile_vals[ty][tx] = val;
        tile_src[ty][tx]  = src;

        __syncthreads();

        for (int stride = NEI_TILE / 2; stride > 0; stride >>= 1) {
            if (ty < stride) {
                float v1 = tile_vals[ty][tx];
                float v2 = tile_vals[ty + stride][tx];
                int s1 = tile_src[ty][tx];
                int s2 = tile_src[ty + stride][tx];
                if (v2 < v1) {
                    tile_vals[ty][tx] = v2;
                    tile_src[ty][tx]  = s2;
                }
            }
            __syncthreads();
        }

        if (ty == 0 && f < d) {
            float tile_best_val = tile_vals[0][tx];
            int tile_best_src = tile_src[0][tx];
            if (tile_best_val < best_val) {
                best_val = tile_best_val;
                best_src = tile_best_src;
            }
        }

        __syncthreads();
    }

    if (ty == 0 && f < d) {
        out[v * d + f] = best_val;
        argmin[v * d + f] = best_src;
    }
}

__global__ void min_aggr_forward(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int num_nodes,
    int d
) {
    int v = blockIdx.x;
    if (v >= num_nodes) {
        return;
    }

    int row_start = edge_ptr[v];
    int row_end = edge_ptr[v + 1];
    int degree = row_end - row_start;

    if (degree <= WARP_SIZE) {
        min_aggr_forward_small(v, row_start, row_end, edge_idx, X, out, argmin, d);
    } else {
        min_aggr_forward_large(v, row_start, row_end, edge_idx, X, out, argmin, d);
    }
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

    #pragma unroll
    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        int src = argmin[block_idx * d + f];
        if (src < 0) {
            continue;
        }

        float grad = grad_out[block_idx * d + f];
        atomicAdd(&grad_x[src * d + f], grad);
    }
}

// void min_aggr_forward_cuda(
//     const at::Tensor& edge_ptr,
//     const at::Tensor& edge_idx,
//     const at::Tensor& X,
//     at::Tensor& out,
//     at::Tensor& argmin
// ) {
//     const int num_nodes = X.size(0);
//     const int d = X.size(1);

//     const dim3 blocks(num_nodes);
//     const dim3 threads(THREADS_PER_BLOCK);
//     min_aggr_forward<<<blocks, threads>>>(
//         edge_ptr.data_ptr<int>(),
//         edge_idx.data_ptr<int>(),
//         X.data_ptr<float>(),
//         out.data_ptr<float>(),
//         argmin.data_ptr<int>(),
//         num_nodes,
//         d
//     );

//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     TORCH_CHECK(err == cudaSuccess, "forward kernel launch failed: ", cudaGetErrorString(err));
// }

void min_aggr_forward_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    at::Tensor& out,
    at::Tensor& argmin
) {
    const int num_nodes = out.size(0);
    const int d = out.size(1);

    const int num_f_blocks = (d + F_TILE - 1) / F_TILE;
    const dim3 blocks(num_nodes, num_f_blocks);
    const dim3 threads(F_TILE, NEI_TILE);
    min_aggr_forward<<<blocks, threads>>>(
        edge_ptr.data_ptr<int>(),
        edge_idx.data_ptr<int>(),
        X.data_ptr<float>(),
        out.data_ptr<float>(),
        argmin.data_ptr<int>(),
        num_nodes,
        d
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "forward kernel launch failed: ", cudaGetErrorString(err));
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
