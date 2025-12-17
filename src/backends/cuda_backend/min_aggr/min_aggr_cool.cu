#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>

#define FULL_WARP_MASK 0xffffffff

constexpr int kWarpSize = 32;

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

constexpr int F_TILE = 32;
constexpr int N_TILE = 32;

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

// __global__ void min_aggr_forward_warp_tiled(
//     const int* __restrict__ edge_ptr,
//     const int* __restrict__ edge_idx,
//     const float* __restrict__ X,
//     float* __restrict__ out,
//     int* __restrict__ argmin,
//     int num_nodes,
//     int d
// ) {
//     const int v = blockIdx.x;
//     if (v >= num_nodes) {
//         return;
//     }

//     const int f_block = blockIdx.y;
//     const int f0 = f_block * F_TILE;
//     const int tid = threadIdx.x;
//     const int lane = tid & (kWarpSize - 1);
//     const int wid = tid >> 5;

//     const int row_start = edge_ptr[v];
//     const int row_end = edge_ptr[v + 1];

//     for (int fi = wid; fi < F_TILE; fi += WARPS_PER_BLOCK) {
//         const int f = f0 + fi;
//         if (f >= d) {
//             continue;
//         }

//         float best_val = INFINITY;
//         int best_src = -1;
//         for (int base = row_start; base < row_end; base += N_TILE) {
//             const int eid = base + lane;
//             float val = INFINITY;
//             int src = -1;
//             if (eid < row_end) {
//                 src = edge_idx[eid];
//                 val = X[src * d + f];
//             }

//             warp_reduce_argmin(val, src);

//             if (lane == 0) {
//                 if (val < best_val) {
//                     best_val = val;
//                     best_src = src;
//                 }
//             }
//         }

//         if (lane == 0) {
//             out[v * d + f] = best_val;
//             argmin[v * d + f] = best_src;
//         }
//     }
// }

__global__ void min_aggr_forward_light(
    const int* __restrict__ nodes,
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int d
) {
    int idx = blockIdx.x;
    int v = nodes[idx];

    int lane = threadIdx.x;
    if (lane >= kWarpSize) {
        return;
    }

    int row_start = edge_ptr[v];
    int row_end = edge_ptr[v + 1];

    for (int f = lane; f < d; f += kWarpSize) {
        float best_val = INFINITY;
        int best_src = -1;
        int deg = row_end - row_start;
        if (lane < deg) {
            int src = edge_idx[row_start + lane];
            best_val = X[src * d + f];
            best_src = src;
        }

        warp_reduce_argmin(best_val, best_src);

        if (lane == 0) {
            out[v * d + f] = best_val;
            argmin[v * d + f] = best_src;
        }
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
    int f = f0 + tx;

    // __shared__ float tile_vals[NEI_TILE][F_TILE];
    // __shared__ int   tile_src[NEI_TILE][F_TILE];


    __shared__ float tile_vals[F_TILE][NEI_TILE];
    __shared__ int   tile_src[F_TILE][NEI_TILE];

    float best_val = INFINITY;
    int best_src = -1;

    for (int base = row_start; base < row_end; base += NEI_TILE) {
        int eid = base + ty;
        float val = INFINITY;
        int src = -1;
        if (eid < row_end && f < d) {

            src = edge_idx[eid]; // !!!!!!!!!!! MB Global memory reads are not broadcasted and this transaction is performed multiple times
            val = X[src * d + f];
        }

        // Steps to switch to warp-reduce
        // !!!!!(0) Fix light/heavy nodes mapping



        // Transposed shared memory layout
        // ++++ !!!!!!!! (0.5) Reduce shared memory banck conflicts similar to matrix tranposition
        tile_vals[tx][ty] = val;
        tile_src[tx][ty]  = src;

        // SHMEM layout:
        // Neighbor_i_feature_j                | Neighbor_{i + 1}_feature_j                | ... | Neighbor_{i + NEI_TILE - 1}_feature_j
        // ...
        // Neighbor_i_feature_{j + F_TILE - 1} | Neighbor_{i + 1}_feature_{j + F_TILE - 1} | ... | Neighbor_{i + NEI_TILE - 1}_feature_{j + F_TILE - 1}

        __syncthreads();

        // !!!!! (0.5) Redefines var and src:
        val = tile_vals[ty][tx];
        src = tile_src[ty][tx];

        // !!!!!(1) switch to `warp_reduce_argmin`:

        for (int stride = NEI_TILE / 2; stride > 0; stride >>= 1) {
            if (tx < stride) {
                float v1 = tile_vals[ty][tx];
                float v2 = tile_vals[ty][tx + stride];
                int s1 = tile_src[ty][tx];
                int s2 = tile_src[ty][tx + stride];
                if (v2 < v1) {
                    tile_vals[ty][tx] = v2;
                    tile_src[ty][tx]  = s2;
                }
            }
            __syncthreads();
        }

        if (tx == 0 && f < d) {
            // !!!!!(2) switch to use warp-reduced `src` and `val`:

            float tile_best_val = tile_vals[ty][0];
            int tile_best_src = tile_src[ty][0];
            if (tile_best_val < best_val) {
                best_val = tile_best_val;
                best_src = tile_best_src;
            }
        }

        __syncthreads();
    }

    if (tx == 0 && f < d) {
        out[v * d + f] = best_val;
        argmin[v * d + f] = best_src;
    }
}




__global__ void min_aggr_forward_heavy(
    const int* __restrict__ edge_ptr,
    const int* __restrict__ edge_idx,
    const float* __restrict__ X,
    float* __restrict__ out,
    int* __restrict__ argmin,
    int num_nodes,
    int d
) {
    const int v = blockIdx.x;
    if (v >= num_nodes) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int wid = tid >> 5;

    const int row_start = edge_ptr[v];
    const int row_end = edge_ptr[v + 1];

    for (int f = wid; f < d; f += WARPS_PER_BLOCK) {
        if (f >= d) {
            continue;
        }

        float best_val = INFINITY;
        int best_src = -1;
        for (int base = row_start; base < row_end; base += N_TILE) {
            const int eid = base + lane;
            float val = INFINITY;
            int src = -1;
            if (eid < row_end) {
                src = edge_idx[eid];
                val = X[src * d + f];
            }

            warp_reduce_argmin(val, src);

            if (lane == 0) {
                if (val < best_val) {
                    best_val = val;
                    best_src = src;
                }
            }
        }

        if (lane == 0) {
            out[v * d + f] = best_val;
            argmin[v * d + f] = best_src;
        }
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

    for (int f = tid; f < d; f += THREADS_PER_BLOCK) {
        int src = argmin[block_idx * d + f];
        if (src < 0) {
            continue;
        }

        float grad = grad_out[block_idx * d + f];
        atomicAdd(&grad_x[src * d + f], grad);
    }
}

void min_aggr_forward_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    at::Tensor& out,
    at::Tensor& argmin
) {
     const int num_nodes = out.size(0);
    const int d = out.size(1);

    auto edge_ptr_cpu = edge_ptr.cpu();
    auto acc = edge_ptr_cpu.accessor<int,1>();

    std::vector<int> light, heavy;
    constexpr int DEG_THRESHOLD = 32;

    for (int v = 0; v < num_nodes; ++v) {
        int deg = acc[v + 1] - acc[v];
        if (deg <= DEG_THRESHOLD) {
            light.push_back(v);
        }
        else heavy.push_back(v);
    }

    if (!light.empty()) {
        auto light_gpu = torch::tensor(light, edge_ptr.options());
        min_aggr_forward_light<<<light.size(), 32>>>(
            light_gpu.data_ptr<int>(),
            edge_ptr.data_ptr<int>(),
            edge_idx.data_ptr<int>(),
            X.data_ptr<float>(),
            out.data_ptr<float>(),
            argmin.data_ptr<int>(),
            d
        );
    }

    if (!heavy.empty()) {
        auto heavy_gpu = torch::tensor(heavy, edge_ptr.options());
        min_aggr_forward_heavy<<<heavy.size(), THREADS_PER_BLOCK>>>(
            edge_ptr.data_ptr<int>(),
            edge_idx.data_ptr<int>(),
            X.data_ptr<float>(),
            out.data_ptr<float>(),
            argmin.data_ptr<int>(),
            heavy.size(),
            d
        );
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
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
