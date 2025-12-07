#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>


constexpr int THREADS_PER_BLOCK = 64;

__global__ void min_aggr_forward(
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

void min_aggr_forward_cuda(
    const at::Tensor& edge_ptr,
    const at::Tensor& edge_idx,
    const at::Tensor& X,
    at::Tensor& out,
    at::Tensor& argmin
) {
    const int num_nodes = X.size(0);
    const int d = X.size(1);

    const dim3 blocks(num_nodes);
    const dim3 threads(THREADS_PER_BLOCK);
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
