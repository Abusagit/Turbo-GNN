#include <cuda_runtime.h>
#include <cmath>

__global__ void min_aggr_forward_1(
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
    if (tid >= d) {
        return;
    }

    int row_start = edge_ptr[block_idx];
    int row_end = edge_ptr[block_idx + 1];

    float best_val = INFINITY;
    int best_src = -1;
    for (int eid = row_start; eid < row_end; ++eid) {
        int src = edge_idx[eid];
        float v = X[src * d + tid];
        if (v < best_val) {
            best_val = v;
            best_src = src;
        }
    }

    out[block_idx * d + tid] = best_val;
    argmin[block_idx * d + tid] = best_src;
}

__global__ void min_aggr_backward_1(
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
    if (tid >= d) {
        return;
    }

    int src = argmin[block_idx * d + tid];
    if (src < 0) {
        return;
    }

    float grad = grad_out[block_idx * d + tid];
    atomicAdd(&grad_x[src * d + tid], grad);
}
