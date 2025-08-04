#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>


enum class NormType {
    NONE = 0,
    RIGHT = 1,
    LEFT = 2,
    BOTH = 3
};


__global__ void compute_degrees_kernel(const int32_t* indptr, const int32_t* indices,
                                     float* in_degrees, float* out_degrees, int32_t num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        // For TRANSPOSED CSR: indptr gives IN-degrees (incoming edges)
        in_degrees[idx] = static_cast<float>(indptr[idx + 1] - indptr[idx]);
        
        // Out-degrees need to be computed by counting occurrences in indices
        out_degrees[idx] = 0.0f;
    }

    __syncthreads();

    // Count in-degrees
    if (idx < num_nodes) {
        for (int32_t i = indptr[idx]; i < indptr[idx + 1]; i++) {
            int32_t neighbor = indices[i];
            atomicAdd(&out_degrees[neighbor], 1.0f);
        }
    }
}

// CUDA kernel to compute normalized edge weights
__global__ void compute_edge_weights_kernel(const int32_t* indptr, const int32_t* indices,
                                          const float* edge_weights, float* normalized_weights,
                                          const float* in_degrees, const float* out_degrees,
                                          int32_t num_nodes, NormType norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        for (int32_t i = indptr[idx]; i < indptr[idx + 1]; i++) {
            int32_t dst = idx;
            int32_t src = indices[i];
            float weight = edge_weights ? edge_weights[i] : 1.0f;
            
            switch (norm) {
                case NormType::NONE:
                    normalized_weights[i] = weight;
                    break;
                case NormType::RIGHT:
                    // Divide by in-degree of destination node
                    normalized_weights[i] = weight / fmaxf(in_degrees[dst], 1.0f);
                    break;
                case NormType::LEFT:
                    // Divide by out-degree of source node
                    normalized_weights[i] = weight / fmaxf(out_degrees[src], 1.0f);
                    break;
                case NormType::BOTH:
                    // Symmetric normalization: 1/sqrt(d_src * d_dst)
                    {
                        float norm_factor = sqrtf(fmaxf(out_degrees[src], 1.0f) * fmaxf(in_degrees[dst], 1.0f));
                        normalized_weights[i] = weight / norm_factor;
                    }
                    break;
            }
        }
    }
}


void launch_compute_degrees(const torch::Tensor& indptr, const torch::Tensor& indices,
                           torch::Tensor& in_degrees, torch::Tensor& out_degrees) {
    int32_t num_nodes = indptr.size(0) - 1;
    
    // Initialize degrees to zero
    in_degrees.zero_();
    out_degrees.zero_();
    
    dim3 block(256);
    dim3 grid((num_nodes + block.x - 1) / block.x);
    
    compute_degrees_kernel<<<grid, block>>>(
        indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        in_degrees.data_ptr<float>(), out_degrees.data_ptr<float>(), num_nodes);
    
    cudaDeviceSynchronize();
}

void launch_compute_normalized_weights(const torch::Tensor& indptr, const torch::Tensor& indices,
                                      const torch::Tensor& edge_weights, torch::Tensor& normalized_weights,
                                      const torch::Tensor& in_degrees, const torch::Tensor& out_degrees,
                                      NormType norm) {
    int32_t num_nodes = indptr.size(0) - 1;
    
    dim3 block(256);
    dim3 grid((num_nodes + block.x - 1) / block.x);
    
    const float* edge_weights_ptr = edge_weights.numel() > 0 ? edge_weights.data_ptr<float>() : nullptr;
    
    compute_edge_weights_kernel<<<grid, block>>>(
        indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        edge_weights_ptr, normalized_weights.data_ptr<float>(),
        in_degrees.data_ptr<float>(), out_degrees.data_ptr<float>(),
        num_nodes, norm);
    
    cudaDeviceSynchronize();
}