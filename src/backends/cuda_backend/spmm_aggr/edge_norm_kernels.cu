#include "../common.cuh"


enum class NormType {
    NONE = 0,
    RIGHT = 1,
    LEFT = 2,
    BOTH = 3
};


template <typename index_t>
__global__ void compute_degrees_kernel(const index_t* indptr, const index_t* indices,
                                     float* in_degrees, float* out_degrees, int32_t num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        // For TRANSPOSED CSR: indptr gives IN-degrees (incoming edges)
        // IN-degree in TRANSPOSED CSR (row = destination, entries = sources)
        in_degrees[idx] = static_cast<float>(indptr[idx + 1] - indptr[idx]);

    }

    // Count in-degrees
    // Count OUT-degrees by scanning incoming lists and attributing to the source
    if (idx < num_nodes) {
        for (index_t i = indptr[idx]; i < indptr[idx + 1]; ++i) {
            index_t src = indices[i];
            atomicAdd(&out_degrees[static_cast<size_t>(src)], 1.0f);
        }
    }
}


template <typename index_t>
__global__ void compute_edge_weights_kernel(const index_t* indptr, const index_t* indices,
                                          const float* edge_weights, float* normalized_weights,
                                          const float* in_degrees, const float* out_degrees,
                                          int32_t num_nodes, NormType norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        for (index_t i = indptr[idx]; i < indptr[idx + 1]; i++) {
            int dst = idx;
            index_t src = indices[i];
            float weight = edge_weights ? edge_weights[static_cast<size_t>(i)] : 1.0f;

            switch (norm) {
                case NormType::NONE:
                    normalized_weights[static_cast<size_t>(i)] = weight;
                    break;
                case NormType::RIGHT:
                    // Divide by in-degree of destination node
                    normalized_weights[static_cast<size_t>(i)] = weight / fmaxf(in_degrees[dst], 1.0f);
                    break;
                case NormType::LEFT:
                    // Divide by out-degree of source node
                    normalized_weights[static_cast<size_t>(i)] = weight / fmaxf(out_degrees[static_cast<size_t>(src)], 1.0f);
                    break;
                case NormType::BOTH:
                    // Symmetric normalization: 1/sqrt(d_src * d_dst)
                    {
                        float norm_factor = sqrtf(fmaxf(out_degrees[static_cast<size_t>(src)], 1.0f) * fmaxf(in_degrees[dst], 1.0f));
                        normalized_weights[static_cast<size_t>(i)] = weight / norm_factor;
                    }
                    break;
            }
        }
    }
}


void launch_compute_degrees(const torch::Tensor& indptr, const torch::Tensor& indices,
                           torch::Tensor& in_degrees, torch::Tensor& out_degrees,
                           int block_dim) {

    TORCH_CHECK(indptr.is_cuda() && indices.is_cuda(), "indptr/indices must be CUDA");
    TORCH_CHECK(in_degrees.is_cuda() && out_degrees.is_cuda(), "degree buffers must be CUDA");
    auto idx_dtype = indptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype),
                "indptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(indices.scalar_type() == idx_dtype, "indices must have same dtype as indptr");
    TORCH_CHECK(in_degrees.scalar_type() == torch::kFloat, "in_degrees must be float32");
    TORCH_CHECK(out_degrees.scalar_type() == torch::kFloat, "out_degrees must be float32");
    TORCH_CHECK(indptr.is_contiguous() && indices.is_contiguous(), "indptr/indices contiguous");


    int32_t num_nodes = indptr.size(0) - 1;

    in_degrees.zero_();
    out_degrees.zero_();

    dim3 block(block_dim);
    dim3 grid((num_nodes + block.x - 1) / block.x);

    std::visit([&](auto idxInfo) {
        using index_t = typename decltype(idxInfo)::Type;
        compute_degrees_kernel<index_t><<<grid, block>>>(
            index_ptr<index_t>(indptr), index_ptr<index_t>(indices),
            in_degrees.data_ptr<float>(), out_degrees.data_ptr<float>(), num_nodes);
    }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "compute_degrees_kernel launch failed: ", cudaGetErrorString(err));

}

void launch_compute_normalized_weights(const torch::Tensor& indptr, const torch::Tensor& indices,
                                      const torch::Tensor& edge_weights, torch::Tensor& normalized_weights,
                                      const torch::Tensor& in_degrees, const torch::Tensor& out_degrees,
                                      NormType norm, int block_dim) {


    TORCH_CHECK(indptr.is_cuda() && indices.is_cuda(), "indptr/indices must be CUDA");
    TORCH_CHECK(in_degrees.is_cuda() && out_degrees.is_cuda(), "degree tensors must be CUDA");
    TORCH_CHECK(normalized_weights.is_cuda(), "normalized_weights must be CUDA");
    auto idx_dtype = indptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype),
                "indptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(indices.scalar_type() == idx_dtype, "indices must have same dtype as indptr");
    TORCH_CHECK(in_degrees.scalar_type() == torch::kFloat, "in_degrees must be float32");
    TORCH_CHECK(out_degrees.scalar_type() == torch::kFloat, "out_degrees must be float32");
    TORCH_CHECK(normalized_weights.scalar_type() == torch::kFloat, "normalized_weights must be float32");
    TORCH_CHECK(indptr.is_contiguous() && indices.is_contiguous() && normalized_weights.is_contiguous(), "inputs contiguous");
    TORCH_CHECK(!edge_weights.defined() || edge_weights.numel() == indices.numel() || edge_weights.numel() == 0,
                "edge_weights must be same length as indices or empty");

    int32_t num_nodes = static_cast<int32_t>(indptr.size(0) - 1);

    dim3 block(block_dim);
    dim3 grid((num_nodes + block.x - 1) / block.x);

    const float* edge_weights_ptr = nullptr;

    if (edge_weights.defined() && edge_weights.numel() > 0) {
            TORCH_CHECK(edge_weights.is_cuda(), "edge_weights must be CUDA if provided");
            TORCH_CHECK(edge_weights.scalar_type() == torch::kFloat, "edge_weights must be float32");
            TORCH_CHECK(edge_weights.is_contiguous(), "edge_weights contiguous");
            edge_weights_ptr = edge_weights.data_ptr<float>();
        }

    std::visit([&](auto idxInfo) {
        using index_t = typename decltype(idxInfo)::Type;
        compute_edge_weights_kernel<index_t><<<grid, block>>>(
            index_ptr<index_t>(indptr), index_ptr<index_t>(indices),
            edge_weights_ptr, normalized_weights.data_ptr<float>(),
            in_degrees.data_ptr<float>(), out_degrees.data_ptr<float>(),
            num_nodes, norm);
    }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype));

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "compute_edge_weights_kernel launch failed: ", cudaGetErrorString(err));

}
