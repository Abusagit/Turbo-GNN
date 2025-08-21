#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }


enum class NormType {
    NONE = 0,
    RIGHT = 1,
    LEFT = 2,
    BOTH = 3
};



void launch_compute_degrees(const torch::Tensor& indptr, const torch::Tensor& indices,
                           torch::Tensor& in_degrees, torch::Tensor& out_degrees);

void launch_compute_normalized_weights(const torch::Tensor& indptr, const torch::Tensor& indices,
                                      const torch::Tensor& edge_weights, torch::Tensor& normalized_weights,
                                      const torch::Tensor& in_degrees, const torch::Tensor& out_degrees,
                                      NormType norm);



// Cache for graph structures and preprocessed data
struct GraphCache {
    cusparseSpMatDescr_t matA = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    torch::Tensor edge_values;  // Stores normalized edge weights
    torch::Tensor in_degrees;   // Cache in-degrees
    torch::Tensor out_degrees;  // Cache out-degrees
    int32_t m, n, nnz;
    cusparseSpMMAlg_t best_alg = CUSPARSE_SPMM_ALG_DEFAULT;
    NormType cached_norm = NormType::NONE;
    bool has_edge_weights = false;
    
    ~GraphCache() {
        if (matA) cusparseDestroySpMat(matA);
        if (workspace) cudaFree(workspace);
    }
};

// Global cache and static buffers
static std::unordered_map<size_t, std::unique_ptr<GraphCache>> graph_cache;
static std::mutex cache_mutex;

// Hash function for graph structure (including normalization type)
size_t hash_graph(const torch::Tensor& indptr, const torch::Tensor& indices, 
                  NormType norm, bool has_edge_weights) {
    size_t h1 = std::hash<int64_t>{}(indptr.size(0));
    size_t h2 = std::hash<int64_t>{}(indices.size(0));
    size_t h3 = std::hash<void*>{}(indptr.data_ptr());
    size_t h4 = std::hash<void*>{}(indices.data_ptr());
    size_t h5 = std::hash<int>{}(static_cast<int>(norm));
    size_t h6 = std::hash<bool>{}(has_edge_weights);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
}





torch::Tensor csr_SPMM_normalized(const torch::Tensor &indptr,
                                 const torch::Tensor &indices, 
                                 const torch::Tensor &features,
                                 const torch::Tensor &edge_weights,
                                 const std::string &norm_str,
                                 int algorithm,
                                 bool use_cache) {
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(features);

    NormType norm;
    if (norm_str == "none") norm = NormType::NONE;
    else if (norm_str == "right") norm = NormType::RIGHT;
    else if (norm_str == "left") norm = NormType::LEFT;
    else if (norm_str == "both") norm = NormType::BOTH;
    else {
        TORCH_CHECK(false, "Invalid normalization type. Must be one of: 'none', 'right', 'left', 'both'");
    }

    bool has_edge_weights = edge_weights.numel() > 0;
    if (has_edge_weights) {
        CHECK_INPUT(edge_weights);
        TORCH_CHECK(edge_weights.size(0) == indices.size(0), "Edge weights must have same length as indices");
    }

    auto handle = at::cuda::getCurrentCUDASparseHandle();

    int32_t m = indptr.size(0) - 1;
    int32_t n = features.size(1);
    int32_t k = features.size(0);
    int64_t nnz = indices.size(0);
    
    TORCH_CHECK(k == m, "Feature matrix first dimension must match number of nodes");

    float alpha = 1.0f;
    float beta = 0.0f;

    auto out = torch::empty({m, n}, features.options());

    cusparseSpMMAlg_t alg;
    switch (algorithm) {
        case 0: alg = CUSPARSE_SPMM_ALG_DEFAULT; break;
        case 1: alg = CUSPARSE_SPMM_CSR_ALG1; break;
        case 2: alg = CUSPARSE_SPMM_CSR_ALG2; break;
        case 3: alg = CUSPARSE_SPMM_CSR_ALG3; break;
        default: alg = CUSPARSE_SPMM_ALG_DEFAULT;
    }

    GraphCache* cache = nullptr;
    size_t graph_hash = 0;

    if (use_cache) {
        graph_hash = hash_graph(indptr, indices, norm, has_edge_weights);
        std::lock_guard<std::mutex> lock(cache_mutex);

        auto it = graph_cache.find(graph_hash);
        if (it != graph_cache.end()) {
            cache = it->second.get();

            if (algorithm == -1) {
                alg = cache->best_alg;
            }
        } else {

            graph_cache[graph_hash] = std::make_unique<GraphCache>();
            cache = graph_cache[graph_hash].get();
            cache->m = m;
            cache->n = n;
            cache->nnz = nnz;
            cache->best_alg = alg;
            cache->cached_norm = norm;
            cache->has_edge_weights = has_edge_weights;
            

            cache->in_degrees = torch::zeros({m}, torch::dtype(torch::kFloat32).device(features.device()));
            cache->out_degrees = torch::zeros({m}, torch::dtype(torch::kFloat32).device(features.device()));
            

            launch_compute_degrees(indptr, indices, cache->in_degrees, cache->out_degrees);
            

            cache->edge_values = torch::empty({nnz}, torch::dtype(torch::kFloat32).device(features.device()));
            launch_compute_normalized_weights(indptr, indices, edge_weights, cache->edge_values,
                                     cache->in_degrees, cache->out_degrees, norm);
            

            CHECK_CUSPARSE(cusparseCreateCsr(
                &cache->matA, m, m, nnz, 
                indptr.data_ptr<int32_t>(),
                indices.data_ptr<int32_t>(), 
                cache->edge_values.data_ptr<float>(), 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_32F));
        }
    }

    // Create descriptors
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    torch::Tensor normalized_weights;
    
    if (cache && use_cache) {
        matA = cache->matA;
    } else {
        // Compute normalized weights on-the-fly
        torch::Tensor in_degrees = torch::zeros({m}, torch::dtype(torch::kFloat32).device(features.device()));
        torch::Tensor out_degrees = torch::zeros({m}, torch::dtype(torch::kFloat32).device(features.device()));
        
        launch_compute_degrees(indptr, indices, in_degrees, out_degrees);
        
        normalized_weights = torch::empty({nnz}, torch::dtype(torch::kFloat32).device(features.device()));
        launch_compute_normalized_weights(indptr, indices, edge_weights, normalized_weights,
                                 in_degrees, out_degrees, norm);
        
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA, m, m, nnz, 
            indptr.data_ptr<int32_t>(),
            indices.data_ptr<int32_t>(), 
            normalized_weights.data_ptr<float>(), 
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, 
            CUSPARSE_INDEX_BASE_ZERO, 
            CUDA_R_32F));
    }

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, m, n, n, features.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, n, out.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Handle workspace
    void* workspace = nullptr;
    size_t workspace_size = 0;
    bool need_free_workspace = false;
    
    if (cache && cache->workspace) {
        // Use cached workspace
        workspace = cache->workspace;
        workspace_size = cache->workspace_size;
    } else {
        // Get required workspace size
        size_t required_size;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, &required_size));
        
        if (cache && use_cache) {
            // Allocate and cache workspace
            if (required_size > 0) {
                cudaMalloc(&cache->workspace, required_size);
                cache->workspace_size = required_size;
                workspace = cache->workspace;
                workspace_size = required_size;
            }
        } else {
            // Temporary workspace
            if (required_size > 0) {
                cudaMalloc(&workspace, required_size);
                workspace_size = required_size;
                need_free_workspace = true;
            }
        }
    }

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, workspace));

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    
    if (!cache || !use_cache) {
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    }
    
    if (need_free_workspace && workspace) {
        cudaFree(workspace);
    }

    return out;
}


int find_best_algorithm_normalized(const torch::Tensor &indptr,
                                  const torch::Tensor &indices, 
                                  const torch::Tensor &features,
                                  const torch::Tensor &edge_weights,
                                  const std::string &norm_str) {
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    

    NormType norm;
    if (norm_str == "none") norm = NormType::NONE;
    else if (norm_str == "right") norm = NormType::RIGHT;
    else if (norm_str == "left") norm = NormType::LEFT;
    else if (norm_str == "both") norm = NormType::BOTH;
    else {
        TORCH_CHECK(false, "Invalid normalization type. Must be one of: 'none', 'right', 'left', 'both'");
    }
    
    int32_t m = indptr.size(0) - 1;
    int32_t n = features.size(1);
    int64_t nnz = indices.size(0);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute normalized weights
    torch::Tensor in_degrees = torch::zeros({m}, torch::dtype(torch::kFloat32).device(features.device()));
    torch::Tensor out_degrees = torch::zeros({m}, torch::dtype(torch::kFloat32).device(features.device()));
    launch_compute_degrees(indptr, indices, in_degrees, out_degrees);

    torch::Tensor normalized_weights = torch::empty({nnz}, torch::dtype(torch::kFloat32).device(features.device()));
    launch_compute_normalized_weights(indptr, indices, edge_weights, normalized_weights, in_degrees, out_degrees, norm);
    
    auto out = torch::empty({m, n}, features.options());
    
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, m, nnz, 
        indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), 
        normalized_weights.data_ptr<float>(), 
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, 
        CUSPARSE_INDEX_BASE_ZERO, 
        CUDA_R_32F));
    
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, m, n, n, features.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, n, out.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    // Test different algorithms
    std::vector<std::pair<int, cusparseSpMMAlg_t>> algorithms = {
        {0, CUSPARSE_SPMM_ALG_DEFAULT},
        {1, CUSPARSE_SPMM_CSR_ALG1},
        {2, CUSPARSE_SPMM_CSR_ALG2},
        {3, CUSPARSE_SPMM_CSR_ALG3}
    };
    
    int best_alg_id = -1;
    float best_time = std::numeric_limits<float>::max();
    
    for (auto& [alg_id, alg] : algorithms) {
        try {
            size_t workspace_size;
            CHECK_CUSPARSE(cusparseSpMM_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, &workspace_size));
            
            void* workspace = nullptr;
            if (workspace_size > 0) {
                cudaMalloc(&workspace, workspace_size);
            }
            
            // Warmup
            for (int i = 0; i < 3; i++) {
                cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, workspace);
            }
            
            // Time it
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            for (int i = 0; i < 10; i++) {
                cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, workspace);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            if (milliseconds < best_time) {
                best_time = milliseconds;
                best_alg_id = alg_id;
            }
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            if (workspace) cudaFree(workspace);
            
        } catch (...) {
            // Algorithm not supported, skip
        }
    }
    
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    
    return best_alg_id;
}


// ************************* BACKWARD COMPATIBILITY STARTS *************************
int find_best_algorithm(const torch::Tensor &indptr,
                        const torch::Tensor &indices, 
                        const torch::Tensor &features) {
    torch::Tensor empty_weights = torch::empty({0}, features.options());
    return find_best_algorithm_normalized(indptr, indices, features, empty_weights, "none");
}


torch::Tensor csr_SPMM(const torch::Tensor &indptr,
                       const torch::Tensor &indices, 
                       const torch::Tensor &features,
                       int algorithm,
                       bool use_cache) {
    torch::Tensor empty_weights = torch::empty({0}, features.options());
    return csr_SPMM_normalized(indptr, indices, features, empty_weights, "none", algorithm, use_cache);
}

// ************************* BACKWARD COMPATIBILITY ENDS *************************


void clear_graph_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    graph_cache.clear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csr_SPMM", &csr_SPMM, "Optimized and cached csr_SPMM (backward compatibility)",
          py::arg("indptr"), py::arg("indices"), py::arg("features"), 
          py::arg("algorithm") = -1, py::arg("use_cache") = true);
    
    m.def("csr_SPMM_normalized", &csr_SPMM_normalized, "Optimized and cached csr_SPMM with normalization",
          py::arg("indptr"), py::arg("indices"), py::arg("features"), py::arg("edge_weights"),
          py::arg("norm") = "none", py::arg("algorithm") = -1, py::arg("use_cache") = true);

    m.def("find_best_algorithm", &find_best_algorithm, "Find best cuSPARSE algorithm for given graph (backward compatibility)");
    
    m.def("find_best_algorithm_normalized", &find_best_algorithm_normalized, 
          "Find best cuSPARSE algorithm for given graph with normalization",
          py::arg("indptr"), py::arg("indices"), py::arg("features"), py::arg("edge_weights"),
          py::arg("norm") = "none");

    m.def("clear_graph_cache", &clear_graph_cache, "Clear graph cache");
}
