#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <unordered_map>
#include <mutex>

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

// Cache for graph structures and preprocessed data
struct GraphCache {
    cusparseSpMatDescr_t matA = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    torch::Tensor ones_values;
    int32_t m, n, nnz;
    cusparseSpMMAlg_t best_alg = CUSPARSE_SPMM_ALG_DEFAULT;
    
    ~GraphCache() {
        if (matA) cusparseDestroySpMat(matA);
        if (workspace) cudaFree(workspace);
    }
};

// Global cache and static buffers
static std::unordered_map<size_t, std::unique_ptr<GraphCache>> graph_cache;
static std::mutex cache_mutex;

// Static buffer for ones (when not using cache)
static torch::Tensor global_ones_buffer;
static int64_t global_ones_buffer_size = 0;
static std::mutex ones_buffer_mutex;

// Hash function for graph structure
size_t hash_graph(const torch::Tensor& indptr, const torch::Tensor& indices) {
    size_t h1 = std::hash<int64_t>{}(indptr.size(0));
    size_t h2 = std::hash<int64_t>{}(indices.size(0));
    size_t h3 = std::hash<void*>{}(indptr.data_ptr());
    size_t h4 = std::hash<void*>{}(indices.data_ptr());
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
}

torch::Tensor csr_SPMM(const torch::Tensor &indptr,
                       const torch::Tensor &indices, 
                       const torch::Tensor &features,
                       int algorithm,
                       bool use_cache) {
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(features);

    auto handle = at::cuda::getCurrentCUDASparseHandle();
    
    int32_t m = indptr.size(0) - 1;
    int32_t n = features.size(1);
    int32_t k = features.size(0);
    int64_t nnz = indices.size(0);
    
    TORCH_CHECK(k == m, "Feature matrix first dimension must match number of nodes");

    float alpha = 1.0f;
    float beta = 0.0f;


    // Pre-allocate output
    auto out = torch::empty({m, n}, features.options());

    // Determine which algorithm to use
    cusparseSpMMAlg_t alg;
    switch (algorithm) {
        case 0: alg = CUSPARSE_SPMM_ALG_DEFAULT; break;
        case 1: alg = CUSPARSE_SPMM_CSR_ALG1; break;
        case 2: alg = CUSPARSE_SPMM_CSR_ALG2; break;
        case 3: alg = CUSPARSE_SPMM_CSR_ALG3; break;
        default: alg = CUSPARSE_SPMM_ALG_DEFAULT;
    }

    // Try to use cached graph structure
    GraphCache* cache = nullptr;
    size_t graph_hash = 0;
    
    if (use_cache) {
        graph_hash = hash_graph(indptr, indices);
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = graph_cache.find(graph_hash);
        if (it != graph_cache.end()) {
            cache = it->second.get();
            // Use cached best algorithm if no specific algorithm requested
            if (algorithm == -1) {
                alg = cache->best_alg;
            }
        } else {
            // Create new cache entry
            graph_cache[graph_hash] = std::make_unique<GraphCache>();
            cache = graph_cache[graph_hash].get();
            cache->m = m;
            cache->n = n;
            cache->nnz = nnz;
            cache->best_alg = alg;
            
            // Pre-allocate ones buffer for this graph
            cache->ones_values = torch::ones({nnz}, torch::dtype(torch::kFloat32).device(features.device()));
            
            // Create sparse matrix descriptor
            CHECK_CUSPARSE(cusparseCreateCsr(
                &cache->matA, m, m, nnz, 
                indptr.data_ptr<int32_t>(),
                indices.data_ptr<int32_t>(), 
                cache->ones_values.data_ptr<float>(), 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_32F));
        }
    }

    // Create descriptors
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    torch::Tensor ones_buffer;
    
    if (cache && use_cache) {
        matA = cache->matA;
    } else {
        // Use global ones buffer when not caching
        {
            std::lock_guard<std::mutex> lock(ones_buffer_mutex);
            if (global_ones_buffer_size < nnz) {
                global_ones_buffer = torch::ones({nnz}, torch::dtype(torch::kFloat32).device(features.device()));
                global_ones_buffer_size = nnz;
            }
            ones_buffer = global_ones_buffer;
        }
        
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA, m, m, nnz, 
            indptr.data_ptr<int32_t>(),
            indices.data_ptr<int32_t>(), 
            ones_buffer.data_ptr<float>(), 
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

// Function to find best algorithm for a given graph
int find_best_algorithm(const torch::Tensor &indptr,
                        const torch::Tensor &indices, 
                        const torch::Tensor &features) {
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    
    int32_t m = indptr.size(0) - 1;
    int32_t n = features.size(1);
    int64_t nnz = indices.size(0);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    auto ones = torch::ones({nnz}, torch::dtype(torch::kFloat32).device(features.device()));
    auto out = torch::empty({m, n}, features.options());
    
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, m, nnz, 
        indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), 
        ones.data_ptr<float>(), 
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

void clear_graph_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    graph_cache.clear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csr_SPMM", &csr_SPMM, "Optimized and cached csr_SPMM",
          py::arg("indptr"), py::arg("indices"), py::arg("features"), 
          py::arg("algorithm") = -1, py::arg("use_cache") = true);
    m.def("find_best_algorithm", &find_best_algorithm, "Find best cuSPARSE algorithm for given graph");
    m.def("clear_graph_cache", &clear_graph_cache, "Clear graph cache");
}
