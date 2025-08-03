#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <vector>

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

const auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
const auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
const auto ALG = CUSPARSE_SPMM_CSR_ALG2;

void CheckStatus(const cudaError_t& status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
    }
}


void cusp_SPMM_csr_impl(torch::Tensor &out,
                        const torch::Tensor &indptr,
                        const torch::Tensor &indices,
                        const torch::Tensor &N) {
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(N);
    CHECK_INPUT(out);

    auto handle = at::cuda::getCurrentCUDASparseHandle();
    
    int32_t m = indptr.size(0) - 1;  // number of nodes
    int32_t n = N.size(1);  // number of features
    int32_t k = N.size(0);  // should equal m
    int64_t nnz = indices.size(0); // number of edges
    
    TORCH_CHECK(k == m, "Feature matrix first dimension must match number of nodes");
    TORCH_CHECK(out.size(0) == m && out.size(1) == n, "Output dimensions mismatch");

    float alpha = 1.0f;
    float beta = 0.0f;

    
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    
    // Create sparse matrix A with all ones
    auto A = torch::ones({nnz}, torch::dtype(torch::kFloat32).device(N.device()));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, m, nnz, 
        indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), 
        A.data_ptr<float>(), 
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, 
        CUSPARSE_INDEX_BASE_ZERO, 
        CUDA_R_32F));

    // Leading dimension should be n for row-major format
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, m, n, n, N.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, n, out.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Get workspace size
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, transA, transB, &alpha, matA,
                                           matB, &beta, matC, CUDA_R_32F, ALG,
                                           &workspace_size));

    // Allocate workspace if needed
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace_ptr, workspace_size);
    }

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta,
                                matC, CUDA_R_32F, ALG, workspace_ptr));

    // Optional: Print non-zero elements for debugging
    if (false) {  // Set to true for debugging
        std::cout << "m=" << m << " k=" << k << " n=" << n << " nnz=" << nnz << std::endl;

        std::vector<float> host_data(m * n);
        CheckStatus(cudaMemcpy(host_data.data(), out.data_ptr<float>(), 
                   m * n * sizeof(float), cudaMemcpyDeviceToHost));
        
        int count = 0;
        for (int i = 0; i < m && count < 100; i++) {
            for (int j = 0; j < n && count < 100; j++) {
                if (host_data[i * n + j] != 0) {
                    std::cout << "out[" << i << "," << j << "] = " 
                              << host_data[i * n + j] << std::endl;
                    count++;
                }
            }
        }
    }

    // Cleanup
    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
    
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
}

torch::Tensor csr_SPMM(const torch::Tensor &indptr,
                       const torch::Tensor &indices, 
                       const torch::Tensor &features) {
    auto out = torch::zeros_like(features);  // Initialize to zero
    cusp_SPMM_csr_impl(out, indptr, indices, features);  // Pass features, not out!
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csr_SPMM", &csr_SPMM, "csr_SPMM");
}
