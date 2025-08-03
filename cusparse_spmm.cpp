#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <nvrtc.h>
#include <stdlib.h>
#include <cstdint>
#include <string>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

// #include <iostream>

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}



void CheckStatus(const cudaError_t& status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define cudaErrchk(err)                                                        \
    { gpuAssert(err, __FILE__, __LINE__); }


void checkCusparseStatus(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}


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

void cusp_SPMM_csr_impl(torch::Tensor &out,
                        const torch::Tensor &indptr,
                        const torch::Tensor &indices,
                        const torch::Tensor &N
                        ) {
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(N);
    CHECK_INPUT(out);

    auto handle = at::cuda::getCurrentCUDASparseHandle();
    cusparsePointerMode_t ptr_mode;
    cusparseGetPointerMode(handle, &ptr_mode);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    int32_t m = indptr.size(0) - 1;  // number of nodes --> A is sparse $m \times m$ matrix
    int32_t n = out.size(-1);  // number of features --> B is dense $m \times n$ matrix
    int32_t k = N.size(0);  // k == n
    int64_t nnz = indices.size(0); // number of edges in a graph
    TORCH_CHECK(k == m, "Feature matrix first dimension must match number of nodes");

    float alpha = 1.0f;
    float beta = 0.0f;

    std::cout << "m=" << m <<  " " << "k=" << k << " " << "n=" << n << " " << "nnz=" << nnz <<std::endl;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    // std::cout << "AAAAAAA\n";
    // sparse matrix A is all one
    // auto A = torch::empty({nnz}, torch::dtype(torch::kFloat32).device(N.device()));
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

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, m, n, n, N.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, n, out.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // workspace
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, transA, transB, &alpha, matA,
                                           matB, &beta, matC, CUDA_R_32F, ALG,
                                           &workspace_size));

    auto workspace = torch::empty(
        {(long)workspace_size}, torch::dtype(torch::kFloat32).device(N.device()));
    float *workspace_ptr = workspace.data_ptr<float>();
    // call SPMM

    CHECK_CUSPARSE(cusparseSpMM(handle, transA, transB, &alpha, matA, matB, &beta,
                                matC, CUDA_R_32F, ALG, workspace_ptr));

    std::vector<float> host_data(m * n);

   // Optional: Print non-zero elements for debugging
    if (true) {  // Set to true for debugging
        std::vector<float> host_data(m * n);
        cudaMemcpy(host_data.data(), out.data_ptr<float>(), 
                   m * n * sizeof(float), cudaMemcpyDeviceToHost);
        
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

    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));

    cusparseSetPointerMode(handle, ptr_mode);

}


torch::Tensor csr_SPMM(const torch::Tensor &indptr,
              const torch::Tensor &indices, const torch::Tensor &features) {
    auto out = torch::empty_like(features);
    cusp_SPMM_csr_impl(out, indptr, indices, features);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // cusparse
    m.def("csr_SPMM", &csr_SPMM, "csr_SPMM");
}

