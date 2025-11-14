// #include <torch/extension.h>
// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <vector>
#include <cusparse.h>
// #include <ATen/cuda/CUDAContext.h>

#include "torch_minimal.h"

std::vector<torch::Tensor> csr2csc_cuda(
    torch::Tensor inPtr,
    torch::Tensor inInd,
    torch::Tensor inVal,
    int num_row
){
    // initialize the output tensor
    auto outPtr = torch::zeros_like(inPtr);
    auto outInd = torch::empty_like(inInd);
    auto outVal = torch::empty_like(inVal);
    int nnz = inInd.size(0);

    // create cusparse handler
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cudaStream_t stream = getCurrentCUDAStream();
    cusparseSetStream(handle, stream);
    /*
    cusparseScsr2csc(handle, num_row, num_row, nnz, inVal.data_ptr<float>(), inPtr.data_ptr<int>(),
    inInd.data_ptr<int>(), outVal.data_ptr<float>(), outInd.data_ptr<int>(), outPtr.data_ptr<int>(), CUSPARSE_ACTION_SYMBOLIC,
    CUSPARSE_INDEX_BASE_ZERO);
    */
    // for CUDA 10.2
    // Determine temporary device storage requirement
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    AT_DISPATCH_FLOATING_TYPES(inVal.type(), "get temporary device storage requirement", ([&]{
        cusparseCsr2cscEx2_bufferSize(handle, num_row, num_row, nnz, inVal.data_ptr<scalar_t>(), inPtr.data_ptr<int>(),
                                  inInd.data_ptr<int>(), outVal.data_ptr<scalar_t>(), outPtr.data_ptr<int>(), outInd.data_ptr<int>(),
                                  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                                  &temp_storage_bytes
                                );
                            }));



    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Do the conversion
    AT_DISPATCH_FLOATING_TYPES(inVal.type(), "type convert", ([&]{
    cusparseCsr2cscEx2(handle, num_row, num_row, nnz, inVal.data_ptr<scalar_t>(), inPtr.data_ptr<int>(),
                       inInd.data_ptr<int>(), outVal.data_ptr<scalar_t>(), outPtr.data_ptr<int>(), outInd.data_ptr<int>(),
                       CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                       d_temp_storage
                    );
                }));

    cusparseDestroy(handle);
    cudaFree(d_temp_storage);
    return {outPtr, outInd, outVal};
}


torch::Tensor coo2csr_cuda(
    torch::Tensor cooRowInd,
    int num_row
){
    // initialize the output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(cooRowInd.device());
    auto csrRowPtr = torch::empty({num_row + 1, }, options);
    int nnz = cooRowInd.size(0);

    // create cusparse handler
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cudaStream_t stream = getCurrentCUDAStream();
    cusparseSetStream(handle, stream);
    cusparseXcoo2csr(handle, cooRowInd.data_ptr<int>(), nnz, num_row, csrRowPtr.data_ptr<int>(), CUSPARSE_INDEX_BASE_ZERO);
    // cudaDeviceSynchronize();

    cusparseDestroy(handle);
    return csrRowPtr;
}
