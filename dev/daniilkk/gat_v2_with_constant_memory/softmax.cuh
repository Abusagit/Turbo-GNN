#include <cuda_runtime.h>

void Softmax(size_t rows, size_t cols, const float* d_input_matrix, size_t input_stride,
             float* d_out, size_t out_stride, cudaStream_t stream);
