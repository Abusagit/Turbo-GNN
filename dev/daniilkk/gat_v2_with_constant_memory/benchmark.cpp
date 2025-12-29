#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>

// Declarations from above kernels
void SoftmaxDotProductGlobal(size_t K, size_t z, const float* d_vectors,
                             size_t vectors_stride, const float* d_A,
                             float* d_out, cudaStream_t stream);

void SoftmaxDotProductConstant(size_t K, size_t z, const float* d_vectors,
                               size_t vectors_stride, const float* h_A,
                               float* d_out, cudaStream_t stream);


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)



void benchmark(size_t K, size_t z) {
    // const size_t K = 10000;    // number of vectors
    // const size_t z = 128;   // dimension
    const int num_iterations = 1000;

    // Allocate host memory
    std::vector<float> h_vectors(K * z);
    std::vector<float> h_A(z);
    std::vector<float> h_out_global(K);
    std::vector<float> h_out_constant(K);

    // Initialize with random values
    for (size_t i = 0; i < K * z; i++) {
        h_vectors[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < z; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_vectors, *d_A_global, *d_out_global, *d_out_constant;
    CUDA_CHECK(cudaMalloc(&d_vectors, K * z * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_global, z * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_global, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_constant, K * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors.data(), K * z * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_global, h_A.data(), z * sizeof(float), cudaMemcpyHostToDevice));

    // Warm up
    CUDA_CHECK(cudaDeviceSynchronize());
    SoftmaxDotProductGlobal(K, z, d_vectors, z, d_A_global, d_out_global, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    SoftmaxDotProductConstant(K, z, d_vectors, z, h_A.data(), d_out_constant, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark global memory version
    cudaEvent_t start_global, stop_global;
    CUDA_CHECK(cudaEventCreate(&start_global));
    CUDA_CHECK(cudaEventCreate(&stop_global));

    cudaEventRecord(start_global);
    for (int i = 0; i < num_iterations; i++) {
        SoftmaxDotProductGlobal(K, z, d_vectors, z, d_A_global, d_out_global, 0);
    }
    cudaEventRecord(stop_global);
    cudaEventSynchronize(stop_global);

    float milliseconds_global = 0;
    cudaEventElapsedTime(&milliseconds_global, start_global, stop_global);

    // Benchmark constant memory version
    cudaEvent_t start_constant, stop_constant;
    cudaEventCreate(&start_constant);
    cudaEventCreate(&stop_constant);

    cudaEventRecord(start_constant);
    for (int i = 0; i < num_iterations; i++) {
        SoftmaxDotProductConstant(K, z, d_vectors, z, h_A.data(), d_out_constant, 0);
    }
    cudaEventRecord(stop_constant);
    cudaEventSynchronize(stop_constant);

    float milliseconds_constant = 0;
    cudaEventElapsedTime(&milliseconds_constant, start_constant, stop_constant);

    // Copy results back
    cudaMemcpy(h_out_global.data(), d_out_global, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_constant.data(), d_out_constant, K * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results match
    float max_diff = 0.0f;
    for (size_t i = 0; i < K; i++) {
        float diff = std::abs(h_out_global[i] - h_out_constant[i]);
        max_diff = std::max(max_diff, diff);

        if (max_diff > 1e-5) {
            std::cout << "Results differ!, max difference between results: " << max_diff << std::endl;
        }
    }

    std::cout << "K = " << K << ", z = " << z << std::endl;
    std::cout << "Global memory version: " << milliseconds_global / num_iterations << " ms/iter" << std::endl;
    std::cout << "Constant memory version: " << milliseconds_constant / num_iterations << " ms/iter" << std::endl;
    std::cout << "Speedup: " << milliseconds_global / milliseconds_constant << "x" << std::endl;

    // Cleanup
    cudaFree(d_vectors);
    cudaFree(d_A_global);
    cudaFree(d_out_global);
    cudaFree(d_out_constant);
    cudaEventDestroy(start_global);
    cudaEventDestroy(stop_global);
    cudaEventDestroy(start_constant);
    cudaEventDestroy(stop_constant);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "argc != 3\n";
    }
    size_t K = std::atoll(argv[1]);
    size_t z = std::atoll(argv[2]);
    benchmark(K, z);
    std::cout << '\n';
    return 0;
}
