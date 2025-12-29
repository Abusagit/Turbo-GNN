#include "softmax.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

constexpr size_t kMaxConstantMemorySize = 16 * 1024; // 16KB = 4096 floats

// Constant memory for query vector
__constant__ float d_const_query[kMaxConstantMemorySize / sizeof(float)];

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel 1: Vector A in GLOBAL memory
__global__ void ComputeLogitsMatrixGlobalKernel(
    size_t N,
    size_t K,
    size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_logits
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K;

    if (global_idx < total_elements) {
        // Convert 1D index to (n, k) indices
        int n = global_idx / K;
        int k = global_idx % K;

        const float* vec_nk = d_input + (n * K + k) * z;

        float dot_product = 0.0f;

        if (z >= 4 && z % 4 == 0) {
            const float4* vec_ptr = reinterpret_cast<const float4*>(vec_nk);
            const float4* a_ptr = reinterpret_cast<const float4*>(d_vector_A);

            int num_float4 = z / 4;
            #pragma unroll 4
            for (int i = 0; i < num_float4; i++) {
                float4 v = vec_ptr[i];
                float4 a = a_ptr[i];

                dot_product += v.x * a.x;
                dot_product += v.y * a.y;
                dot_product += v.z * a.z;
                dot_product += v.w * a.w;
            }
        } else {
            for (int i = 0; i < z; i++) {
                dot_product += vec_nk[i] * d_vector_A[i];
            }
        }

        d_logits[n * K + k] = dot_product;
    }
}

// Kernel 2: Vector A in CONSTANT memory
__global__ void ComputeLogitsMatrixConstantKernel(
    size_t N,
    size_t K,
    size_t z,
    const float* d_input,
    float* d_logits
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K;

    if (global_idx < total_elements) {
        int n = global_idx / K;
        int k = global_idx % K;

        const float* vec_nk = d_input + (n * K + k) * z;

        float dot_product = 0.0f;

        if (z >= 4 && z % 4 == 0) {
            const float4* vec_ptr = reinterpret_cast<const float4*>(vec_nk);
            const float4* a_ptr = reinterpret_cast<const float4*>(d_const_query);

            int num_float4 = z / 4;
            #pragma unroll 4
            for (int i = 0; i < num_float4; i++) {
                float4 v = vec_ptr[i];
                float4 a = a_ptr[i];

                dot_product += v.x * a.x;
                dot_product += v.y * a.y;
                dot_product += v.z * a.z;
                dot_product += v.w * a.w;
            }
        } else {
            for (int i = 0; i < z; i++) {
                dot_product += vec_nk[i] * d_const_query[i];
            }
        }

        d_logits[n * K + k] = dot_product;
    }
}


// Host wrapper for GLOBAL memory version
void ComputeLogitsThenSoftmaxGlobal(
    size_t N,
    size_t K,
    size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_logits,
    float* d_output,
    cudaStream_t stream
) {
    // Strategy: each thread processes one (n, k) pair
    // Total work: N * K elements

    int threads_per_block = 32;
    int total_elements = N * K;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    ComputeLogitsMatrixGlobalKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        N, K, z, d_input, d_vector_A, d_logits
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Softmax(N, K, d_logits, K, d_output, K, stream);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Same for constant version
void ComputeLogitsThenSoftmaxConstant(
    size_t N,
    size_t K,
    size_t z,
    const float* h_vector_A,
    const float* d_input,
    float* d_logits,
    float* d_output,
    cudaStream_t stream,
    bool copy_to_constant
) {
    if (copy_to_constant) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_const_query, h_vector_A, z * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int threads_per_block = 32;
    int total_elements = N * K;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    ComputeLogitsMatrixConstantKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        N, K, z, d_input, d_logits
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Softmax(N, K, d_logits, K, d_output, K, stream);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


void comprehensiveBenchmark() {
    std::cout << "Benchmarking: Dot Product + Softmax (reusing your kernel)\n";
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(8) << "z"
              << std::setw(15) << "Global (ms)"
              << std::setw(15) << "Constant (ms)"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        // N, K, z - ensure K is divisible by 4 for your softmax kernel
        {10000, 4 , 128},       // Small
        {10000, 8 , 128},
        {10000, 16, 128},

        {500000, 4 , 128},       // Medium
        {500000, 8 , 128},
        {500000, 16, 128},

        {1000000, 4 , 128},       // Large
        {1000000, 8 , 128},
        {1000000, 16, 128},
    };

    const int num_iterations = 100;

    for (auto [N, K, z] : test_cases) {
        if (K % 4 != 0) {
            std::cout << "Skipping N=" << N << ", K=" << K << ", z=" << z
                      << " (K not divisible by 4)" << std::endl;
            continue;
        }

        std::cout << "Testing N=" << N << ", K=" << K << ", z=" << z << "..." << std::flush;

        // Allocate host memory for (N, K, z) tensor and vector A
        std::vector<float> h_input(N * K * z);
        std::vector<float> h_vector_A(z);

        // Initialize
        for (size_t i = 0; i < N * K * z; i++) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        for (size_t i = 0; i < z; i++) {
            h_vector_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }

        // Allocate device memory
        float *d_input, *d_vector_A_global;
        float *d_logits_global, *d_logits_constant;
        float *d_output_global, *d_output_constant;

        CUDA_CHECK(cudaMalloc(&d_input, N * K * z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vector_A_global, z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits_global, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits_constant, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_global, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_constant, N * K * sizeof(float)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * K * z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vector_A_global, h_vector_A.data(), z * sizeof(float), cudaMemcpyHostToDevice));

        // Pre-copy to constant memory (one-time cost, not measured)
        CUDA_CHECK(cudaMemcpyToSymbol(d_const_query, h_vector_A.data(), z * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Warm up
        ComputeLogitsThenSoftmaxGlobal(N, K, z, d_input, d_vector_A_global,
                                       d_logits_global, d_output_global, 0);
        ComputeLogitsThenSoftmaxConstant(N, K, z, h_vector_A.data(), d_input,
                                         d_logits_constant, d_output_constant, 0, false);

        std::cout << " warmup done..." << std::flush;

        // Benchmark GLOBAL memory version
        cudaEvent_t start_global, stop_global;
        CUDA_CHECK(cudaEventCreate(&start_global));
        CUDA_CHECK(cudaEventCreate(&stop_global));

        CUDA_CHECK(cudaEventRecord(start_global));
        for (int i = 0; i < num_iterations; i++) {
            ComputeLogitsThenSoftmaxGlobal(N, K, z, d_input, d_vector_A_global,
                                          d_logits_global, d_output_global, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop_global));
        CUDA_CHECK(cudaEventSynchronize(stop_global));

        float ms_global = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_global, start_global, stop_global));
        ms_global /= num_iterations;

        std::cout << " global done..." << std::flush;

        // Benchmark CONSTANT memory version
        cudaEvent_t start_constant, stop_constant;
        CUDA_CHECK(cudaEventCreate(&start_constant));
        CUDA_CHECK(cudaEventCreate(&stop_constant));

        CUDA_CHECK(cudaEventRecord(start_constant));
        for (int i = 0; i < num_iterations; i++) {
            ComputeLogitsThenSoftmaxConstant(N, K, z, h_vector_A.data(), d_input,
                                           d_logits_constant, d_output_constant, 0, false);
        }
        CUDA_CHECK(cudaEventRecord(stop_constant));
        CUDA_CHECK(cudaEventSynchronize(stop_constant));

        float ms_constant = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_constant, start_constant, stop_constant));
        ms_constant /= num_iterations;

        float speedup = ms_global / ms_constant;

        std::cout << "\r" << std::setw(8) << N
                  << std::setw(8) << K
                  << std::setw(8) << z
                  << std::setw(15) << std::fixed << std::setprecision(4) << ms_global
                  << std::setw(15) << ms_constant
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::endl;

        // Cleanup
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_vector_A_global));
        CUDA_CHECK(cudaFree(d_logits_global));
        CUDA_CHECK(cudaFree(d_logits_constant));
        CUDA_CHECK(cudaFree(d_output_global));
        CUDA_CHECK(cudaFree(d_output_constant));
        CUDA_CHECK(cudaEventDestroy(start_global));
        CUDA_CHECK(cudaEventDestroy(stop_global));
        CUDA_CHECK(cudaEventDestroy(start_constant));
        CUDA_CHECK(cudaEventDestroy(stop_constant));
    }
}


int main() {
    // Check device properties
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout << std::endl;

    comprehensiveBenchmark();
    return 0;
}
