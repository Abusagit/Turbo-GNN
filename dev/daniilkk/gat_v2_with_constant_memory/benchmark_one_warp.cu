#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

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

#define CUDART_MINF_F __int_as_float(0xff800000)

__constant__ float d_const_vector_A[4096]; // z up to 4096 floats


// Single warp per row - no __syncthreads__ needed!
__global__ void GATFinalPartKernel_SingleWarp(
    size_t N,
    size_t K,
    size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_out,
    bool is_constant
) {
    // No shared memory needed for vector_A - just use constant or global
    // Only need shared memory for logits
    extern __shared__ float logits[];

    int row_idx = blockIdx.x;
    int lane_id = threadIdx.x;  // 0-31

    if (row_idx >= N) return;

    const float* vec_A_ptr = is_constant ? d_const_vector_A : d_vector_A;
    const float4* a_ptr = reinterpret_cast<const float4*>(vec_A_ptr);
    int num_float4 = z / 4;

    // ==========================================
    // PHASE 1: Compute K dot products
    // ==========================================

    for (int k = 0; k < K; ++k) {
        const float* vec_nk = d_input + (row_idx * K + k) * z;
        const float4* vec_ptr = reinterpret_cast<const float4*>(vec_nk);

        float dot_product = 0.0f;

        // All 32 threads collaborate on z dimension
        for (int i = lane_id; i < num_float4; i += 32) {
            float4 v = vec_ptr[i];
            float4 a = a_ptr[i];
            dot_product += v.x * a.x + v.y * a.y + v.z * a.z + v.w * a.w;
        }

        // Warp shuffle reduction - no __syncthreads__ needed!
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
        }

        // Lane 0 writes result
        if (lane_id == 0) {
            logits[k] = dot_product;
        }
        // Implicit warp sync here - all lanes wait for shuffles to complete
    }

    // ==========================================
    // PHASE 2: Softmax - Find max value
    // ==========================================

    float max_val = CUDART_MINF_F;

    // Each thread processes some logits
    for (int i = lane_id; i < K; i += 32) {
        max_val = fmaxf(max_val, logits[i]);
    }

    // Warp shuffle reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    // Broadcast max to all lanes
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // ==========================================
    // PHASE 3: Compute exp and sum
    // ==========================================

    float sum_exp = 0.0f;

    for (int i = lane_id; i < K; i += 32) {
        float exp_val = __expf(logits[i] - max_val);
        logits[i] = exp_val;  // Store back
        sum_exp += exp_val;
    }

    // Warp shuffle reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }

    // Broadcast sum to all lanes
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    float inv_sum = 1.0f / sum_exp;

    // ==========================================
    // PHASE 4: Normalize and write output
    // ==========================================

    for (int i = lane_id; i < K; i += 32) {
        d_out[row_idx * K + i] = logits[i] * inv_sum;
    }
}

void FusedDotProductSoftmaxGlobal_SingleWarp(
    size_t N, size_t K, size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_output,
    cudaStream_t stream = 0
) {
    dim3 nThreads(32);  // Single warp
    dim3 nBlocks(N);    // One block per row

    // Shared memory: only K floats for logits
    size_t shared_mem_size = K * sizeof(float);

    GATFinalPartKernel_SingleWarp<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, K, z, d_input, d_vector_A, d_output, false
    );
}

void FusedDotProductSoftmaxConstant_SingleWarp(
    size_t N, size_t K, size_t z,
    const float* h_vector_A,
    const float* d_input,
    float* d_output,
    cudaStream_t stream = 0
) {
    cudaMemcpyToSymbol(d_const_vector_A, h_vector_A, z * sizeof(float));

    dim3 nThreads(32);
    dim3 nBlocks(N);

    size_t shared_mem_size = K * sizeof(float);

    GATFinalPartKernel_SingleWarp<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, K, z, d_input, nullptr, d_output, true
    );
}


void comprehensiveBenchmark() {
    std::cout << "Benchmarking: Fused Dot Product + Softmax (Single-Warp-Per-Row)\n";
    std::cout << std::string(130, '=') << std::endl;
    std::cout << std::setw(10) << "N"
              << std::setw(8) << "K"
              << std::setw(8) << "z"
              << std::setw(15) << "Global (ms)"
              << std::setw(15) << "Constant (ms)"
              << std::setw(20) << "Speedup (constant)"
              << std::setw(17) << "Bandwidth (GB/s)"
              << std::endl;
    std::cout << std::string(130, '-') << std::endl;

    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        {10000, 4, 128},
        {10000, 8, 128},
        {10000, 16, 128},
        {10000, 32, 128},
        {10000, 64, 128},

        {100000, 4, 128},
        {100000, 8, 128},
        {100000, 16, 128},
        {100000, 32, 128},
        {100000, 64, 128},

        {500000, 4, 128},
        {500000, 8, 128},
        {500000, 16, 128},
        {500000, 32, 128},
        {500000, 64, 128},

        {1000000, 4, 128},
        {1000000, 8, 128},
        {1000000, 16, 128},
        {1000000, 32, 128},
        {1000000, 64, 128},

        {10000, 4, 1024},
        {10000, 8, 1024},
        {10000, 16, 1024},
        {10000, 32, 1024},
        {10000, 64, 1024},

        {100000, 4, 1024},
        {100000, 8, 1024},
        {100000, 16, 1024},
        {100000, 32, 1024},
        {100000, 64, 1024},
    };

    const int num_iterations = 100;
    const int warmup_iterations = 10;

    for (auto [N, K, z] : test_cases) {
        if (K % 4 != 0 || z % 4 != 0) {
            std::cout << "Skipping N=" << N << ", K=" << K << ", z=" << z << std::endl;
            continue;
        }

        std::vector<float> h_input(N * K * z);
        std::vector<float> h_vector_A(z);
        std::vector<float> h_output_global(N * K);
        std::vector<float> h_output_constant(N * K);

        srand(42);
        for (size_t i = 0; i < N * K * z; i++) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        for (size_t i = 0; i < z; i++) {
            h_vector_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }

        float *d_input, *d_vector_A_global;
        float *d_output_global, *d_output_constant;

        CUDA_CHECK(cudaMalloc(&d_input, N * K * z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vector_A_global, z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_global, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_constant, N * K * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                              N * K * z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vector_A_global, h_vector_A.data(),
                              z * sizeof(float), cudaMemcpyHostToDevice));

        FusedDotProductSoftmaxConstant_SingleWarp(N, K, z, h_vector_A.data(), d_input, d_output_constant, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Warm up
        for (int i = 0; i < warmup_iterations; i++) {
            FusedDotProductSoftmaxGlobal_SingleWarp(N, K, z, d_input, d_vector_A_global, d_output_global, 0);
            FusedDotProductSoftmaxConstant_SingleWarp(N, K, z, h_vector_A.data(), d_input, d_output_constant, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark GLOBAL memory version
        cudaEvent_t start_global, stop_global;
        CUDA_CHECK(cudaEventCreate(&start_global));
        CUDA_CHECK(cudaEventCreate(&stop_global));

        CUDA_CHECK(cudaEventRecord(start_global));
        for (int i = 0; i < num_iterations; i++) {
            FusedDotProductSoftmaxGlobal_SingleWarp(N, K, z, d_input, d_vector_A_global, d_output_global, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop_global));
        CUDA_CHECK(cudaEventSynchronize(stop_global));

        float ms_global = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_global, start_global, stop_global));
        ms_global /= num_iterations;

        // Benchmark CONSTANT memory version
        cudaEvent_t start_constant, stop_constant;
        CUDA_CHECK(cudaEventCreate(&start_constant));
        CUDA_CHECK(cudaEventCreate(&stop_constant));

        CUDA_CHECK(cudaEventRecord(start_constant));
        for (int i = 0; i < num_iterations; i++) {
            FusedDotProductSoftmaxConstant_SingleWarp(N, K, z, h_vector_A.data(), d_input, d_output_constant, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop_constant));
        CUDA_CHECK(cudaEventSynchronize(stop_constant));

        float ms_constant = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_constant, start_constant, stop_constant));
        ms_constant /= num_iterations;

        // Copy results back for verification
        CUDA_CHECK(cudaMemcpy(h_output_global.data(),   d_output_global,   N * K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output_constant.data(), d_output_constant, N * K * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results match
        float max_diff = 0.0f;
        for (size_t i = 0; i < std::min(N * K, size_t(10000)); i++) {
            float diff = std::abs(h_output_global[i] - h_output_constant[i]);
            max_diff = std::max(max_diff, diff);
        }

        // Calculate speedup
        float speedup_constant = ms_global / ms_constant;

        // Calculate bandwidth
        size_t bytes_read = N * K * z * sizeof(float) + N * K * z * sizeof(float);
        size_t bytes_written = N * K * sizeof(float);
        float bandwidth_gbs = ((bytes_read + bytes_written) / 1e9) / (ms_global / 1000.0f);

        std::cout << std::setw(10) << N
                  << std::setw(8) << K
                  << std::setw(8) << z
                  << std::setw(15) << std::fixed << std::setprecision(4) << ms_global
                  << std::setw(15) << ms_constant
                  << std::setw(19) << std::setprecision(2) << speedup_constant << "x"
                  << std::setw(17) << std::setprecision(1) << bandwidth_gbs;

        if (max_diff > 1e-5) {
            std::cout << "  MISMATCH! max_diff=" << std::setprecision(6) << max_diff;
        }
        std::cout << std::endl;

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_vector_A_global));
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
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1e9) << " GB" << std::endl;
    std::cout << "Total Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Memory Clock Rate: " << (prop.memoryClockRate / 1e6) << " GHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: "
              << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6)
              << " GB/s" << std::endl;
    std::cout << std::endl;

    comprehensiveBenchmark();

    std::cout << "\n✓ Benchmark completed successfully!" << std::endl;

    return 0;
}
