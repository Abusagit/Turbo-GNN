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

constexpr size_t kThreadsInWarp = 32;

__constant__ float d_const_vector_A[4096]; // z up to 4096 floats

// Fused Kernel: Dot Product + Softmax with GLOBAL memory vector A
__global__ void GATFinalPartKernel(
    size_t N,
    size_t K,
    size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_out,
    bool is_constant,
    bool global_to_shared
) {
    extern __shared__ float shared_mem[];
    float* logits = shared_mem;
    float* vector_A_shared = logits + K;
    float* reduction = vector_A_shared + static_cast<int>(global_to_shared) * z;

    int row_idx = blockIdx.x;
    int thread_id = threadIdx.y;
    int threads_per_block = blockDim.y;
    int warp_id = thread_id / kThreadsInWarp;
    int num_warps = (threads_per_block + kThreadsInWarp - 1) / kThreadsInWarp;

    // Load vector A to shared memory if needed
    if (global_to_shared) {
        int num_float4 = z / 4;
        float4* vec_A_shared_f4 = reinterpret_cast<float4*>(vector_A_shared);
        const float4* vec_A_global_f4 = reinterpret_cast<const float4*>(d_vector_A);

        for (int i = thread_id; i < num_float4; i += threads_per_block) {
            vec_A_shared_f4[i] = vec_A_global_f4[i];
        }
        __syncthreads();
    }

    if (row_idx < N) {
        // ==========================================
        // PHASE 1: Compute K dot products (logits)
        // ==========================================
        for (int k = thread_id; k < K; k += threads_per_block) {
            const float* vec_nk = d_input + (row_idx * K + k) * z;
            float dot_product = 0.0f;

            // Select correct vector A source
            const float* vec_A_ptr = nullptr;
            if (is_constant) {
                vec_A_ptr = d_const_vector_A;
            } else if (global_to_shared) {
                vec_A_ptr = vector_A_shared;
            } else {
                vec_A_ptr = d_vector_A;
            }

            // Vectorized dot product (z always divisible by 4)
            const float4* vec_ptr = reinterpret_cast<const float4*>(vec_nk);
            const float4* a_ptr = reinterpret_cast<const float4*>(vec_A_ptr);
            int num_float4 = z / 4;

            #pragma unroll 4
            for (int i = 0; i < num_float4; i++) {
                float4 v = vec_ptr[i];
                float4 a = a_ptr[i];
                dot_product += v.x * a.x + v.y * a.y + v.z * a.z + v.w * a.w;
            }

            logits[k] = dot_product;
        }

        __syncthreads();

        // ==========================================
        // PHASE 2: Softmax - Find max value
        // ==========================================
        float max_val = CUDART_MINF_F;

        // Process K/4 float4 elements (K divisible by 4)
        int num_float4 = K / 4;
        for (int idx = thread_id; idx < num_float4; idx += threads_per_block) {
            float4 val = reinterpret_cast<const float4*>(logits)[idx];
            max_val = fmaxf(max_val, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        }

        // Warp-level max reduction
        #pragma unroll
        for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
        }

        // Store per-warp result
        if (thread_id % kThreadsInWarp == 0) {
            reduction[warp_id] = max_val;
        }
        __syncthreads();

        // Final reduction across warps
        if (warp_id == 0) {
            float val = (thread_id < num_warps) ? reduction[thread_id] : CUDART_MINF_F;
            #pragma unroll
            for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            }
            if (thread_id == 0) {
                reduction[0] = val;
            }
        }
        __syncthreads();
        max_val = reduction[0];

        // ==========================================
        // PHASE 3: Compute exp and sum
        // ==========================================
        float sum_exp = 0.0f;

        for (int idx = thread_id; idx < num_float4; idx += threads_per_block) {
            float4 val = reinterpret_cast<const float4*>(logits)[idx];
            sum_exp += __expf(val.x - max_val) + __expf(val.y - max_val) +
                       __expf(val.z - max_val) + __expf(val.w - max_val);
        }

        // Warp-level sum reduction
        #pragma unroll
        for (unsigned mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask);
        }

        if (thread_id % kThreadsInWarp == 0) {
            reduction[warp_id] = sum_exp;
        }
        __syncthreads();

        if (warp_id == 0) {
            sum_exp = (thread_id < num_warps) ? reduction[thread_id] : 0.0f;
            #pragma unroll
            for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
                sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask);
            }
            if (thread_id == 0) {
                reduction[0] = sum_exp;
            }
        }
        __syncthreads();
        float inv_sum = 1.0f / reduction[0];

        // ==========================================
        // PHASE 4: Compute final softmax and write output
        // ==========================================
        float4* d_out_f4 = reinterpret_cast<float4*>(d_out + row_idx * K);

        for (int idx = thread_id; idx < num_float4; idx += threads_per_block) {
            float4 val = reinterpret_cast<const float4*>(logits)[idx];
            val.x = __expf(val.x - max_val) * inv_sum;
            val.y = __expf(val.y - max_val) * inv_sum;
            val.z = __expf(val.z - max_val) * inv_sum;
            val.w = __expf(val.w - max_val) * inv_sum;
            d_out_f4[idx] = val;
        }
    }
}


void FusedDotProductSoftmaxGlobal(
    size_t N, size_t K, size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_output,
    cudaStream_t stream = 0
) {
    size_t threads_per_block = z / 4;
    dim3 nThreads(1, threads_per_block, 1);  // threadIdx.y as in your kernel
    dim3 nBlocks(N, 1, 1);  // One block per row

    // Shared memory: K floats for logits + workspace for reduction
    size_t shared_mem_size = K * sizeof(float) + (threads_per_block / kThreadsInWarp) * sizeof(float);

    GATFinalPartKernel<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, K, z, d_input, d_vector_A, d_output, false, false
    );
}

void FusedDotProductSoftmaxConstant(
    size_t N, size_t K, size_t z,
    const float* h_vector_A,
    const float* d_input,
    float* d_output,
    cudaStream_t stream = 0
) {
    size_t threads_per_block = z / 4;
    cudaMemcpyToSymbol(d_const_vector_A, h_vector_A, z * sizeof(float));

    dim3 nThreads(1, threads_per_block, 1);
    dim3 nBlocks(N, 1, 1);

    size_t shared_mem_size = K * sizeof(float) + (threads_per_block / kThreadsInWarp) * sizeof(float);

    GATFinalPartKernel<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, K, z, d_input, nullptr, d_output, true, false
    );
}

void FusedDotProductSoftmaxShared(
    size_t N, size_t K, size_t z,
    const float* d_input,
    const float* d_vector_A,
    float* d_output,
    cudaStream_t stream = 0
) {
    size_t threads_per_block = z / 4;
    dim3 nThreads(1, threads_per_block, 1);  // threadIdx.y as in your kernel
    dim3 nBlocks(N, 1, 1);  // One block per row

    // Shared memory: K floats for logits + workspace for reduction + vector_A
    size_t shared_mem_size = K * sizeof(float) + (threads_per_block / kThreadsInWarp) * sizeof(float) + z * sizeof(float);

    GATFinalPartKernel<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, K, z, d_input, d_vector_A, d_output, false, true
    );
}


void comprehensiveBenchmark() {
    std::cout << "Benchmarking: Fused Dot Product + Softmax\n";
    std::cout << std::string(130, '=') << std::endl;
    std::cout << std::setw(10) << "N"
              << std::setw(8) << "K"
              << std::setw(8) << "z"
              << std::setw(15) << "Global (ms)"
              << std::setw(15) << "Shared (ms)"
              << std::setw(15) << "Constant (ms)"
              << std::setw(20) << "Speedup (shared)"
              << std::setw(20) << "Speedup (constant)"
              << std::setw(17) << "Bandwidth (GB/s)"
              << std::endl;
    std::cout << std::string(130, '-') << std::endl;

    std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
        // N, K, z

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

        // {500000, 4, 1024},
        // {500000, 8, 1024},
        // {500000, 16, 1024},
        // {500000, 32, 1024},

        // {1000000, 4, 1024},
        // {1000000, 8, 1024},
        // {1000000, 16, 1024},
    };

    const int num_iterations = 100;
    const int warmup_iterations = 10;

    for (auto [N, K, z] : test_cases) {
        if (K % 4 != 0 || z % 4 != 0) {
            std::cout << "Skipping N=" << N << ", K=" << K << ", z=" << z << std::endl;
            continue;
        }

        // Allocate host memory
        std::vector<float> h_input(N * K * z);
        std::vector<float> h_vector_A(z);
        std::vector<float> h_output_global(N * K);
        std::vector<float> h_output_shared(N * K);
        std::vector<float> h_output_constant(N * K);

        // Initialize with small random values
        srand(42);
        for (size_t i = 0; i < N * K * z; i++) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        for (size_t i = 0; i < z; i++) {
            h_vector_A[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }

        // Allocate device memory
        float *d_input, *d_vector_A_global;
        float *d_output_global, *d_output_shared, *d_output_constant;

        CUDA_CHECK(cudaMalloc(&d_input, N * K * z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vector_A_global, z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_global, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_shared, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_constant, N * K * sizeof(float)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                              N * K * z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vector_A_global, h_vector_A.data(),
                              z * sizeof(float), cudaMemcpyHostToDevice));

        FusedDotProductSoftmaxConstant(N, K, z, h_vector_A.data(), d_input, d_output_constant, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Warm up
        for (int i = 0; i < warmup_iterations; i++) {
            FusedDotProductSoftmaxGlobal(N, K, z, d_input, d_vector_A_global, d_output_global, 0);
            FusedDotProductSoftmaxShared(N, K, z, d_input, d_vector_A_global, d_output_global, 0);
            FusedDotProductSoftmaxConstant(N, K, z, h_vector_A.data(), d_input, d_output_constant, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark GLOBAL memory version
        cudaEvent_t start_global, stop_global;
        CUDA_CHECK(cudaEventCreate(&start_global));
        CUDA_CHECK(cudaEventCreate(&stop_global));

        CUDA_CHECK(cudaEventRecord(start_global));
        for (int i = 0; i < num_iterations; i++) {
            FusedDotProductSoftmaxGlobal(N, K, z, d_input, d_vector_A_global, d_output_global, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop_global));
        CUDA_CHECK(cudaEventSynchronize(stop_global));

        float ms_global = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_global, start_global, stop_global));
        ms_global /= num_iterations;

        // Benchmark SHARED memory version
        cudaEvent_t start_shared, stop_shared;
        CUDA_CHECK(cudaEventCreate(&start_shared));
        CUDA_CHECK(cudaEventCreate(&stop_shared));

        CUDA_CHECK(cudaEventRecord(start_shared));
        for (int i = 0; i < num_iterations; i++) {
            FusedDotProductSoftmaxShared(N, K, z, d_input, d_vector_A_global, d_output_shared, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop_shared));
        CUDA_CHECK(cudaEventSynchronize(stop_shared));

        float ms_shared = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start_shared, stop_shared));
        ms_shared /= num_iterations;

        // Benchmark CONSTANT memory version
        cudaEvent_t start_constant, stop_constant;
        CUDA_CHECK(cudaEventCreate(&start_constant));
        CUDA_CHECK(cudaEventCreate(&stop_constant));

        CUDA_CHECK(cudaEventRecord(start_constant));
        for (int i = 0; i < num_iterations; i++) {
            FusedDotProductSoftmaxConstant(N, K, z, h_vector_A.data(), d_input, d_output_constant, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop_constant));
        CUDA_CHECK(cudaEventSynchronize(stop_constant));

        float ms_constant = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_constant, start_constant, stop_constant));
        ms_constant /= num_iterations;

        // Copy results back for verification
        CUDA_CHECK(cudaMemcpy(h_output_global.data(),   d_output_global,   N * K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output_shared.data(),   d_output_shared,   N * K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output_constant.data(), d_output_constant, N * K * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results match
        float max_diff = 0.0f;
        for (size_t i = 0; i < std::min(N * K, size_t(10000)); i++) {
            float diff1 = std::abs(h_output_global[i] - h_output_constant[i]);
            float diff2 = std::abs(h_output_global[i] - h_output_shared[i]);
            float diff3 = std::abs(h_output_constant[i] - h_output_shared[i]);
            max_diff = std::max(max_diff, diff1);
            max_diff = std::max(max_diff, diff2);

            // if (diff1 > 1e-3) {
            //     std::cout << "(global - constant mismatch) ";
            // }
            // if (diff2 > 1e-3) {
            //     std::cout << "(global - shared mismatch) ";
            // }
            // if (diff3 > 1e-3) {
            //     std::cout << "(constant - shared mismatch) ";
            // }
            // break;
        }

        // Calculate speedup
        float speedup_constant = ms_global / ms_constant;
        float speedup_shared = ms_global / ms_shared;

        // Calculate bandwidth (bytes read + written)
        // Read: N*K*z (input) + N*K*z (for each access to vector A, conservative estimate)
        // Write: N*K (output)
        size_t bytes_read = N * K * z * sizeof(float) + N * K * z * sizeof(float);
        size_t bytes_written = N * K * sizeof(float);
        float bandwidth_gbs = ((bytes_read + bytes_written) / 1e9) / (ms_global / 1000.0f);

        std::cout << std::setw(10) << N
                  << std::setw(8) << K
                  << std::setw(8) << z
                  << std::setw(15) << std::fixed << std::setprecision(4) << ms_global
                  << std::setw(15) << ms_shared
                  << std::setw(15) << ms_constant
                  << std::setw(19) << std::setprecision(2) << speedup_shared << "x"
                  << std::setw(19) << speedup_constant << "x"
                  << std::setw(17) << std::setprecision(1) << bandwidth_gbs;

        if (max_diff > 1e-6) {
            std::cout << "  MISMATCH! max_diff=" << std::setprecision(5) << max_diff;
        }
        std::cout << std::endl;

        // Cleanup
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_vector_A_global));
        CUDA_CHECK(cudaFree(d_output_global));
        CUDA_CHECK(cudaFree(d_output_shared));
        CUDA_CHECK(cudaFree(d_output_constant));
        CUDA_CHECK(cudaEventDestroy(start_global));
        CUDA_CHECK(cudaEventDestroy(stop_global));
        CUDA_CHECK(cudaEventDestroy(start_shared));
        CUDA_CHECK(cudaEventDestroy(stop_shared));
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
