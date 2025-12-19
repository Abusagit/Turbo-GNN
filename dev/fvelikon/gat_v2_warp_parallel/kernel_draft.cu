#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define FULL_WARP_MASK 0xffffffff
constexpr int kMaxThreadsInWarp = 32;

__constant__ float d_const_attn_vec[4096];

// =============================================================================
// GATv2 Kernel with CSR Graph Format
// =============================================================================


__device__ __forceinline__ float leaky_relu_elementwise(float x, float negative_slope) {
    return  (x > 0.0f) ? x : negative_slope * x;
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_WARP_MASK, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = kMaxThreadsInWarp / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(FULL_WARP_MASK, x, offset));
    }
    return x;
}


struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;

    __device__ __forceinline__ OnlineSoftmaxState() : max_val(-FLT_MAX), sum_exp(0.0f) {}

    __device__ __forceinline__ void update(float logit) {
        float old_max = max_val;
        max_val = fmaxf(max_val, logit);

        // correction factor for previous sum when max changes
        float correction = __expf(old_max - max_val);
        sum_exp = sum_exp * correction + __expf(logit - max_val);
    }

    __device__ __forceinline__ float get_alpha(float logit) const {
        return __expf(logit - max_val) / sum_exp;
    }
};


__global__ void GATv2Kernel_CSR(
    size_t N,
    size_t z,
    const float* __restrict__ d_l,          // [N, z] - left features
    const float* __restrict__ d_r,          // [N, z] - right features
    const int* __restrict__ d_row_ptr,      // [N+1] - CSR row pointers
    const int* __restrict__ d_col_idx,      // [E] - CSR column indices (neighbor IDs)
    const float* __restrict__ d_attn_vec,   // [z] - attention vector (nullptr if constant)
    float* __restrict__ d_h_out,            // [N, z] - output node features
    float* __restrict__ d_logits_out,        // [E] - attention weights per edge -- can be used for backward
    float negative_slope
) {
    // shared memory layout:
    // [0, z): l_i cached
    extern __shared__ float shared[];
    float* l_shared = shared;

    int node_i = blockIdx.x;
    int lane_id = threadIdx.x;

    if (node_i >= N) {
        return;
    }

    // neighbor range from CSR
    int edge_start = d_row_ptr[node_i];
    int edge_end = d_row_ptr[node_i + 1];
    int num_neighbors = edge_end - edge_start;

    // skip isolated nodes (no neighbors)
    if (num_neighbors == 0) {
        return;
    }

    const float4* attn_ptr = reinterpret_cast<const float4*>(d_attn_vec);
    int num_float4 = z / 4;

    // ==========================================
    // 0: Load l_i into shared memory
    // ==========================================
    {
        const float4* l_ptr = reinterpret_cast<const float4*>(d_l + node_i * z);
        float4* l_shared_f4 = reinterpret_cast<float4*>(l_shared);

        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            l_shared_f4[i] = l_ptr[i];
        }
    }
    __syncthreads();


    // ==========================================
    // 1: compute attention logits for each neighbor
    // e_ij = a^T @ LeakyReLU(l_i + r_j)
    // ONLINE SOFTMAX
    // ==========================================
    const float4* l_f4 = reinterpret_cast<const float4*>(l_shared);
    OnlineSoftmaxState softmax_state;


    // ==========================================
    // 2: compute attention weights and aggregate
    // h_i = sum_j a_ij * r_j
    // ==========================================

    // register accumulators for output
    int float4_per_thread = (num_float4 + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;  // Ceiling division
    float4 h_acc[8];  // Support up to z=1024 NOTE TODO THIS IS A WORKAROUND!!!!!!!
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        h_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // accumulate over all neighbors
    for (int k = 0; k < num_neighbors; ++k) {
        int neighbor_j = d_col_idx[edge_start + k];
        const float4* r_ptr = reinterpret_cast<const float4*>(d_r + neighbor_j * z);

        float dot = 0.0f;
        for (int i = lane_id; i < num_float4; i += kMaxThreadsInWarp) {
            float4 l_val = l_f4[i];
            float4 r_val = r_ptr[i];
            float4 a_val = attn_ptr[i];

            float4 sum;
            sum.x = l_val.x + r_val.x;
            sum.y = l_val.y + r_val.y;
            sum.z = l_val.z + r_val.z;
            sum.w = l_val.w + r_val.w;

            sum.x = leaky_relu_elementwise(sum.x, negative_slope);
            sum.y = leaky_relu_elementwise(sum.y, negative_slope);
            sum.z = leaky_relu_elementwise(sum.z, negative_slope);
            sum.w = leaky_relu_elementwise(sum.w, negative_slope);
            dot += sum.x * a_val.x + sum.y * a_val.y + sum.z * a_val.z + sum.w * a_val.w;
        }

        dot = warp_reduce_sum(dot);
        // save old max before update
        float old_max = softmax_state.max_val;
        // update softmax statistics
        softmax_state.update(dot);
        // compute rescaling factor (this is computed in update() but we need it here too)
        float rescale = __expf(old_max - softmax_state.max_val);

        // rescale previously accumulated values
        #pragma unroll
        for (int i = 0; i < float4_per_thread; ++i) {  // NOTE currently thys loop has at most 8 iterations
                                                    // TODO support dimension > 1024, maybe use more blocks?
            h_acc[i].x *= rescale;
            h_acc[i].y *= rescale;
            h_acc[i].z *= rescale;
            h_acc[i].w *= rescale;
        }

        // compute unnormalized contribution for this neighbor
        float contribution = __expf(dot - softmax_state.max_val);

        // add weighted contribution from this neighbor
        for (int i = 0; i < float4_per_thread; ++i) {
            int idx = lane_id + i * kMaxThreadsInWarp;
            if (idx < num_float4) {
                float4 r_val = r_ptr[idx];

                h_acc[i].x += contribution * r_val.x;
                h_acc[i].y += contribution * r_val.y;
                h_acc[i].z += contribution * r_val.z;
                h_acc[i].w += contribution * r_val.w;
            }
        }
    }

    float4* h_out_ptr = reinterpret_cast<float4*>(d_h_out + node_i * z);

    for (int i = 0; i < float4_per_thread; ++i) {
        int idx = lane_id + i * kMaxThreadsInWarp;
        if (idx < num_float4) {
            h_out_ptr[idx] = h_acc[i];
        }
    }
}


// =============================================================================
// Launcher
// =============================================================================

void GATv2Forward_CSR(
    size_t N, size_t z,
    const float* d_l,
    const float* d_r,
    const int* d_row_ptr,
    const int* d_col_idx,
    const float* d_attn_vec,
    float* d_h_out,
    float* d_logits_out,
    float negative_slope,
    int max_neighbors,
    cudaStream_t stream = 0
) {
    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N);

    // shared memory: z floats for l_i + max_neighbors floats for logits
    size_t shared_mem_size = z * sizeof(float);

    GATv2Kernel_CSR<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        N, z, d_l, d_r, d_row_ptr, d_col_idx, d_attn_vec,
        d_h_out, d_logits_out, negative_slope
    );
}


// =============================================================================
// Generate random graph in CSR format
// =============================================================================

void generateRandomCSR(
    size_t N,
    size_t avg_degree,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    int& max_degree
) {
    std::mt19937 rng(42);
    std::poisson_distribution<int> degree_dist(avg_degree);
    std::uniform_int_distribution<int> neighbor_dist(0, N - 1);

    row_ptr.resize(N + 1);
    col_idx.clear();

    row_ptr[0] = 0;
    max_degree = 0;

    for (size_t i = 0; i < N; ++i) {
        int degree = std::max(1, std::min((int)avg_degree * 3, degree_dist(rng)));
        max_degree = std::max(max_degree, degree);

        std::vector<int> neighbors;
        for (int j = 0; j < degree; ++j) {
            int neighbor = neighbor_dist(rng);
            if (neighbor != (int)i) {  // No self-loops
                neighbors.push_back(neighbor);
            }
        }

        // Remove duplicates
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        for (int n : neighbors) {
            col_idx.push_back(n);
        }

        row_ptr[i + 1] = col_idx.size();
    }
}


// =============================================================================
// Benchmark
// =============================================================================

void benchmarkGATv2_CSR() {
    std::cout << "\nBenchmarking: GATv2 with CSR Graph Format\n";
    std::cout << std::string(120, '=') << std::endl;
    std::cout << std::setw(10) << "N"
              << std::setw(12) << "Avg Degree"
              << std::setw(12) << "Max Degree"
              << std::setw(10) << "Edges"
              << std::setw(8) << "z"
              << std::setw(14) << "Time (ms)"
              << std::setw(18) << "Throughput (MN/s)"
              << std::setw(16) << "Edges/s (M)"
              << std::setw(12) << "Verify"
              << std::endl;
    std::cout << std::string(120, '-') << std::endl;

    struct TestCase {
        size_t N;
        size_t avg_degree;
        size_t z;
    };

    std::vector<TestCase> test_cases = {

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


        {1000000, 4, 128},
        {1000000, 8, 128},
        {1000000, 16, 128},
        {1000000, 32, 128},

        {100000, 8, 64},
        {100000, 8, 128},
        {100000, 8, 256},
        {100000, 8, 512},
        {100000, 8, 1024},
    };

    const int num_iterations = 50;
    const int warmup_iterations = 10;
    const float negative_slope = 0.2f;

    for (const auto& tc : test_cases) {
        size_t N = tc.N;
        size_t avg_degree = tc.avg_degree;
        size_t z = tc.z;

        // random graph
        std::vector<int> h_row_ptr, h_col_idx;
        int max_degree;
        generateRandomCSR(N, avg_degree, h_row_ptr, h_col_idx, max_degree);
        size_t E = h_col_idx.size();

        std::vector<float> h_l(N * z);
        std::vector<float> h_r(N * z);
        std::vector<float> h_attn_vec(z);
        std::vector<float> h_output(N * z);
        std::vector<float> h_alpha(E);

        srand(42);
        for (size_t i = 0; i < N * z; i++) {
            h_l[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
            h_r[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        for (size_t i = 0; i < z; i++) {
            h_attn_vec[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }

        float *d_l, *d_r, *d_attn_vec, *d_h_out, *d_alpha_out;
        int *d_row_ptr, *d_col_idx;

        CUDA_CHECK(cudaMalloc(&d_l, N * z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r, N * z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_vec, z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_h_out, N * z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha_out, E * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx, E * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_l, h_l.data(), N * z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), N * z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn_vec, h_attn_vec.data(), z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), E * sizeof(int), cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup_iterations; i++) {
            GATv2Forward_CSR(N, z, d_l, d_r, d_row_ptr, d_col_idx, d_attn_vec,
                            d_h_out, d_alpha_out, negative_slope, max_degree);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iterations; i++) {
            GATv2Forward_CSR(N, z, d_l, d_r, d_row_ptr, d_col_idx, d_attn_vec,
                            d_h_out, d_alpha_out, negative_slope, max_degree);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= num_iterations;

        CUDA_CHECK(cudaMemcpy(h_alpha.data(), d_alpha_out, E * sizeof(float), cudaMemcpyDeviceToHost));

        bool valid = true;
        for (size_t i = 0; i < std::min(N, size_t(1000)); i++) {
            int start_idx = h_row_ptr[i];
            int end_idx = h_row_ptr[i + 1];
            if (end_idx > start_idx) {
                float sum = 0.0f;
                for (int j = start_idx; j < end_idx; j++) {
                    sum += h_alpha[j];
                }
                if (std::abs(sum - 1.0f) > 1e-3f) {
                    valid = false;
                    break;
                }
            }
        }

        float throughput_nodes = (N / 1e6) / (ms / 1000.0f);
        float throughput_edges = (E / 1e6) / (ms / 1000.0f);

        std::cout << std::setw(10) << N
                  << std::setw(12) << avg_degree
                  << std::setw(12) << max_degree
                  << std::setw(10) << (E / 1000) << "K"
                  << std::setw(8) << z
                  << std::setw(14) << std::fixed << std::setprecision(4) << ms
                  << std::setw(18) << std::setprecision(2) << throughput_nodes
                  << std::setw(16) << throughput_edges
                  << std::setw(12) << (valid ? "✓" : "✗")
                  << std::endl;

        CUDA_CHECK(cudaFree(d_l));
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaFree(d_attn_vec));
        CUDA_CHECK(cudaFree(d_h_out));
        CUDA_CHECK(cudaFree(d_alpha_out));
        CUDA_CHECK(cudaFree(d_row_ptr));
        CUDA_CHECK(cudaFree(d_col_idx));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}


int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "L2 Cache Size: " << (prop.l2CacheSize / 1024) << " KB" << std::endl;
    std::cout << "Peak Bandwidth: "
              << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6)
              << " GB/s" << std::endl;

    benchmarkGATv2_CSR();

    return 0;
}
