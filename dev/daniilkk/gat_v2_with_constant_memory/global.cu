#include <cuda_runtime.h>

#define CUDART_MINF_F __int_as_float(0xff800000)

constexpr size_t kMaxThreadsInBlock = 32;
constexpr size_t kThreadsInWarp = 32;
constexpr int URF = 8;

// Kernel 1: Vector A in global memory
// Input: K vectors of dimension z stored as (K, z) matrix
// Vector A of dimension z in global memory
// Output: K softmax values
__global__ void SoftmaxDotProductGlobalKernel(
    size_t K,           // number of vectors
    size_t z,           // dimension of each vector
    const float* d_vectors,  // K x z matrix
    float* d_logits,
    size_t vectors_stride,
    const float* d_A,   // vector A of size z (global memory)
    float* d_out        // output K softmax values
) {
    extern __shared__ float reduction[];

    int thread_id = threadIdx.x;
    int warp_id = thread_id / kThreadsInWarp;

    // Step 1: Compute dot products (logits) for all K vectors
    // Each thread computes partial dot products
    float max_val = CUDART_MINF_F;

    for (int k = thread_id; k < K; k += blockDim.x) {
        float dot_product = 0.0f;

        // Compute dot product using float4 for vectorized loads
        const float4* vec_ptr = reinterpret_cast<const float4*>(d_vectors + k * vectors_stride);
        const float4* a_ptr = reinterpret_cast<const float4*>(d_A);

        int num_float4 = z / 4;

        #pragma unroll URF
        for (int i = 0; i < num_float4; i++) {
            float4 vec_val = vec_ptr[i];
            float4 a_val = a_ptr[i];

            dot_product += vec_val.x * a_val.x;
            dot_product += vec_val.y * a_val.y;
            dot_product += vec_val.z * a_val.z;
            dot_product += vec_val.w * a_val.w;
        }

        // Handle remaining elements if z % 4 != 0
        for (int i = num_float4 * 4; i < z; i++) {
            dot_product += d_vectors[k * vectors_stride + i] * d_A[i];
        }

        if (k < K) {
            d_logits[k] = dot_product;
            max_val = fmaxf(max_val, dot_product);
        }
    }

    // Step 2: Max reduction across threads
    #pragma unroll
    for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
    }

    if (thread_id % kThreadsInWarp == 0) {
        reduction[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val_to_reduce = (thread_id < (blockDim.x / kThreadsInWarp)) ? reduction[thread_id] : CUDART_MINF_F;
        #pragma unroll
        for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
            max_val_to_reduce = fmaxf(max_val_to_reduce, __shfl_xor_sync(0xffffffff, max_val_to_reduce, mask));
        }
        if (thread_id == 0) {
            reduction[0] = max_val_to_reduce;
        }
    }
    __syncthreads();
    max_val = reduction[0];

    // Step 3: Sum-exp reduction
    float sum_exp = 0.0f;

    for (int k = thread_id; k < K; k += blockDim.x) {
        float exp_val = __expf(d_logits[k] - max_val);
        d_logits[k] = exp_val;  // Store exp values for final computation
        sum_exp += exp_val;
    }

    #pragma unroll
    for (unsigned mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask);
    }

    if (thread_id % kThreadsInWarp == 0) {
        reduction[warp_id] = sum_exp;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (thread_id < (blockDim.x / kThreadsInWarp)) ? reduction[thread_id] : 0.0f;
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

    // Step 4: Write softmax output
    for (int k = thread_id; k < K; k += blockDim.x) {
        d_out[k] = d_logits[k] * inv_sum;
    }
}


// Host wrapper for global memory version
void SoftmaxDotProductGlobal(
    size_t K, size_t z,
    const float* d_vectors, size_t vectors_stride,
    const float* d_A,
    float* d_out,
    cudaStream_t stream = 0
) {
    dim3 nThreads(kMaxThreadsInBlock);
    dim3 nBlocks(1);

    float* d_logits;
    cudaMalloc(&d_logits, K * sizeof(float));

    size_t shared_mem_size = 4 * (kMaxThreadsInBlock / kThreadsInWarp) * sizeof(float);

    SoftmaxDotProductGlobalKernel<<<nBlocks, nThreads, shared_mem_size, stream>>>(
        K, z, d_vectors, d_logits, vectors_stride, d_A, d_out
    );
}
