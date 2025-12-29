#include <cuda_runtime.h>

#define CUDART_MINF_F __int_as_float(0xff800000)

constexpr size_t kMaxThreadsInBlock = 1024;
constexpr size_t kThreadsInWarp = 32;
constexpr int URF{8};

__global__ void SoftmaxKernel(size_t rows, size_t cols, const float* d_input_matrix,
                              size_t input_stride, float* d_out, size_t out_stride) {
    extern __shared__ float reduction[];

    int i = blockIdx.x;

    if (i < rows) {
        int thread_id = threadIdx.y;
        int warp_id = thread_id / kThreadsInWarp;

        float max_val = CUDART_MINF_F;

        // load 4 float values (128 bytes at the same time):
        const float4* f4_input_ptr = reinterpret_cast<const float4*>(d_input_matrix + i * input_stride + thread_id * 4);

#pragma unroll URF
        for (int shift = thread_id; shift < cols / 4; shift += blockDim.y) { // for each thread do max-reduce with the corresponding thread in other thread blocks
            float4 val = *f4_input_ptr; // return float4 value
            max_val = fmaxf(max_val, val.x);
            max_val = fmaxf(max_val, val.y);
            max_val = fmaxf(max_val, val.z);
            max_val = fmaxf(max_val, val.w);
            f4_input_ptr += blockDim.y; // move pointer
        }
#pragma unroll
        // reduce max_val within the warp: mask determines which threads participate in the reduction
        // first step: for the first warp (warp idx 0) at thr first step the mask is FFFF meaning that each thread reduces the value from 16 threads to the right half of the warp
        // second step:for the first warp the mask is 7FFF meaning that each thread reduces the value from 8 threads to the right half of the warp
        // and so on
        for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
        }
        // for each warp, store reduced max value in the shared memory:
        if (thread_id % kThreadsInWarp == 0) {
            reduction[warp_id] = max_val; // shared memory size is equal to the warp size, so each warp loads its reduced value
        }
        __syncthreads();

        // for the first warp, reduce maximums obtained from the warps using warp-level reduction:
        if (warp_id == 0) {
            float max_val_to_reduce_from_warps = reduction[thread_id];
#pragma unroll
            for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
                max_val_to_reduce_from_warps = fmaxf(max_val_to_reduce_from_warps, __shfl_xor_sync(0xffffffff, max_val_to_reduce_from_warps, mask));
            }

            if (thread_id == 0) {
                reduction[0] = max_val_to_reduce_from_warps;
            }
        }
        __syncthreads();
        max_val = reduction[0];
        // >>>>>>>>>>>>>>>>>>>>>>>>>>
        // MAX REDUCE END
        // >>>>>>>>>>>>>>>>>>>>>>>>>>

        // >>>>>>>>>>>>>>>>>>>>>>>>>>
        // SUM-EXP REDUCE START
        // >>>>>>>>>>>>>>>>>>>>>>>>>>
        float divisor = 0.f;
        f4_input_ptr = reinterpret_cast<const float4*>(d_input_matrix + i * input_stride + thread_id * 4);


        // reduce to the size of thread block
#pragma unroll URF
        for (int s = thread_id; s < cols / 4; s += blockDim.y) {
            float4 val = *f4_input_ptr;
            divisor += __expf(val.x - max_val);
            divisor += __expf(val.y - max_val);
            divisor += __expf(val.z - max_val);
            divisor += __expf(val.w - max_val);
            f4_input_ptr += blockDim.y;
        }

        // reduce in each warp separately
#pragma unroll
        for (unsigned mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
            divisor += __shfl_xor_sync(0xffffffff, divisor, mask);
        }

        // save warp-reduced value in the sahred memory
        if (thread_id % kThreadsInWarp == 0) {
            reduction[warp_id] = divisor;
        }


        // first warp reduces results of other warps
        __syncthreads();
        if (warp_id == 0) {
            divisor = reduction[thread_id];

#pragma unroll
            for (unsigned int mask = kThreadsInWarp / 2; mask > 0; mask >>= 1) {
                divisor += __shfl_xor_sync(0xffffffff, divisor, mask);
            }
            if (thread_id == 0) {
                reduction[0] = divisor;
            }
        }

        __syncthreads();
        float exponents_sum = 1.0f / reduction[0];
        // >>>>>>>>>>>>>>>>>>>>>>>>>>
        // SUM-EXP REDUCE END
        // >>>>>>>>>>>>>>>>>>>>>>>>>>

        // COMPUTE SOFTMAX
        f4_input_ptr = reinterpret_cast<const float4*>(d_input_matrix + i * input_stride + thread_id * 4);
        float4* f4_out_ptr = reinterpret_cast<float4*>(d_out + i * out_stride + thread_id * 4); // initialize resulting array pointer (points to a specific location - row i and 4 cols starting from thread_id)
#pragma unroll URF
        for (int s = thread_id; s < cols / 4; s += blockDim.y) {
            float4 val = *f4_input_ptr;
            val.x = __expf(val.x - max_val) * exponents_sum;
            val.y = __expf(val.y - max_val) * exponents_sum;
            val.z = __expf(val.z - max_val) * exponents_sum;
            val.w = __expf(val.w - max_val) * exponents_sum;
            *f4_out_ptr = val; // write aligned values to the poiter location

            // move input and p=output pointers
            f4_input_ptr += blockDim.y;
            f4_out_ptr += blockDim.y;
        }
    }
}

void Softmax(size_t rows, size_t cols, const float* d_input_matrix, size_t input_stride,
             float* d_out, size_t out_stride, cudaStream_t stream) {
    dim3 nThreads(1, kMaxThreadsInBlock, 1);
    dim3 nBlocks(rows, 1, 1);
    SoftmaxKernel<<<nBlocks, nThreads, kMaxThreadsInBlock / kThreadsInWarp * sizeof(float), stream>>>(
        rows, cols, d_input_matrix, input_stride, d_out, out_stride);
}
