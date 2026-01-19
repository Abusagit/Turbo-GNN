#ifndef computeUtil_H
#define computeUtil_H
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

#define WARP_SIZE 32

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF
// The max number of threads per block
#define CUDA_MAX_NUM_THREADS 256

constexpr unsigned int full_mask = 0xffffffff;

#define CEIL(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)
#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))

#define checkCudaError(a)                                                      \
  do {                                                                         \
    if (cudaSuccess != (a)) {                                                  \
      fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n",                                                                  \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkCuSparseError(a)                                                  \
  do {                                                                         \
    if (CUSPARSE_STATUS_SUCCESS != (a)) {                                      \
      fprintf(stderr, "CuSparse runTime error in line %d of file %s \
    : %s \n",                                                                  \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() { return 0; }

__device__ __forceinline__ int findRow(const int *S_csrRowPtr, int eid,
                                       int start, int end) {
  int low = start, high = end;
  if (low == high)
    return low;
  while (low < high) {
    int mid = (low + high) >> 1;
    if (S_csrRowPtr[mid] <= eid)
      low = mid + 1;
    else
      high = mid;
  }
  if (S_csrRowPtr[high] == eid)
    return high;
  else
    return high - 1;
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(ldType &tmp, data *array, int offset) {
  tmp = *(reinterpret_cast<ldType *>(array + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(data *lhd, data *rhd, int offset) {
  *(reinterpret_cast<ldType *>(lhd)) =
      *(reinterpret_cast<ldType *>(rhd + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Store(data *lhd, data *rhd, int offset) {
  *(reinterpret_cast<ldType *>(lhd + offset)) =
      *(reinterpret_cast<ldType *>(rhd));
}

__device__ inline void st_global(void *addr, const uint8_t (&x)[16]) {
  using store_t = uint32_t;
  const store_t *x_store_arr = reinterpret_cast<const store_t *>(x);
  auto *s1 = reinterpret_cast<void *>(addr);
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s1), "r"(x_store_arr[0]), "r"(x_store_arr[1]),
                 "r"(x_store_arr[2]), "r"(x_store_arr[3]));
}

template <typename dataType, int N>
__device__ inline void stg(dataType *addr, const dataType (&x)[N]) {
  using arr_t = const uint8_t(&)[sizeof(dataType) * N];
  st_global(static_cast<void *>(addr), reinterpret_cast<arr_t>(x));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load4(ldType *tmp, data *array, int *offset,
                                      int offset2 = 0) {
  Load(tmp[0], array, offset[0] + offset2);
  Load(tmp[1], array, offset[1] + offset2);
  Load(tmp[2], array, offset[2] + offset2);
  Load(tmp[3], array, offset[3] + offset2);
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot2(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y;
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot4(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y + lhd.z * rhd.z + lhd.w * rhd.w;
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec4Dot4(data *cal, vecData *lhd,
                                         vecData *rhd) {
  cal[0] += vecDot4<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot4<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot4<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot4<vecData, data>(lhd[3], rhd[3]);
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec2Dot4(data *cal, vecData *lhd,
                                         vecData *rhd) {
  cal[0] += vecDot2<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot2<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot2<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot2<vecData, data>(lhd[3], rhd[3]);
}

template <typename data>
__device__ __forceinline__ void Dot4(data *cal, data *lhd, data *rhd) {
  cal[0] += lhd[0] * rhd[0];
  cal[1] += lhd[1] * rhd[1];
  cal[2] += lhd[2] * rhd[2];
  cal[3] += lhd[3] * rhd[3];
}

template <typename data>
__device__ __forceinline__ void Mul4_const(data *cal, const data *lhd,
                                           data Const) {
  cal[0] += lhd[0] * Const;
  cal[1] += lhd[1] * Const;
  cal[2] += lhd[2] * Const;
  cal[3] += lhd[3] * Const;
}

template <typename data>
__device__ __forceinline__ void selfMul4(data *lhd, data *rhd) {
  lhd[0] *= rhd[0];
  lhd[1] *= rhd[1];
  lhd[2] *= rhd[2];
  lhd[3] *= rhd[3];
}

template <typename data>
__device__ __forceinline__ void selfMulConst4(data *lhd, data Const) {
  lhd[0] = lhd[0] * Const;
  lhd[1] = lhd[1] * Const;
  lhd[2] = lhd[2] * Const;
  lhd[3] = lhd[3] * Const;
}

template <typename data>
__device__ __forceinline__ void selfAddConst4(data *lhd, data Const) {
  lhd[0] += Const;
  lhd[1] += Const;
  lhd[2] += Const;
  lhd[3] += Const;
}

template <typename data>
__device__ __forceinline__ void AllReduce4(data *multi, int stride,
                                           int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi[0] += __shfl_xor_sync(0xffffffff, multi[0], stride, warpSize);
    multi[1] += __shfl_xor_sync(0xffffffff, multi[1], stride, warpSize);
    multi[2] += __shfl_xor_sync(0xffffffff, multi[2], stride, warpSize);
    multi[3] += __shfl_xor_sync(0xffffffff, multi[3], stride, warpSize);
  }
}

template <typename data>
__device__ __forceinline__ void AllReduce(data multi, int stride,
                                          int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi += __shfl_xor_sync(0xffffffff, multi, stride, warpSize);
  }
}

inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  // CHECK_GE(dim, 0);
  if (dim == 0)
    return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

/**
 * @brief Find number of blocks is smaller than nblks and max_nblks
 * on the given axis ('x', 'y' or 'z').
 */
template <char axis> inline int FindNumBlocks(int nblks, int max_nblks = -1) {
  int default_max_nblks = -1;
  switch (axis) {
  case 'x':
    default_max_nblks = CUDA_MAX_NUM_BLOCKS_X;
    break;
  case 'y':
    default_max_nblks = CUDA_MAX_NUM_BLOCKS_Y;
    break;
  case 'z':
    default_max_nblks = CUDA_MAX_NUM_BLOCKS_Z;
    break;
  }
  if (max_nblks == -1)
    max_nblks = default_max_nblks;
  // CHECK_NE(nblks, 0);
  if (nblks < max_nblks)
    return nblks;
  return max_nblks;
}

inline bool isMul32(int x) { return (x > 0 && x % 32 == 0); }

#define roundup(x, y)                                                          \
  ({                                                                           \
    typeof(y) __y = y;                                                         \
    (((x) + (__y - 1)) / __y) * __y;                                           \
  })

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA: " << cudaGetErrorString(e);                                  \
  }

#define CUSPARSE_CALL(func)                                                    \
  {                                                                            \
    cusparseStatus_t e = (func);                                               \
    CHECK(e == CUSPARSE_STATUS_SUCCESS) << "CUSPARSE ERROR: " << e;            \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, ...)                     \
  {                                                                            \
    {                                                                          \
      (kernel)<<<(nblks), (nthrs), (shmem)>>>(__VA_ARGS__);                    \
      cudaError_t e = cudaGetLastError();                                      \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                 \
          << "CUDA kernel launch error: " << cudaGetErrorString(e);            \
    }                                                                          \
  }

__device__ __forceinline__ float warpReduceSum(float sum, int blockSize) {
  if (blockSize > 16)
    sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (blockSize > 8)
    sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (blockSize > 4)
    sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (blockSize > 2)
    sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (blockSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

__device__ __forceinline__ float warpReduceMax(float val) {
  const unsigned int FULL_MASK = 0xffffffff;

  for (int mask = warpSize / 2; mask > 0; mask /= 2) {
    val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
  }

  return val;
}

__device__ __forceinline__ void namedBarrierSync(int name, int numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

#endif // computeUtil_H
