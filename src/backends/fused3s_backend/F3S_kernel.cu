#include "config.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>
#include "ptx.h"

using namespace nvcuda;

typedef uint64_t clocktype;
struct Dur {
  clocktype begin;
  clocktype end;
  int smid = -1;
  Dur(clocktype x, clocktype y, int outsm) {
    begin = x;
    end = y;
    smid = outsm;
  }
};

bool cmp(Dur x, Dur y) { return (x.end > y.end); }

// gimped version of globalTimer64 from DTC-Spmm because it's adding too much overhead
static __device__ inline uint64_t globalTimer64(void) {
  volatile uint64_t first_reading;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  return first_reading;
}

__device__ inline uint getSMId() {
  uint smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

// timer is a 3 * nRowWindow array in GPU memory
void printSMTime(clocktype* timer, int nRowWindow, std::string filename){
  clocktype* cpu_timer = new clocktype[3 * nRowWindow];
  memset(cpu_timer, 0, 3 * nRowWindow * sizeof(clocktype));
  cudaMemcpy(cpu_timer, timer, sizeof(clocktype) * 3 * nRowWindow, cudaMemcpyDeviceToHost);
  std::vector<Dur> v;
  for(int j = 0; j < nRowWindow; ++j) {
    v.push_back(Dur(cpu_timer[j * 3], cpu_timer[j * 3 + 1], (int)(cpu_timer[j * 3 + 2])));
  }
  clocktype* start_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		start_time_sm[i] = LONG_LONG_MAX;
	}
	clocktype* end_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		end_time_sm[i] = 0;
	}
	for(auto item : v) {
    if (item.begin < start_time_sm[item.smid]) {
		  start_time_sm[item.smid] = item.begin;
	  }
	  if (item.end > end_time_sm[item.smid]) {
		  end_time_sm[item.smid] = item.end;
	  }
	}
	std::ofstream out(filename);
	if (out.is_open()) {
    out << std::fixed << std::setprecision(15);
	  for (int i = 0; i < NUM_SM_GPU; i++) {
      out << (double)(start_time_sm[i])/1e6 << " " << (double)(end_time_sm[i])/1e6 << std::endl;
	  }
  }
	out.close();
}

union Half2Uint32 {
    half2 h2;
    uint32_t u32;
};

struct Scheduler {
  int ind = blockIdx.x;
  int targetRw;

  __device__ bool
  next_iter(const int* sortedRowWindows, int nRw){
    if(ind < nRw){
      targetRw = sortedRowWindows[ind];
      ind += gridDim.x;
      return true;
    }
    else{
      return false;
    }
  }
};

__device__ void saveSddmmResult(float* sum, float* sddmm_result, int tcbId, int last_block){
  int warpId = threadIdx.y;    // warp_index handling multi-dimension > 16.
  int laneId = threadIdx.x; // lanid of each warp.
  // have warp 0 load sum into sddmm_result, which is block row major
  if(warpId == 0){
    int offset = tcbId * BLK_M * BLK_N;
    for(int i = 0; i < 2; i++){ // 2 16x8 blocks
      if(!last_block || i == 0){
        for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
          int sum_offset = i*BLK_M*BLK_N + j*BLK_N*BLK_N + laneId*2;
          sddmm_result[offset + sum_offset] = sum[sum_offset];
          sddmm_result[offset + sum_offset + 1] = sum[sum_offset + 1];
        }
      }
    }
  }
}

// produce the max value for every subwarp of 4 threads
__device__ void reduceMax(float& max, volatile int laneId){
  // Round 1: Compare with neighbor at offset 1
  max = fmaxf(max, __shfl_xor_sync(0xF << (laneId/4)*4, max, 1, 4));
  // Round 2: Compare with neighbor at offset 2
  max = fmaxf(max, __shfl_xor_sync(0xF << (laneId/4)*4, max, 2, 4));
}

// sum up "sum" for every 4 consecutive threads in a warp
// result is only valid for the first thread in the warp
__device__ void reduceSum(float& sum){
  // offset = 1
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1, 4);
  // offset = 2
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2, 4);
}

//sum should be BLK_M*BLK_M*number of warps.
__device__ void storePartialSumShm(int i, float* sum, float* D_frag, int warpId, int laneId){
  int sumOffset = warpId*BLK_M*BLK_M + i*BLK_M*BLK_N + laneId*2;
  sum[sumOffset] = D_frag[0];
  sum[sumOffset + 1] = D_frag[1];
  sum[sumOffset + BLK_N*BLK_N] = D_frag[2];
  sum[sumOffset + BLK_N*BLK_N + 1] = D_frag[3];
}

__device__ void addPartialSums(float* sum, int tcbId, int tid, int n_warps, const uint64_t* TCblock_bit_map, bool last_block){
  for(int ind = tid; ind < BLK_M*BLK_M; ind += blockDim.x*blockDim.y){
    int block_id = ind / 64; // which 8x8 block ind belongs to
    if(!last_block || block_id < 2){
      int block_offset = ind % 64; // which element in the 8x8 block ind belongs to
      uint64_t bit_mask = 1ULL << (63 - block_offset);
      if((TCblock_bit_map[tcbId*2+block_id] & bit_mask) != 0){
        //skip first warp because it's already loaded into sum
        for(int i = 1; i < n_warps; i++){
          sum[ind] += sum[i*BLK_M*BLK_M + ind];
        }
      }
      else{
        sum[ind] = 0.0f;
      }
    }
    else{
      sum[ind] = 0.0f;
    }
  }
}

__device__ void setQFragPermute(volatile uint32_t* Q_frag, uint64_t* Q, int bid, int warpId, int laneId, int numNodes, int embeddingDim){
   // Threads of a warp for fetching a 16X16 block of Q.
  // DOC: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // Here I'm swapping columns of Q to make the memory access more coalesced.
  // So when loading K, we have to swap the rows accordingly in order to get the same result.
  int rowIdx = bid * BLK_M + laneId/4;
  // /4 because half4.
  int colIdx = warpId * BLK_K/4 + (laneId%4);
  if(rowIdx < numNodes){
    uint64_t val = Q[rowIdx * embeddingDim/4 + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
  }
  if(rowIdx + 8 < numNodes){
    uint64_t val = Q[(rowIdx+8) * embeddingDim/4 + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
  }
}

__device__ void setQFrag(volatile uint32_t* Q_frag, uint32_t* Q, int bid, int warpId, int laneId, int numNodes, int embeddingDim){
  // Threads of a warp for fetching a 16X16 block of Q.
  // DOC: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  // Here I'm swapping columns of Q to make the memory access more coalesced.
  // So when loading K, we have to swap the rows accordingly in order to get the same result.
  int64_t rowIdx = bid * BLK_M + laneId/4;
  // /4 because half4.
  int colIdx = warpId * BLK_K/2 + (laneId%4);
  if(rowIdx < numNodes){
    int64_t qIdx = rowIdx * embeddingDim/2 + colIdx;
    Q_frag[0] = Q[qIdx];
    Q_frag[2] = Q[qIdx + BLK_K/4];
  }
  if(rowIdx + 8 < numNodes){
    int64_t qIdx = (rowIdx+8) * embeddingDim/2 + colIdx;
    Q_frag[1] = Q[qIdx];
    Q_frag[3] = Q[qIdx + BLK_K/4];
  }
}

__device__ void setQFragShm(float* dynShm, uint64_t* Q, int bid, int warpId, int laneId, int numNodes, int embeddingDim){
  uint32_t* Q_frag = reinterpret_cast<uint32_t*>(dynShm + BLK_M*2 + warpId*blockDim.x*4);
  int rowIdx = bid * BLK_M + laneId/4;
  // /4 because half4.
  int colIdx = warpId * BLK_K/4 + (laneId%4);
  if(rowIdx < numNodes){
    uint64_t val = Q[rowIdx * embeddingDim/4 + colIdx];
    Q_frag[laneId] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[blockDim.x*2 + laneId] = static_cast<uint32_t>(val >> 32);
  }
  if(rowIdx + 8 < numNodes){
    uint64_t val = Q[(rowIdx+8) * embeddingDim/4 + colIdx];
    Q_frag[blockDim.x + laneId] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[blockDim.x*3 + laneId] = static_cast<uint32_t>(val >> 32);
  }
}

__global__ void f3sKernel1tb1tcb(
		const int *__restrict__ rowWindowOffset, 		 // offset of each row window.
		const int *__restrict__ sparseAToXidx,     // colid of each TC block nonzero element.
    const uint64_t *__restrict__ tcbBitMap,
    const int numNodes,
    const int embeddingDim,
		torch::Half *__restrict__ Q,
    torch::Half *__restrict__ K,
    torch::Half *__restrict__ V,
		float *output,              // output feature matrix.
		float *sddmmResult, // result of SDDMM
    bool applySoftmax
);

__global__ void f3sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    uint32_t *__restrict__ Q,
    uint32_t *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void f3sKernel1tb1rwClocked(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult,
    clocktype* timer);

__global__ void f3sKernel1tb1rwScheduled(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    uint32_t *__restrict__ Q,
    uint32_t *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void f3sKernel1tb1rwScheduledClocked(
  const int *__restrict__ rowWindowOffset,
  const int *__restrict__ sortedRowWindows,
  const int *__restrict__ sparseAToXidx,
  const uint64_t *__restrict__ tcbBitMap,
  int embeddingDim,
  int nRw,
  ulonglong2 *__restrict__ Q,
  ulonglong2 *__restrict__ K,
  half *__restrict__ V,
  float2 *output,
  float2 *sddmmResult,
  clocktype* timer);

__global__ void f3sKernel1tb1rwScheduledPermutedQKV(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void f3sKernel1tb1rwScheduledPermutedQKVScaleQK(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    float scalingFactor,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

__global__ void f2sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult);

std::vector<torch::Tensor>
f3sCuda1tb1tcb(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int numNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool applySoftmax,
    bool saveSddmmResult){
  int nBlockEmbeddingDim = (embeddingDim + BLK_N - 1) / BLK_N;
  int nWarpPerBlock =  (nBlockEmbeddingDim + 2 - 1) / 2;
  const int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  torch::Tensor sddmmResult;
  if(saveSddmmResult){
    uint64_t nTcb = sparseAToXidx.size(0)/BLK_N;
    int64_t sddmmResultSize = nTcb*BLK_M*BLK_N;
    sddmmResult = torch::zeros({sddmmResultSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  }
  else{
    sddmmResult = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  }
  float* sddmmResultPtr = sddmmResult.data_ptr<float>();
  int fixedSharedSize = nWarpPerBlock * BLK_M * BLK_M * sizeof(float);
  int dynamicSharedSize = applySoftmax ? fixedSharedSize + 2 * BLK_M * sizeof(float) : fixedSharedSize;
  torch::Tensor time;
  #if BLK_M == 16 && BLK_N == 8 && BLK_K == 16
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  f3sKernel1tb1tcb<<<grid, block, dynamicSharedSize>>>(
    rowWindowOffset.data_ptr<int>(),
    sparseAToXidx.data_ptr<int>(),
    tcbBitMap.data_ptr<uint64_t>(),
    numNodes, embeddingDim,
    Q.data_ptr<torch::Half>(),
    K.data_ptr<torch::Half>(),
    V.data_ptr<torch::Half>(),
    output.data_ptr<float>(),
    sddmmResultPtr,
    applySoftmax);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  time = torch::tensor(milliseconds, torch::TensorOptions().dtype(torch::kFloat32));
  #else
  printf("only m16n8k16 is supported\n");
  #endif

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error in f3sCuda1tb1tcb: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, numNodes), torch::indexing::Slice()});
  return {time, output, sddmmResult};
}

std::vector<torch::Tensor>
f3sCuda1tb1rw(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock,
    bool applySoftmax){
  uint64_t nTcb = sparseAToXidx.size(0)/BLK_M;
  // int64_t sddmmResultSize = nTcb*BLK_M*BLK_M;
	// torch::Tensor sddmmResult = torch::zeros({sddmmResultSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor sddmmResult = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  if(applySoftmax){
    f3sKernel1tb1rw<<<grid, block, sharedSize>>>(
      rowWindowOffset.data_ptr<int>(),
      sparseAToXidx.data_ptr<int>(),
      tcbBitMap.data_ptr<uint64_t>(),
      embeddingDim,
      reinterpret_cast<uint32_t*>(Q.data_ptr<torch::Half>()),
      reinterpret_cast<uint32_t*>(K.data_ptr<torch::Half>()),
      reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
      reinterpret_cast<float2*>(output.data_ptr<float>()),
      reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  }
  else{
    f2sKernel1tb1rw<<<grid, block, sharedSize>>>(
      rowWindowOffset.data_ptr<int>(),
      sparseAToXidx.data_ptr<int>(),
      tcbBitMap.data_ptr<uint64_t>(),
      embeddingDim,
      reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()),
      reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()),
      reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
      reinterpret_cast<float2*>(output.data_ptr<float>()),
      reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()));
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  torch::Tensor time = torch::tensor(milliseconds, torch::TensorOptions().dtype(torch::kFloat32));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, nNodes), torch::indexing::Slice()});
  return {time, output, sddmmResult};
}

std::vector<torch::Tensor>
f3sCuda1tb1rwClocked(
    torch::Tensor rowWindowOffset,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock){
  uint64_t nTcb = sparseAToXidx.size(0)/BLK_M;
  // int64_t sddmmResultSize = nTcb*BLK_M*BLK_M;
	// torch::Tensor sddmmResult = torch::zeros({sddmmResultSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor sddmmResult = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  clocktype* timer;
	cudaMalloc((void**)&timer, sizeof(clocktype) * 3 * nRowWindow);
	cudaMemset(timer, 0, 3 * nRowWindow * sizeof(clocktype));
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  f3sKernel1tb1rwClocked<<<grid, block, sharedSize>>>(
    rowWindowOffset.data_ptr<int>(),
    sparseAToXidx.data_ptr<int>(),
    tcbBitMap.data_ptr<uint64_t>(),
    embeddingDim,
    reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()),
    reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()),
    reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
    reinterpret_cast<float2*>(output.data_ptr<float>()),
    reinterpret_cast<float2*>(sddmmResult.data_ptr<float>()),
    timer);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  torch::Tensor time = torch::tensor(milliseconds, torch::TensorOptions().dtype(torch::kFloat32));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, nNodes), torch::indexing::Slice()});
  // print the time of each SM
  printSMTime(timer, nRowWindow, "SM_time_f3s1tb1rw.csv");
  return {time, output, sddmmResult};
}

std::vector<torch::Tensor>
f3sCuda1tb1rwScheduledClocked(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock){
  uint64_t nTcb = sparseAToXidx.size(0)/BLK_M;
  // int64_t sddmmResultSize = nTcb*BLK_M*BLK_M;
	// torch::Tensor sddmmResult = torch::zeros({sddmmResultSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor sddmmResult = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  float2* sddmmResultPtr = reinterpret_cast<float2*>(sddmmResult.data_ptr<float>());
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  clocktype* timer;
	cudaMalloc((void**)&timer, sizeof(clocktype) * 3 * nRowWindow);
	cudaMemset(timer, 0, 3 * nRowWindow * sizeof(clocktype));
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  // create cuda event
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  f3sKernel1tb1rwScheduledClocked<<<grid, block, sharedSize>>>(
    rowWindowOffset.data_ptr<int>(),
    sortedRowWindows.data_ptr<int>(),
    sparseAToXidx.data_ptr<int>(),
    tcbBitMap.data_ptr<uint64_t>(),
    embeddingDim,
    nRowWindow,
    reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()),
    reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()),
    reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
    reinterpret_cast<float2*>(output.data_ptr<float>()),
    sddmmResultPtr,
    timer);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  torch::Tensor time = torch::tensor(milliseconds, torch::TensorOptions().dtype(torch::kFloat32));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, nNodes), torch::indexing::Slice()});
  // print the time of each SM
  printSMTime(timer, nRowWindow, "SM_time_f3s1tb1rw_scheduled.csv");
  return {time, output, sddmmResult};
}

std::vector<torch::Tensor>
f3sCuda1tb1rwScheduled(
    torch::Tensor rowWindowOffset,
    torch::Tensor sortedRowWindows,
    torch::Tensor sparseAToXidx,
    torch::Tensor tcbBitMap,
    int nNodes,
    int embeddingDim,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int nWarpPerBlock,
    bool permuteV,
    float scalingFactor){
  uint64_t nTcb = sparseAToXidx.size(0)/BLK_M;
  // int64_t sddmmResultSize = nTcb*BLK_M*BLK_M;
	// torch::Tensor sddmmResult = torch::zeros({sddmmResultSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor sddmmResult = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  float2* sddmmResultPtr = reinterpret_cast<float2*>(sddmmResult.data_ptr<float>());
  int nRowWindow = rowWindowOffset.size(0) - 1;
  int paddedLength = nRowWindow * BLK_M;
  auto output = torch::zeros({paddedLength, embeddingDim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  int sharedSize = BLK_M * embeddingDim * sizeof(half); // Q
  sharedSize += nWarpPerBlock * BLK_M * BLK_N * sizeof(half); // E
  sharedSize += nWarpPerBlock * 2 * BLK_M * sizeof(float); // row_max, row_sum, old_max, old_sum
  sharedSize += BLK_M * embeddingDim * sizeof(float); // O_frag
  dim3 grid(nRowWindow, 1, 1);
  dim3 block(WARP_SIZE, nWarpPerBlock, 1);
  // create cuda event
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  if(permuteV){
    if(scalingFactor != 1.0){
      f3sKernel1tb1rwScheduledPermutedQKVScaleQK<<<grid, block, sharedSize>>>(
        rowWindowOffset.data_ptr<int>(),
        sortedRowWindows.data_ptr<int>(),
        sparseAToXidx.data_ptr<int>(),
        tcbBitMap.data_ptr<uint64_t>(),
        embeddingDim,
        nRowWindow,
        scalingFactor,
        reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()),
        reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
        reinterpret_cast<float2*>(output.data_ptr<float>()),
        sddmmResultPtr);
    }
    else{
      f3sKernel1tb1rwScheduledPermutedQKV<<<grid, block, sharedSize>>>(
        rowWindowOffset.data_ptr<int>(),
        sortedRowWindows.data_ptr<int>(),
        sparseAToXidx.data_ptr<int>(),
        tcbBitMap.data_ptr<uint64_t>(),
        embeddingDim,
        nRowWindow,
        reinterpret_cast<ulonglong2*>(Q.data_ptr<torch::Half>()),
        reinterpret_cast<ulonglong2*>(K.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
        reinterpret_cast<float2*>(output.data_ptr<float>()),
        sddmmResultPtr);
    }
  }
  else{
    f3sKernel1tb1rwScheduled<<<grid, block, sharedSize>>>(
      rowWindowOffset.data_ptr<int>(),
      sortedRowWindows.data_ptr<int>(),
      sparseAToXidx.data_ptr<int>(),
      tcbBitMap.data_ptr<uint64_t>(),
      embeddingDim,
      nRowWindow,
      reinterpret_cast<uint32_t*>(Q.data_ptr<torch::Half>()),
      reinterpret_cast<uint32_t*>(K.data_ptr<torch::Half>()),
      reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
      reinterpret_cast<float2*>(output.data_ptr<float>()),
      sddmmResultPtr);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  torch::Tensor time = torch::tensor(milliseconds, torch::TensorOptions().dtype(torch::kFloat32));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // remove padding
  output = output.index(
      {torch::indexing::Slice(0, nNodes), torch::indexing::Slice()});
  return {time, output, sddmmResult};
}

// #if defined(BLK_M) && defined(BLK_N) && defined(BLK_K) && \
//     BLK_M == 16 && BLK_N == 8 && BLK_K == 16
#define bid blockIdx.x
#define tid (threadIdx.x + threadIdx.y * blockDim.x)
__global__ void f3sKernel1tb1tcb(
		const int *__restrict__ rowWindowOffset, 		 // offset of each row window.
		const int *__restrict__ sparseAToXidx,     // colid of each TC block nonzero element.
    const uint64_t *__restrict__ tcbBitMap,
    const int numNodes,
    const int embeddingDim,
		torch::Half *__restrict__ Q,
    torch::Half *__restrict__ K,
    torch::Half *__restrict__ V,
		float *output,
		float *sddmmResult,
    bool applySoftmax){
  volatile int warpId = threadIdx.y;    // warp_index handling multi-dimension > 16.
  volatile int laneId = threadIdx.x; // lanid of each warp.

  uint32_t* K_uint32 = reinterpret_cast<uint32_t*>(K);

  // row_max, row_sum (size BLK_M each) for online-softmax,
  // then Q_frag, then O_frag
  extern __shared__ float dynShm[];
  for(int i = tid; i < BLK_M*2+blockDim.y*BLK_M*BLK_M; i += blockDim.x*blockDim.y){
    dynShm[i] = 0.0f;
  }

  float* sum = dynShm + BLK_M*2;
  float O_frag[8] = {0};// spmm result
  uint32_t Q_frag[4] = {0};
  setQFrag(Q_frag, reinterpret_cast<uint32_t*>(Q), bid, warpId, laneId, numNodes, embeddingDim);

  /////////////////////////////////
  // main loop
  /////////////////////////////////
  volatile bool lastBlock = false;
  for (int tcbId = rowWindowOffset[bid]; tcbId < rowWindowOffset[bid + 1]; tcbId+=2) {
    if((rowWindowOffset[bid + 1] - rowWindowOffset[bid]) % 2 && tcbId == rowWindowOffset[bid + 1] - 1){
      lastBlock = true;
    }
    {// sddmm
      uint32_t B_frag[2];
      float D_frag[4];
      // int colIdx = (warpId * BLK_M)/4 + (laneId % 4);
      int colIdx = (warpId * BLK_M)/2 + (laneId % 4);
      for(int i = 0; i < 2; i++){
        if(!lastBlock || i == 0){
          // Initialize B_frag from K
          int64_t rowIdx = sparseAToXidx[(tcbId+i) * BLK_N + laneId / 4];
          int64_t kIdx = rowIdx * embeddingDim/2 + colIdx;
	  B_frag[0] = K_uint32[kIdx];
          B_frag[1] = K_uint32[kIdx + BLK_M/4];
          HMMA16816(D_frag[0], D_frag[1], D_frag[2], D_frag[3],
                    Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3],
                    B_frag[0], B_frag[1],
                    0.0f, 0.0f, 0.0f, 0.0f);
          storePartialSumShm(i, sum, D_frag, warpId, laneId);
        }
      }
      __syncthreads();
      addPartialSums(sum, tcbId, tid, blockDim.y, tcbBitMap, lastBlock);
    }
    __syncthreads();

    if(sddmmResult != nullptr){
      saveSddmmResult(sum, sddmmResult, tcbId, lastBlock);
    }

    {// softmax + spmm
      uint32_t S_frag[4];// softmax/sddmm result
      if(applySoftmax){
        float D_frag[4];
        float2* sumFloat2 = reinterpret_cast<float2*>(sum);
        for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
          int sumOffset = j*BLK_N*BLK_N/2;
          for(int i = 0; i < 2; i++){// 2 16x8 blocks
            if(!lastBlock || i == 0){
              float2 temp = sumFloat2[i*BLK_M*BLK_N/2 + sumOffset + laneId];
              D_frag[i*2] = temp.x;
              D_frag[i*2 + 1] = temp.y;
            }
            else{
              D_frag[i*2] = 0.0f;
              D_frag[i*2 + 1] = 0.0f;
            }
          }
          //max of the 4 elements in the same row across 2 16x8 blocks
          //need every warp to do this because they will need it for the next computation
          float maxOld = dynShm[j*BLK_N + laneId/4];

          float max = fmaxf(
            fmaxf(fmaxf(D_frag[0], D_frag[1]), fmaxf(D_frag[2], D_frag[3])), maxOld);
          reduceMax(max, laneId);

          for(int i = 0; i < 4; i++){
            if(D_frag[i] != 0.0f){
              D_frag[i] = __expf(D_frag[i] - max);
            }
          }

          float expMaxDiff = __expf(maxOld - max);

          if(warpId == 0){
            float sum = D_frag[0] + D_frag[1] + D_frag[2] + D_frag[3];
            reduceSum(sum);
            if(laneId % 4 == 0){
              dynShm[BLK_M + j*BLK_N + laneId/4] = dynShm[BLK_M + j*BLK_N + laneId/4] * expMaxDiff + sum;
            }
          }

          O_frag[j*2]   = O_frag[j*2]   * expMaxDiff;
          O_frag[j*2+1] = O_frag[j*2+1] * expMaxDiff;
          O_frag[j*2+4] = O_frag[j*2+4] * expMaxDiff;
          O_frag[j*2+5] = O_frag[j*2+5] * expMaxDiff;

          if(warpId == 0 && laneId % 4 == 0){
            dynShm[j*BLK_N + laneId/4] = max;
          }

          Half2Uint32 h2U32Converter;
          for(int i = 0; i < 2; i++){
            h2U32Converter.h2.x = __float2half(D_frag[i*2]);
            h2U32Converter.h2.y = __float2half(D_frag[i*2+1]);
            S_frag[i*2 + j] = h2U32Converter.u32;
          }
        }
      }
      else{
        float2* sumFloat2 = reinterpret_cast<float2*>(sum);
        for(int i = 0; i < 2; i++){// 2 16x8 blocks
          int sumOffset = i*BLK_M*BLK_N/2;
          Half2Uint32 h2U32Converter;
          if(!lastBlock || i == 0){
            for(int j = 0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
              float2 temp = sumFloat2[sumOffset + j*BLK_N*BLK_N/2 + laneId];
              h2U32Converter.h2.x = __float2half(temp.x);
              h2U32Converter.h2.y = __float2half(temp.y);
              S_frag[i*2+j] = h2U32Converter.u32;
            }
          }
          else{
            S_frag[i*2] = 0;
            S_frag[i*2+1] = 0;
          }
        }
      }
      __syncthreads();
      for(int i = laneId; i < BLK_M*BLK_M; i += blockDim.x){
        sum[warpId*BLK_M*BLK_M + i] = 0.0f;
      }
      /////////
      // SpMM
      /////////
      {
        uint32_t B_frag[2];
        Half2Uint32 h2U32Converter;
        half temp_V[2];
        // float* O_frag = dynShm + BLK_M*2 + blockDim.x*blockDim.y*4 + warpId*blockDim.x*8;
        for(int j = 0; j < 2; j++){// 2 16x8 blocks
          int colIdx = (warpId*2+j) * BLK_N + laneId/4;
          for(int i = 0; i < 2; i++){// 2 8x8 blocks in each 16x8 block
            if(!lastBlock || i == 0){
              for(int k = 0; k < 2; k++){// 2 halfs in each 8x8 block
                int64_t rowIdx = sparseAToXidx[(tcbId+i) * BLK_N + (laneId%4)*2 + k];
                int64_t vIdx = rowIdx * embeddingDim + colIdx;
		temp_V[k] = V[vIdx];
              }
              h2U32Converter.h2 = __halves2half2(temp_V[0], temp_V[1]);
              B_frag[i] = h2U32Converter.u32;
            }
            else{
              B_frag[i] = 0;
            }
          }
          HMMA16816(O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3],
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[4*j], O_frag[4*j+1], O_frag[4*j+2], O_frag[4*j+3]);
        }
      }
    }
  }
  if(applySoftmax){
    for(int i = 0; i < 2; i++){
      float rowSum = dynShm[BLK_M + laneId/4 + i*BLK_N ];
      if(rowSum != 0.0f){
        O_frag[i*2] = O_frag[i*2] * (1.0f/rowSum);
        O_frag[i*2+1] = O_frag[i*2+1] * (1.0f/rowSum);
        O_frag[i*2+4] = O_frag[i*2+4] * (1.0f/rowSum);
        O_frag[i*2+5] = O_frag[i*2+5] * (1.0f/rowSum);

      }
    }
  }
  for(int j=0; j < 2; j++){// 2 8x8 blocks in each 16x8 block
    int64_t rowIdx = bid * BLK_M + (laneId / 4) + j * BLK_M/2;
    for(int i =0; i < 2; i++){// 2 16x8 blocks
      int colIdx = (warpId * 2 + i) * BLK_N + (laneId % 4) * 2;
      int64_t outIdx = rowIdx * embeddingDim + colIdx;
      output[outIdx] = O_frag[i*4 + j*2];
      output[outIdx + 1] = O_frag[i*4 + j*2 + 1];
    }
  }
}

// load Q from HBM to register. Permute columns
__device__ void loadQFragPermuteCol(volatile uint32_t *Q_frag, uint64_t *Q, int embeddingDim, int rowIdx, int colIdx) {
    uint64_t val = Q[rowIdx * embeddingDim + colIdx];
    Q_frag[0] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[2] = static_cast<uint32_t>(val >> 32);
    val = Q[(rowIdx+8) * embeddingDim + colIdx];
    Q_frag[1] = static_cast<uint32_t>(val & 0xFFFFFFFFull);
    Q_frag[3] = static_cast<uint32_t>(val >> 32);
}

// ind is in 128b interval
// Assume Q is stored in row-major order. We divide Q into N 16x32 element block.
__device__ void loadQHbm2Shm128b(uint64_t* qShm, ulonglong2* Q, int embeddingDim, int ind){
  int qWidth = embeddingDim/8;
  ulonglong2 val = Q[ind];
  ulonglong2 val2 = Q[ind + BLK_M/2 * qWidth];
  qWidth *= 2; //convert to 64b width
  ind *= 2; //convert to 64b index
  int rid = ind / qWidth;
  int cid = ind % qWidth;
  // 8 because 32*sizeof(half)/sizeof(uint64_t) = 8
  int blockCid = cid / 8;
  int cidInBlock = cid % 8;
  int blockOffset = blockCid*BLK_M*8 + rid*8 + cidInBlock;
  qShm[blockOffset] = val.x;
  qShm[blockOffset + 1] = val2.x;
  qShm[blockOffset + 64] = val.y;
  qShm[blockOffset + 65] = val2.y;
}

// Pair with loadQHbm2Shm128b.
// This function has each warp read a 16x4 block of uint64_t
//Not following the register layout in ptx doc but reordering it to match how K is loaded.
__device__ void loadQFragShm(uint64_t* Q_frag, uint64_t* dynShm, int ind, int laneId) {
  // 4 because BLK_M*sizeof(half)/sizeof(uint64_t) = 4
  int offset = ind*BLK_M*4 + laneId*2;
  Q_frag[0] = dynShm[offset];
  Q_frag[1] = dynShm[offset + 1];
}

// Q should already have the TB offset
// Q is in row major
// qShm is in block column major, inside block it's row major. Block size is 8x4.
__device__ void loadQHbm2Shm32b(uint32_t* qShm, uint32_t* Q, int embeddingDim, int ind){
  // int blockWidth = BLK_M/4;
  // int blockHeight = BLK_M/2;
  int qWidth = embeddingDim/2;
  uint32_t val = Q[ind];
  int rowInd = ind/qWidth;
  int colInd = ind%qWidth;
  int blockRowInd = rowInd/(BLK_M/2);
  int blockColInd = colInd/(BLK_M/4);
  int blockInd = blockColInd * 2 + blockRowInd;
  int rowIndInBlock = rowInd%(BLK_M/2);
  int colIndInBlock = colInd%(BLK_M/4);
  int indInBlock = rowIndInBlock*(BLK_M/4) + colIndInBlock;
  qShm[blockInd*((BLK_M/4)*(BLK_M/2)) + indInBlock] = val;
}

__device__ void storeEFragShm(float* E_frag, uint32_t* dynShm) {
  Half2Uint32 h2U32Converter;
  for(int i = 0; i < 2; i++){
    h2U32Converter.h2.x = __float2half(E_frag[i*2]);
    h2U32Converter.h2.y = __float2half(E_frag[i*2+1]);
    dynShm[i*32] = h2U32Converter.u32;
  }
}
__device__ void loadEFragShm(uint32_t* E_frag, uint32_t* dynShm) {
  E_frag[0] = dynShm[0];
  E_frag[1] = dynShm[32];
  E_frag[2] = dynShm[64];
  E_frag[3] = dynShm[96];
}

__device__ void loadEFragShm2(volatile uint32_t* E_frag, uint32_t* dynShm) {
  E_frag[0] = dynShm[0];
  E_frag[1] = dynShm[32];
}

__device__ void loadOFragShm(float* O_frag, float* dynShm, float* mTilde) {
  O_frag[0] = dynShm[0] * mTilde[0];
  O_frag[1] = dynShm[32] * mTilde[0];
  O_frag[2] = dynShm[64] * mTilde[BLK_M/2];
  O_frag[3] = dynShm[96] * mTilde[BLK_M/2];
}

__device__ void storeOFragShm(volatile float* O_frag, float* dynShm) {
  dynShm[0] = O_frag[0];
  dynShm[32] = O_frag[1];
  dynShm[64] = O_frag[2];
  dynShm[96] = O_frag[3];
}

__device__ void loadOFragShm2(float* O_frag, float* dynShm, float* mTilde) {
  O_frag[0] = dynShm[0] * mTilde[0];
  O_frag[1] = dynShm[32] * mTilde[0];
  O_frag[2] = dynShm[64] * mTilde[BLK_M/2];
  O_frag[3] = dynShm[96] * mTilde[BLK_M/2];
  O_frag[4] = dynShm[128] * mTilde[0];
  O_frag[5] = dynShm[160] * mTilde[0];
  O_frag[6] = dynShm[192] * mTilde[BLK_M/2];
  O_frag[7] = dynShm[224] * mTilde[BLK_M/2];
}

__device__ void storeOFragShm2(volatile float* O_frag, float* dynShm) {
  for(int i = 0; i < 8; i++){
    dynShm[i*32] = O_frag[i];
  }
}

__global__ void f3sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    uint32_t *__restrict__ Q,
    uint32_t *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
    //each thread loads a 32bit elements
    int64_t qOffset = (int64_t)bid*BLK_M*embeddingDim/2;
    for(int i = tid; i < BLK_M*embeddingDim/2; i += blockDim.x*blockDim.y){
      loadQHbm2Shm32b(reinterpret_cast<uint32_t*>(dynShm1tb1rw), Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();
  int niter = ((rowWindowOffset[bid+1] - rowWindowOffset[bid])*2 + blockDim.y - 1)/blockDim.y;
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[bid] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[bid+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] * embeddingDim/2 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i++) {
          uint32_t b_1 = K[kOffset + i*BLK_K/2];
          uint32_t b_2 = K[kOffset + i*BLK_K/2 + BLK_K/4];
          uint32_t Q_frag[4];
          loadEFragShm(Q_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+i*BLK_M*BLK_M/2+laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3],
                    b_1, b_2,
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      // {//save sddmm result
      //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
      //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      //     int sumOffset = j*BLK_N*BLK_N;
      //     float2 val;
      //     val.x = S_frag[j*2];
      //     val.y = S_frag[j*2+1];
      //     sddmmResult[(offset + sumOffset)/2] = val;
      //   }
      // }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            Half2Uint32 h2U32Converter;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.x = V[offset];
            offset = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.y = V[offset];
            B_frag[k] = h2U32Converter.u32;
          }
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : 1.0f/sumOld[laneId/4];
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : 1.0f/sumOld[BLK_M/2 + laneId/4];
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int64_t outputOffset = (int64_t)(bid*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[32] * invR0;
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[96] * invR1;
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
}

// this permutes QK and not V
__global__ void f3sKernel1tb1rwClocked(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult,
    clocktype* timer) {
  clocktype begin = globalTimer64();
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
    //BLK_M/2 because each thread loads 2 128b elements
    int64_t qOffset = (int64_t)bid*BLK_M*embeddingDim/8;
    for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
      loadQHbm2Shm128b(dynShm1tb1rw, Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();

  int niter = ((rowWindowOffset[bid+1] - rowWindowOffset[bid])*2 + blockDim.y - 1)/blockDim.y;
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[bid] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[bid+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.x >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.y >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            Half2Uint32 h2U32Converter;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.x = V[offset];
            offset = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim+ i*BLK_N + laneId/4;
            h2U32Converter.h2.y = V[offset];
            B_frag[k] = h2U32Converter.u32;
          }
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : 1.0f/sumOld[laneId/4];
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : 1.0f/sumOld[BLK_M/2 + laneId/4];
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int64_t outputOffset = ((int64_t)bid*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[32] * invR0;
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[96] * invR1;
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
  if(threadIdx.x == 0){
    timer[3 * blockIdx.x] = begin;
    timer[3 * blockIdx.x + 1] = globalTimer64();
    timer[3 * blockIdx.x + 2] = (clocktype)(getSMId());
  }
}

// Permuted columns of V
__global__ void f3sKernel1tb1rwScheduledPermutedQKV(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  Scheduler scheduler;
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];

  scheduler.next_iter(sortedRowWindows, nRw);
  int niter = ((rowWindowOffset[scheduler.targetRw+1] - rowWindowOffset[scheduler.targetRw])*2
                + blockDim.y - 1)/blockDim.y;
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
    //BLK_M/2 because each thread loads 2 128b elements
    int64_t qOffset = (int64_t)scheduler.targetRw*BLK_M*embeddingDim/8;
    for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
      loadQHbm2Shm128b(dynShm1tb1rw, Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[scheduler.targetRw] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[scheduler.targetRw+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4]
                      * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.x >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.y >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      // {//save sddmm result
      //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
      //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      //     int sumOffset = j*BLK_N*BLK_N;
      //     float2 val;
      //     val.x = S_frag[j*2];
      //     val.y = S_frag[j*2+1];
      //     sddmmResult[(offset + sumOffset)/2] = val;
      //   }
      // }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      uint32_t* V_uint32 = reinterpret_cast<uint32_t*>(V);
      for(int i=warpId; i<embeddingDim/BLK_M; i+=blockDim.y){
        int vRowOffset = i*BLK_M + (laneId/4)*2; //*2 because each thread loads 2 half now
        float* warpOPtr = reinterpret_cast<float*>(dynShm1tb1rw)+oOffset_base + i*BLK_M*BLK_M;
        float O_frag[8];
        loadOFragShm2(O_frag, warpOPtr, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[4];
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            uint32_t temp[2];
            sparseAToXidxOffset += k*BLK_N;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim + vRowOffset;
            int64_t offset2 = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim + vRowOffset;
            temp[0] = V_uint32[offset/2];
            temp[1] = V_uint32[offset2/2];
            // combine lower 16 bits of temp[0] and temp[1]
            B_frag[k] = (temp[0] & 0xFFFF) | ((temp[1] & 0xFFFF) << 16);
            B_frag[k+2] = (temp[0] >> 16) | ((temp[1] >> 16) << 16);
          }
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
          HMMA16816(O_frag[4], O_frag[5], O_frag[6], O_frag[7],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[2], B_frag[3],
                    O_frag[4], O_frag[5], O_frag[6], O_frag[7]);
        }
        storeOFragShm2(O_frag, warpOPtr);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[laneId/4]);
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[BLK_M/2 + laneId/4]);
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 4 to get the index of the float4
  int64_t outputOffset = ((int64_t)scheduler.targetRw*BLK_M + laneId/4)*embeddingDim + (laneId%4)*4;
  float4* output_float4 = reinterpret_cast<float4*>(output);
  for(int i = warpId; i < embeddingDim/BLK_M; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_M;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float4 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[128] * invR0;
    val.z = oPtr[32] * invR0;
    val.w = oPtr[160] * invR0;
    output_float4[(outputOffset + i*BLK_M)/4] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[192] * invR1;
    val.z = oPtr[96] * invR1;
    val.w = oPtr[224] * invR1;
    output_float4[(outputOffset + i*BLK_M + BLK_M/2*embeddingDim)/4] = val;
  }
}

// Permuted columns of V
__global__ void f3sKernel1tb1rwScheduledPermutedQKVScaleQK(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    float scalingFactor,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  Scheduler scheduler;
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];

  scheduler.next_iter(sortedRowWindows, nRw);
  int niter = ((rowWindowOffset[scheduler.targetRw+1] - rowWindowOffset[scheduler.targetRw])*2
                + blockDim.y - 1)/blockDim.y;
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
    //BLK_M/2 because each thread loads 2 128b elements
    int64_t qOffset = (int64_t)scheduler.targetRw*BLK_M*embeddingDim/8;
    for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
      loadQHbm2Shm128b(dynShm1tb1rw, Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[scheduler.targetRw] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[scheduler.targetRw+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4]
                      * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.x >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.y >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ?
                      0.0f : S_frag[i]/scalingFactor;
        }
      }
      // {//save sddmm result
      //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
      //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      //     int sumOffset = j*BLK_N*BLK_N;
      //     float2 val;
      //     val.x = S_frag[j*2];
      //     val.y = S_frag[j*2+1];
      //     sddmmResult[(offset + sumOffset)/2] = val;
      //   }
      // }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      uint32_t* V_uint32 = reinterpret_cast<uint32_t*>(V);
      for(int i=warpId; i<embeddingDim/BLK_M; i+=blockDim.y){
        int vRowOffset = i*BLK_M + (laneId/4)*2; //*2 because each thread loads 2 half now
        float* warpOPtr = reinterpret_cast<float*>(dynShm1tb1rw)+oOffset_base + i*BLK_M*BLK_M;
        float O_frag[8];
        loadOFragShm2(O_frag, warpOPtr, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[4];
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            uint32_t temp[2];
            sparseAToXidxOffset += k*BLK_N;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim + vRowOffset;
            int64_t offset2 = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim + vRowOffset;
            temp[0] = V_uint32[offset/2];
            temp[1] = V_uint32[offset2/2];
            // combine lower 16 bits of temp[0] and temp[1]
            B_frag[k] = (temp[0] & 0xFFFF) | ((temp[1] & 0xFFFF) << 16);
            B_frag[k+2] = (temp[0] >> 16) | ((temp[1] >> 16) << 16);
          }
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
          HMMA16816(O_frag[4], O_frag[5], O_frag[6], O_frag[7],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[2], B_frag[3],
                    O_frag[4], O_frag[5], O_frag[6], O_frag[7]);
        }
        storeOFragShm2(O_frag, warpOPtr);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[laneId/4]);
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[BLK_M/2 + laneId/4]);
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 4 to get the index of the float4
  int64_t outputOffset = ((int64_t)scheduler.targetRw*BLK_M + laneId/4)*embeddingDim + (laneId%4)*4;
  float4* output_float4 = reinterpret_cast<float4*>(output);
  for(int i = warpId; i < embeddingDim/BLK_M; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_M;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float4 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[128] * invR0;
    val.z = oPtr[32] * invR0;
    val.w = oPtr[160] * invR0;
    output_float4[(outputOffset + i*BLK_M)/4] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[192] * invR1;
    val.z = oPtr[96] * invR1;
    val.w = oPtr[224] * invR1;
    output_float4[(outputOffset + i*BLK_M + BLK_M/2*embeddingDim)/4] = val;
  }
}

// doesn't permute columns of V
__global__ void f3sKernel1tb1rwScheduledClocked(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult,
    clocktype* timer) {
  clocktype begin = globalTimer64();
  Scheduler scheduler;
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];

  scheduler.next_iter(sortedRowWindows, nRw);
  int niter = ((rowWindowOffset[scheduler.targetRw+1] - rowWindowOffset[scheduler.targetRw])*2
                + blockDim.y - 1)/blockDim.y;
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
    //BLK_M/2 because each thread loads 2 128b elements
    int64_t qOffset = (int64_t)scheduler.targetRw*BLK_M*embeddingDim/8;
    for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
      loadQHbm2Shm128b(dynShm1tb1rw, Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[scheduler.targetRw] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[scheduler.targetRw+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4]
                      * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.x >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.y >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      uint16_t* V_uint16 = reinterpret_cast<uint16_t*>(V);
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int vRowOffset = i*BLK_N + laneId/4;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset_base + i*BLK_M*BLK_N, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim + vRowOffset;
            int64_t offset2 = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim + vRowOffset;
            B_frag[k] = ((uint32_t)V_uint16[offset2] << 16) | V_uint16[offset];
          }
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset_base + i*BLK_M*BLK_N);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[laneId/4]);
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[BLK_M/2 + laneId/4]);
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int64_t outputOffset = ((int64_t)scheduler.targetRw*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[32] * invR0;
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[96] * invR1;
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
  if(tid == 0){
    timer[3 * blockIdx.x] = begin;
    timer[3 * blockIdx.x + 1] = globalTimer64();
    timer[3 * blockIdx.x + 2] = (clocktype)(getSMId());
  }
}

__global__ void f3sKernel1tb1rwScheduled(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sortedRowWindows,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    int nRw,
    uint32_t *__restrict__ Q,
    uint32_t *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  Scheduler scheduler;
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float maxOld[BLK_M];
  // r_b in Alg 1
  __shared__ float sumOld[BLK_M];
  __shared__ float mTilde[BLK_M];

  scheduler.next_iter(sortedRowWindows, nRw);
  int niter = ((rowWindowOffset[scheduler.targetRw+1] - rowWindowOffset[scheduler.targetRw])*2
                + blockDim.y - 1)/blockDim.y;
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      maxOld[i] = 0.0f;
      sumOld[i] = 0.0f;
      mTilde[i] = 0.0f;
    }
    //each thread loads a 32bit elements
    int64_t qOffset = (int64_t)scheduler.targetRw*BLK_M*embeddingDim/2;
    for(int i = tid; i < BLK_M*embeddingDim/2; i += blockDim.x*blockDim.y){
      loadQHbm2Shm32b(reinterpret_cast<uint32_t*>(dynShm1tb1rw), Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[scheduler.targetRw] + iter*blockDim.y/2;
    // number of 16x16 blocks in S/E being computed in this iteration.
    int nBlock = min(blockDim.y/2, rowWindowOffset[scheduler.targetRw+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    if(warpId < nBlock*2){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] * embeddingDim/2 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i++) {
          uint32_t b_1 = K[kOffset + i*BLK_K/2];
          uint32_t b_2 = K[kOffset + i*BLK_K/2 + BLK_K/4];
          uint32_t Q_frag[4];
          loadEFragShm(Q_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+i*BLK_M*BLK_M/2+laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    Q_frag[0], Q_frag[1], Q_frag[2], Q_frag[3],
                    b_1, b_2,
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      // {//save sddmm result
      //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
      //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      //     int sumOffset = j*BLK_N*BLK_N;
      //     float2 val;
      //     val.x = S_frag[j*2];
      //     val.y = S_frag[j*2+1];
      //     sddmmResult[(offset + sumOffset)/2] = val;
      //   }
      // }
      {//online softmax
        float* maxPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId/4 + blockDim.y*BLK_M;
        //save max of each row within the warp to shared memory for cross-warp communication
        for(int i=0; i<2; i++){
          float localMax = fmaxf(S_frag[i*2], S_frag[i*2+1]);
          reduceMax(localMax, laneId);
          if(laneId % 4 == 0){
            maxPtr[warpId*BLK_M + i*BLK_M/2] = localMax;
          }
        }
      }
    }
    __syncthreads();
    if(warpId < nBlock*2){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2;
      for(int i=0; i<2; i++){
        int offset = i*BLK_M/2 + laneId/4;
        float* maxPtr = sumPtr + blockDim.y*BLK_M + offset;
        float newGlobalMax = maxOld[offset];
        //we have blockDim.y columns reserved for local sum of each warp
        //but only nBlock*2 are used
        for(int j=0; j<nBlock*2; j++){
          newGlobalMax = fmaxf(maxPtr[j*BLK_M], newGlobalMax);
        }
        if(warpId == 0 && laneId % 4 == 0){
          mTilde[offset] = __expf(maxOld[offset] - newGlobalMax);
          maxOld[offset] = newGlobalMax;
        }
        //compute E, ignore 0s
        S_frag[i*2] = S_frag[i*2]==0.0f ? 0.0f : __expf(S_frag[i*2] - newGlobalMax);
        S_frag[i*2+1] = S_frag[i*2+1]==0.0f ? 0.0f : __expf(S_frag[i*2+1] - newGlobalMax);
        //compute row sum and save to shared memory
        float localSum = S_frag[i*2] + S_frag[i*2+1];
        reduceSum(localSum);
        if(laneId % 4 == 0){
          sumPtr[warpId*BLK_M + offset] = localSum;
        }
      }
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    //Could try moving this block to the end of the loop
    //Without warpId == 0, the compiler gets confused and the long scoreboard stall will increase significantly.
    if(warpId == 0 && tid < BLK_M){
      float* sumPtr = reinterpret_cast<float*>(dynShm1tb1rw) + (embeddingDim + blockDim.y*BLK_N)*BLK_M/2 + laneId;
      //update r_b
      float rowSum = 0.0f;
      //we have blockDim.y columns reserved for local sum of each warp
      //but only nBlock*2 are used
      for(int j=0; j<nBlock*2; j++){
        rowSum += sumPtr[j*BLK_M];
      }
      sumOld[laneId] = fmaf(mTilde[laneId], sumOld[laneId], rowSum);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      uint16_t* V_uint16 = reinterpret_cast<uint16_t*>(V);
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int vRowOffset = i*BLK_N + laneId/4;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset_base + i*BLK_M*BLK_N, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim + vRowOffset;
            int64_t offset2 = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim + vRowOffset;
            B_frag[k] = ((uint32_t)V_uint16[offset2] << 16) | V_uint16[offset];
          }
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                    E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                    B_frag[0], B_frag[1],
                    O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset_base + i*BLK_M*BLK_N);
      }
    }
  }
  __syncthreads();
  float invR0 = sumOld[laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[laneId/4]);
  float invR1 = sumOld[BLK_M/2 + laneId/4] == 0.0f ? 0.0f : __frcp_rn(sumOld[BLK_M/2 + laneId/4]);
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int64_t outputOffset = ((int64_t)scheduler.targetRw*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0] * invR0;
    val.y = oPtr[32] * invR0;
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64] * invR1;
    val.y = oPtr[96] * invR1;
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
}

// Each warp computes 1 tcb of S.
// Assumption: At least 2 warps per block because we need 2 tcbs to go to the spmm stage.
__global__ void f2sKernel1tb1rw(
    const int *__restrict__ rowWindowOffset,
    const int *__restrict__ sparseAToXidx,
    const uint64_t *__restrict__ tcbBitMap,
    int embeddingDim,
    ulonglong2 *__restrict__ Q,
    ulonglong2 *__restrict__ K,
    half *__restrict__ V,
    float2 *output,
    float2 *sddmmResult) {
  volatile int laneId = threadIdx.x;
  int warpId = threadIdx.y;
  // contains a RW of Q
  extern __shared__ __align__(16) uint64_t dynShm1tb1rw[];
  __shared__ float mTilde[BLK_M];
  {
    //initialize everything to 0
    int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/4 + 2*blockDim.y*BLK_M/2;
    for(int i = tid; i < oOffset + embeddingDim*BLK_M/2; i += blockDim.x*blockDim.y){
      dynShm1tb1rw[i] = 0;
    }
    for(int i = tid; i < BLK_M; i += blockDim.x*blockDim.y){
      mTilde[i] = 1.0f;
    }
    //BLK_M/2 because each thread loads 2 128b elements
    int64_t qOffset = (int64_t)bid*BLK_M*embeddingDim/8;
    for(int i = tid; i < (BLK_M/2)*embeddingDim/8; i += blockDim.x*blockDim.y){
      loadQHbm2Shm128b(dynShm1tb1rw, Q+qOffset, embeddingDim, i);
    }
  }
  __syncthreads();

  int niter = ((rowWindowOffset[bid+1] - rowWindowOffset[bid]) + (blockDim.y/2) - 1)/(blockDim.y/2);
  #pragma unroll 1
  for(int iter = 0; iter < niter; iter++){
    int iterTcbStart = rowWindowOffset[bid] + iter*(blockDim.y/2);
    // number of 16x16 blocks in S/E being computed in this iteration.
    // This is a check in case the last iteration is not full
    int nBlock = min(blockDim.y/2, rowWindowOffset[bid+1]-iterTcbStart);
    float S_frag[4] = {0.0f};
    int warpTcbId = warpId/2 + iterTcbStart;
    __syncthreads();//DEBUG
    if(warpTcbId < rowWindowOffset[bid+1]){
      {//sddmm
        int64_t kOffset = (int64_t)sparseAToXidx[warpTcbId*BLK_M + (warpId%2)*BLK_N + laneId/4] * embeddingDim/8 + laneId % 4;
        for(int i = 0; i < embeddingDim/BLK_K; i+=2) {
          //load K with permuted columns
          ulonglong2 val = K[kOffset + i*BLK_K/8];
          uint64_t Q_frag[2];
          loadQFragShm(Q_frag, dynShm1tb1rw, i, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.x & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.x >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
          loadQFragShm(Q_frag, dynShm1tb1rw, i+1, laneId);
          HMMA16816(S_frag[0], S_frag[1], S_frag[2], S_frag[3],
                    static_cast<uint32_t>(Q_frag[0] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[1] & 0xFFFFFFFFull),
                    static_cast<uint32_t>(Q_frag[0] >> 32),
                    static_cast<uint32_t>(Q_frag[1] >> 32),
                    static_cast<uint32_t>(val.y & 0xFFFFFFFFull),
                    static_cast<uint32_t>(val.y >> 32),
                    S_frag[0], S_frag[1], S_frag[2], S_frag[3]);
        }
        int bitIdx = 63 - laneId*2;
        for(int i = 0; i < 4; i++){
          uint64_t bitMask = 1ULL << (bitIdx - i%2);
          S_frag[i] = (tcbBitMap[warpTcbId*4+(warpId%2)*2+i/2] & bitMask) == 0 ? 0.0f : S_frag[i];
        }
      }
      // {//save sddmm result
      //   int offset = warpTcbId*BLK_M*BLK_M + (warpId%2)*BLK_M*BLK_N + laneId*2;
      //   for(int j = 0; j < 2; j++){ // 2 8x8 blocks in each 16x8 block
      //     int sumOffset = j*BLK_N*BLK_N;
      //     float2 val;
      //     val.x = S_frag[j*2];
      //     val.y = S_frag[j*2+1];
      //     sddmmResult[(offset + sumOffset)/2] = val;
      //   }
      // }
    }
    __syncthreads();
    if(warpTcbId < rowWindowOffset[bid+1]){
      int eOffset = (embeddingDim + warpId*BLK_N)*BLK_M/2+laneId;
      storeEFragShm(S_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
    }
    __syncthreads();
    {//SpMM
      int oOffset_base = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
      for(int i=warpId; i<embeddingDim/BLK_N; i+=blockDim.y){
        int oOffset = oOffset_base + i*BLK_M*BLK_N;
        float O_frag[4];
        loadOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset, mTilde+laneId/4);
        for(int j=0; j<nBlock; j++){
          uint32_t E_frag[4];
          uint32_t B_frag[2];
          //load E
          int eOffset = (embeddingDim+j*BLK_M)*BLK_M/2 + laneId;
          loadEFragShm(E_frag, reinterpret_cast<uint32_t*>(dynShm1tb1rw)+eOffset);
          //load V
          int sparseAToXidxOffset = (iterTcbStart+j)*BLK_M + (laneId%4)*2;
          for(int k = 0; k < 2; k++){
            sparseAToXidxOffset += k*BLK_N;
            Half2Uint32 h2U32Converter;
            int64_t offset = (int64_t)sparseAToXidx[sparseAToXidxOffset]*embeddingDim + i*BLK_N + laneId/4;
            h2U32Converter.h2.x = V[offset];
            offset = (int64_t)sparseAToXidx[sparseAToXidxOffset+1]*embeddingDim + i*BLK_N + laneId/4;
            h2U32Converter.h2.y = V[offset];
            B_frag[k] = h2U32Converter.u32;
          }
          HMMA16816(O_frag[0], O_frag[1], O_frag[2], O_frag[3],
                   E_frag[0], E_frag[1], E_frag[2], E_frag[3],
                   B_frag[0], B_frag[1],
                   O_frag[0], O_frag[1], O_frag[2], O_frag[3]);
        }
        storeOFragShm(O_frag, reinterpret_cast<float*>(dynShm1tb1rw)+oOffset);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  //points to (laneId)th element of O
  int oOffset = (embeddingDim+blockDim.y*BLK_N)*BLK_M/2 + 2*blockDim.y*BLK_M + laneId;
  //offset in terms of number of elements,
  //have to be divided by 2 to get the index of the float2
  int64_t outputOffset = ((int64_t)bid*BLK_M + laneId/4)*embeddingDim + (laneId%4)*2;
  for(int i = warpId; i < embeddingDim/BLK_N; i += blockDim.y){
    int offset = oOffset + i*BLK_M*BLK_N;
    float* oPtr = reinterpret_cast<float*>(dynShm1tb1rw) + offset;
    float2 val;
    val.x = oPtr[0];
    val.y = oPtr[32];
    output[(outputOffset + i*BLK_N)/2] = val;
    val.x = oPtr[64];
    val.y = oPtr[96];
    output[(outputOffset + i*BLK_N + BLK_M/2*embeddingDim)/2] = val;
  }
}
