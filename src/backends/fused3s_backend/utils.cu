#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <torch/extension.h>

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////
// assuming each tcblock can be divided into 8x8 (64 bits) sub-blocks
// sub_blocks are organized in column-major order
// elements inside each sub-block are organized in row-major order
__device__ void update_bitmap(uint64_t* bitmap,
                              int tcblock_id, int n_sub_blocks_per_tcblock,
                              int blockSize_h, int blockSize_w,
                              int row_local, int col_local) {
  int n_sub_blocks_per_row = blockSize_w / 8;
  int n_sub_blocks_per_col = blockSize_h / 8;
  unsigned long long int *ull_bitmap = reinterpret_cast<unsigned long long int*>(bitmap);
  int sub_block_row_id = row_local / 8;
  int sub_block_col_id = col_local / 8;
  int sub_block_id = sub_block_col_id * n_sub_blocks_per_col + sub_block_row_id;
  int sub_block_local_id = row_local % 8 * 8 + col_local % 8;
  uint64_t mask = 1ULL << (63 - sub_block_local_id);
  atomicOr(&ull_bitmap[tcblock_id * n_sub_blocks_per_tcblock + sub_block_id], mask);
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid,
                                          int *padded_offset,
                                          uint8_t *padded_tileid, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int s = ori_offset[tid];
    int e = ori_offset[tid + 1];
    int s1 = padded_offset[tid];
    for (int i = 0; i < e - s; i++) {
      padded_tileid[s1 + i] = ori_tileid[s + i];
    }
  }
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset,
                                       int *tcblock_rowid,
                                       int num_row_windows) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
    tcblock_rowid[block_start + idx] = winId;
  }
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid,
                                 int num_row_windows) {
  int block_size = 512;
  int window_count = num_row_windows;
  generate_tcblock_rowid<<<window_count, block_size>>>(
      rowwindow_offset, tcblock_rowid, num_row_windows);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in generate_tcblock_rowid_cuda: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

// assign row to thread block such that each thread block owns at least minTCBPerWarp*nWarpPerTB TCBs
// also make sure each thread block own entire row windows and no straddling
std::vector<int> assign_row_to_tb(int *rowwindow_offset, int nRowWindows){
  // Each TB must have at least this many TCBs
  int minTCB = MIN_TCBLOCK_PER_WARP * WARP_PER_TB;
  std::vector<int> tb_boundaries;
  tb_boundaries.push_back(0); // first TB starts at window
  int currentTBStart = 0;
  for (int i = 1; i <= nRowWindows; ++i) {
    if (rowwindow_offset[i] - rowwindow_offset[currentTBStart] >= minTCB) {
        tb_boundaries.push_back(i);
        currentTBStart = i;
    }
  }
  // If the last TB boundary is not at the end, add the end
  if (tb_boundaries.back() != nRowWindows) {
      tb_boundaries.push_back(nRowWindows);
  }
  return tb_boundaries;
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target) {
  int left = 0;
  int right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      while (mid > 0 && arr[mid - 1] == target) {
        mid--;
      }
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length,
                                                      int *loc) {
  int cur = 1;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      (*loc)++;
      array[(*loc)] = array[cur];
    }
    cur++;
  }
}
__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  int *start = edgelist_sort + block_start;
  int size = 0;
  inplace_deduplication(start, num_window_edges, &size);
  int num = (size + blockSize_w) / blockSize_w;
  atomicAdd(blocknum, num);
  blockpartition[winId] = num;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  generate_edgetocolumn<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
      blockSize_h, blockSize_w, num_nodes);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in generate_edgetocolumn_cuda: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
    int *sparseatob, uint64_t *tcblock_bit_map,
    int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
  extern __shared__ int pos_ptr[];
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  if (num_blocks == 0) {
    return;
  }
  int *tcblock_offset_ptr = pos_ptr + num_blocks;
  int *tcblock_offset_global_ptr = tcblock_offset + block_start;
  int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  for (int i = 0; i < 2 * num_blocks + 1; i++) {
    pos_ptr[i] = 0;
  }
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    tcblock_nnz_ptr[col / blockSize_w]++;
  }
  for (int i = 0; i < num_blocks; i++) {
    tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
  }
  auto tileid = tcblocktile_id + element_start;
  auto sparse_AToB = sparseatob + block_start * blockSize_w;
  for (int i = 0; i < num_blocks; i++) {
    tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
  }
  int n_sub_blocks_per_tcblock = blockSize_w * blockSize_h / 64;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;
    tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
        (uint8_t)(row_local * blockSize_w + col_local);
    update_bitmap(tcblock_bit_map, block_start + tcblock_id, n_sub_blocks_per_tcblock, blockSize_w, blockSize_h, row_local, col_local);
    sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    pos_ptr[tcblock_id]++;
  }
}
void generate_tcoffset_id_atob_cuda(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, int *tcblock_offset,
                                    uint8_t *tcblock_tileid, int *sparseatob,
                                    uint64_t *tcblock_bit_map,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
  std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
  if (dynamic_shared_size > 204800) {
    printf("dynamic_shared_size: %d is greater than 204800\n", dynamic_shared_size);
    exit(-1);
  } else if (dynamic_shared_size > 131072) {
    int maxbytes = 204800; // 200 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 98304) {
    int maxbytes = 131072; // 128 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 65536) {
    int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 32768) {
    int maxbytes = 65536; // 64 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  }
  generate_tcoffset_id_atob<<<window_count, block_size, dynamic_shared_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
      tcblock_offset, tcblock_tileid, sparseatob, tcblock_bit_map, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in generate_tcoffset_id_atob_cuda: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid,
                        int *padded_offset, uint8_t *padded_tileid, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}
/*main function*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockpartition, int *block_num,
              int *rowwindow_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num) {
  thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
  thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
  thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
  thrust::device_vector<int> deviceEL(EL, EL + num_edges);
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(deviceSeg.end(), deviceEL.end()));
  thrust::sort(thrust::device, begin, end);
  generate_edgetocolumn_cuda(
      nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), edgetocol,
      blockpartition, block_num, blockSize_h, blockSize_w, num_nodes);
  thrust::device_ptr<int> blockpartition_ptr =
      thrust::device_pointer_cast(blockpartition);
  thrust::device_ptr<int> rowwindow_offset_ptr =
      thrust::device_pointer_cast(rowwindow_offset + 1);
  thrust::device_vector<int> blockpartition_vector(
      blockpartition_ptr, blockpartition_ptr + rowwindow_num);
  thrust::inclusive_scan(blockpartition_vector.begin(),
                         blockpartition_vector.end(), rowwindow_offset_ptr);
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto options_gpu_uint64 =
      torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
  thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
  thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
  int block_counter = bnum_vector[0];
  auto tcblock_rowid_tensor = torch::zeros({block_counter}, options_gpu);
  auto tcblock_rowid = tcblock_rowid_tensor.data_ptr<int>();
  generate_tcblock_rowid_cuda(rowwindow_offset, tcblock_rowid, rowwindow_num);
  auto max_element =
      thrust::max_element(thrust::device, blockpartition_vector.begin(),
                          blockpartition_vector.end());
  int max_blocks = *max_element;
  auto tcblocktile_id_tensor = torch::zeros({num_edges}, options_gpu_unit8);
  auto tcblock_offset_tensor = torch::zeros({block_counter + 1}, options_gpu);
  auto sparse_AToX_index_tensor =
      torch::zeros({block_counter * blockSize_w}, options_gpu);
  int bit_map_size = blockSize_h * blockSize_w;
  assert(bit_map_size % 64 == 0);
  int bit_map_int64_size = bit_map_size / 64;
  auto tcblock_bit_map_tensor = torch::zeros({block_counter*bit_map_int64_size}, options_gpu_uint64);
  auto tcblock_bit_map = tcblock_bit_map_tensor.data_ptr<uint64_t>();
  auto tcblock_offset = tcblock_offset_tensor.data_ptr<int>();
  auto sparse_AToX_index = sparse_AToX_index_tensor.data_ptr<int>();
  auto tcblocktile_id = tcblocktile_id_tensor.data_ptr<uint8_t>();
  generate_tcoffset_id_atob_cuda(
      nodepointer, rowwindow_offset, edgetocol, edgetorow, edgeLists,
      tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, tcblock_bit_map, max_blocks,
      num_nodes, blockSize_h, blockSize_w, rowwindow_num);
  thrust::device_ptr<int> tcblock_offset_ptr =
      thrust::device_pointer_cast(tcblock_offset);
  thrust::inclusive_scan(tcblock_offset_ptr,
                         tcblock_offset_ptr + block_counter + 1,
                         tcblock_offset_ptr);
  return std::make_tuple(tcblock_offset_tensor, tcblock_rowid_tensor,
                         tcblocktile_id_tensor, sparse_AToX_index_tensor,
                         tcblock_bit_map_tensor, block_counter);
}

__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer,
                               int num_nodes) {
  int wid = threadIdx.y;
  int bid = blockIdx.x;
  for(int nid = bid*blockDim.y + wid; nid < num_nodes; nid += blockDim.y*gridDim.x) {
    for (int eid = nodePointer[nid] + threadIdx.x; eid < nodePointer[nid + 1]; eid += blockDim.x) {
      edgeToRow[eid] = nid;
    }
  }
}

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int numNodes) {
  int warpSize = 32;
  int nWarpPerBlock = 32;
  dim3 blockSize(warpSize, nWarpPerBlock);
  long long gridSize = (static_cast<long long>(numNodes) + nWarpPerBlock - 1) / nWarpPerBlock;
  int gridSize_int = std::min(gridSize, static_cast<long long>(10000));
  fill_edgeToRow<<<gridSize_int, blockSize>>>(edgeToRow, nodePointer, numNodes);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error in fill_edgeToRow_cuda: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h,
                             int blockSize_w, int num_nodes) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
    seg_out[block_start + idx] = winId;
  }
}
void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes) {
  int block_size = 512;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h,
                                             blockSize_w, num_nodes);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in fill_segment_cuda: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

// Sort row windows by number of TCBlocks in descending order
// Returns array where sorted_row_window[i] contains index of row window with i-th most TCBlocks
torch::Tensor sort_row_windows_by_tcb_count(const int* rowwindow_offset, int num_row_windows) {
  auto options_gpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

  // Create vectors for (count, index) pairs
  thrust::device_vector<int> counts(num_row_windows+1);
  thrust::device_vector<int> indices(num_row_windows);
  thrust::sequence(indices.begin(), indices.end());

  // Calculate TCBlock counts for each window
  // the first element of count = rowwindow_offset[0], has to be skipped
  thrust::adjacent_difference(
      rowwindow_offset,                         // Input start
      rowwindow_offset + num_row_windows + 1,  // Input end
      counts.begin()                            // Output start
  );

  // Sort by counts in descending order while keeping track of original indices
  thrust::sort_by_key(
    thrust::device,
    counts.begin()+1,
    counts.end(),
    indices.begin(),
    thrust::greater<int>()
  );

  // Create and return tensor with sorted indices
  auto sorted_row_window = torch::zeros({num_row_windows}, options_gpu);
  thrust::copy(indices.begin(), indices.end(),
               thrust::device_pointer_cast(sorted_row_window.data_ptr<int>()));
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in sort_row_windows_by_tcb_count: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  return sorted_row_window;
}
