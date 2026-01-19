#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType, int colSize>
__global__ void fused_gt_tiling(const int h, const int f, const int *indptr,
                                const int *indices, const DType *val,
                                const DType *Q, const DType *K, const DType *V,
                                DType *out_feat) {
  const int rid = blockIdx.x; // loop over row of adj matrix
  const int hid = blockIdx.y; // loop over heads
  const int warpId = threadIdx.y;
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim
  const int blockSize = blockDim.y;

  const int lb = indptr[rid]; // row rid elements
  const int num_neighbor = indptr[rid + 1] - lb;
  static __shared__ DType neigh_nodes_weight[colSize];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  DType acc = 0, partial_sum = 0;
  DType weightMax_old = -INFINITY, weightMax = -INFINITY;

  int loop_neighbor = (num_neighbor + colSize - 1) / colSize;

  for (int j = 0; j < loop_neighbor; j++) {
    const int *indicesoff = indices + lb + j * colSize;
    const DType *valoff = val + lb + j * colSize;
    for (int i = 0; i < colSize / blockSize; i++) {
      int eid = i * blockSize + warpId;
      if (eid + j * colSize < num_neighbor) {
        int dst = __ldg(indicesoff + eid);
        // // the Q feature of row node
        const DType *Qoff = Q + rid * f * h + hid * f;
        // the K feature of col node
        const DType *Koff = K + dst * f * h + hid * f;

        DType att_val = 0;
        for (int j = threadIdx.x; j < f; j += 32) {
          att_val += Qoff[j] * Koff[j];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          att_val += __shfl_down_sync(full_mask, att_val, offset);
        if (threadIdx.x == 0) {
          neigh_nodes_weight[eid] = att_val * valoff[eid];
        }
      }
    }
    __syncthreads();

    for (int i = 0; i < colSize / WARP_SIZE; i++) {
      DType weight = -INFINITY;
      int pid = threadIdx.x + (i << 5);
      if (pid < num_neighbor) {
        weight = neigh_nodes_weight[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }
    __syncthreads();

    DType expweightMax =
        (weightMax_old == weightMax) ? 1.0f : exp(weightMax_old - weightMax);
    partial_sum = partial_sum * expweightMax;
    acc = acc * expweightMax;
    for (int i = 0; i < colSize; i++) {
      if (j * colSize + i < num_neighbor) {
        int cid = indicesoff[i];
        DType weight = neigh_nodes_weight[i];
        DType expweight = exp(weight - weightMax);
        acc += expweight * V[cid * hf + hfid];
        partial_sum += expweight;
      }
    }
    __syncthreads();
    weightMax_old = weightMax;
  }
  partial_sum = (partial_sum != 0) ? 1 / partial_sum : 0;
  out_feat[rid * hf + hfid] = acc * partial_sum;
}


std::vector<torch::Tensor>
gt_tiling_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor val, int smem_consume, torch::Tensor Q,
                         torch::Tensor K, torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  const int ntx = WARP_SIZE;
  const int nty = (f + WARP_SIZE - 1) / WARP_SIZE;

  const dim3 nblks(m, h);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL((fused_gt_tiling<float, 32>), nblks, nthrs,
                   (smem_consume) * sizeof(float), h, f, indptr.data_ptr<int>(),
                   indices.data_ptr<int>(), val.data_ptr<float>(),
                   Q.data_ptr<float>(), K.data_ptr<float>(),
                   V.data_ptr<float>(), out_feat.data_ptr<float>());
  return {out_feat};
}
