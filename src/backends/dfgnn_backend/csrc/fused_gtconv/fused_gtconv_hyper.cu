#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const Idx *array, Idx length,
                                               Idx eid) {
  Idx lo = 0, hi = length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (__ldg(array + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (__ldg(array + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename DType>
__global__ void fused_gt_hyper(const int m, const int nnz, const int h,
                               const int f, const int *row, const int *indptr,
                               const int *indices, const DType *val,
                               const DType *Q, const DType *K, const DType *V,
                               DType *attn_edge, DType *out_feat) {
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = indptr[blk_node_lb];

  // the num of edges in this block
  const int blk_num_edge = indptr[blk_node_hb] - blk_edge_lb;
  // the num of edges each warp need to process
  int nnz_per_warp = (blk_num_edge + blockSize - 1) / blockSize;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = indices + blk_edge_lb;
  const DType *valoff = val + blk_edge_lb;

  // SDDMM, edge parallel
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      const int src = __ldg(rowoff + curr_edge);
      const int dst = __ldg(indicesoff + curr_edge);

      // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 64) {
        // float2 Q2 = reinterpret_cast<const float2*>(Qoff)[j];
        // float2 K2 = reinterpret_cast<const float2*>(Koff)[j];
        // att_val += vecDot2<float2, float>(Q2, K2);
        att_val += Qoff[j] * Koff[j];
        if (j + 32 < f)
          att_val += Qoff[j + 32] * Koff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
      }
    }
  }
  __syncthreads();

  // Softmax+SPMM, node parallel
  int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = indptr[curr_node];
    const int num_edge = indptr[curr_node + 1] - edge_lb;
    const int hf = h * f;

    DType weightMax = -1e38;
    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    // compute the max val of SDDMM result
    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -1e38;
      int pid = tidx + (j << 5);
      if (pid < num_edge) {
        weight = neigh_nodes_weight_off[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }

    // compute the sum of exp
    DType expAll = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5);
      DType exptmp = 0;
      if (pid < num_edge) {
        DType weight = neigh_nodes_weight_off[pid];
        exptmp = exp(weight - weightMax);
        neigh_nodes_weight_off[pid] = exptmp;
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }

    // handle the node with no neighbor
    expAll = (expAll != 0) ? 1.0f / expAll : 0;

    // write out intermediate value
    for (int j = tidx; j < num_edge; j += 32) {
      DType attn_val = neigh_nodes_weight_off[j];
      attn_edge[hid * nnz + edge_lb + j] = attn_val * expAll;
    }

    // compute the output
    // TODO check the performance change
    for (int i = tidx; i < f; i += WARP_SIZE) {
      DType acc = 0;
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        acc += attn_val * V[cid * hf + hid * f + i];
      }
      out_feat[curr_node * hf + hid * f + i] = acc * expAll;
    }
  }
}

std::vector<torch::Tensor>
gt_hyper_forward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind,
                      torch::Tensor rows, torch::Tensor val,
                      torch::Tensor col_ptr, torch::Tensor row_ind,
                      torch::Tensor val_idx, int smem_consume, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V) {
  const auto m = row_ptr.size(0) - 1; // num of nodes
  const auto nnz = col_ind.size(0);   // num of edges
  const auto h = Q.size(1);           // num of heads
  const auto f = Q.size(2);           // num of feats
  auto devid = row_ptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  auto attn_edge = torch::empty({h, nnz}, options);

  const int ntx = 32;
  const int nty = 8;

  const int nbx = (m + nty - 1) / nty;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = smem_consume * sizeof(float);

  CUDA_KERNEL_CALL((fused_gt_hyper<float>), nblks, nthrs, smem_size, m, nnz, h,
                   f, rows.data_ptr<int>(), row_ptr.data_ptr<int>(),
                   col_ind.data_ptr<int>(), val.data_ptr<float>(),
                   Q.data_ptr<float>(), K.data_ptr<float>(),
                   V.data_ptr<float>(), attn_edge.data_ptr<float>(),
                   out_feat.data_ptr<float>());

  return {out_feat, attn_edge};
}
