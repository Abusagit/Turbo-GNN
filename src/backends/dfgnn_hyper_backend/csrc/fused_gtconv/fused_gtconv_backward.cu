#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

// csc format spmm kernel
// efeat (nnz, h, efeat_len), ufeat (n, h, ufeat_len), in csr format
template <typename DType>
__global__ void
spmm_csc_kernel(int n, int ufeat_len, int efeat_len, int out_len,
                const int *indptr, const int *indices, const int *val_idx,
                const DType *ufeat, const DType *efeat, DType *out) {
  int hid = blockIdx.y;
  int ty = blockIdx.x * blockDim.y + threadIdx.y; // 0-n
  const int stride_x = blockDim.x * gridDim.y;    // f*h

  if (ty < n) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x; // 0-f
    while (tx < out_len) {
      DType acc = 0;
      for (int i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const int eid = val_idx[i];
        const int cid = __ldg(indices + i);
        const DType *uoff = ufeat + cid * ufeat_len;
        const DType *eoff = efeat + hid * efeat_len;
        DType tmp_out = uoff[tx] * eoff[eid];
        acc += tmp_out;
      }
      out[ty * out_len + tx] = acc;
      tx += stride_x;
    }
  }
}

// SPMM backward A @ B = C -> dB = (A^T) @ dC
template <typename DType>
__global__ void spmm_backward_kernel(int h, int f, const int *col_ptr,
                                     const int *row_ind, const int *val_idx,
                                     const DType *Q, const DType *attn_edge,
                                     const DType *grad_edge, const DType *grad,
                                     DType *grad_V, DType *grad_K) {
  const int cid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature di
  const int lb = col_ptr[cid];
  const int hb = col_ptr[cid + 1];
  const int num_neighbor = hb - lb;

  // compute the output
  DType acc = 0;
  DType acc2 = 0;

  for (int j = 0; j < num_neighbor; j++) {
    int rid = row_ind[lb + j];
    DType weight = attn_edge[val_idx[lb + j]];
    DType weight2 = grad_edge[val_idx[lb + j]];
    if (fid < f) {
      acc += weight * grad[rid * h * f + hid * f + fid];
      acc2 += weight2 * Q[rid * h * f + hid * f + fid];
    }
  }
  if (fid < f) {
    grad_V[cid * h * f + hid * f + fid] = acc;
    grad_K[cid * h * f + hid * f + fid] = acc2;
  }
}

// SPMM backward A @ B = C -> dA = (dC @ B^T) * A
template <typename DType>
__global__ void fused_backward_kernel(int m, int h, int f, const int *row,
                                      const int *row_ptr, const int *col_ind,
                                      const DType *K, const DType *rhs,
                                      const DType *attn_edge, const DType *lhs,
                                      DType *grad_edge, DType *grad_Q) {
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tid = threadIdx.y * 32 + tidx;

  // the node bound of this block
  const int blk_node_lb = blockDim.y * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockDim.y, m);

  // the edge bound of this block
  const int blk_edge_lb = row_ptr[blk_node_lb];
  // const int blk_edge_hb = row_ptr[blk_node_hb];

  // the num of edges in this block
  const int blk_num_edge = row_ptr[blk_node_hb] - blk_edge_lb;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  int nnz_per_warp = (blk_num_edge + blockDim.y - 1) / blockDim.y;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = col_ind + blk_edge_lb;

  // SDDMM, edge parallel
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      int src = __ldg(rowoff + curr_edge);
      int dst = __ldg(indicesoff + curr_edge);

      const DType *lhsoff = lhs + src * f * h + hid * f;
      const DType *rhsoff = rhs + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 64) {
        att_val += lhsoff[j] * rhsoff[j];
        if (j + 32 < f)
          att_val += lhsoff[j + 32] * rhsoff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight[curr_edge] = att_val;
      }
    }
  }
  __syncthreads();

  for (int i = tid; i < blk_num_edge; i += 256) {
    DType grad_i = neigh_nodes_weight[i];
    DType out = attn_edge[blk_edge_lb + i];
    neigh_nodes_weight[i] = grad_i * out;
  }
  __syncthreads();

  int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = row_ptr[curr_node];
    const int edge_hb = row_ptr[curr_node + 1];
    const int num_edge = edge_hb - edge_lb;

    const int hf = h * f;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    // compute the output
    int loop_f = (f + WARP_SIZE - 1) / WARP_SIZE;

    DType prodsum = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5); // node need to process in loop j
      DType prod = 0;
      if (pid < num_edge) {
        prod = neigh_nodes_weight_off[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        prod += __shfl_xor_sync(0xffffffff, prod, stride, 32);
      }
      __syncwarp();
      prodsum += prod;
    }

    // write imtermediate value
    for (int j = tidx; j < num_edge; j += 32) {
      DType attn_score = attn_edge[edge_lb + j];
      DType attn_val = neigh_nodes_weight_off[j] - prodsum * attn_score;
      grad_edge[edge_lb + j] = attn_val;
      neigh_nodes_weight_off[j] = attn_val;
    }

    for (int i = 0; i < loop_f; i += 1) {
      DType acc = 0;
      int pid = tidx + (i << 5);
      for (int j = 0; j < num_edge; j++) {
        int cid = col_ind[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        if (pid < f)
          acc += attn_val * K[cid * hf + hid * f + pid];
      }
      if (pid < f)
        grad_Q[curr_node * hf + hid * f + pid] = acc;
    }
  }
}

void gt_backward_launch(int m, int n, int nnz, int h, int f, int smem_consume,
                        int *row, int *row_ptr, int *col_ind, float *val,
                        int *col_ptr, int *row_ind, int *val_idx, float *Q,
                        float *K, float *V, float *attn_edge, float *grad_edge,
                        float *grad,   // input grad
                        float *grad_Q, // output grad
                        float *grad_K, // output grad
                        float *grad_V) // output grad
{

  const dim3 nblks2((m + 7) / 8, h, 1);
  const dim3 nthrs2(32, 8, 1);
  const int smem_size = smem_consume * sizeof(float);
  CUDA_KERNEL_CALL((fused_backward_kernel<float>), nblks2, nthrs2, smem_size, m,
                   h, f, row, row_ptr, col_ind, K, V, attn_edge, grad,
                   grad_edge, grad_Q);

  // const int ntx = FindNumThreads(f);
  // const int nty = CUDA_MAX_NUM_THREADS / ntx;
  // const int nby = h;
  // const int nbx = (n + nty - 1) / nty;
  // // printf("launch dim %d %d %d %d \n", ntx, nty, nbx, nby);
  // const dim3 nblks(nbx, nby, 1);
  // const dim3 nthrs(ntx, nty, 1);

  // CUDA_KERNEL_CALL((spmm_csc_kernel<float>), nblks, nthrs, 0, n, f * h, h,
  //                  f * h, col_ptr, row_ind, val_idx, Q, grad_edge, grad_K);
  // CUDA_KERNEL_CALL((spmm_csc_kernel<float>), nblks, nthrs, 0, n, f * h, nnz,
  //                  f * h, col_ptr, row_ind, val_idx, grad, attn_edge,
  //                  grad_V);

  const dim3 nblks(n, h, 1);
  const dim3 nthrs(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL((spmm_backward_kernel<float>), nblks, nthrs,
                   512 * sizeof(float), h, f, col_ptr, row_ind, val_idx, Q,
                   attn_edge, grad_edge, grad, grad_V, grad_K);
}

std::vector<torch::Tensor>
gt_backward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind,
                 torch::Tensor rows, torch::Tensor val, torch::Tensor col_ptr,
                 torch::Tensor row_ind, torch::Tensor val_idx, int smem_consume,
                 torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                 torch::Tensor attn_edge, torch::Tensor grad) {

  const auto m = row_ptr.size(0) - 1;
  const auto n = col_ptr.size(0) - 1;
  // if (m != n) {
  //   printf("m %d n %d\n", m, n);
  // }
  const auto nnz = col_ind.size(0);
  const auto h = Q.size(1);
  const auto f = Q.size(2);
  auto devid = row_ptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

  auto grad_edge = torch::empty({h, nnz}, options);
  auto grad_Q = torch::empty({m, h, f}, options);
  auto grad_K = torch::zeros({m, h, f}, options);
  auto grad_V = torch::zeros({m, h, f}, options);

  gt_backward_launch(
      m, n, nnz, h, f, smem_consume, rows.data_ptr<int>(),
      row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(), val.data_ptr<float>(),
      col_ptr.data_ptr<int>(), row_ind.data_ptr<int>(), val_idx.data_ptr<int>(),
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      attn_edge.data_ptr<float>(), grad_edge.data_ptr<float>(),
      grad.data_ptr<float>(), grad_Q.data_ptr<float>(),
      grad_K.data_ptr<float>(), grad_V.data_ptr<float>());

  return {grad_Q, grad_K, grad_V};
}
