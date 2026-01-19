#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_DEVICE(x)                                                        \
  TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<torch::Tensor>
gt_tiling_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor val, int smem_consume, torch::Tensor Q,
                         torch::Tensor K, torch::Tensor V);
std::vector<torch::Tensor>
gt_tiling_inference(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor val, int smem_consume, torch::Tensor Q,
                    torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(indices.size(0) == val.size(0));

  return gt_tiling_inference_cuda(indptr, indices, val, smem_consume, Q, K, V);
}


PYBIND11_MODULE(fused_gtconv_tiling, m) {
  m.doc() = "fuse sparse ops in graph transformer into one kernel.";
  m.def("gt_tiling_inference", &gt_tiling_inference,
        "fused graph transformer forward op in hyper format, one kernel");
}
