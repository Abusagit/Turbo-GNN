// #include <torch/extension.h>
// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <vector>
// #include <ATen/cuda/CUDAContext.h>

#include "torch_minimal.h"


#define BLOCKS(N, T) (N + T - 1)/T


template <typename scalar_t>
__device__ void smem_reduce_v2(volatile scalar_t* sdata, unsigned int tid, unsigned int reduce_len){
    if (reduce_len > 512){
        if(tid < 512){
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (reduce_len > 256){
        if(tid < 256){
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (reduce_len > 128){
        if(tid < 128){
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (reduce_len > 64){
        if(tid < 64){
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32){
        if (reduce_len > 32) sdata[tid] += sdata[tid + 32];
        if (reduce_len > 16) sdata[tid] += sdata[tid + 16];
        if (reduce_len > 8)  sdata[tid] += sdata[tid + 8];
        if (reduce_len > 4)  sdata[tid] += sdata[tid + 4];
        if (reduce_len > 2)  sdata[tid] += sdata[tid + 2];
        if (reduce_len > 1)  sdata[tid] += sdata[tid + 1];
    }
}


template <typename scalar_t, unsigned int blockSize>
__device__ void smem_reduce_v3(volatile scalar_t* sdata, unsigned int tid){
    if (blockSize >= 1024){
        if (tid < 512){
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512){
        if (tid < 256){
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128){
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64){
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32){
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void gar_get_alpha(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> e_pre,
    const int* __restrict__ src_ptr,
    const int* __restrict__ tar_index,
    scalar_t* alpha,
    scalar_t* mask_lrelu,
    scalar_t* e_sum,
    scalar_t* e_,
    float negative_slope
){
    // each thread block handles a single target
    unsigned int src_id = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int e_start = src_ptr[src_id];
    unsigned int e_bound = src_ptr[src_id + 1];
    scalar_t e_pre_src = e_pre[src_id][0];
    scalar_t e = 0;

    for (unsigned int e_id = e_start + tid; e_id < e_bound; e_id += blockDim.x){
        unsigned int tar_id = tar_index[e_id];
        // add attention factors
        e = e_pre_src + e_pre[tar_id][1];
        // leaky RelU
        if (e < 0){
            e *= negative_slope;
            mask_lrelu[e_id] = negative_slope;
        }else{
            mask_lrelu[e_id] = 1;
        }
        e = exp(e);
        atomicAdd(&e_sum[tar_id], e);
        alpha[e_id] = e;
        e_[e_id] = e;
    }
}


template <typename scalar_t>
__global__ void get_alpha_self(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> e_pre,
    scalar_t* e_sum, scalar_t* e_self, scalar_t* mask_lrelu_self, scalar_t* alpha_self,
    unsigned int num_node, float negative_slope
){
    for (unsigned int src_id = blockIdx.x * blockDim.x + threadIdx.x; src_id < num_node; src_id += blockDim.x * gridDim.x){
        scalar_t e = e_pre[src_id][0] + e_pre[src_id][1];
        if (e < 0){
            e *= negative_slope;
            mask_lrelu_self[src_id] = negative_slope;
        }else{
            mask_lrelu_self[src_id] = 1;
        }
        e = exp(e);
        e_self[src_id] = e;
        scalar_t sum = e_sum[src_id] + e;
        e_sum[src_id] = sum;
        alpha_self[src_id] = e / sum;
    }
}


#define THREADS 256


template <typename scalar_t>
__global__ void alpha_normalize_kernel(
    scalar_t* alpha,
    const scalar_t* __restrict__ e_sum,
    const int* __restrict__ tar_index,
    unsigned int num_edge
){
    for (unsigned int e_id = blockIdx.x * blockDim.x + threadIdx.x; e_id < num_edge; e_id += blockDim.x * gridDim.x){
        alpha[e_id] /= (e_sum[tar_index[e_id]]);
    }
}


std::vector<torch::Tensor> gat_gar_edge_weight_cuda(
    torch::Tensor e_pre,
    torch::Tensor src_ptr,
    torch::Tensor tar_index,
    float negative_slope
){
    unsigned int num_node = e_pre.size(0);
    unsigned int num_edge = tar_index.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(e_pre.device());
    auto alpha = torch::empty({num_edge, 1}, options);
    auto mask_lrelu = torch::empty_like(alpha);
    auto alpha_self = torch::empty({num_node, 1}, options);
    auto e_sum = torch::zeros_like(alpha_self);
    auto mask_lrelu_self = torch::empty_like(alpha_self);

    auto e = torch::empty_like(alpha);
    auto e_self = torch::empty_like(alpha_self);

    AT_DISPATCH_FLOATING_TYPES(e_pre.type(), "gcn_aggregate_f_kernel", ([&]{
        gar_get_alpha<scalar_t, THREADS><<<num_node, THREADS>>>(
            e_pre.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            src_ptr.data_ptr<int>(), tar_index.data_ptr<int>(),
            alpha.data_ptr<scalar_t>(), mask_lrelu.data_ptr<scalar_t>(),
            e_sum.data_ptr<scalar_t>(), e.data_ptr<scalar_t>(), negative_slope
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(e_pre.type(), "gcn_aggregate_f_kernel", ([&]{
        get_alpha_self<scalar_t><<<BLOCKS(num_node, THREADS), THREADS>>>(
            e_pre.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            e_sum.data_ptr<scalar_t>(), e_self.data_ptr<scalar_t>(), mask_lrelu_self.data_ptr<scalar_t>(),
            alpha_self.data_ptr<scalar_t>(), num_node, negative_slope
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(e_pre.type(), "gcn_aggregate_f_kernel", ([&]{
        alpha_normalize_kernel<scalar_t><<<BLOCKS(num_edge, THREADS), THREADS>>>(
            alpha.data_ptr<scalar_t>(), e_sum.data_ptr<scalar_t>(),
            tar_index.data_ptr<int>(), num_edge
        );
    }));

    return {alpha, alpha_self, mask_lrelu, mask_lrelu_self, e_sum, e, e_self};
}


template <typename scalar_t>
__global__ void gas_get_alpha(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> e_pre,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    scalar_t* alpha,
    scalar_t* mask_lrelu,
    scalar_t* e_sum,
    scalar_t* e_,
    float negative_slope,
    unsigned int num_edge
){
    for (unsigned int e_id = blockIdx.x * blockDim.x + threadIdx.x; e_id < num_edge; e_id += blockDim.x * gridDim.x){
        unsigned int tar_id = tar_index[e_id];
        unsigned int src_id = src_index[e_id];
        scalar_t e = e_pre[src_id][0] + e_pre[tar_id][1];
        if (e < 0){
            e *= negative_slope;
            mask_lrelu[e_id] = negative_slope;
        }else{
            mask_lrelu[e_id] = 1;
        }
        e = exp(e);
        atomicAdd(&e_sum[tar_id], e);
        alpha[e_id] = e;
        e_[e_id] = e;
    }
}



std::vector<torch::Tensor> gat_gas_edge_weight_cuda(
    torch::Tensor e_pre,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    float negative_slope
){
    unsigned int num_node = e_pre.size(0);
    unsigned int num_edge = tar_index.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(e_pre.device());
    auto alpha = torch::empty({num_edge, 1}, options);
    auto mask_lrelu = torch::empty_like(alpha);
    auto alpha_self = torch::empty({num_node, 1}, options);
    auto e_sum = torch::zeros_like(alpha_self);
    auto mask_lrelu_self = torch::empty_like(alpha_self);

    auto e = torch::empty_like(alpha);
    auto e_self = torch::empty_like(alpha_self);

    AT_DISPATCH_FLOATING_TYPES(e_pre.type(), "gcn_aggregate_f_kernel", ([&]{
        gas_get_alpha<scalar_t><<<BLOCKS(num_edge, THREADS), THREADS>>>(
            e_pre.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
            alpha.data_ptr<scalar_t>(), mask_lrelu.data_ptr<scalar_t>(),
            e_sum.data_ptr<scalar_t>(), e.data_ptr<scalar_t>(), negative_slope, num_edge
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(e_pre.type(), "gcn_aggregate_f_kernel", ([&]{
        get_alpha_self<scalar_t><<<BLOCKS(num_node, THREADS), THREADS>>>(
            e_pre.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            e_sum.data_ptr<scalar_t>(), e_self.data_ptr<scalar_t>(), mask_lrelu_self.data_ptr<scalar_t>(),
            alpha_self.data_ptr<scalar_t>(), num_node, negative_slope
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(e_pre.type(), "gcn_aggregate_f_kernel", ([&]{
        alpha_normalize_kernel<scalar_t><<<BLOCKS(num_edge, THREADS), THREADS>>>(
            alpha.data_ptr<scalar_t>(), e_sum.data_ptr<scalar_t>(),
            tar_index.data_ptr<int>(), num_edge
        );
    }));

    return {alpha, alpha_self, mask_lrelu, mask_lrelu_self, e_sum, e, e_self};
}




template <typename scalar_t>
__global__ void gat_gar_edge_weight_b_kernel(
    const scalar_t* __restrict__ grad_alpha,
    const scalar_t* __restrict__ alpha,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ e_sum,
    scalar_t* grad_e_sum,
    scalar_t* grad_e,
    unsigned int num_edge
){
    for (unsigned int e_id = blockIdx.x * blockDim.x + threadIdx.x; e_id < num_edge; e_id += blockDim.x * gridDim.x){
        unsigned int tar_id = tar_index[e_id];
        scalar_t e_sum_tar = e_sum[tar_id];
        scalar_t grad_alpha_buffer = grad_alpha[e_id];
        scalar_t g_e_sum = -(grad_alpha_buffer * alpha[e_id]);
        atomicAdd(&grad_e_sum[tar_id], g_e_sum);
        grad_e[e_id] = grad_alpha_buffer / e_sum_tar;
    }
}

template <typename scalar_t>
__global__ void gat_gar_edge_weight_b2_kernel(
    const scalar_t* __restrict__ grad_alpha_self,
    const scalar_t* __restrict__ alpha_self,
    const scalar_t* __restrict__ e_sum,
    scalar_t* grad_e_sum, //scalar_t* grad_e_self,
    const scalar_t* __restrict__ e_self,
    const scalar_t* __restrict__ mask_lrelu_self,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_e_pre,
    unsigned int num_node
){
    for (unsigned int n_id = blockIdx.x * blockDim.x + threadIdx.x; n_id < num_node; n_id += blockDim.x * gridDim.x){
        scalar_t e_sum_ = e_sum[n_id];
        scalar_t grad_alpha_self_ = grad_alpha_self[n_id];
        scalar_t grad_e_sum_ = (grad_e_sum[n_id] - grad_alpha_self_ * alpha_self[n_id]) / e_sum_;
        grad_e_sum[n_id] = grad_e_sum_;
        scalar_t grad_e_self = (grad_alpha_self_ / e_sum_ + grad_e_sum_) * e_self[n_id] * mask_lrelu_self[n_id];
        grad_e_pre[n_id][0] = grad_e_self;
        grad_e_pre[n_id][1] = grad_e_self;
    }
}


template <typename scalar_t>
__global__ void gat_gar_edge_weight_b3_kernel(
    scalar_t* grad_e,
    const scalar_t* __restrict__ e,
    const scalar_t* __restrict__ grad_e_sum,
    const scalar_t* __restrict__ mask_lrelu,
    const int* __restrict__ tar_index,
    const int* __restrict__ src_index,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_e_pre,
    unsigned int num_edge
){
    for (unsigned int e_id = blockIdx.x * blockDim.x + threadIdx.x; e_id < num_edge; e_id += blockDim.x * gridDim.x){
        unsigned int tar_id = tar_index[e_id];
        scalar_t grad_e_ = (grad_e[e_id] + grad_e_sum[tar_id]) * e[e_id] * mask_lrelu[e_id];
        atomicAdd(&grad_e_pre[tar_id][1], grad_e_);
        atomicAdd(&grad_e_pre[src_index[e_id]][0], grad_e_);
    }

}


std::vector<torch::Tensor> gat_gar_edge_weight_b_cuda(
    torch::Tensor grad_alpha_self,
    torch::Tensor grad_alpha,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor mask_lrelu,
    torch::Tensor mask_lrelu_self,
    torch::Tensor e,
    torch::Tensor e_self,
    torch::Tensor e_sum,
    torch::Tensor alpha_self,
    torch::Tensor alpha
){
    unsigned int num_node = alpha_self.size(0);
    unsigned int num_edge = alpha.size(0);
    auto grad_e_sum = torch::zeros_like(e_sum);
    auto grad_e = torch::empty_like(e);
    // auto grad_e_self = torch::empty_like(e_self);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(alpha.device());
    auto grad_e_pre = torch::empty({num_node, 2}, options);

    AT_DISPATCH_FLOATING_TYPES(grad_alpha.type(), "gat_gar_edge_weight_b_kernel", ([&]{
        gat_gar_edge_weight_b_kernel<scalar_t><<<BLOCKS(num_edge, THREADS), THREADS>>>(
            grad_alpha.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(),
            tar_index.data_ptr<int>(), e_sum.data_ptr<scalar_t>(),
            grad_e_sum.data_ptr<scalar_t>(), grad_e.data_ptr<scalar_t>(), num_edge
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(grad_alpha.type(), "gat_gar_edge_weight_b_kernel", ([&]{
        gat_gar_edge_weight_b2_kernel<scalar_t><<<BLOCKS(num_node, THREADS), THREADS>>>(
            grad_alpha_self.data_ptr<scalar_t>(), alpha_self.data_ptr<scalar_t>(),
            e_sum.data_ptr<scalar_t>(), grad_e_sum.data_ptr<scalar_t>(),
            e_self.data_ptr<scalar_t>(), mask_lrelu_self.data_ptr<scalar_t>(),
            grad_e_pre.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(), num_node
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(grad_alpha.type(), "gat_gar_edge_weight_b_kernel", ([&]{
        gat_gar_edge_weight_b3_kernel<scalar_t><<<BLOCKS(num_edge, THREADS), THREADS>>>(
            grad_e.data_ptr<scalar_t>(), e.data_ptr<scalar_t>(), grad_e_sum.data_ptr<scalar_t>(),
            mask_lrelu.data_ptr<scalar_t>(), tar_index.data_ptr<int>(),
            src_index.data_ptr<int>(),
            grad_e_pre.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            num_edge
        );
    }));

    return {grad_e_sum, grad_e_pre};
}
