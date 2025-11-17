#include "torch_minimal.h"



template <typename scalar_t, unsigned int blockSize>
__device__ void smem_reduce_v1(volatile scalar_t* sdata, unsigned int tid, unsigned int reduce_len, unsigned int f_dim){
    while (reduce_len > 1){
        __syncthreads();
        // add the remainer
        if ((tid < f_dim) && (reduce_len % 2 == 1)){
            sdata[tid] += sdata[tid + f_dim * (reduce_len - 1)];
        }
        reduce_len /= 2;
        if (tid < f_dim * reduce_len){
            sdata[tid] += sdata[tid + f_dim * reduce_len];
        }
    }
}


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


/*
 * The fused forward kernel for GAR aggregator
 */


template <typename scalar_t, unsigned int blockSize>
__global__ void fused_gar_f_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> out,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_ptr,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int f_dim,
    const unsigned int stride
){
    // shared memory for feature reduction
    __shared__ scalar_t s_feature[blockSize];
    // Registers
    unsigned int tid = threadIdx.x;
    unsigned int tar_id = blockIdx.x;  // each block handles a single target
    unsigned int f_idx = tid % f_dim;
    unsigned int e_start = tar_ptr[tar_id];
    unsigned int e_bound = tar_ptr[tar_id + 1];

    // Step 0: initialize shared memory
    s_feature[tid] = 0;
    // Step 1: reduce the feature vectors into shared memory
    for (unsigned int e_id = e_start + tid / f_dim; e_id < e_bound; e_id += stride){
        s_feature[tid] += feature[src_index[e_id]][f_idx] * edge_weight[e_id];
    }

    // Step 2: Reduction
    unsigned int reduce_len = min(stride, e_bound - e_start);

    smem_reduce_v1<scalar_t, blockSize>(s_feature, tid, reduce_len, f_dim);

    // Step 3: Write out
    if (tid < f_dim){
        out[tar_id][f_idx] = s_feature[tid];
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void fused_gar_f_large_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> out,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_ptr,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int f_dim
){
    unsigned int tid = threadIdx.x;
    unsigned int tar_id = blockIdx.x;

    for (unsigned int f_idx = tid; f_idx < f_dim; f_idx += blockSize){
        scalar_t s_feature = 0;
        for (unsigned int e_id=tar_ptr[tar_id]; e_id < tar_ptr[tar_id + 1]; e_id ++){
            s_feature += feature[src_index[e_id]][f_idx] * edge_weight[e_id];
        }
        out[tar_id][f_idx] = s_feature;
    }
}

#define GAR_THREADS 128

// CUDA GAR model forward
torch::Tensor fused_gar_f_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor edge_weight
){
    unsigned int f_dim = feature.size(1);
    unsigned int num_node = feature.size(0);

    auto out = torch::empty_like(feature);

    if (f_dim <= GAR_THREADS){
        unsigned int stride = GAR_THREADS / f_dim;
        AT_DISPATCH_FLOATING_TYPES(feature.type(), "aggregation gar forward", ([&]{
            fused_gar_f_kernel<scalar_t, GAR_THREADS><<<num_node, GAR_THREADS>>>(
                feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data_ptr<int>(),
                tar_ptr.data_ptr<int>(),
                edge_weight.data_ptr<scalar_t>(),
                f_dim, stride
            );
        }));
    }else{
        AT_DISPATCH_FLOATING_TYPES(feature.type(), "aggregation gar forward", ([&]{
            fused_gar_f_large_kernel<scalar_t, GAR_THREADS><<<num_node, GAR_THREADS>>>(
                feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data_ptr<int>(),
                tar_ptr.data_ptr<int>(),
                edge_weight.data_ptr<scalar_t>(),
                f_dim
            );
        }));
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "compute_edge_weights_kernel launch failed: ", cudaGetErrorString(err));

    return out;
}


/*
 * The fused backward kernel for GAR aggregator
 */

 template <typename scalar_t, unsigned int blockSize>
 __global__ void fused_gar_b_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_out,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ tar_index,
    const int* __restrict__ src_ptr,
    const scalar_t* __restrict__ edge_weight,
    unsigned int f_dim, unsigned int stride
 ){
    // shared memory for gradient reduction
    __shared__ scalar_t s_grad_feature[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int src_id = blockIdx.x;
    unsigned int f_idx = tid % f_dim;
    unsigned int e_start = src_ptr[src_id];
    unsigned int e_bound = src_ptr[src_id+1];

    // initialize the shared memory
    s_grad_feature[tid] = 0;

    for (unsigned int e_id = e_start + tid / f_dim; e_id < e_bound; e_id += stride){
        s_grad_feature[tid] += grad_out[tar_index[e_id]][f_idx] * edge_weight[e_id];
    }

    // Step 2: reduction
    unsigned int reduce_len = min(stride, e_bound - e_start);

    smem_reduce_v1<scalar_t, blockSize>(s_grad_feature, tid, reduce_len, f_dim);

    if (tid < f_dim){
        grad_feature[src_id][f_idx] = s_grad_feature[tid];
    }
 }



 template <typename scalar_t, unsigned int blockSize>
 __global__ void fused_gar_b_kernelv2(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_out,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ tar_index,
    const int* __restrict__ src_ptr,
    scalar_t* grad_edge_weight,
    const scalar_t* __restrict__ edge_weight,
    unsigned int f_dim, unsigned int stride
 ){
    // shared memory for gradient reduction
    __shared__ scalar_t s_grad_feature[blockSize];
    __shared__ scalar_t s_grad_edge_weight[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int src_id = blockIdx.x;
    unsigned int f_idx = tid % f_dim;
    unsigned int e_start = src_ptr[src_id];
    unsigned int e_bound = src_ptr[src_id+1];
    unsigned int group_id = tid / f_dim;
    unsigned int e_idx = e_start + group_id;

    unsigned int reduce_len = f_dim;

    // initialize the shared memory
    s_grad_feature[tid] = 0;
    s_grad_edge_weight[tid] = 0;

    scalar_t grad_out_buffer = 0;
    scalar_t src_feature_buffer = feature[src_id][f_idx];

    unsigned int total_strides = (e_bound - e_start + stride - 1) / stride;

    for (unsigned int i=e_start; i < total_strides * stride + e_start; i += stride){
        __syncthreads();
        if ((tid < stride * f_dim) && (e_idx < e_bound)){
            grad_out_buffer = grad_out[tar_index[e_idx]][f_idx];
            s_grad_feature[tid] += grad_out_buffer * edge_weight[e_idx];
            // interleavely load the intermediate results into s_grad_edge_weight
            s_grad_edge_weight[group_id + f_idx * stride] = grad_out_buffer * src_feature_buffer;
        }
        __syncthreads();
        reduce_len = f_dim;
        smem_reduce_v1<scalar_t, blockSize>(s_grad_edge_weight, tid, reduce_len, stride);
        __syncthreads();
        if ((tid < stride) && (i + tid < e_bound)){
            grad_edge_weight[i+tid] = s_grad_edge_weight[tid];
        }
        s_grad_edge_weight[tid] = 0;
        e_idx += stride;
    }

    // Step 2: reduction
    reduce_len = min(stride, e_bound - e_start);

    smem_reduce_v1<scalar_t, blockSize>(s_grad_feature, tid, reduce_len, f_dim);

    if (tid < f_dim){
        grad_out_buffer = grad_out[src_id][f_idx];
        s_grad_edge_weight[tid] = grad_out_buffer * src_feature_buffer;
        grad_feature[src_id][f_idx] = s_grad_feature[tid];
    }
    __syncthreads();

    smem_reduce_v2<scalar_t>(s_grad_edge_weight, tid, f_dim);
 }


 template <typename scalar_t, unsigned int blockSize>
 __global__ void fused_gar_b_large_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_out,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ tar_index,
    const int* __restrict__ src_ptr,
    const scalar_t* __restrict__ edge_weight,
    unsigned int f_dim
 ){
    unsigned int tid = threadIdx.x;
    unsigned int src_id = blockIdx.x;

    for (unsigned int f_idx = tid; f_idx < f_dim; f_idx += blockSize){
        scalar_t s_grad_feature = 0;
        for (unsigned int e_id=src_ptr[src_id]; e_id < src_ptr[src_id + 1]; e_id ++){
            s_grad_feature += grad_out[tar_index[e_id]][f_idx] * edge_weight[e_id];
        }
        grad_feature[src_id][f_idx] = s_grad_feature;
    }
 }


 template <typename scalar_t, unsigned int blockSize>
 __global__ void fused_gar_b_large_kernelv2(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_out,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ tar_index,
    const int* __restrict__ src_ptr,
    scalar_t* grad_edge_weight,
    const scalar_t* __restrict__ edge_weight,
    unsigned int f_dim
 ){
    // shared memory to buffer the src feature
    __shared__ scalar_t s_src_feature[1024];
    __shared__ scalar_t s_grad_feature[1024];
    // shared memory for edge weight reduction
    __shared__ scalar_t s_grad_edge_weight[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int src_id = blockIdx.x;
    scalar_t grad_out_buffer = 0;
    s_grad_edge_weight[tid] = 0;

    for (unsigned int f_idx=tid; f_idx < f_dim; f_idx += blockSize){
        grad_out_buffer = grad_out[src_id][f_idx];
        s_src_feature[f_idx] = feature[src_id][f_idx];
        s_grad_feature[f_idx] = 0;
        s_grad_edge_weight[tid] += s_src_feature[f_idx] * grad_out_buffer;
    }
    __syncthreads();
    smem_reduce_v3<scalar_t, blockSize>(s_grad_edge_weight, tid);
    __syncthreads();


    for (unsigned int e_id=src_ptr[src_id]; e_id < src_ptr[src_id + 1]; e_id ++){
        // for each edge
        s_grad_edge_weight[tid] = 0;
        scalar_t weight = edge_weight[e_id];
        unsigned int tar_id = tar_index[e_id];
        for (unsigned int f_idx=tid; f_idx < f_dim; f_idx += blockSize){
            grad_out_buffer = grad_out[tar_id][f_idx];
            s_grad_feature[f_idx] += grad_out_buffer * weight;
            s_grad_edge_weight[tid] += grad_out_buffer * s_src_feature[f_idx];
        }
        __syncthreads();
        smem_reduce_v3<scalar_t, blockSize>(s_grad_edge_weight, tid);
        __syncthreads();
        if (tid == 0){
            grad_edge_weight[e_id] = s_grad_edge_weight[0];
        }
    }

    for (unsigned int f_idx = tid; f_idx < f_dim; f_idx += blockSize){
        grad_feature[src_id][f_idx] = s_grad_feature[f_idx];
    }
 }


 std::vector<torch::Tensor> fused_gar_b_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor tar_index,
    torch::Tensor src_ptr,
    torch::Tensor edge_weight,
    bool require_edge_weight
){
    unsigned int f_dim = grad_out.size(1);
    unsigned int num_node = grad_out.size(0);

    auto grad_feature = torch::empty_like(grad_out);
    auto grad_edge_weight = torch::empty_like(edge_weight);

    if (f_dim <= GAR_THREADS){
        unsigned int stride = GAR_THREADS / f_dim;
        if (require_edge_weight){
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "fused_gar_b_kernel", ([&]{
                fused_gar_b_kernelv2<scalar_t, GAR_THREADS><<<num_node, GAR_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    tar_index.data_ptr<int>(), src_ptr.data_ptr<int>(), grad_edge_weight.data_ptr<scalar_t>(),
                    edge_weight.data_ptr<scalar_t>(),
                    f_dim, stride
                );
            }));
        }else{
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "fused_gar_b_larger_kernel", ([&]{
                fused_gar_b_kernel<scalar_t, GAR_THREADS><<<num_node, GAR_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    tar_index.data_ptr<int>(), src_ptr.data_ptr<int>(), edge_weight.data_ptr<scalar_t>(),
                    f_dim, stride
                );
            }));
        }
    }else{
        if (require_edge_weight){
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "aggregation gar forward", ([&]{
                fused_gar_b_large_kernelv2<scalar_t, GAR_THREADS><<<num_node, GAR_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    tar_index.data_ptr<int>(), src_ptr.data_ptr<int>(), grad_edge_weight.data_ptr<scalar_t>(),
                    edge_weight.data_ptr<scalar_t>(),
                    f_dim
                );
            }));
        }else{
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "aggregation gar forward", ([&]{
                fused_gar_b_large_kernel<scalar_t, GAR_THREADS><<<num_node, GAR_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    tar_index.data_ptr<int>(), src_ptr.data_ptr<int>(), edge_weight.data_ptr<scalar_t>(),
                    f_dim
                );
            }));
        }
    }
    return {grad_feature, grad_edge_weight};
}

/***********************************************************************************************************/

/*
 * The fused forward kernel for GAS aggregator
 */

template <typename scalar_t>
__global__ void fused_gas_f_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> out,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int edge_stride, // each thread block handles how many edges
    const unsigned int f_dim, const unsigned int num_edge
){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid < f_dim * edge_stride){
        unsigned int f_idx = tid % f_dim;
        for (unsigned int e_id = bid * edge_stride + tid / f_dim; e_id < num_edge; e_id += gridDim.x * edge_stride){
            atomicAdd(&out[tar_index[e_id]][f_idx], feature[src_index[e_id]][f_idx] * edge_weight[e_id]);
        }
    }
}

template <typename scalar_t>
__global__ void fused_gas_f_large_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> out,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int f_dim, const unsigned int num_edge
){
    unsigned int e_id = blockIdx.x;
    scalar_t weight = edge_weight[e_id];
    for (unsigned int f_idx = threadIdx.x; f_idx < f_dim; f_idx += blockDim.x){
        atomicAdd(&out[tar_index[e_id]][f_idx], feature[src_index[e_id]][f_idx] * weight);
    }
}


#define BLOCKS(N, T) (N + T - 1)/T
#define GAS_THREADS 256

torch::Tensor fused_gas_f_cuda(
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight
){
    unsigned int f_dim = feature.size(1);
    unsigned int num_edge = edge_weight.size(0);

    auto out = torch::zeros_like(feature);

    if (f_dim <= GAS_THREADS){
        unsigned int stride = GAS_THREADS/ f_dim;
        AT_DISPATCH_FLOATING_TYPES(feature.type(), "fused_gas_f_kernel", ([&]{
            fused_gas_f_kernel<scalar_t><<<BLOCKS(num_edge, stride), GAS_THREADS>>>(
                feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
                edge_weight.data_ptr<scalar_t>(), stride, f_dim, num_edge
            );
        }));
    }else{
        AT_DISPATCH_FLOATING_TYPES(feature.type(), "fused_gas_f_kernel_g", ([&]{
            fused_gas_f_large_kernel<scalar_t><<<num_edge, GAS_THREADS>>>(
                feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
                edge_weight.data_ptr<scalar_t>(), f_dim, num_edge
            );
        }));
    }
    return out;
}



/*
 * The fused backward kernel for GAS aggregator
 */

 template <typename scalar_t>
__global__ void fused_gas_b_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int edge_stride,  // each thread block handles how many edges
    const unsigned int f_dim, const unsigned int num_edge
){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    if (tid < f_dim * edge_stride){
        unsigned int f_idx = tid % f_dim;  // which entry to handle
        for (unsigned int e_id = bid* edge_stride + tid / f_dim; e_id < num_edge; e_id += gridDim.x * edge_stride){
            atomicAdd(&grad_feature[src_index[e_id]][f_idx], grad_out[tar_index[e_id]][f_idx] * edge_weight[e_id]);
        }
    }
}



template <typename scalar_t, unsigned int blockSize>
__global__ void fused_gas_b_kernelv2(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    scalar_t* grad_edge_weight,
    const unsigned int edge_stride,  // each thread block handles how many edges
    const unsigned int f_dim, const unsigned int num_edge
){
    __shared__ scalar_t s_grad_edge_weight[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int group_id = tid / f_dim;
    scalar_t grad_out_buffer = 0;
    unsigned int reduce_len = f_dim;

    unsigned int e_start = bid * edge_stride;
    unsigned int e_bound = num_edge;
    unsigned int f_idx = tid % f_dim;
    unsigned int e_idx = e_start + group_id;
    unsigned stride = edge_stride * gridDim.x;

    unsigned int total_strides = (e_bound - e_start + stride- 1) / stride;
    s_grad_edge_weight[tid] = 0;

    for (unsigned int i=e_start; i < total_strides * stride + e_start; i += stride){
        __syncthreads();
        if ((tid < edge_stride * f_dim) && (e_idx < e_bound)){
            grad_out_buffer = grad_out[tar_index[e_idx]][f_idx];
            atomicAdd(&grad_feature[src_index[e_idx]][f_idx], grad_out_buffer * edge_weight[e_idx]);
            s_grad_edge_weight[group_id + f_idx * edge_stride] = grad_out_buffer * feature[src_index[e_idx]][f_idx];
        }
        __syncthreads();
        reduce_len = f_dim;
        smem_reduce_v1<scalar_t, blockSize>(s_grad_edge_weight, tid, reduce_len, edge_stride);
        __syncthreads();
        if ((tid < edge_stride) && (i + tid < e_bound)){
            grad_edge_weight[i + tid] = s_grad_edge_weight[tid];
        }
        s_grad_edge_weight[tid] = 0;
        e_idx += stride;
    }
}


template <typename scalar_t>
__global__ void fused_gas_b_large_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    const unsigned int f_dim, const unsigned int num_edge
){
    unsigned int e_id = blockIdx.x;
    scalar_t weight = edge_weight[e_id];
    unsigned int tid = threadIdx.x;

    for (unsigned int f_idx = tid; f_idx < f_dim; f_idx += blockDim.x){
        atomicAdd(&grad_feature[src_index[e_id]][f_idx], grad_out[tar_index[e_id]][f_idx] * weight);
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void fused_gas_b_large_kernelv2(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_feature,
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    scalar_t* grad_edge_weight,
    const unsigned int f_dim, const unsigned int num_edge
){
    __shared__ scalar_t s_grad_edge_weight[blockSize];

    unsigned int e_id = blockIdx.x;
    scalar_t weight = edge_weight[e_id];
    unsigned int tid = threadIdx.x;
    scalar_t grad_out_buffer = 0;

    s_grad_edge_weight[tid] = 0;

    for (unsigned int f_idx = tid; f_idx < f_dim; f_idx += blockDim.x){
        grad_out_buffer = grad_out[tar_index[e_id]][f_idx];
        s_grad_edge_weight[tid] += grad_out_buffer * feature[src_index[e_id]][f_idx];
        atomicAdd(&grad_feature[src_index[e_id]][f_idx], grad_out_buffer * weight);
    }
    __syncthreads();
    smem_reduce_v3<scalar_t, blockSize>(s_grad_edge_weight, tid);
    if (tid == 0){
        grad_edge_weight[e_id] = s_grad_edge_weight[0];
    }
}


std::vector<torch::Tensor> fused_gas_b_cuda(
    torch::Tensor grad_out,
    torch::Tensor feature,
    torch::Tensor src_index,
    torch::Tensor tar_index,
    torch::Tensor edge_weight,
    bool require_edge_weight
){
    unsigned int f_dim = grad_out.size(1);
    auto grad_feature = torch::zeros_like(grad_out);
    unsigned int num_edge = edge_weight.size(0);
    unsigned int num_node = feature.size(0);
    auto grad_edge_weight = torch::empty_like(edge_weight);

    // gradient from the neighbors (this includes self-loops if they're in the edge list)
    if (f_dim <= GAS_THREADS){
        unsigned int stride = GAS_THREADS / f_dim;
        if (require_edge_weight){
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "fused_gas_b_kernelv2", ([&]{
                fused_gas_b_kernelv2<scalar_t, GAS_THREADS><<<BLOCKS(num_edge, stride), GAS_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
                    edge_weight.data_ptr<scalar_t>(), grad_edge_weight.data_ptr<scalar_t>(),
                    stride, f_dim, num_edge
                );
            }));
        }else{
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "fused_gas_b_kernel", ([&]{
                fused_gas_b_kernel<scalar_t><<<BLOCKS(num_edge, stride), GAS_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
                    edge_weight.data_ptr<scalar_t>(), stride, f_dim, num_edge
                );
            }));
        }
    }else{
        if (require_edge_weight){
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "fused_gas_b_large_kernel", ([&]{
                fused_gas_b_large_kernelv2<scalar_t, GAS_THREADS><<<num_edge, GAS_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
                    edge_weight.data_ptr<scalar_t>(), grad_edge_weight.data_ptr<scalar_t>(),
                    f_dim, num_edge
                );
            }));
        }else{
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "fused_gas_b_large_kernel", ([&]{
                fused_gas_b_large_kernel<scalar_t><<<num_edge, GAS_THREADS>>>(
                    grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_feature.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
                    src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
                    edge_weight.data_ptr<scalar_t>(), f_dim, num_edge
                );
            }));
        }
    }

    return {grad_feature, grad_edge_weight};
}
