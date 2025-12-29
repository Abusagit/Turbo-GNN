#include "torch_minimal.h"


#define THREADS 256
#define BLOCKS(N, T) (N + T - 1)/T

// when using CSR/CSC format
template <typename scalar_t>
__global__ void from_ptr(
    int* __restrict__ tar_ptr,
    scalar_t* __restrict__ degree,
    int num_nodes
){
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_nodes; tid += stride){
        degree[tid] = tar_ptr[tid + 1] - tar_ptr[tid];
    }
}


template <typename scalar_t, unsigned int blockSize>
__device__ void smem_reduction(volatile scalar_t* sdata, unsigned int tid){
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
        if (blockSize >= 64)sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32)sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16)sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)sdata[tid] += sdata[tid + 1];
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void from_weight(
    scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ degree,
    int* __restrict__ tar_ptr
){
    __shared__ scalar_t deg[blockSize];
    unsigned int tid = threadIdx.x;
    deg[tid] = 0;
    unsigned int tar_id = blockIdx.x;
    for (unsigned int e_idx = tar_ptr[tar_id] + threadIdx.x; e_idx < tar_ptr[tar_id + 1]; e_idx += blockDim.x){
        deg[tid] += edge_weight[e_idx];
    }
    __syncthreads();
    smem_reduction<scalar_t, blockSize>(deg, tid);
    if (tid == 0) degree[tar_id] = deg[0];
}

// When using COO format
template <typename scalar_t>
__global__ void scatter_add(
    scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ degree,
    int* __restrict__ index,
    unsigned int num_edge
){
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += stride){
        atomicAdd(&degree[index[tid]], edge_weight[tid]);
    }
}

template <typename scalar_t>
__global__ void clamp_min_kernel(scalar_t* data, scalar_t min_val, int size) {
    for(unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += gridDim.x * blockDim.x){
        if (data[tid] < min_val) data[tid] = min_val;
    }
}

torch::Tensor get_degree_cuda(
    torch::Tensor tar_ptr,
    torch::Tensor src_index,
    torch::optional<torch::Tensor> optional_edge_weight,
    int num_nodes
){
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tar_ptr.device());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor degree;

    if (optional_edge_weight.has_value()){
        torch::Tensor edge_weight;
        edge_weight = optional_edge_weight.value().contiguous();
        degree = torch::zeros({num_nodes,}, options);

        // Count only from tar_ptr (incoming edges for each node)
        AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from ptr", ([&]{
            from_ptr<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
                tar_ptr.data_ptr<int>(), degree.data_ptr<scalar_t>(), num_nodes
            );
        }));

        // Clamp zeros to ones
        AT_DISPATCH_FLOATING_TYPES(degree.type(), "clamp min", ([&]{
            clamp_min_kernel<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
                degree.data_ptr<scalar_t>(), 1.0, num_nodes);
        }));
    }
    else {
        degree = torch::empty({num_nodes,}, options);
        AT_DISPATCH_FLOATING_TYPES(degree.type(), "get degree from ptr", ([&]{
            from_ptr<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
                tar_ptr.data_ptr<int>(), degree.data_ptr<scalar_t>(), num_nodes
            );
        }));
    }

    return degree;
}


template <typename scalar_t>
__global__ void update_weight(
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ out_edge_weight,
    const scalar_t* __restrict__ degree,
    int num_edge
){
    for(unsigned int tid=blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += gridDim.x * blockDim.x){
        scalar_t res = edge_weight[tid] / sqrtf(degree[src_index[tid]])/sqrtf(degree[tar_index[tid]]);
        if (isinf(res)) res = 0;
        out_edge_weight[tid] = res;
    }
}

template <typename scalar_t>
__global__ void update_weight_v2(
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    const scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ out_edge_weight,
    const scalar_t* __restrict__ src_degree,  // outgoing degrees
    const scalar_t* __restrict__ tar_degree,  // incoming degrees
    int num_edge
){
    for(unsigned int tid=blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += gridDim.x * blockDim.x){
        scalar_t res = edge_weight[tid] / sqrtf(src_degree[src_index[tid]]) / sqrtf(tar_degree[tar_index[tid]]);
        if (isinf(res)) res = 0;
        out_edge_weight[tid] = res;
    }
}


template <typename scalar_t>
__global__ void get_weight(
    const int* __restrict__ src_index,
    const int* __restrict__ tar_index,
    scalar_t* __restrict__ edge_weight,
    scalar_t* __restrict__ degree,
    int num_edge
){
    for(unsigned int tid=blockIdx.x * blockDim.x + threadIdx.x; tid < num_edge; tid += gridDim.x * blockDim.x){
        scalar_t res = 1/sqrtf(degree[src_index[tid]] * degree[tar_index[tid]]);
        if (isinf(res)) res = 0;
        edge_weight[tid] = res;
    }
}


// CUDA Edge processing declaration

torch::Tensor gcn_gar_edge_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_ptr,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src_index.device());

    // Compute TWO separate degree arrays
    torch::Tensor src_degree = torch::zeros({num_nodes,}, options);  // outgoing degrees
    torch::Tensor tar_degree = torch::zeros({num_nodes,}, options);  // incoming degrees

    unsigned int Ne = src_index.size(0);
    torch::Tensor edge_weight;

    if (optional_edge_weight.has_value()){
        edge_weight = optional_edge_weight.value().contiguous();
    } else {
        edge_weight = torch::ones({Ne,}, options);
    }

    // Count outgoing degrees (COUNT edges, not sum weights)
    torch::Tensor ones = torch::ones({Ne,}, options);
    AT_DISPATCH_FLOATING_TYPES(src_degree.type(), "get src degree", ([&]{
        scatter_add<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
            ones.data_ptr<scalar_t>(),  // Changed: use ones instead of edge_weight
            src_degree.data_ptr<scalar_t>(),
            src_index.data_ptr<int>(), Ne);
    }));

    // Count incoming degrees (from tar_ptr - this counts edges, not weights)
    AT_DISPATCH_FLOATING_TYPES(tar_degree.type(), "get tar degree", ([&]{
        from_ptr<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
            tar_ptr.data_ptr<int>(), tar_degree.data_ptr<scalar_t>(), num_nodes
        );
    }));

    // Clamp both to minimum 1
    AT_DISPATCH_FLOATING_TYPES(src_degree.type(), "clamp src min", ([&]{
        clamp_min_kernel<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
            src_degree.data_ptr<scalar_t>(), 1.0, num_nodes);
    }));

    AT_DISPATCH_FLOATING_TYPES(tar_degree.type(), "clamp tar min", ([&]{
        clamp_min_kernel<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
            tar_degree.data_ptr<scalar_t>(), 1.0, num_nodes);
    }));

    // Compute normalized edge weights
    auto out_edge_weight = torch::empty_like(edge_weight);
    AT_DISPATCH_FLOATING_TYPES(edge_weight.type(), "update weight", ([&]{
        update_weight_v2<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
            src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
            edge_weight.data_ptr<scalar_t>(),
            out_edge_weight.data_ptr<scalar_t>(),
            src_degree.data_ptr<scalar_t>(),
            tar_degree.data_ptr<scalar_t>(),
            Ne
        );
    }));

    // For self-edge weights, use incoming degree (tar_degree)
    return out_edge_weight;
}


// CUDA Edge processing declaration
torch::Tensor gcn_gas_edge_weight_cuda(
    torch::Tensor src_index,
    torch::Tensor tar_index,
    int num_nodes,
    torch::optional<torch::Tensor> optional_edge_weight
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unsigned int Ne = src_index.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(src_index.device());

    // Get edge weights
    torch::Tensor edge_weight;
    if (optional_edge_weight.has_value()){
        edge_weight = optional_edge_weight.value().contiguous();
    } else {
        edge_weight = torch::ones({Ne,}, options);
    }

    // Compute TWO separate degree arrays
    torch::Tensor src_degree = torch::zeros({num_nodes,}, options);  // outgoing degrees
    torch::Tensor tar_degree = torch::zeros({num_nodes,}, options);  // incoming degrees

    // Count edges (not sum weights!)
    torch::Tensor ones = torch::ones({Ne,}, options);

    // Count outgoing degrees (scatter over src_index)
    AT_DISPATCH_FLOATING_TYPES(src_degree.type(), "get src degree", ([&]{
        scatter_add<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
            ones.data_ptr<scalar_t>(),
            src_degree.data_ptr<scalar_t>(),
            src_index.data_ptr<int>(), Ne);
    }));

    // Count incoming degrees (scatter over tar_index)
    AT_DISPATCH_FLOATING_TYPES(tar_degree.type(), "get tar degree", ([&]{
        scatter_add<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
            ones.data_ptr<scalar_t>(),
            tar_degree.data_ptr<scalar_t>(),
            tar_index.data_ptr<int>(), Ne);
    }));

    // Clamp both to minimum 1
    AT_DISPATCH_FLOATING_TYPES(src_degree.type(), "clamp src min", ([&]{
        clamp_min_kernel<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
            src_degree.data_ptr<scalar_t>(), 1.0, num_nodes);
    }));

    AT_DISPATCH_FLOATING_TYPES(tar_degree.type(), "clamp tar min", ([&]{
        clamp_min_kernel<scalar_t><<<BLOCKS(num_nodes, THREADS), THREADS, 0, stream>>>(
            tar_degree.data_ptr<scalar_t>(), 1.0, num_nodes);
    }));

    // Compute normalized edge weights using separate degrees
    auto out_edge_weight = torch::empty_like(edge_weight);
    AT_DISPATCH_FLOATING_TYPES(edge_weight.type(), "update weight", ([&]{
        update_weight_v2<scalar_t><<<BLOCKS(Ne, THREADS), THREADS, 0, stream>>>(
            src_index.data_ptr<int>(), tar_index.data_ptr<int>(),
            edge_weight.data_ptr<scalar_t>(),
            out_edge_weight.data_ptr<scalar_t>(),
            src_degree.data_ptr<scalar_t>(),
            tar_degree.data_ptr<scalar_t>(),
            Ne
        );
    }));

    return out_edge_weight;
}
