#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void min_aggr_forward(
    const int* edge_ptr,
    const int* edge_idx,
    const float* X,
    float* out,
    int* argmin,
    int num_nodes,
    int d
);

__global__ void min_aggr_backward(
    const float* grad_out,
    const int* argmin,
    float* grad_x,
    int num_nodes,
    int d
);

int main() {
    int num_nodes = 3;
    int d = 2;

    // 0 <- 0,1
    // 1 <- 1,2
    // 2 <- 0

    //
    std::vector<int> h_edge_ptr  = {0, 2, 4, 5};
    std::vector<int> h_edge_idx = {0, 1, 1, 2, 0};

    std::vector<float> h_X = {
        3.f,  1.f,
        2.f,  4.f,
        5.f, -1.f
    };
    std::vector<float> h_grad_out = {
        10.f, 20.f,
        30.f, 40.f,
        50.f, 60.f
    };

    int *d_edge_ptr = nullptr;
    int *d_edge_idx = nullptr;
    float *d_X = nullptr;
    float *d_grad_out = nullptr;

    cudaMalloc(&d_edge_ptr, h_edge_ptr.size() * sizeof(int));
    cudaMemcpy(d_edge_ptr, h_edge_ptr.data(), h_edge_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_edge_idx, h_edge_idx.size() * sizeof(int));
    cudaMemcpy(d_edge_idx, h_edge_idx.data(), h_edge_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_X, h_X.size() * sizeof(float));
    cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_grad_out, h_grad_out.size() * sizeof(float));
    cudaMemcpy(d_grad_out, h_grad_out.data(), h_grad_out.size() * sizeof(float), cudaMemcpyHostToDevice);

    float *d_out = nullptr;
    float *d_grad_x = nullptr;
    int *d_argmin = nullptr;

    cudaMalloc(&d_out, num_nodes * d * sizeof(float));
    cudaMalloc(&d_argmin, num_nodes * d * sizeof(int));
    cudaMalloc(&d_grad_x, num_nodes * d * sizeof(float));
    cudaMemset(d_grad_x, 0, num_nodes * d * sizeof(float));

    {
        int THREADS = 64;
        dim3 blocks(num_nodes);
        dim3 threads(THREADS);

        min_aggr_forward<<<blocks, threads>>>(
            d_edge_ptr, d_edge_idx, d_X, d_out, d_argmin, num_nodes, d
        );
        min_aggr_backward<<<blocks, threads>>>(
            d_grad_out, d_argmin, d_grad_x, num_nodes, d
        );
        cudaDeviceSynchronize();
    }

    std::vector<float> h_out(num_nodes * d);
    std::vector<int> h_argmin(num_nodes * d);
    std::vector<float> h_grad_x(num_nodes * d);

    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_argmin.data(), d_argmin, h_argmin.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_x.data(), d_grad_x, h_grad_x.size() * sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << "\n=== forward out ===\n";
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << "node " << i << ": ";
        for (int f = 0; f < d; ++f) {
            std::cout << h_out[i * d + f] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== argmin ===\n";
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << "node " << i << ": ";
        for (int f = 0; f < d; ++f) {
            std::cout << h_argmin[i * d + f] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== grad ===\n";
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << "node " << i << ": ";
        for (int f = 0; f < d; ++f) {
            std::cout << h_grad_x[i * d + f] << " ";
        }
        std::cout << "\n";
    }


    cudaFree(d_edge_ptr);
    cudaFree(d_edge_idx);
    cudaFree(d_X);
    cudaFree(d_grad_out);
    cudaFree(d_out);
    cudaFree(d_argmin);
    cudaFree(d_grad_x);

    return 0;
}


// вывести ручки в плюсах и достать в питоне через load (см cusparse backend)
// сравнить с  min_aggr в dgl (посмотреть в какую сторону текут сообщения -- )
