#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>



#define ROW_WINDOW_SIZE 16
#define TCB_WIDTH 8
#define TCB_SIZE (ROW_WINDOW_SIZE * TCB_WIDTH)  // 16 * 8 = 128

// Build WSB format from CSR adjacency (CPU implementation inside a CUDA extension).
// This mirrors the Python WSBFormat.build_wsb_format() implementation.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_wsb_format_cpu(torch::Tensor adj_csr, torch::Dtype dtype) {
    TORCH_CHECK(adj_csr.is_sparse_csr(), "Input must be sparse CSR tensor");
    TORCH_CHECK(adj_csr.dim() == 2, "Adjacency must be 2D");
    TORCH_CHECK(adj_csr.size(0) == adj_csr.size(1), "Adjacency must be square");

    auto device = adj_csr.device();

    // Work on CPU for simplicity & correctness, then move result back.
    auto adj = adj_csr.to(torch::kCPU);

    auto indptr = adj.crow_indices();   // int64
    auto indices = adj.col_indices();   // int64
    auto values = adj.values();         // usually float

    TORCH_CHECK(indptr.scalar_type() == torch::kInt64,
                "crow_indices must be int64");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64,
                "col_indices must be int64");

    if (values.scalar_type() != torch::kFloat32) {
        values = values.to(torch::kFloat32);
    }

    int64_t N = adj.size(0);
    int64_t num_edges = indices.size(0);

    const int num_row_windows =
        static_cast<int>((N + ROW_WINDOW_SIZE - 1) / ROW_WINDOW_SIZE);

    const int64_t* indptr_ptr   = indptr.data_ptr<int64_t>();
    const int64_t* indices_ptr  = indices.data_ptr<int64_t>();
    const float*   values_ptr   = values.data_ptr<float>();

    // Global WSB buffers (CPU-side)
    std::vector<int32_t> tcb_row_offset;
    tcb_row_offset.reserve(num_row_windows + 1);
    tcb_row_offset.push_back(0);  // offset[0] = 0

    std::vector<int32_t> all_col_idx;
    all_col_idx.reserve(static_cast<size_t>(num_edges)); // heuristic

    std::vector<uint64_t> all_bitmaps;
    all_bitmaps.reserve(static_cast<size_t>(2 * num_edges / TCB_SIZE + 2));

    std::vector<float> all_weights;
    all_weights.reserve(static_cast<size_t>(
        num_edges * (ROW_WINDOW_SIZE * 1.0 / TCB_SIZE) + TCB_SIZE));

    // Per-window temporary buffers (reused)
    std::vector<int64_t> cols_window;
    std::vector<int32_t> rows_window;
    std::vector<float>   weights_window;

    for (int rw = 0; rw < num_row_windows; ++rw) {
        int row_start = rw * ROW_WINDOW_SIZE;
        int row_end   = static_cast<int>(std::min<int64_t>(row_start + ROW_WINDOW_SIZE, N));
        int num_rows_in_window = row_end - row_start;

        int64_t edge_start = indptr_ptr[row_start];
        int64_t edge_end   = indptr_ptr[row_end];
        int64_t num_edges_window = edge_end - edge_start;

        if (num_edges_window == 0) {
            // No edges in this row window
            tcb_row_offset.push_back(tcb_row_offset.back());
            continue;
        }

        cols_window.clear();
        rows_window.clear();
        weights_window.clear();

        cols_window.reserve(static_cast<size_t>(num_edges_window));
        rows_window.reserve(static_cast<size_t>(num_edges_window));
        weights_window.reserve(static_cast<size_t>(num_edges_window));

        // === Gather edges in this row window (local_row, col, weight) ===
        for (int local_row = 0; local_row < num_rows_in_window; ++local_row) {
            int64_t global_row = row_start + local_row;
            int64_t e0 = indptr_ptr[global_row];
            int64_t e1 = indptr_ptr[global_row + 1];

            for (int64_t e = e0; e < e1; ++e) {
                cols_window.push_back(indices_ptr[e]);
                rows_window.push_back(local_row);
                weights_window.push_back(values_ptr[e]);
            }
        }

        // get unique columns and sort them
        std::vector<int64_t> unique_cols = cols_window;
        std::sort(unique_cols.begin(), unique_cols.end());
        unique_cols.erase(std::unique(unique_cols.begin(), unique_cols.end()),
                          unique_cols.end());

        int num_unique_cols = static_cast<int>(unique_cols.size());
        if (num_unique_cols == 0) {
            tcb_row_offset.push_back(tcb_row_offset.back());
            continue;
        }

        // number of TCBs for this row window
        int num_tcbs_in_rw = (num_unique_cols + TCB_WIDTH - 1) / TCB_WIDTH;
        int cur_tcb_base   = tcb_row_offset.back();
        tcb_row_offset.push_back(cur_tcb_base + num_tcbs_in_rw);

        // column -> global local index mapping (0..num_unique_cols-1)
        std::unordered_map<int64_t, int> col_to_idx;
        col_to_idx.reserve(static_cast<size_t>(num_unique_cols * 2));
        for (int i = 0; i < num_unique_cols; ++i) {
            col_to_idx.emplace(unique_cols[i], i);
        }

        // Per-TCB bitmaps & weights for this row window
        std::vector<uint64_t> bm_lo_rw(num_tcbs_in_rw, 0);
        std::vector<uint64_t> bm_hi_rw(num_tcbs_in_rw, 0);
        std::vector<float>    weights_rw(num_tcbs_in_rw * TCB_SIZE, 0.0f);

        // === Fill bitmaps and weights from edges ===
        int num_edges_window_int = static_cast<int>(cols_window.size());
        for (int e = 0; e < num_edges_window_int; ++e) {
            int32_t local_row = rows_window[e];
            int64_t col       = cols_window[e];
            float   w         = weights_window[e];

            auto it = col_to_idx.find(col);
            if (it == col_to_idx.end()) {
                continue;  // should not happen if col in unique_cols
            }

            int global_local_col   = it->second;          // index in unique_cols
            int tcb_idx_in_rw      = global_local_col / TCB_WIDTH;
            int local_col_in_tcb   = global_local_col % TCB_WIDTH;

            int bit_pos = (local_row % 8) * TCB_WIDTH + local_col_in_tcb;
            uint64_t mask = 1ull << static_cast<uint64_t>(bit_pos);

            if (local_row < 8) {
                bm_lo_rw[tcb_idx_in_rw] |= mask;
            } else {
                bm_hi_rw[tcb_idx_in_rw] |= mask;
            }

            int weight_idx = local_row * TCB_WIDTH + local_col_in_tcb;
            int base       = tcb_idx_in_rw * TCB_SIZE;
            weights_rw[base + weight_idx] = w;
        }

        // === Append TCB data to global arrays in the same order as Python code ===
        for (int tcb_idx = 0; tcb_idx < num_tcbs_in_rw; ++tcb_idx) {
            int col_start = tcb_idx * TCB_WIDTH;
            int col_end   = std::min(col_start + TCB_WIDTH, num_unique_cols);

            // Column indices [TCB_WIDTH], padded with 0
            for (int i = col_start; i < col_end; ++i) {
                all_col_idx.push_back(static_cast<int32_t>(unique_cols[i]));
            }
            for (int i = col_end; i < col_start + TCB_WIDTH; ++i) {
                all_col_idx.push_back(0);
            }

            // Bitmap: 2 x uint64 per TCB
            all_bitmaps.push_back(bm_lo_rw[tcb_idx]);
            all_bitmaps.push_back(bm_hi_rw[tcb_idx]);

            // Weights: [TCB_SIZE] per TCB
            int base = tcb_idx * TCB_SIZE;
            for (int k = 0; k < TCB_SIZE; ++k) {
                all_weights.push_back(weights_rw[base + k]);
            }
        }
    }

    // === Convert vectors to tensors (CPU) ===
    auto tcb_row_offset_tensor = torch::from_blob(
        tcb_row_offset.data(),
        {static_cast<int64_t>(tcb_row_offset.size())},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).clone();

    auto col_idx_tensor = torch::from_blob(
        all_col_idx.data(),
        {static_cast<int64_t>(all_col_idx.size())},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).clone();

    auto bitmap_tensor = torch::from_blob(
        all_bitmaps.data(),
        {static_cast<int64_t>(all_bitmaps.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    ).clone();

    auto weights_fp32_tensor = torch::from_blob(
        all_weights.data(),
        {static_cast<int64_t>(all_weights.size())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    auto weights_out = weights_fp32_tensor.to(dtype);

    // === Move result back to original device (CPU or CUDA) ===
    if (device.is_cuda()) {
        tcb_row_offset_tensor = tcb_row_offset_tensor.to(device);
        col_idx_tensor        = col_idx_tensor.to(device);
        bitmap_tensor         = bitmap_tensor.to(device);
        weights_out           = weights_out.to(device);
    }

    return std::make_tuple(tcb_row_offset_tensor,
                           col_idx_tensor,
                           bitmap_tensor,
                           weights_out);
}


// Count total edges per row-window
__global__ void count_edges_per_rw_kernel(
    const int64_t* __restrict__ indptr,
    int* __restrict__ edge_counts,
    int num_row_windows,
    int64_t num_nodes
) {
    int rw = blockIdx.x * blockDim.x + threadIdx.x;
    if (rw >= num_row_windows) return;

    int row_start = rw * ROW_WINDOW_SIZE;
    int row_end = row_start + ROW_WINDOW_SIZE;
    if (row_end > num_nodes) row_end = (int)num_nodes;

    int64_t edge_start = indptr[row_start];
    int64_t edge_end   = indptr[row_end];
    edge_counts[rw] = static_cast<int>(edge_end - edge_start);
}

// Extract edges per row-window into flat buffers
__global__ void extract_rw_edges_kernel(
    const int64_t* __restrict__ indptr,
    const int64_t* __restrict__ indices,
    const float*   __restrict__ weights,
    int* __restrict__ rw_cols_edges,
    int* __restrict__ rw_local_rows,
    float* __restrict__ rw_weights,
    const int* __restrict__ rw_edge_offsets,
    int num_row_windows,
    int64_t num_nodes
) {
    int rw = blockIdx.x;
    if (rw >= num_row_windows) return;

    int row_start = rw * ROW_WINDOW_SIZE;
    int row_end = row_start + ROW_WINDOW_SIZE;
    if (row_end > num_nodes) row_end = (int)num_nodes;

    int64_t edge_start = indptr[row_start];
    int64_t edge_end   = indptr[row_end];
    int num_edges = static_cast<int>(edge_end - edge_start);
    int out_offset = rw_edge_offsets[rw];

    for (int tid = threadIdx.x; tid < num_edges; tid += blockDim.x) {
        int64_t edge_idx = edge_start + tid;

        // find which row this edge belongs to (within this window)
        int global_row = row_start;
        while (global_row + 1 < row_end && indptr[global_row + 1] <= edge_idx) {
            ++global_row;
        }
        int local_row = global_row - row_start;

        int out_idx = out_offset + tid;
        rw_cols_edges[out_idx]   = static_cast<int>(indices[edge_idx]);
        rw_local_rows[out_idx]   = local_row;
        rw_weights[out_idx]      = weights[edge_idx];
    }
}

// Build TCB data (col_idx, bitmap, weights) from per-window edges & unique cols
__global__ void build_tcb_data_kernel(
    const int* __restrict__ unique_cols,        // [total_temp_edges]
    const int* __restrict__ unique_col_counts,  // [num_row_windows]
    const int* __restrict__ rw_edge_offsets,    // [num_row_windows + 1]
    const int* __restrict__ rw_cols_edges,      // [total_temp_edges]
    const float* __restrict__ rw_weights,       // [total_temp_edges]
    const int* __restrict__ rw_local_rows,      // [total_temp_edges]
    const int* __restrict__ tcb_row_offset_in,  // [num_row_windows + 1]
    int* __restrict__ col_idx_out,              // [num_tcbs * TCB_WIDTH]
    uint64_t* __restrict__ bitmap_out,          // [num_tcbs * 2]
    float* __restrict__ weights_out,            // [num_tcbs * TCB_SIZE]
    int num_row_windows
) {
    int rw = blockIdx.x;
    if (rw >= num_row_windows) return;

    int num_unique = unique_col_counts[rw];
    if (num_unique == 0) return;

    int num_tcbs = (num_unique + TCB_WIDTH - 1) / TCB_WIDTH;
    int tcb_base = tcb_row_offset_in[rw];

    int edge_start = rw_edge_offsets[rw];
    int edge_end   = rw_edge_offsets[rw + 1];

    // each thread handles one TCB in this row window
    for (int tcb_idx = threadIdx.x; tcb_idx < num_tcbs; tcb_idx += blockDim.x) {
        int global_tcb_idx = tcb_base + tcb_idx;

        int col_start = tcb_idx * TCB_WIDTH;
        int col_end   = min(col_start + TCB_WIDTH, num_unique);

        int edge_offset_base = edge_start;

        // write column indices
        for (int i = 0; i < TCB_WIDTH; ++i) {
            int out_idx = global_tcb_idx * TCB_WIDTH + i;
            if (col_start + i < num_unique) {
                col_idx_out[out_idx] =
                    unique_cols[edge_offset_base + col_start + i];
            } else {
                col_idx_out[out_idx] = 0;
            }
        }

        // init bitmap + weights
        bitmap_out[global_tcb_idx * 2 + 0] = 0;
        bitmap_out[global_tcb_idx * 2 + 1] = 0;

        for (int i = 0; i < TCB_SIZE; ++i) {
            weights_out[global_tcb_idx * TCB_SIZE + i] = 0.0f;
        }

        // local mapping: index in TCB -> global column id
        int tcb_cols[TCB_WIDTH];
        for (int i = 0; i < TCB_WIDTH; ++i) {
            if (col_start + i < num_unique) {
                tcb_cols[i] = unique_cols[edge_offset_base + col_start + i];
            } else {
                tcb_cols[i] = -1;
            }
        }

        // process edges of this rw and fill bitmap+weights
        for (int e = edge_start; e < edge_end; ++e) {
            int col        = rw_cols_edges[e];
            int local_row  = rw_local_rows[e];
            float weight   = rw_weights[e];

            // find column inside this TCB (linear search, TCB_WIDTH==8)
            int local_col_in_tcb = -1;
            for (int i = 0; i < TCB_WIDTH && col_start + i < num_unique; ++i) {
                if (tcb_cols[i] == col) {
                    local_col_in_tcb = i;
                    break;
                }
            }
            if (local_col_in_tcb < 0) continue;

            int bit_pos = (local_row % 8) * TCB_WIDTH + local_col_in_tcb;
            uint64_t mask = (uint64_t)1 << bit_pos;

            if (local_row < 8) {
                bitmap_out[global_tcb_idx * 2 + 0] |= mask;
            } else {
                bitmap_out[global_tcb_idx * 2 + 1] |= mask;
            }

            int weight_idx = local_row * TCB_WIDTH + local_col_in_tcb;
            weights_out[global_tcb_idx * TCB_SIZE + weight_idx] = weight;
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_wsb_format_cuda(torch::Tensor adj_csr, torch::Dtype dtype) {
    TORCH_CHECK(adj_csr.is_sparse_csr(), "adj_csr must be sparse CSR");
    TORCH_CHECK(adj_csr.is_cuda(), "adj_csr must be on CUDA for GPU builder");
    TORCH_CHECK(adj_csr.dim() == 2, "adj_csr must be 2D");
    TORCH_CHECK(adj_csr.size(0) == adj_csr.size(1), "adj_csr must be square");

    auto device = adj_csr.device();
    int64_t N = adj_csr.size(0);

    auto indptr  = adj_csr.crow_indices();  // int64, cuda
    auto indices = adj_csr.col_indices();   // int64, cuda
    auto values  = adj_csr.values();        // usually float or half

    TORCH_CHECK(indptr.scalar_type() == torch::kInt64,
                "crow_indices must be int64");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64,
                "col_indices must be int64");

    if (values.scalar_type() != torch::kFloat32) {
        values = values.to(torch::kFloat32);
    }

    int64_t num_edges = indices.size(0);
    int num_row_windows = static_cast<int>((N + ROW_WINDOW_SIZE - 1) / ROW_WINDOW_SIZE);

    const int64_t* indptr_ptr   = indptr.data_ptr<int64_t>();
    const int64_t* indices_ptr  = indices.data_ptr<int64_t>();
    const float*   values_ptr   = values.data_ptr<float>();

    auto options_i32_cuda = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_f32_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // === Step 1: count edges per row-window ===
    auto edge_counts = torch::empty({num_row_windows}, options_i32_cuda);
    auto rw_edge_offsets = torch::zeros({num_row_windows + 1}, options_i32_cuda);

    int threads = 256;
    int blocks  = (num_row_windows + threads - 1) / threads;

    count_edges_per_rw_kernel<<<blocks, threads>>>(
        indptr_ptr,
        edge_counts.data_ptr<int>(),
        num_row_windows,
        N
    );
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "count_edges_per_rw_kernel failed");

    // === prefix-sum to get rw_edge_offsets ===
    {
        thrust::device_ptr<int> edge_counts_ptr(edge_counts.data_ptr<int>());
        thrust::device_ptr<int> offsets_ptr(rw_edge_offsets.data_ptr<int>() + 1);
        // inclusive scan into offsets[1..]
        thrust::inclusive_scan(edge_counts_ptr,
                               edge_counts_ptr + num_row_windows,
                               offsets_ptr);
    }

    // copy offsets to CPU to get total_temp_edges & per-rw offsets
    auto rw_edge_offsets_cpu = rw_edge_offsets.to(torch::kCPU);
    auto rw_edge_offsets_cpu_ptr = rw_edge_offsets_cpu.data_ptr<int>();

    int64_t total_temp_edges = static_cast<int64_t>(
        rw_edge_offsets_cpu_ptr[num_row_windows]
    );

    // === Step 2: allocate per-edge buffers ===
    auto rw_cols_edges = torch::empty({total_temp_edges}, options_i32_cuda);
    auto unique_cols   = torch::empty({total_temp_edges}, options_i32_cuda);
    auto rw_local_rows = torch::empty({total_temp_edges}, options_i32_cuda);
    auto rw_weights    = torch::empty({total_temp_edges}, options_f32_cuda);

    // === Step 3: extract edges into buffers ===
    extract_rw_edges_kernel<<<num_row_windows, 256>>>(
        indptr_ptr,
        indices_ptr,
        values_ptr,
        rw_cols_edges.data_ptr<int>(),
        rw_local_rows.data_ptr<int>(),
        rw_weights.data_ptr<float>(),
        rw_edge_offsets.data_ptr<int>(),
        num_row_windows,
        N
    );
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "extract_rw_edges_kernel failed");

    // copy columns into unique_cols (we'll sort/unique in-place)
    unique_cols.copy_(rw_cols_edges);

    // === Step 4: Thrust sort + unique per row-window on GPU, build unique_col_counts & tcb_row_offset (on host) ===
    std::vector<int32_t> unique_col_counts_host(num_row_windows, 0);
    std::vector<int32_t> tcb_row_offset_host(num_row_windows + 1, 0);

    int* unique_cols_ptr = unique_cols.data_ptr<int>();

    int32_t total_tcbs = 0;

    for (int rw = 0; rw < num_row_windows; ++rw) {
        int start = rw_edge_offsets_cpu_ptr[rw];
        int end   = rw_edge_offsets_cpu_ptr[rw + 1];
        int count = end - start;

        if (count <= 0) {
            unique_col_counts_host[rw] = 0;
            tcb_row_offset_host[rw + 1] = total_tcbs;
            continue;
        }

        thrust::device_ptr<int> cols_begin(unique_cols_ptr + start);
        thrust::sort(cols_begin, cols_begin + count);

        auto new_end = thrust::unique(cols_begin, cols_begin + count);
        int unique_count = static_cast<int>(new_end - cols_begin);

        unique_col_counts_host[rw] = unique_count;

        int num_tcbs_in_rw =
            (unique_count + TCB_WIDTH - 1) / TCB_WIDTH;

        total_tcbs += num_tcbs_in_rw;
        tcb_row_offset_host[rw + 1] = total_tcbs;
    }

    // move unique_col_counts & tcb_row_offset to device
    auto unique_col_counts = torch::from_blob(
        unique_col_counts_host.data(),
        {static_cast<int64_t>(num_row_windows)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).clone().to(device);

    auto tcb_row_offset = torch::from_blob(
        tcb_row_offset_host.data(),
        {static_cast<int64_t>(num_row_windows + 1)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    ).clone().to(device);

    int64_t num_tcbs = total_tcbs;

    // === Step 5: allocate final WSB buffers on GPU ===
    auto col_idx = torch::empty({num_tcbs * TCB_WIDTH}, options_i32_cuda);
    auto bitmap  = torch::zeros({num_tcbs * 2},
                                torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto weights_fp32 = torch::empty({num_tcbs * TCB_SIZE}, options_f32_cuda);

    // === Step 6: build TCBs via kernel ===
    build_tcb_data_kernel<<<num_row_windows, 256>>>(
        unique_cols.data_ptr<int>(),
        unique_col_counts.data_ptr<int>(),
        rw_edge_offsets.data_ptr<int>(),
        rw_cols_edges.data_ptr<int>(),
        rw_weights.data_ptr<float>(),
        rw_local_rows.data_ptr<int>(),
        tcb_row_offset.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        reinterpret_cast<uint64_t*>(bitmap.data_ptr<int64_t>()),
        weights_fp32.data_ptr<float>(),
        num_row_windows
    );
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "build_tcb_data_kernel failed");

    // === Step 7: cast weights to requested dtype ===
    auto weights_out = weights_fp32.to(dtype);

    return std::make_tuple(tcb_row_offset, col_idx, bitmap, weights_out);
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "build_wsb_format_cpu",
        &build_wsb_format_cpu,
        "Build WSB block-sparse format from CSR adjacency (CPU implementation)"
    );
    m.def("build_wsb_format_cuda",
          &build_wsb_format_cuda,
          "Build WSB format (GPU builder using CSR on CUDA)");
}
