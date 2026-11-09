#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "common/row_kernels.cuh"
#include "reduction/reduction_backward.cuh"
#include "reduction/reduction_launch.cuh"

// =============================================================================
// reduction_aggr host side: validation and the launches. All work goes to the caller's current
// stream on X's device.
//
// Forward: the walk of reduction/reduction_launch.cuh with reduction_aggr's message, the neighbour's
// row. Backward (min/max): one scatter over grad_out.
// =============================================================================

namespace {

namespace red  = turbo_gnn::reduction;
namespace rows = turbo_gnn::rows;

// Longest feature row: the slice merge's grid covers a row in at most 65535 blocks of
// rows::kMergeThreads vectors, and 2^22 leaves room for any vector width.
constexpr int64_t kMaxFeatureDim = int64_t{1} << 22;

ReductionOp reduce_op(std::string const& reduce) {
    if (reduce == "min") {
        return ReductionOp::MIN;
    }
    if (reduce == "max") {
        return ReductionOp::MAX;
    }
    TORCH_CHECK(reduce == "sum", "reduction_aggr: unsupported reduce '", reduce, "'; expected 'min', 'max' or 'sum'");
    return ReductionOp::SUM;
}

void check_values(at::Tensor const& t, char const *what) {
    TORCH_CHECK(t.is_cuda() && t.dim() == 2, "reduction_aggr: ", what, " must be a 2-D CUDA tensor");
    TORCH_CHECK(
        t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16, "reduction_aggr: ", what,
        " must be float32, float16 or bfloat16, got ", t.scalar_type()
    );
    TORCH_CHECK(t.size(1) <= kMaxFeatureDim, "reduction_aggr: feature dim ", t.size(1), " exceeds the supported ", kMaxFeatureDim);
}

void check_csr(at::Tensor const& row_ptr, at::Tensor const& col_idx, at::Device device) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "reduction_aggr: edge_ptr and edge_idx must be CUDA tensors");
    TORCH_CHECK(row_ptr.device() == device && col_idx.device() == device, "reduction_aggr: the CSR must be on the device of the features");
    TORCH_CHECK(row_ptr.is_contiguous() && col_idx.is_contiguous(), "reduction_aggr: edge_ptr and edge_idx must be contiguous");
    TORCH_CHECK(is_supported_index_type(row_ptr.scalar_type()), "reduction_aggr: index tensors must be int32, int64, uint32 or uint64");
    TORCH_CHECK(col_idx.scalar_type() == row_ptr.scalar_type(), "reduction_aggr: edge_idx must have the dtype of edge_ptr");
    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.numel() >= 1, "reduction_aggr: edge_ptr must be [N + 1]");
    TORCH_CHECK(col_idx.dim() == 1, "reduction_aggr: edge_idx must be [E]");
    rows::check_index_range(row_ptr, col_idx, "reduction_aggr");
}

}  // namespace

std::vector<at::Tensor> reduction_aggr_forward_cuda(
    at::Tensor const& edge_ptr,
    at::Tensor const& edge_idx,
    at::Tensor X,
    at::Tensor const& light_nodes,
    at::Tensor const& heavy_nodes,
    int warps_per_block,
    int64_t slice_size,
    std::string const& reduce,
    at::Tensor const& chunk_node,
    at::Tensor const& chunk_start,
    at::Tensor const& node_chunk_offset,
    int64_t light_degree_bound
) {
    check_values(X, "X");
    const at::cuda::CUDAGuard device_guard(X.device());
    check_csr(edge_ptr, edge_idx, X.device());
    rows::check_node_list(light_nodes, edge_ptr, "light_nodes");
    rows::check_node_list(heavy_nodes, edge_ptr, "heavy_nodes");
    TORCH_CHECK(warps_per_block >= 1, "reduction_aggr: warps_per_block must be positive, got ", warps_per_block);
    TORCH_CHECK(
        slice_size >= 1 && slice_size <= std::numeric_limits<int32_t>::max(),
        "reduction_aggr: edges_per_block_heavy_nodes must be positive, got ", slice_size
    );
    TORCH_CHECK(heavy_nodes.numel() <= std::numeric_limits<int32_t>::max(), "reduction_aggr: too many heavy nodes");

    const ReductionOp op  = reduce_op(reduce);
    const bool tracks_arg = op != ReductionOp::SUM;
    const int64_t D       = X.size(1);
    // Elements must be unit-strided, and the row stride fit the forward's 32-bit gather offsets.
    if ((D > 1 && X.stride(1) != 1) || X.stride(0) > std::numeric_limits<uint32_t>::max()) {
        X = X.contiguous();
    }

    const red::Slices slices = red::slices_for(edge_ptr, heavy_nodes, slice_size, chunk_node, chunk_start, node_chunk_offset, "reduction_aggr");
    const red::ForwardLaunch L = red::plan_forward(
        edge_ptr, edge_idx, light_nodes, heavy_nodes, slices, D, X.element_size(), rows::rows_16b_aligned(X), op, warps_per_block,
        light_degree_bound, X.options()
    );
    if (L.items == 0) {
        return {L.out, L.arg};
    }

    std::visit(
        [&](auto idx_c, auto type_c, auto op_c) {
            using index_t            = typename decltype(idx_c)::Type;
            using cuda_t             = typename decltype(type_c)::CudaType;
            constexpr ReductionOp OP = decltype(op_c)::value;

            red::ForwardParams<cuda_t, index_t> p{};
            p.X        = reinterpret_cast<cuda_t const *>(X.data_ptr());
            p.stride_x = X.stride(0);
            L.fill(p, edge_ptr, edge_idx, light_nodes, heavy_nodes, slice_size, tracks_arg);
            red::run_forward<OP, red::NodeRows>(L, p, X.element_size(), "reduction_aggr");
        },
        MakeIndexWidthVariant(edge_ptr.scalar_type()), MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
        MakeEnumVariant<ReductionOp, ReductionOp::MIN, ReductionOp::MAX, ReductionOp::SUM>(op)
    );
    CUDA_KERNEL_CHECK();
    return {L.out, L.arg};
}

at::Tensor reduction_aggr_backward_cuda(at::Tensor grad_out, at::Tensor const& arg_idx, int64_t num_src_nodes) {
    check_values(grad_out, "grad_out");
    const at::cuda::CUDAGuard device_guard(grad_out.device());
    TORCH_CHECK(arg_idx.is_cuda() && arg_idx.device() == grad_out.device(), "reduction_aggr: arg_idx must be on the device of grad_out");
    TORCH_CHECK(is_supported_index_type(arg_idx.scalar_type()), "reduction_aggr: arg_idx must be int32, int64, uint32 or uint64");
    TORCH_CHECK(arg_idx.sizes() == grad_out.sizes(), "reduction_aggr: arg_idx and grad_out shapes must match");
    TORCH_CHECK(arg_idx.is_contiguous(), "reduction_aggr: arg_idx must be contiguous");
    TORCH_CHECK(num_src_nodes >= 0, "reduction_aggr: num_src_nodes must be non-negative");

    // Autograd hands over whatever layout the next op produced -- out.sum() passes an expanded
    // tensor of zero strides. Rows may have any stride; elements must be unit-strided.
    const int64_t D = grad_out.size(1);
    if (D > 1 && grad_out.stride(1) != 1) {
        grad_out = grad_out.contiguous();
    }
    // Accumulated in fp32 (see arg_scatter_kernel) and cast to the input dtype at the end.
    at::Tensor grad_x  = at::zeros({num_src_nodes, D}, grad_out.options().dtype(at::kFloat));
    const int64_t rows = grad_out.size(0);
    if (rows == 0 || D == 0 || num_src_nodes == 0) {
        return grad_x.to(grad_out.scalar_type());
    }

    // One thread per element, grid-striding only past 2^31 - 1 blocks.
    const int64_t total   = rows * D;
    const int64_t blocks  = std::min<int64_t>(ceil_div(total, int64_t{red::kScatterThreads}), std::numeric_limits<int32_t>::max());
    const cudaStream_t st = at::cuda::getCurrentCUDAStream();

    std::visit(
        [&](auto idx_c, auto type_c) {
            using index_t = typename decltype(idx_c)::Type;
            using cuda_t  = typename decltype(type_c)::CudaType;
            red::arg_scatter_kernel<cuda_t, index_t><<<static_cast<unsigned>(blocks), red::kScatterThreads, 0, st>>>(
                reinterpret_cast<cuda_t const *>(grad_out.data_ptr()), grad_out.stride(0), index_ptr<index_t>(arg_idx),
                grad_x.data_ptr<float>(), rows, static_cast<uint32_t>(D)
            );
        },
        MakeIndexWidthVariant(arg_idx.scalar_type()), MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type())
    );
    CUDA_KERNEL_CHECK();
    return grad_x.to(grad_out.scalar_type());
}
