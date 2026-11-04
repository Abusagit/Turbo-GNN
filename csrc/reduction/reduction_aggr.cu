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
#include "reduction/reduction_forward.cuh"

// =============================================================================
// reduction_aggr host side: validation, the slice table of the heavy bucket, and the launches. All
// work goes to the caller's current stream on X's device.
//
// Forward: one launch over every work item -- the heavy bucket's slices first, being the longest,
// then the light rows -- and, when there are slices, one merge folding them into their rows.
// Backward (min/max): one scatter over grad_out.
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

// Signed view of an index tensor, for the torch ops below (they have no unsigned kernels).
at::Tensor signed_view(at::Tensor const& t) {
    if (t.scalar_type() == c10::ScalarType::UInt32) {
        return t.view(at::kInt);
    }
    if (t.scalar_type() == c10::ScalarType::UInt64) {
        return t.view(at::kLong);
    }
    return t;
}

struct Slices {
    at::Tensor chunk_node;         // [num] int32: the slice's slot in the heavy bucket
    at::Tensor chunk_start;        // [num] int32: its first edge, relative to the row
    at::Tensor node_chunk_offset;  // [num_heavy + 1] int32: slot k owns slices [offset[k], offset[k + 1])
    int64_t num = 0;
};

// The heavy bucket cut into slices of `slice` edges, the table
// AdjacencyForwardBackwardWithNodeBuckets.heavy_edge_slices builds (and caches per graph): a row
// with no edges keeps one empty slice, so that the merge still writes it. Costs one host sync, for
// the slice count.
Slices build_slices(at::Tensor const& row_ptr, at::Tensor const& heavy, int64_t slice) {
    const int64_t num_heavy = heavy.numel();
    const at::Tensor ptr    = signed_view(row_ptr);
    const at::Tensor h      = signed_view(heavy).to(at::kLong);
    const at::Tensor deg    = (ptr.index_select(0, h + 1) - ptr.index_select(0, h)).to(at::kLong);
    const at::Tensor counts = (deg + (slice - 1)).div(slice, "floor").clamp_min(1);

    at::Tensor offset = at::zeros({num_heavy + 1}, h.options());
    offset.slice(0, 1).copy_(counts.cumsum(0));
    const int64_t num = offset[num_heavy].item<int64_t>();
    TORCH_CHECK(num <= std::numeric_limits<int32_t>::max(), "reduction_aggr: too many heavy-row slices (", num, "); raise the slice size");

    // The indices overload (slot k repeated counts[k] times); a plain int64_t would select the one
    // repeating every element of `counts` that many times.
    const at::Tensor chunk_node  = at::repeat_interleave(counts, std::optional<int64_t>(num));
    const at::Tensor chunk_start = (at::arange(num, h.options()) - offset.index_select(0, chunk_node)) * slice;
    return {chunk_node.to(at::kInt), chunk_start.to(at::kInt), offset.to(at::kInt), num};
}

Slices slices_for(
    at::Tensor const& row_ptr, at::Tensor const& heavy, int64_t slice, at::Tensor const& chunk_node, at::Tensor const& chunk_start,
    at::Tensor const& node_chunk_offset
) {
    if (heavy.numel() == 0) {
        return {};
    }
    if (!chunk_node.defined() || chunk_node.numel() == 0) {
        return build_slices(row_ptr, heavy, slice);
    }
    for (at::Tensor const *t : {&chunk_node, &chunk_start, &node_chunk_offset}) {
        TORCH_CHECK(
            t->is_cuda() && t->device() == row_ptr.device() && t->is_contiguous() && t->scalar_type() == at::kInt && t->dim() == 1,
            "reduction_aggr: the slice table must be contiguous 1-D int32 tensors on the CSR's device"
        );
    }
    TORCH_CHECK(chunk_start.numel() == chunk_node.numel(), "reduction_aggr: chunk_node and chunk_start must have the same length");
    TORCH_CHECK(node_chunk_offset.numel() == heavy.numel() + 1, "reduction_aggr: node_chunk_offset must be [num_heavy + 1]");
    return {chunk_node, chunk_start, node_chunk_offset, chunk_node.numel()};
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
    const int64_t n_dst   = edge_ptr.numel() - 1;
    const int64_t D       = X.size(1);
    // Elements must be unit-strided, and the row stride fit the forward's 32-bit gather offsets.
    if ((D > 1 && X.stride(1) != 1) || X.stride(0) > std::numeric_limits<uint32_t>::max()) {
        X = X.contiguous();
    }

    // One output row per CSR row. Accumulating reducers have no arg: at [N, D] of the index dtype
    // it would cost as much as the output itself.
    at::Tensor out = at::empty({n_dst, D}, X.options());
    at::Tensor arg = tracks_arg ? at::empty({n_dst, D}, edge_ptr.options()) : at::empty({0}, edge_ptr.options());

    const int64_t num_light = light_nodes.numel();
    const Slices slices     = slices_for(edge_ptr, heavy_nodes, slice_size, chunk_node, chunk_start, node_chunk_offset);
    if (n_dst == 0 || D == 0 || num_light + slices.num == 0) {
        return {out, arg};
    }

    // Row layout: 16-byte vectors when every row of X allows them (out and the partials, fresh
    // allocations, then do as well), single elements otherwise.
    const bool wide       = rows::rows_16b_aligned(X);
    const uint32_t vec    = wide ? static_cast<uint32_t>(16 / X.element_size()) : 1u;
    const uint32_t nvec   = static_cast<uint32_t>(D) / vec;
    const uint32_t passes = rows::sweep_passes(rows::plan_sweep_shape(D, X.element_size(), wide), nvec);

    // Heavy rows of more than one slice: per-slice partial sums, or one packed min/max candidate per
    // element (see red::pack_candidate), starting out empty. A row of one slice is written in place,
    // and when every heavy row is (num_slices == num_heavy) nothing needs merging at all.
    const int64_t num_heavy = heavy_nodes.numel();
    const bool merge        = slices.num > num_heavy;
    at::Tensor part, packed;
    if (merge && tracks_arg) {
        packed = op == ReductionOp::MIN ? at::full({num_heavy, D}, -1, X.options().dtype(at::kLong))
                                        : at::zeros({num_heavy, D}, X.options().dtype(at::kLong));
    } else if (merge) {
        part = at::empty({slices.num, D}, X.options().dtype(at::kFloat));
    }
    const int warps = std::min(warps_per_block, static_cast<int>(red::kForwardMaxWarps));
    // The build for whole-warp rows: unbounded when the longest light row -- one warp's walk -- is at
    // least 1.5 times each resident warp's share of all the edges, so that the finish waits on long
    // rows rather than on throughput (see kForwardMinBlocks). Measured on real graphs: tolokers-2
    // (ratio 2.2-6.7) and hm-categories at D = 128 (2.5) ran 14-28% faster unbounded; hm-categories
    // at D = 256 (1.2) 3% faster; flickr, ogbn-arxiv, twitch-views, reddit and ogbn-products (0.03-
    // 0.57) 1-9% slower. An unknown bound keeps the bounded build.
    cudaDeviceProp const *const props = at::cuda::getCurrentDeviceProperties();
    const int64_t resident_warps      = int64_t{props->multiProcessorCount} * (props->maxThreadsPerMultiProcessor / kWarpSize);
    const bool long_rows              = light_degree_bound > 0 && 2 * light_degree_bound * resident_warps >= 3 * edge_idx.numel() * passes;
    const int64_t items               = (slices.num + num_light) * passes;
    const cudaStream_t stream         = at::cuda::getCurrentCUDAStream();

    std::visit(
        [&](auto idx_c, auto type_c, auto op_c) {
            using index_t            = typename decltype(idx_c)::Type;
            using cuda_t             = typename decltype(type_c)::CudaType;
            constexpr ReductionOp OP = decltype(op_c)::value;

            red::ForwardParams<cuda_t, index_t> p{};
            p.X           = reinterpret_cast<cuda_t const *>(X.data_ptr());
            p.stride_x    = X.stride(0);
            p.D           = static_cast<uint32_t>(D);
            p.nvec        = nvec;
            p.passes      = passes;
            p.row_ptr     = index_ptr<index_t>(edge_ptr);
            p.col_idx     = index_ptr<index_t>(edge_idx);
            p.num_slices  = slices.num;
            p.num_light   = num_light;
            p.light_nodes = num_light > 0 ? index_ptr<index_t>(light_nodes) : nullptr;
            p.out         = reinterpret_cast<cuda_t *>(out.data_ptr());
            p.arg         = tracks_arg ? index_ptr_mut<index_t>(arg) : nullptr;
            if (slices.num > 0) {
                p.heavy_nodes = index_ptr<index_t>(heavy_nodes);
                p.chunk_node  = slices.chunk_node.data_ptr<int>();
                p.chunk_start = slices.chunk_start.data_ptr<int>();
                p.slice_size  = static_cast<uint32_t>(slice_size);
                p.part        = part.defined() ? part.data_ptr<float>() : nullptr;
                p.packed      = packed.defined() ? reinterpret_cast<unsigned long long *>(packed.data_ptr<int64_t>()) : nullptr;
            }

            rows::with_sweep_shape(D, X.element_size(), wide, [&](auto shape_c) {
                constexpr int SHAPE = decltype(shape_c)::value;
                // Items per block: one per lane group.
                const int64_t per_block = static_cast<int64_t>(warps) * static_cast<int64_t>(kWarpSize / red::Shape<SHAPE>::G);
                const int64_t blocks    = ceil_div(items, per_block);
                TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(), "reduction_aggr: too much work for one launch (", items, " items)");
                auto launch = [&]<bool OCCUPANCY>() {
                    red::reduction_forward_kernel<OP, SHAPE, cuda_t, index_t, OCCUPANCY>
                        <<<static_cast<unsigned>(blocks), static_cast<unsigned>(warps) * kWarpSize, 0, stream>>>(p);
                };
                if constexpr (red::kHasIlpTwin<OP, SHAPE, cuda_t, index_t>) {
                    if (long_rows) {
                        launch.template operator()<false>();
                        return;
                    }
                }
                launch.template operator()<true>();
            });

            if (merge) {
                int const *const offsets = slices.node_chunk_offset.data_ptr<int>();
                if constexpr (OP == ReductionOp::SUM) {
                    red::launch_sum_slices(p, static_cast<int>(num_heavy), offsets, wide, stream);
                } else {
                    red::launch_finalize_heavy<OP>(p, static_cast<int>(num_heavy), offsets, wide, stream);
                }
            }
        },
        MakeIndexWidthVariant(edge_ptr.scalar_type()), MakeTypeVariant<float, at::Half, at::BFloat16>(X.scalar_type()),
        MakeEnumVariant<ReductionOp, ReductionOp::MIN, ReductionOp::MAX, ReductionOp::SUM>(op)
    );
    CUDA_KERNEL_CHECK();
    return {out, arg};
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
