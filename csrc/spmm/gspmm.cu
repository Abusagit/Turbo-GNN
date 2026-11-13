#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include "spmm/gspmm.h"
#include "spmm/gspmm_kernels.cuh"
#include "spmm/gspmm_launch.h"

// =============================================================================
// g-SpMM host side: validation, planning and the launches. All work goes to the caller's current
// stream on the device of the operands.
//
//   forward          the reduction walk of reduction/reduction_launch.cuh with a g-SpMM message --
//                    lane groups or warps per row, 16-byte vectors where every row allows them,
//                    heavy rows in slices -- instantiated per op and dtype by the shards
//                    (gspmm_launch.h). The sum backward's node gradient is this same forward on the
//                    transposed CSR.
//   backward_edge    the sum backward's edge gradient (gspmm_edge_grad.cuh), on the forward's items.
//   backward_arg     the min/max backward (gspmm_kernels.cuh): a walk over the transposed CSR, or
//                    a scatter over the winning edges on a dense graph.
// =============================================================================

namespace {

namespace red  = turbo_gnn::reduction;
namespace rows = turbo_gnn::rows;
namespace gsp  = turbo_gnn::gspmm;

ReductionOp reduce_op_from_string(const std::string& reduce) {
    if (reduce == "min") return ReductionOp::MIN;
    if (reduce == "max") return ReductionOp::MAX;
    TORCH_CHECK(reduce == "sum", "gspmm: unsupported reduce '", reduce, "'; expected 'sum', 'min' or 'max'");
    return ReductionOp::SUM;
}

bool op_uses_lhs(BinaryOp op) { return op != BinaryOp::COPY_E; }
bool op_uses_rhs(BinaryOp op) { return op != BinaryOp::COPY_U; }

struct FeatureLayout {
    int64_t d;
    bool rhs_broadcast;
};

bool deduce_rhs_broadcast(BinaryOp op, const torch::Tensor& rhs, int64_t d) {
    if (!op_uses_rhs(op) || d <= 1) {
        return false;
    }
    return (rhs.dim() == 1) || (rhs.size(1) == 1);
}

FeatureLayout deduce_feature_layout(BinaryOp op, const torch::Tensor& lhs, const torch::Tensor& rhs) {
    int64_t d = 1;
    if (op_uses_lhs(op)) {
        TORCH_CHECK(lhs.dim() == 2, "gspmm: lhs must be 2-D [N, d], got ", lhs.dim(), "-D");
        d = lhs.size(1);
    } else {
        TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "gspmm: rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
        d = (rhs.dim() > 1) ? rhs.size(1) : 1;
    }
    return {d, deduce_rhs_broadcast(op, rhs, d)};
}

void check_csr(const torch::Tensor& edge_ptr, const torch::Tensor& edge_idx, at::Device device, const char *what) {
    TORCH_CHECK(edge_ptr.is_cuda() && edge_idx.is_cuda(), what, ": CSR tensors must be CUDA");
    TORCH_CHECK(edge_ptr.device() == device && edge_idx.device() == device, what, ": the CSR must be on the device of the operands");
    TORCH_CHECK(edge_ptr.is_contiguous() && edge_idx.is_contiguous(), what, ": CSR tensors must be contiguous");
    TORCH_CHECK(
        is_supported_index_type(edge_ptr.scalar_type()), what, ": index tensors must be int32, int64, uint32 or uint64, got ",
        edge_ptr.scalar_type()
    );
    TORCH_CHECK(edge_idx.scalar_type() == edge_ptr.scalar_type(), what, ": edge_idx must have the same dtype as edge_ptr");
    TORCH_CHECK(edge_ptr.dim() == 1 && edge_ptr.numel() >= 1, what, ": edge_ptr must be [N + 1]");
    rows::check_index_range(edge_ptr, edge_idx, what);
}

void check_values(const torch::Tensor& t, const char *name) {
    TORCH_CHECK(t.is_cuda(), "gspmm: ", name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.scalar_type() == torch::kFloat || t.scalar_type() == torch::kHalf || t.scalar_type() == torch::kBFloat16, "gspmm: ", name,
        " must be float32/float16/bfloat16, got ", t.scalar_type()
    );
}

void check_common_inputs(
    const torch::Tensor& edge_ptr, const torch::Tensor& edge_idx, const torch::Tensor& lhs, const torch::Tensor& rhs, BinaryOp bop,
    const FeatureLayout& layout, at::Device device
) {
    check_csr(edge_ptr, edge_idx, device, "gspmm");
    const int64_t num_edges = edge_idx.numel();

    if (op_uses_lhs(bop)) {
        check_values(lhs, "lhs");
        TORCH_CHECK(lhs.dim() == 2, "gspmm: lhs must be 2-D [N, d], got ", lhs.dim(), "-D");
        TORCH_CHECK(lhs.size(0) == edge_ptr.numel() - 1, "gspmm: lhs.size(0) (", lhs.size(0), ") must equal N = edge_ptr.numel() - 1");
        TORCH_CHECK(lhs.device() == device, "gspmm: lhs must be on the device of the operands");
    }

    if (op_uses_rhs(bop)) {
        check_values(rhs, "rhs");
        TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "gspmm: rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
        TORCH_CHECK(rhs.size(0) == num_edges, "gspmm: rhs.size(0) (", rhs.size(0), ") must equal the edge count E = ", num_edges);
        TORCH_CHECK(rhs.device() == device, "gspmm: rhs must be on the device of the operands");
        if (!layout.rhs_broadcast && rhs.dim() == 2) {
            TORCH_CHECK(rhs.size(1) == layout.d, "gspmm: rhs.size(1) (", rhs.size(1), ") must equal d = ", layout.d, " or 1 to broadcast");
        }
        if (op_uses_lhs(bop)) {
            TORCH_CHECK(
                rhs.scalar_type() == lhs.scalar_type(), "gspmm: rhs dtype (", rhs.scalar_type(), ") must match lhs dtype (", lhs.scalar_type(),
                ")"
            );
        }
    }
}

// A node-operand view the kernels can address: unit stride over the features and a row stride that
// fits their 32-bit gather offsets -- any view meeting both is read in place.
torch::Tensor as_node_rows(torch::Tensor t) {
    if ((t.size(1) > 1 && t.stride(1) != 1) || t.stride(0) < 0 || t.stride(0) > std::numeric_limits<uint32_t>::max()) {
        return t.contiguous();
    }
    return t;
}

// Whether a node-operand or edge-operand view takes the 16-byte layout.
bool wide_rows(const torch::Tensor& t) { return t.dim() == 2 && rows::rows_16b_aligned(t); }

// The dtype the kernels are templated on: whichever operand the op reads.
at::ScalarType value_dtype(BinaryOp bop, const torch::Tensor& lhs, const torch::Tensor& rhs) {
    return op_uses_lhs(bop) ? lhs.scalar_type() : rhs.scalar_type();
}

// The shard entry point of an op and dtype (gspmm_launch.h).
#define GSPMM_ROUTE(fn, args)                                                                   \
    switch (bop) {                                                                              \
        case BinaryOp::COPY_U: GSPMM_ROUTE_DTYPE(fn, copy_u, args); break;                      \
        case BinaryOp::COPY_E: GSPMM_ROUTE_DTYPE(fn, copy_e, args); break;                      \
        case BinaryOp::ADD: GSPMM_ROUTE_DTYPE(fn, add, args); break;                            \
        case BinaryOp::SUB: GSPMM_ROUTE_DTYPE(fn, sub, args); break;                            \
        case BinaryOp::MUL: GSPMM_ROUTE_DTYPE(fn, mul, args); break;                            \
        case BinaryOp::DIV: GSPMM_ROUTE_DTYPE(fn, div, args); break;                            \
    }
#define GSPMM_ROUTE_DTYPE(fn, name, args)                                                       \
    switch (dtype) {                                                                            \
        case torch::kFloat: gspmm_##fn##_##name##_f32(args); break;                             \
        case torch::kHalf: gspmm_##fn##_##name##_f16(args); break;                              \
        case torch::kBFloat16: gspmm_##fn##_##name##_bf16(args); break;                         \
        default: TORCH_CHECK(false, "gspmm: unsupported dtype ", dtype);                         \
    }

void route_forward(BinaryOp bop, at::ScalarType dtype, const gsp::ForwardArgs& args) { GSPMM_ROUTE(forward, args) }
void route_edge_grad(BinaryOp bop, at::ScalarType dtype, const gsp::EdgeGradArgs& args) { GSPMM_ROUTE(edge_grad, args) }

#undef GSPMM_ROUTE_DTYPE
#undef GSPMM_ROUTE

// g-SpMM picks its block shape at runtime, so it pins the __launch_bounds__ template parameter of
// the min/max backward's scatter kernel to the CUDA maximum rather than instantiating one kernel per
// warp count.
constexpr size_t kGSpMMWarpsPerBlock = 1024 / kWarpSize;

// Edges one block of the min/max backward's sliced heavy kernel takes. Only graphs whose top degree
// exceeds this get sliced at all: below it every node would come out as a single slice, and the
// partials buffer and its reduce pass would be pure overhead.
constexpr int64_t kGSpMMHeavySliceEdges = 1024;

// ...and a large top degree is not enough by itself.  Measured on a T4 by splitting a fixed edge
// mass across a varying number of hubs (d=32, fp16, copy_u/sum): one hub holding all of it runs
// 3.0-5.2x faster sliced, five hubs 1.37x, and the two paths break even where the top degree is
// about a sixteenth of the whole edge list.  Below that the partials buffer and its reduce pass are
// pure cost, so the launcher slices only above it.
constexpr int64_t kGSpMMHeavySliceShare = 16;

// The light and heavy buckets own disjoint output rows, so their launches are independent.  Given a
// stream each, the light kernel can occupy the SMs that the heavy kernel leaves idle: a heavy launch
// degenerates into a few long-running blocks on a skewed graph, and back-to-back on one stream that
// tail is dead time.  Construct before the launches, join() after them; with one bucket empty both
// members are the caller's stream and join() is a no-op.
struct BucketStreams {
    at::cuda::CUDAStream main;
    at::cuda::CUDAStream heavy;
    bool overlap;

    BucketStreams(bool overlap_buckets, std::initializer_list<const torch::Tensor *> touched_on_side)
        : main(at::cuda::getCurrentCUDAStream()), heavy(main), overlap(overlap_buckets) {
        if (!overlap) {
            return;
        }
        heavy = at::cuda::getStreamFromPool(main.device_index());
        // The heavy launch must not overtake whatever produced the inputs on the caller's stream.
        at::cuda::CUDAEvent inputs_ready;
        inputs_ready.record(main);
        inputs_ready.block(heavy);
        // These were allocated against the caller's stream but are touched on another one, so the
        // caching allocator has to be told before it can consider recycling them.
        for (const torch::Tensor *t : touched_on_side) {
            if (t->defined() && t->numel() > 0) {
                t->record_stream(heavy);
            }
        }
    }

    void join() const {
        if (!overlap) {
            return;
        }
        at::cuda::CUDAEvent heavy_done;
        heavy_done.record(heavy);
        heavy_done.block(main);
    }
};

// Block shape of a light-bucket launch of the min/max backward: features along x, capped at the
// block; nodes fill y.  A broadcast edge operand pins x to one warp, which the shuffle fold relies on.
struct LightShape {
    dim3 block;
    unsigned blocks;
};

LightShape light_launch_shape(int warps_per_block, int64_t d, bool rhs_broadcast, int64_t num_light) {
    const size_t threads = static_cast<size_t>(warps_per_block) * kWarpSize;
    const size_t tile_x  = rhs_broadcast ? kWarpSize : std::min<size_t>(std::max<size_t>(static_cast<size_t>(d), 1), threads);
    const size_t node_y  = std::max<size_t>(threads / tile_x, 1);
    return {dim3(static_cast<unsigned>(tile_x), static_cast<unsigned>(node_y)),
        static_cast<unsigned>((static_cast<size_t>(num_light) + node_y - 1) / node_y)};
}

// Checks of the min/max backward's buckets: CUDA index tensors of the CSR's width that together
// cover every node, and a heavy block shape that fits.
void check_backward_buckets(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    int64_t num_nodes,
    int warps_per_block,
    int features_per_block,
    int tiles_y,
    bool tiles_pow2
) {
    rows::check_node_list(light_nodes, edge_ptr, "gspmm: light_nodes");
    rows::check_node_list(heavy_nodes, edge_ptr, "gspmm: heavy_nodes");
    TORCH_CHECK(
        light_nodes.numel() + heavy_nodes.numel() == num_nodes, "gspmm: light_nodes (", light_nodes.numel(), ") + heavy_nodes (",
        heavy_nodes.numel(), ") must cover all ", num_nodes, " nodes -- gradient rows outside both buckets would be left unwritten"
    );
    TORCH_CHECK(warps_per_block > 0 && warps_per_block <= 32, "gspmm: warps_per_block must be in [1, 32]");
    if (heavy_nodes.numel() > 0) {
        TORCH_CHECK(tiles_y > 0 && tiles_y <= 32, "gspmm: tiles_y must be in [1, 32]");
        TORCH_CHECK(!tiles_pow2 || (tiles_y & (tiles_y - 1)) == 0, "gspmm: tiles_y must be a power of 2 (shared-memory tree reduction)");
        TORCH_CHECK(features_per_block > 0 && features_per_block <= 1024, "gspmm: features_per_block must be in [1, 1024]");
        TORCH_CHECK(
            static_cast<size_t>(features_per_block) * static_cast<size_t>(tiles_y) <= kGSpMMArgHeavyMaxThreads,
            "gspmm: features_per_block * tiles_y must be <= ", kGSpMMArgHeavyMaxThreads
        );
    }
}

struct HeavySlices {
    torch::Tensor offsets;
    int64_t max_slices;
};

HeavySlices heavy_slice_offsets(const torch::Tensor& edge_ptr, const torch::Tensor& heavy_nodes, int64_t num_edges, int64_t slice_edges) {
    const int64_t num_heavy = heavy_nodes.numel();
    const auto heavy_long   = red::signed_view(heavy_nodes).to(torch::kLong);
    const auto ptr          = red::signed_view(edge_ptr);
    const auto degrees      = ptr.index_select(0, heavy_long + 1) - ptr.index_select(0, heavy_long);
    const auto counts       = (degrees.to(torch::kLong) + (slice_edges - 1)).div(slice_edges, "trunc");

    auto offsets = torch::zeros({num_heavy + 1}, ptr.options());
    offsets.slice(0, 1, num_heavy + 1).copy_(counts.cumsum(0));
    return {offsets, num_heavy + (num_edges + slice_edges - 1) / slice_edges};
}

constexpr int64_t kGSpMMArgGatherMaxAvgDegree = 10;

// Longest feature row: the min/max slice merge covers a row in at most 65535 blocks of
// rows::kMergeThreads vectors, as in reduction_aggr.
constexpr int64_t kMaxFeatureDim = int64_t{1} << 22;

void check_feature_dim(int64_t d) { TORCH_CHECK(d <= kMaxFeatureDim, "gspmm: feature dim ", d, " exceeds the supported ", kMaxFeatureDim); }

// The heavy-row slice size checks shared by the forward and the edge gradient.
void check_slice_size(int64_t slice_size, const torch::Tensor& heavy) {
    TORCH_CHECK(
        slice_size >= 1 && slice_size <= std::numeric_limits<int32_t>::max(), "gspmm: slice_size must be in [1, 2^31), got ", slice_size
    );
    TORCH_CHECK(heavy.numel() <= std::numeric_limits<int32_t>::max(), "gspmm: too many heavy nodes");
}

}  // namespace

std::vector<torch::Tensor> gspmm_forward(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    const std::string& reduce,
    int warps_per_block,
    int64_t slice_size,
    const torch::Tensor& chunk_node,
    const torch::Tensor& chunk_start,
    const torch::Tensor& node_chunk_offset,
    int64_t light_degree_bound,
    const torch::Tensor& edge_map
) {
    const BinaryOp bop           = binary_op_from_string(op);
    const ReductionOp rop        = reduce_op_from_string(reduce);
    const auto layout            = deduce_feature_layout(bop, lhs, rhs);
    const torch::Tensor& val_ref = op_uses_lhs(bop) ? lhs : rhs;
    check_values(val_ref, op_uses_lhs(bop) ? "lhs" : "rhs");
    const at::cuda::CUDAGuard device_guard(val_ref.device());
    check_common_inputs(edge_ptr, edge_idx, lhs, rhs, bop, layout, val_ref.device());
    check_feature_dim(layout.d);

    const int64_t num_nodes = edge_ptr.numel() - 1;
    rows::check_node_list(light_nodes, edge_ptr, "gspmm: light_nodes");
    rows::check_node_list(heavy_nodes, edge_ptr, "gspmm: heavy_nodes");
    TORCH_CHECK(
        light_nodes.numel() + heavy_nodes.numel() == num_nodes, "gspmm: light_nodes (", light_nodes.numel(), ") + heavy_nodes (",
        heavy_nodes.numel(), ") must cover all ", num_nodes, " nodes -- output rows outside both buckets would be left uninitialized"
    );
    TORCH_CHECK(warps_per_block >= 1, "gspmm: warps_per_block must be positive, got ", warps_per_block);
    check_slice_size(slice_size, heavy_nodes);

    // An edge map re-indexes the edge operand, which only matters for the ops whose message actually
    // reads it and whose gradient needs its value; the rest degenerate to copy_u on the transposed pass
    // and never ask for one.
    const bool use_edge_map = edge_map.defined() && edge_map.numel() > 0;
    if (use_edge_map) {
        TORCH_CHECK(
            bop == BinaryOp::MUL || bop == BinaryOp::DIV, "gspmm: an edge map is only supported for op 'mul' or 'div' (got '", op,
            "'); the other ops do not read the edge operand on a transposed pass"
        );
        TORCH_CHECK(rop == ReductionOp::SUM, "gspmm: an edge map is only supported for reduce='sum' (got '", reduce, "')");
        TORCH_CHECK(
            edge_map.is_cuda() && edge_map.device() == val_ref.device() && edge_map.is_contiguous(),
            "gspmm: edge_map must be a contiguous CUDA tensor on the device of the operands"
        );
        TORCH_CHECK(
            edge_map.element_size() == edge_ptr.element_size(), "gspmm: edge_map dtype (", edge_map.scalar_type(),
            ") must have the width of edge_ptr's (", edge_ptr.scalar_type(), ")"
        );
        TORCH_CHECK(
            edge_map.numel() == edge_idx.numel(), "gspmm: edge_map must have one entry per edge (", edge_map.numel(), " vs ", edge_idx.numel(),
            ")"
        );
    }

    gsp::ForwardArgs args;
    if (op_uses_lhs(bop)) {
        args.X = as_node_rows(lhs);
    }
    if (op_uses_rhs(bop)) {
        args.E = rhs.contiguous();
    }
    // 16-byte vectors when every row the walk reads allows them: the node rows, and the edge rows
    // unless the edge operand is one scalar per edge.
    const bool wide = (!args.X.defined() || wide_rows(args.X)) && (!args.E.defined() || layout.rhs_broadcast || wide_rows(args.E));

    const red::Slices slices      = red::slices_for(edge_ptr, heavy_nodes, slice_size, chunk_node, chunk_start, node_chunk_offset, "gspmm");
    const red::ForwardLaunch plan = red::plan_forward(
        edge_ptr, edge_idx, light_nodes, heavy_nodes, slices, layout.d, val_ref.element_size(), wide, rop, warps_per_block, light_degree_bound,
        val_ref.options()
    );
    if (plan.items > 0) {
        args.edge_map   = use_edge_map ? edge_map : torch::Tensor();
        args.row_ptr    = edge_ptr;
        args.col_idx    = edge_idx;
        args.light      = light_nodes;
        args.heavy      = heavy_nodes;
        args.plan       = &plan;
        args.slice_size = slice_size;
        args.op         = rop;
        args.bcast      = layout.rhs_broadcast;
        args.elem_size  = val_ref.element_size();
        route_forward(bop, value_dtype(bop, lhs, rhs), args);
        CUDA_KERNEL_CHECK();
    }
    return {plan.out, plan.arg};
}

namespace {

std::vector<torch::Tensor> gspmm_backward_arg_scatter(
    const torch::Tensor& grad_out,
    const torch::Tensor& arg_eid,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    BinaryOp bop,
    bool bcast,
    int warps_per_block
) {
    const int64_t num_nodes = grad_out.size(0);
    const int64_t d         = grad_out.size(1);
    const bool uses_lhs     = op_uses_lhs(bop);
    const bool uses_rhs     = op_uses_rhs(bop);

    const auto value_opts = grad_out.options();
    const auto accum_opts = value_opts.dtype(torch::kFloat);

    auto grad_lhs = uses_lhs ? torch::zeros({num_nodes, d}, accum_opts) : torch::empty({0}, value_opts);
    auto grad_rhs = !uses_rhs ? torch::empty({0}, value_opts) : torch::zeros(rhs.sizes(), bcast ? accum_opts : value_opts);

    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto bcast_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr bool BCAST   = decltype(bcast_c)::value;
            using BOps             = BinaryOps<BOP>;

            if constexpr (BCAST && !BOps::USE_RHS) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                cuda_t const *rhs_ptr = nullptr;
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }

                using grad_rhs_t   = std::conditional_t<BCAST, float, cuda_t>;
                auto *grad_rhs_ptr = uses_rhs ? reinterpret_cast<grad_rhs_t *>(grad_rhs.data_ptr()) : static_cast<grad_rhs_t *>(nullptr);

                const size_t threads = static_cast<size_t>(warps_per_block) * kWarpSize;
                const unsigned row_threads =
                    static_cast<unsigned>(std::min<size_t>(threads, ((static_cast<size_t>(d) + kWarpSize - 1) / kWarpSize) * kWarpSize));

                auto *grad_lhs_ptr = uses_lhs ? grad_lhs.data_ptr<float>() : nullptr;

                reduction_aggr_backward_typed<
                    kGSpMMWarpsPerBlock, cuda_t, index_t, BOP, BCAST, /*ARG_IS_EDGE=*/true, /*grad_t=*/float, /*accum_t=*/float, grad_rhs_t
                ><<<static_cast<unsigned>(num_nodes), row_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>()),
                    index_ptr<index_t>(arg_eid),
                    grad_lhs_ptr,
                    static_cast<size_t>(num_nodes),
                    static_cast<size_t>(d),
                    index_ptr<index_t>(edge_idx),
                    lhs_ptr,
                    rhs_ptr,
                    grad_rhs_ptr
                );
            }
        },
        MakeIndexWidthVariant(edge_idx.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeBoolVariant<false, true>(bcast)
    );

    CUDA_KERNEL_CHECK();

    if (uses_lhs) {
        grad_lhs = grad_lhs.to(value_opts.dtype());
    }
    if (uses_rhs && bcast) {
        grad_rhs = grad_rhs.to(value_opts.dtype());
    }
    return {grad_lhs, grad_rhs};
}

}  // namespace

std::vector<torch::Tensor> gspmm_backward_arg(
    const torch::Tensor& grad_out,
    const torch::Tensor& arg_eid,
    const torch::Tensor& edge_idx,
    const torch::Tensor& bwd_edge_ptr,
    const torch::Tensor& bwd_edge_idx,
    const torch::Tensor& bwd_edge_map,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    int warps_per_block,
    int features_per_block,
    int tiles_y,
    int max_degree,
    int pipeline_stages
) {
    const BinaryOp bop = binary_op_from_string(op);

    TORCH_CHECK(grad_out.is_cuda() && arg_eid.is_cuda() && bwd_edge_ptr.is_cuda() && bwd_edge_idx.is_cuda(), "inputs must be CUDA");
    const at::cuda::CUDAGuard device_guard(grad_out.device());
    rows::check_index_range(bwd_edge_ptr, bwd_edge_idx, "gspmm backward");
    TORCH_CHECK(grad_out.dim() == 2 && grad_out.is_contiguous(), "grad_out must be a contiguous 2-D [N, d]");
    TORCH_CHECK(arg_eid.sizes() == grad_out.sizes(), "arg_eid must be shaped like grad_out");
    TORCH_CHECK(is_supported_index_type(bwd_edge_ptr.scalar_type()), "bwd_edge_ptr must be int32/int64");
    TORCH_CHECK(
        arg_eid.scalar_type() == bwd_edge_ptr.scalar_type() && bwd_edge_idx.scalar_type() == bwd_edge_ptr.scalar_type() &&
            bwd_edge_map.scalar_type() == bwd_edge_ptr.scalar_type(),
        "arg_eid, bwd_edge_idx and bwd_edge_map must have the dtype of bwd_edge_ptr"
    );

    const int64_t num_nodes = grad_out.size(0);
    const int64_t d         = grad_out.size(1);
    const int64_t num_edges = bwd_edge_idx.numel();
    TORCH_CHECK(bwd_edge_ptr.numel() == num_nodes + 1, "bwd_edge_ptr must have N + 1 = ", num_nodes + 1, " entries");
    if (d == 0) {
        // Nothing reached any output, so nothing flows back -- not even to an edge scalar.
        const auto opts = grad_out.options();
        return {op_uses_lhs(bop) ? torch::zeros({num_nodes, 0}, opts) : torch::empty({0}, opts),
            op_uses_rhs(bop) ? torch::zeros(rhs.sizes(), opts) : torch::empty({0}, opts)};
    }
    TORCH_CHECK(bwd_edge_map.numel() == num_edges, "bwd_edge_map must have one entry per edge (", bwd_edge_map.numel(), " vs ", num_edges, ")");

    const bool uses_lhs = op_uses_lhs(bop);
    const bool uses_rhs = op_uses_rhs(bop);
    const bool bcast    = deduce_rhs_broadcast(bop, rhs, d);
    if (uses_rhs) {
        TORCH_CHECK(rhs.size(0) == num_edges, "rhs must have one row per edge (", rhs.size(0), " vs ", num_edges, ")");
    }
    TORCH_CHECK(
        edge_idx.numel() == num_edges && edge_idx.scalar_type() == bwd_edge_ptr.scalar_type(),
        "edge_idx must be the forward CSR's column index, in the dtype of bwd_edge_ptr"
    );

    if (!(uses_rhs && num_edges <= kGSpMMArgGatherMaxAvgDegree * num_nodes)) {
        return gspmm_backward_arg_scatter(grad_out, arg_eid, edge_idx, lhs, rhs, bop, bcast, warps_per_block);
    }

    const int64_t num_light = light_nodes.numel();
    const int64_t num_heavy = heavy_nodes.numel();
    check_backward_buckets(
        bwd_edge_ptr, light_nodes, heavy_nodes, num_nodes, warps_per_block, features_per_block, tiles_y, /*tiles_pow2=*/true
    );

    // Walking the transpose hands every (u, f) to exactly one thread, so the
    // node gradient is written once and goes out in the operand dtype -- no
    // float staging, no atomics, no zero fill.  So does a full-width edge
    // gradient: every forward edge sits at exactly one backward position.  A
    // broadcast edge operand folds d features into one slot, which the kernel
    // accumulates across feature chunks in float; that one is zeroed here and
    // cast below.
    const auto value_opts = grad_out.options();
    const auto accum_opts = value_opts.dtype(torch::kFloat);

    auto grad_lhs = uses_lhs ? torch::empty({num_nodes, d}, value_opts) : torch::empty({0}, value_opts);
    auto grad_rhs = !uses_rhs ? torch::empty({0}, value_opts)
                    : bcast   ? torch::zeros(rhs.sizes(), accum_opts)
                              : torch::empty(rhs.sizes(), value_opts);

    // Same gate as the forward, on the transposed CSR: a source hub holding a
    // sixteenth of the edge list gets its out-edges cut into blocks of
    // kGSpMMHeavySliceEdges, and the chunks' partial node gradients are folded
    // by a second pass.  Below the gate every heavy source is one block.
    // max_degree is the top *out*-degree; -1 means "unknown" and leaves
    // slicing off.
    const bool slice_heavy =
        (num_heavy > 0 && max_degree > kGSpMMHeavySliceEdges && static_cast<int64_t>(max_degree) * kGSpMMHeavySliceShare > num_edges);

    torch::Tensor slice_offsets;
    torch::Tensor slice_partials;
    if (slice_heavy) {
        const HeavySlices slices = heavy_slice_offsets(bwd_edge_ptr, heavy_nodes, num_edges, kGSpMMHeavySliceEdges);
        slice_offsets            = slices.offsets;
        if (uses_lhs) {
            // Sized from the bound, not the real count, which only the device knows.
            slice_partials = torch::empty({slices.max_slices, d}, accum_opts);
        }
    }

    BucketStreams streams(
        num_light > 0 && num_heavy > 0,
        {&grad_out, &arg_eid, &bwd_edge_ptr, &bwd_edge_idx, &bwd_edge_map, &lhs, &rhs, &heavy_nodes, &grad_lhs, &grad_rhs, &slice_offsets,
            &slice_partials}
    );

    // No reducer axis here: min and max share one gather.
    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto bcast_c, auto stages_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr bool BCAST   = decltype(bcast_c)::value;
            constexpr int STAGES   = decltype(stages_c)::value;
            using BOps             = BinaryOps<BOP>;

            if constexpr (BCAST && !BOps::USE_RHS) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                cuda_t const *rhs_ptr = nullptr;
                // Only mul and div differentiate to something that reads the
                // operands; the others would be loading from a null pointer.
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }
                cuda_t *grad_lhs_ptr = uses_lhs ? reinterpret_cast<cuda_t *>(grad_lhs.data_ptr<torch_t>()) : nullptr;

                using grad_rhs_t   = std::conditional_t<BCAST, float, cuda_t>;
                auto *grad_rhs_ptr = uses_rhs ? reinterpret_cast<grad_rhs_t *>(grad_rhs.data_ptr()) : static_cast<grad_rhs_t *>(nullptr);

                auto const *grad_out_ptr = reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>());

                if (num_light > 0) {
                    const LightShape shape = light_launch_shape(warps_per_block, d, BCAST, num_light);
                    const size_t shmem_l =
                        gspmm_arg_grad_shmem_bytes<index_t, STAGES>(static_cast<size_t>(shape.block.x) * static_cast<size_t>(shape.block.y));
                    if constexpr (STAGES > 0) {
                        ensure_dynamic_shmem(
                            gspmm_backward_arg_light_kernel<BOP, BCAST, cuda_t, index_t, grad_rhs_t, float, STAGES>, shmem_l,
                            "gspmm backward arg light"
                        );
                    }
                    gspmm_backward_arg_light_kernel<BOP, BCAST, cuda_t, index_t, grad_rhs_t, float, STAGES>
                        <<<shape.blocks, shape.block, shmem_l, streams.main>>>(
                            index_ptr<index_t>(light_nodes),
                            index_ptr<index_t>(bwd_edge_ptr),
                            index_ptr<index_t>(bwd_edge_idx),
                            index_ptr<index_t>(bwd_edge_map),
                            index_ptr<index_t>(arg_eid),
                            grad_out_ptr,
                            lhs_ptr,
                            rhs_ptr,
                            grad_lhs_ptr,
                            grad_rhs_ptr,
                            static_cast<size_t>(num_light),
                            static_cast<size_t>(d)
                        );
                }

                if (num_heavy > 0) {
                    const dim3 block_h(
                        BCAST ? static_cast<unsigned>(kWarpSize) : static_cast<unsigned>(features_per_block), static_cast<unsigned>(tiles_y)
                    );
                    // Sliced: sized for occupancy, the kernel grid-strides over
                    // a chunk count only the device knows.  Unsliced: a block
                    // per heavy source.
                    unsigned grid_h = static_cast<unsigned>(num_heavy);
                    if (slice_heavy) {
                        const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
                        grid_h             = static_cast<unsigned>(std::max(sm_count * 16, 1));
                    }
                    const size_t shmem_h =
                        gspmm_arg_grad_shmem_bytes<index_t, STAGES>(static_cast<size_t>(block_h.x) * static_cast<size_t>(block_h.y));
                    if constexpr (STAGES > 0) {
                        ensure_dynamic_shmem(
                            gspmm_backward_arg_heavy_kernel<
                                BOP, BCAST, cuda_t, index_t, static_cast<size_t>(kGSpMMHeavySliceEdges), grad_rhs_t, float, STAGES
                            >,
                            shmem_h, "gspmm backward arg heavy"
                        );
                    }
                    gspmm_backward_arg_heavy_kernel<
                        BOP, BCAST, cuda_t, index_t, static_cast<size_t>(kGSpMMHeavySliceEdges), grad_rhs_t, float, STAGES
                    ><<<grid_h, block_h, shmem_h, streams.heavy>>>(
                        index_ptr<index_t>(heavy_nodes),
                        slice_offsets.defined() ? index_ptr<index_t>(slice_offsets) : nullptr,
                        index_ptr<index_t>(bwd_edge_ptr),
                        index_ptr<index_t>(bwd_edge_idx),
                        index_ptr<index_t>(bwd_edge_map),
                        index_ptr<index_t>(arg_eid),
                        grad_out_ptr,
                        lhs_ptr,
                        rhs_ptr,
                        grad_lhs_ptr,
                        grad_rhs_ptr,
                        slice_partials.defined() ? slice_partials.data_ptr<float>() : nullptr,
                        static_cast<size_t>(num_heavy),
                        static_cast<size_t>(d)
                    );

                    if (slice_heavy && uses_lhs) {
                        // Fold each hub's chunk partials in chunk order into
                        // grad_lhs[u].  The forward's slice reducer, with the
                        // sum reducer and no arg to carry.
                        const unsigned reduce_threads = 256;
                        const unsigned reduce_blocks =
                            static_cast<unsigned>(std::min<int64_t>((num_heavy * d + reduce_threads - 1) / reduce_threads, 65535));
                        gspmm_heavy_reduce_slices_kernel<cuda_t, index_t, ReductionOp::SUM>
                            <<<std::max(reduce_blocks, 1u), reduce_threads, 0, streams.heavy>>>(
                                index_ptr<index_t>(heavy_nodes),
                                index_ptr<index_t>(slice_offsets),
                                slice_partials.data_ptr<float>(),
                                nullptr,
                                grad_lhs_ptr,
                                nullptr,
                                static_cast<size_t>(num_heavy),
                                static_cast<size_t>(d)
                            );
                    }
                }
            }
        },
        MakeIndexWidthVariant(bwd_edge_ptr.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeBoolVariant<false, true>(bcast),
        MakeIntVariant<0, 1, 2, 4>(pipeline_stages)
    );

    streams.join();
    CUDA_KERNEL_CHECK();

    // Uniform contract: the edge gradient always comes back in the operand
    // dtype, so only the broadcast path pays for a cast, and it is [E] wide.
    if (uses_rhs && bcast) {
        grad_rhs = grad_rhs.to(value_opts.dtype());
    }

    return {grad_lhs, grad_rhs};
}

torch::Tensor gspmm_backward_edge(
    const torch::Tensor& edge_ptr,
    const torch::Tensor& edge_idx,
    const torch::Tensor& grad_out,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& light_nodes,
    const torch::Tensor& heavy_nodes,
    const std::string& op,
    int warps_per_block,
    int64_t slice_size,
    const torch::Tensor& chunk_node,
    const torch::Tensor& chunk_start,
    const torch::Tensor& node_chunk_offset
) {
    const BinaryOp bop = binary_op_from_string(op);

    // The operand dtype, not float: every slot of this gradient is written exactly once, so there is
    // no atomic accumulation to protect and no reason to stage it through a float buffer.
    const auto grad_opts = grad_out.options();
    if (!op_uses_rhs(bop)) {
        return torch::empty({0}, grad_opts);  // copy_u has no edge operand
    }

    check_values(grad_out, "grad_out");
    const at::cuda::CUDAGuard device_guard(grad_out.device());
    TORCH_CHECK(grad_out.dim() == 2, "gspmm: grad_out must be 2-D [N, d]");
    check_csr(edge_ptr, edge_idx, grad_out.device(), "gspmm backward");
    TORCH_CHECK(grad_out.size(0) == edge_ptr.numel() - 1, "gspmm: grad_out.size(0) must equal N = edge_ptr.numel() - 1");
    check_values(rhs, "rhs");
    TORCH_CHECK(
        rhs.device() == grad_out.device() && rhs.scalar_type() == grad_out.scalar_type(), "gspmm: rhs must match grad_out's device and dtype"
    );

    const int64_t num_nodes = grad_out.size(0);
    const int64_t num_edges = edge_idx.numel();
    const int64_t d         = grad_out.size(1);
    check_feature_dim(d);
    TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "gspmm: rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
    TORCH_CHECK(rhs.size(0) == num_edges, "gspmm: rhs must have one row per edge (", rhs.size(0), " vs ", num_edges, ")");
    TORCH_CHECK(rhs.dim() == 1 || rhs.size(1) == 1 || rhs.size(1) == d, "gspmm: rhs.size(1) (", rhs.size(1), ") must equal d = ", d, " or 1");
    rows::check_node_list(light_nodes, edge_ptr, "gspmm: light_nodes");
    rows::check_node_list(heavy_nodes, edge_ptr, "gspmm: heavy_nodes");
    TORCH_CHECK(
        light_nodes.numel() + heavy_nodes.numel() == num_nodes, "gspmm: light_nodes (", light_nodes.numel(), ") + heavy_nodes (",
        heavy_nodes.numel(), ") must cover all ", num_nodes, " nodes -- edge gradient rows outside both buckets would be left unwritten"
    );
    TORCH_CHECK(warps_per_block >= 1, "gspmm: warps_per_block must be positive, got ", warps_per_block);
    check_slice_size(slice_size, heavy_nodes);

    // Not zeroed: the work items cover every edge exactly once, and each writes its slots -- given the
    // slice table of these heavy rows at slice_size, as graph.walk_partition builds it. With no features
    // there is nothing to write, and the gradient of an edge scalar is the empty sum.
    if (d == 0) {
        return torch::zeros(rhs.sizes(), grad_opts);
    }
    auto grad_rhs = torch::empty(rhs.sizes(), grad_opts);
    if (num_edges == 0) {
        return grad_rhs;
    }

    const bool reads_x = bop == BinaryOp::MUL || bop == BinaryOp::DIV;
    const bool bcast   = deduce_rhs_broadcast(bop, rhs, d);
    gsp::EdgeGradArgs args;
    args.G = as_node_rows(grad_out);
    if (reads_x) {
        check_values(lhs, "lhs");
        TORCH_CHECK(
            lhs.dim() == 2 && lhs.size(0) == num_nodes && lhs.size(1) == d && lhs.device() == grad_out.device() &&
                lhs.scalar_type() == grad_out.scalar_type(),
            "gspmm: lhs must be [N, d] like grad_out"
        );
        args.X = as_node_rows(lhs);
    }
    args.E  = rhs.contiguous();
    args.dE = grad_rhs;
    // 16-byte vectors when every row read or written per edge allows them; a broadcast operand is
    // one scalar per edge and imposes nothing.
    args.wide = wide_rows(args.G) && (!reads_x || wide_rows(args.X)) && (bcast || (wide_rows(args.E) && wide_rows(args.dE)));

    const red::Slices slices = red::slices_for(edge_ptr, heavy_nodes, slice_size, chunk_node, chunk_start, node_chunk_offset, "gspmm");
    args.row_ptr             = edge_ptr;
    args.col_idx             = edge_idx;
    args.light               = light_nodes;
    args.heavy               = heavy_nodes;
    args.slices              = &slices;
    args.slice_size          = slice_size;
    args.D                   = d;
    args.bcast               = bcast;
    args.warps               = warps_per_block;
    route_edge_grad(bop, grad_out.scalar_type(), args);
    CUDA_KERNEL_CHECK();
    return grad_rhs;
}
