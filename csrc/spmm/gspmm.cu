#include <ATen/cuda/CUDAEvent.h>

#include <algorithm>
#include <array>
#include <type_traits>

#include "spmm/gspmm.h"
#include "spmm/gspmm_kernels.cuh"

namespace {

ReductionOp reduce_op_from_string(const std::string& reduce) {
    if (reduce == "sum") return ReductionOp::SUM;
    if (reduce == "min") return ReductionOp::MIN;
    if (reduce == "max") return ReductionOp::MAX;
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
        TORCH_CHECK(lhs.dim() == 2, "lhs must be 2-D [N, d] for op '", static_cast<int>(op), "', got ", lhs.dim(), "-D");
        d = lhs.size(1);
    } else {
        TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
        d = (rhs.dim() > 1) ? rhs.size(1) : 1;
    }

    return {d, deduce_rhs_broadcast(op, rhs, d)};
}

void check_common_inputs(
    const torch::Tensor& edge_ptr, const torch::Tensor& edge_idx, const torch::Tensor& lhs, const torch::Tensor& rhs, BinaryOp bop,
    const FeatureLayout& layout
) {
    TORCH_CHECK(edge_ptr.is_cuda() && edge_idx.is_cuda(), "CSR tensors must be CUDA");
    TORCH_CHECK(edge_ptr.is_contiguous() && edge_idx.is_contiguous(), "CSR tensors must be contiguous");

    const auto idx_dtype = edge_ptr.scalar_type();
    TORCH_CHECK(
        idx_dtype == torch::kInt || idx_dtype == torch::kLong, "g-SpMM index tensors must be int32 or int64 (got ", idx_dtype,
        "); unsigned index types are supported by reduction_aggr only"
    );
    TORCH_CHECK(edge_idx.scalar_type() == idx_dtype, "edge_idx must have the same dtype as edge_ptr");

    const int64_t num_edges = edge_idx.numel();

    if (op_uses_lhs(bop)) {
        TORCH_CHECK(lhs.is_cuda() && lhs.is_contiguous(), "lhs must be a contiguous CUDA tensor");
        TORCH_CHECK(lhs.dim() == 2, "lhs must be 2-D [N, d], got ", lhs.dim(), "-D");
        TORCH_CHECK(lhs.size(0) == edge_ptr.numel() - 1, "lhs.size(0) (", lhs.size(0), ") must equal N = edge_ptr.numel() - 1");
        TORCH_CHECK(
            lhs.scalar_type() == torch::kFloat || lhs.scalar_type() == torch::kHalf || lhs.scalar_type() == torch::kBFloat16,
            "lhs must be float32/float16/bfloat16"
        );
    }

    if (op_uses_rhs(bop)) {
        TORCH_CHECK(rhs.is_cuda() && rhs.is_contiguous(), "rhs must be a contiguous CUDA tensor");
        TORCH_CHECK(rhs.dim() == 1 || rhs.dim() == 2, "rhs must be [E], [E, 1] or [E, d], got ", rhs.dim(), "-D");
        TORCH_CHECK(rhs.size(0) == num_edges, "rhs.size(0) (", rhs.size(0), ") must equal the edge count E = ", num_edges);
        if (!layout.rhs_broadcast && rhs.dim() == 2) {
            TORCH_CHECK(rhs.size(1) == layout.d, "rhs.size(1) (", rhs.size(1), ") must equal d = ", layout.d, " or 1 to broadcast");
        }
        if (op_uses_lhs(bop)) {
            TORCH_CHECK(
                rhs.scalar_type() == lhs.scalar_type(), "rhs dtype (", rhs.scalar_type(), ") must match lhs dtype (", lhs.scalar_type(), ")"
            );
        }
        TORCH_CHECK(
            rhs.scalar_type() == torch::kFloat || rhs.scalar_type() == torch::kHalf || rhs.scalar_type() == torch::kBFloat16,
            "rhs must be float32/float16/bfloat16"
        );
    }
}

// The dtype the kernels are templated on: whichever operand the op reads.
at::ScalarType value_dtype(BinaryOp bop, const torch::Tensor& lhs, const torch::Tensor& rhs) {
    return op_uses_lhs(bop) ? lhs.scalar_type() : rhs.scalar_type();
}

// g-SpMM picks its block shape at runtime, so it pins the __launch_bounds__
// template parameter of the shared kernels to the CUDA maximum rather than
// instantiating one kernel per warp count -- the cross product is already six
// ops by three reducers wide.  The kernels stride by blockDim.x, so the only
// consequence is the register budget nvcc plans for, and with VECTORIZE ==
// false these bodies are nowhere near the 64 registers a 1024-thread block
// leaves them.
constexpr size_t kGSpMMWarpsPerBlock = 1024 / kWarpSize;

// The vectorized TileOps path casts to a 16-byte aligned type, which traps
// unless every row start is 16-byte aligned.  g-SpMM accepts any feature width
// (d = 65 among them), so it always takes the scalar path.
constexpr bool kGSpMMVectorize = false;

// Neighbors one block of the sliced heavy-node kernel takes.  Only graphs
// whose top degree exceeds this get sliced at all: below it every node would
// come out as a single slice, and the partials buffer and its reduce pass
// would be pure overhead.
constexpr int64_t kGSpMMHeavySliceEdges = 1024;

// ...and a large top degree is not enough by itself.  Measured on a T4 by
// splitting a fixed edge mass across a varying number of hubs (d=32, fp16,
// copy_u/sum): one hub holding all of it runs 3.0-5.2x faster sliced, five hubs
// 1.37x, and the two paths break even where the top degree is about a
// sixteenth of the whole edge list.  Below that the partials buffer and its
// reduce pass are pure cost -- 4-7% on every real graph measured, sum and
// min/max alike -- so the launcher slices only above it.
constexpr int64_t kGSpMMHeavySliceShare = 16;

// Edges one warp of the edge-gradient kernel walks before the grid stride.
// Long enough to amortize the binary search that locates the first edge's
// destination, short enough to keep tens of thousands of warps in flight.
constexpr int64_t kGSpMMEdgesPerWarp = 32;

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
    int features_per_block,
    int tiles_y,
    int pipeline_stages,
    const torch::Tensor& edge_map,
    int max_degree
) {
    const BinaryOp bop    = binary_op_from_string(op);
    const ReductionOp rop = reduce_op_from_string(reduce);
    const auto layout     = deduce_feature_layout(bop, lhs, rhs);

    check_common_inputs(edge_ptr, edge_idx, lhs, rhs, bop, layout);

    TORCH_CHECK(light_nodes.is_cuda() && heavy_nodes.is_cuda(), "node buckets must be CUDA");
    TORCH_CHECK(
        light_nodes.scalar_type() == edge_ptr.scalar_type() && heavy_nodes.scalar_type() == edge_ptr.scalar_type(),
        "node buckets must have the same dtype as edge_ptr"
    );

    const int64_t num_nodes = edge_ptr.numel() - 1;
    const int64_t num_light = light_nodes.numel();
    const int64_t num_heavy = heavy_nodes.numel();
    TORCH_CHECK(
        num_light + num_heavy == num_nodes, "light_nodes (", num_light, ") + heavy_nodes (", num_heavy, ") must cover all ", num_nodes,
        " nodes -- output rows outside both buckets would be left uninitialized"
    );

    // An edge map re-indexes the edge operand, which only matters for the ops
    // whose message actually reads it and whose gradient needs its value; the
    // rest degenerate to copy_u on the transposed pass and never ask for one.
    const bool use_edge_map = edge_map.defined() && edge_map.numel() > 0;
    if (use_edge_map) {
        TORCH_CHECK(
            bop == BinaryOp::MUL || bop == BinaryOp::DIV, "an edge map is only supported for op 'mul' or 'div' (got '", op,
            "'); the other ops do not read the edge operand on a transposed pass"
        );
        TORCH_CHECK(rop == ReductionOp::SUM, "an edge map is only supported for reduce='sum' (got '", reduce, "')");
        TORCH_CHECK(edge_map.is_cuda() && edge_map.is_contiguous(), "edge_map must be a contiguous CUDA tensor");
        TORCH_CHECK(
            edge_map.scalar_type() == edge_ptr.scalar_type(), "edge_map dtype (", edge_map.scalar_type(),
            ") must match edge_ptr dtype (", edge_ptr.scalar_type(), ")"
        );
        TORCH_CHECK(
            edge_map.numel() == edge_idx.numel(), "edge_map must have one entry per edge (", edge_map.numel(), " vs ", edge_idx.numel(), ")"
        );
    }

    TORCH_CHECK(warps_per_block > 0 && warps_per_block <= 32, "warps_per_block must be in [1, 32]");
    if (num_heavy > 0) {
        TORCH_CHECK(tiles_y > 0 && tiles_y <= 32, "tiles_y must be in [1, 32]");
        TORCH_CHECK((tiles_y & (tiles_y - 1)) == 0, "tiles_y must be a power of 2 (shared-memory tree reduction)");
        TORCH_CHECK(features_per_block > 0 && features_per_block <= 1024, "features_per_block must be in [1, 1024]");
        TORCH_CHECK(features_per_block * tiles_y <= 1024, "features_per_block * tiles_y must be <= 1024");
    }

    // Slicing pays off only when a handful of nodes hold the edge mass, so that
    // the blocks drawing them set the makespan on their own.  Every reducer can
    // take it: a comparison reducer carries the winning edge position alongside
    // each partial value, in slice_args.
    const bool slice_heavy =
        (num_heavy > 0 && max_degree > kGSpMMHeavySliceEdges &&
         static_cast<int64_t>(max_degree) * kGSpMMHeavySliceShare > edge_idx.numel());

    const torch::Tensor& val_ref = op_uses_lhs(bop) ? lhs : rhs;

    auto out = torch::empty({num_nodes, layout.d}, val_ref.options());
    // min/max need the winning edge for their backward; sum does not, and at
    // [N, d] of index dtype that allocation would cost as much as the output.
    const bool tracks_arg = (rop != ReductionOp::SUM);
    auto arg_eid          = tracks_arg ? torch::empty({num_nodes, layout.d}, edge_ptr.options()) : torch::empty({0}, edge_ptr.options());

    torch::Tensor slice_offsets;
    torch::Tensor slice_partials;
    torch::Tensor slice_args;
    if (slice_heavy) {
        // Exclusive prefix sum of each heavy node's slice count, shaped like a
        // CSR row pointer so the kernel can binary-search it.  Its last entry
        // is the total slice count, which the kernel reads on the device --
        // fetching it here would mean a device-to-host sync per call.
        const auto heavy_long = heavy_nodes.to(torch::kLong);
        const auto degrees    = edge_ptr.index_select(0, heavy_long + 1) - edge_ptr.index_select(0, heavy_long);
        const auto counts     = (degrees.to(torch::kLong) + (kGSpMMHeavySliceEdges - 1)).div(kGSpMMHeavySliceEdges, "trunc");

        slice_offsets = torch::zeros({num_heavy + 1}, edge_ptr.options());
        slice_offsets.slice(0, 1, num_heavy + 1).copy_(counts.cumsum(0));

        // Σ ceil(deg / SLICE) <= num_heavy + E / SLICE, and that bound is
        // known here, so the buffer can be sized without reading the real
        // count back from the device.
        const int64_t max_slices = num_heavy + (edge_idx.numel() + kGSpMMHeavySliceEdges - 1) / kGSpMMHeavySliceEdges;
        slice_partials           = torch::empty({max_slices, layout.d}, out.options().dtype(torch::kFloat));
        if (tracks_arg) {
            slice_args = torch::empty({max_slices, layout.d}, edge_ptr.options());
        }
    }

    // The two buckets own disjoint output rows, so their launches are
    // independent.  Given a stream each, the light kernel can occupy the SMs
    // that the heavy kernel leaves idle: a heavy launch degenerates into a few
    // long-running blocks on a skewed graph, and back-to-back on one stream
    // that tail is dead time.
    const bool overlap_buckets        = (num_light > 0 && num_heavy > 0);
    const auto main_stream            = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAStream heavy_stream = main_stream;
    if (overlap_buckets) {
        heavy_stream = at::cuda::getStreamFromPool(main_stream.device_index());
        // The heavy launch must not overtake whatever produced the inputs on
        // the caller's stream.
        at::cuda::CUDAEvent inputs_ready;
        inputs_ready.record(main_stream);
        inputs_ready.block(heavy_stream);
        // These were allocated against the caller's stream but are touched on
        // another one, so the caching allocator has to be told before it can
        // consider recycling them.
        const std::array<const torch::Tensor *, 10> touched_on_side = {
            &out, &arg_eid, &lhs, &rhs, &edge_ptr, &edge_idx, &heavy_nodes, &slice_offsets, &slice_partials, &slice_args
        };
        for (const torch::Tensor *t : touched_on_side) {
            if (t->defined() && t->numel() > 0) {
                t->record_stream(heavy_stream);
            }
        }
    }

    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto rop_c, auto bcast_c, auto stages_c, auto emap_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP    = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr ReductionOp ROP = static_cast<ReductionOp>(decltype(rop_c)::value);
            constexpr bool BCAST      = decltype(bcast_c)::value;
            constexpr int STAGES      = decltype(stages_c)::value;
            constexpr bool EMAP       = decltype(emap_c)::value;
            using BOps                = BinaryOps<BOP>;

            // deduce_rhs_broadcast never raises the flag for an operation that
            // ignores edge data, so this half of the cross product is
            // unreachable -- pruned here rather than instantiated and skipped.
            // The edge-map half is pruned to the combination the checks above
            // admit, which keeps it from doubling the whole instantiation
            // table for a path only mul/div sum backward takes.
            if constexpr (BCAST && !BOps::USE_RHS) {
                return;
            } else if constexpr (EMAP && !(BOps::GRAD_USES_OPERANDS && ROP == ReductionOp::SUM)) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                if constexpr (BOps::USE_LHS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                }
                cuda_t const *rhs_ptr = nullptr;
                if constexpr (BOps::USE_RHS) {
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }
                cuda_t *out_ptr = reinterpret_cast<cuda_t *>(out.data_ptr<torch_t>());
                const size_t d  = static_cast<size_t>(layout.d);

                index_t const *emap_ptr = nullptr;
                if constexpr (EMAP) {
                    emap_ptr = index_ptr<index_t>(edge_map);
                }

                if (num_light > 0) {
                    // Features along x, capped at the block; nodes fill y.
                    const size_t threads = static_cast<size_t>(warps_per_block) * kWarpSize;
                    const size_t tile_x  = std::min<size_t>(std::max<size_t>(d, 1), threads);
                    const size_t node_y  = std::max<size_t>(threads / tile_x, 1);

                    const dim3 block_l(static_cast<unsigned>(tile_x), static_cast<unsigned>(node_y));
                    const unsigned blocks_l = static_cast<unsigned>((static_cast<size_t>(num_light) + node_y - 1) / node_y);
                    const size_t shmem_l = tile_x * node_y * STAGES * aggr_tile_width<kGSpMMVectorize, cuda_t> * sizeof(cuda_t);

                    if constexpr (STAGES > 0) {
                        ensure_dynamic_shmem(
                            reduction_aggr_forward_light_kernel_1d<
                                kGSpMMWarpsPerBlock, cuda_t, ROP, index_t, float, STAGES, BOP, BCAST, true, kGSpMMVectorize, EMAP>,
                            shmem_l, "gspmm light"
                        );
                    }

                    reduction_aggr_forward_light_kernel_1d<
                        kGSpMMWarpsPerBlock, cuda_t, ROP, index_t, float, STAGES, BOP, BCAST, /*ARG_IS_EDGE=*/true,
                        kGSpMMVectorize, EMAP
                    ><<<blocks_l, block_l, shmem_l, main_stream>>>(
                        index_ptr<index_t>(light_nodes),
                        index_ptr<index_t>(edge_ptr),
                        index_ptr<index_t>(edge_idx),
                        lhs_ptr,
                        out_ptr,
                        index_ptr_mut<index_t>(arg_eid),
                        d,
                        static_cast<size_t>(num_light),
                        rhs_ptr,
                        emap_ptr
                    );
                }

                if (num_heavy > 0 && slice_heavy) {
                    constexpr bool TRACKS_ARG = ReductionOps<ROP>::TRACKS_ARG;

                    const dim3 block_h(static_cast<unsigned>(features_per_block), static_cast<unsigned>(tiles_y));
                    const size_t slots_s = static_cast<size_t>(features_per_block) * static_cast<size_t>(tiles_y);
                    const size_t shmem_s = gspmm_sliced_shmem_bytes<cuda_t, index_t, TRACKS_ARG, STAGES>(slots_s);

                    // Sized for occupancy: the kernel grid-strides over the
                    // slice count, which only the device knows.
                    const int sm_count      = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
                    const unsigned blocks_s = static_cast<unsigned>(std::max(sm_count * 16, 1));

                    ensure_dynamic_shmem(
                        gspmm_heavy_sliced_kernel<
                            cuda_t, index_t, ROP, BOP, BCAST, EMAP, static_cast<size_t>(kGSpMMHeavySliceEdges), STAGES>,
                        shmem_s, "gspmm heavy sliced"
                    );

                    gspmm_heavy_sliced_kernel<
                        cuda_t, index_t, ROP, BOP, BCAST, EMAP, static_cast<size_t>(kGSpMMHeavySliceEdges), STAGES
                    ><<<blocks_s, block_h, shmem_s, heavy_stream>>>(
                        index_ptr<index_t>(heavy_nodes),
                        index_ptr<index_t>(slice_offsets),
                        index_ptr<index_t>(edge_ptr),
                        index_ptr<index_t>(edge_idx),
                        lhs_ptr,
                        rhs_ptr,
                        emap_ptr,
                        slice_partials.data_ptr<float>(),
                        slice_args.defined() ? index_ptr_mut<index_t>(slice_args) : nullptr,
                        static_cast<size_t>(num_heavy),
                        d
                    );

                    const unsigned reduce_threads = 256;
                    const unsigned reduce_blocks  = static_cast<unsigned>(
                        std::min<int64_t>((num_heavy * layout.d + reduce_threads - 1) / reduce_threads, 65535)
                    );
                    gspmm_heavy_reduce_slices_kernel<cuda_t, index_t, ROP>
                        <<<std::max(reduce_blocks, 1u), reduce_threads, 0, heavy_stream>>>(
                            index_ptr<index_t>(heavy_nodes),
                            index_ptr<index_t>(slice_offsets),
                            slice_partials.data_ptr<float>(),
                            slice_args.defined() ? index_ptr<index_t>(slice_args) : nullptr,
                            out_ptr,
                            index_ptr_mut<index_t>(arg_eid),
                            static_cast<size_t>(num_heavy),
                            d
                        );
                } else if (num_heavy > 0) {
                    const dim3 grid_h(static_cast<unsigned>(num_heavy));
                    const dim3 block_h(static_cast<unsigned>(features_per_block), static_cast<unsigned>(tiles_y));
                    // Same expressions the kernel places its arrays with.
                    const size_t slots = static_cast<size_t>(tiles_y) * static_cast<size_t>(features_per_block) *
                                         aggr_tile_width<kGSpMMVectorize, cuda_t>;
                    const size_t shmem = aggr_heavy_shmem_bytes<index_t>(slots) + aggr_pipeline_bytes<cuda_t>(slots, STAGES);

                    ensure_dynamic_shmem(
                        reduction_aggr_forward_heavy_kernel_2d<
                            cuda_t, ROP, index_t, float, BOP, BCAST, /*ARG_IS_EDGE=*/true, kGSpMMVectorize, EMAP, STAGES>,
                        shmem, "gspmm heavy"
                    );

                    reduction_aggr_forward_heavy_kernel_2d<
                        cuda_t, ROP, index_t, float, BOP, BCAST, /*ARG_IS_EDGE=*/true, kGSpMMVectorize, EMAP, STAGES>
                        <<<grid_h, block_h, shmem, heavy_stream>>>(
                            index_ptr<index_t>(heavy_nodes),
                            index_ptr<index_t>(edge_ptr),
                            index_ptr<index_t>(edge_idx),
                            lhs_ptr,
                            out_ptr,
                            index_ptr_mut<index_t>(arg_eid),
                            d,
                            rhs_ptr,
                            emap_ptr
                        );
                }
            }
        },
        MakeIndexVariant<int32_t, int64_t>(edge_ptr.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(value_dtype(bop, lhs, rhs)),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeIntVariant<0, 1, 2>(static_cast<int>(rop)),
        MakeBoolVariant<false, true>(layout.rhs_broadcast),
        MakeIntVariant<0, 1, 2, 4>(pipeline_stages),
        MakeBoolVariant<false, true>(use_edge_map)
    );

    if (overlap_buckets) {
        at::cuda::CUDAEvent heavy_done;
        heavy_done.record(heavy_stream);
        heavy_done.block(main_stream);
    }

    CUDA_KERNEL_CHECK();

    return {out, arg_eid};
}

std::vector<torch::Tensor> gspmm_backward_arg(
    const torch::Tensor& grad_out,
    const torch::Tensor& arg_eid,
    const torch::Tensor& edge_idx,
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const std::string& op,
    int warps_per_block
) {
    const BinaryOp bop = binary_op_from_string(op);

    const int64_t num_nodes = grad_out.size(0);
    const int64_t d         = grad_out.size(1);

    const bool uses_lhs = op_uses_lhs(bop);
    const bool uses_rhs = op_uses_rhs(bop);
    const bool bcast    = deduce_rhs_broadcast(bop, rhs, d);

    // The node gradient accumulates (many destinations can share a winning
    // source), so it is staged in float and cast by the caller.  The edge
    // gradient only accumulates for a broadcast operand; at full width it is
    // written once per slot and goes out directly in the operand dtype, which
    // for an [E, d] operand is the difference between touching 6 bytes per
    // element and 12.
    const auto accum_opts = grad_out.options().dtype(torch::kFloat);
    const auto value_opts = grad_out.options();

    auto grad_lhs = uses_lhs ? torch::zeros({num_nodes, d}, accum_opts) : torch::empty({0}, accum_opts);
    auto grad_rhs = !uses_rhs ? torch::empty({0}, value_opts)
                              : torch::zeros(rhs.sizes(), bcast ? accum_opts : value_opts);

    // No reducer axis here: min and max share one scatter.
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
                // Only mul and div differentiate to something that reads the
                // operands; the others would be loading from a null pointer.
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }

                const unsigned threads = static_cast<unsigned>(static_cast<size_t>(warps_per_block) * kWarpSize);

                // A broadcast edge gradient stays in the float staging buffer
                // because its slots are shared; a full-width one is written in
                // the operand dtype.
                using grad_rhs_t = std::conditional_t<BCAST, float, cuda_t>;
                auto *grad_rhs_ptr =
                    uses_rhs ? reinterpret_cast<grad_rhs_t *>(grad_rhs.data_ptr()) : static_cast<grad_rhs_t *>(nullptr);

                // One block per node, and no wider than the row it walks: this
                // scatter's work is d elements per node whatever the degree is,
                // so a 256-thread block at d=64 would leave three quarters of
                // itself idle.  Reshaping it further does not pay -- a grid
                // strided over [N, d] measured 5-7% slower on every real graph
                // (city-reviews 2.73 -> 2.91 ms), and so did the same kernel
                // with rows along y.
                const unsigned row_threads = static_cast<unsigned>(
                    std::min<size_t>(threads, ((static_cast<size_t>(d) + kWarpSize - 1) / kWarpSize) * kWarpSize)
                );

                reduction_aggr_backward_typed<
                    kGSpMMWarpsPerBlock, cuda_t, index_t, BOP, BCAST, /*ARG_IS_EDGE=*/true, /*grad_t=*/float, /*accum_t=*/float, grad_rhs_t
                ><<<static_cast<unsigned>(num_nodes), row_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>()),
                    index_ptr<index_t>(arg_eid),
                    grad_lhs.data_ptr<float>(),
                    static_cast<size_t>(num_nodes),
                    static_cast<size_t>(d),
                    index_ptr<index_t>(edge_idx),
                    lhs_ptr,
                    rhs_ptr,
                    grad_rhs_ptr
                );
            }
        },
        MakeIndexVariant<int32_t, int64_t>(edge_idx.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeBoolVariant<false, true>(bcast)
    );

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
    const std::string& op,
    int warps_per_block
) {
    const BinaryOp bop = binary_op_from_string(op);

    // The operand dtype, not float: every slot of this gradient is written
    // exactly once, so there is no atomic accumulation to protect and no
    // reason to stage it through a float buffer the caller then has to cast.
    const auto grad_opts = grad_out.options();
    if (!op_uses_rhs(bop)) {
        return torch::empty({0}, grad_opts);  // copy_u has no edge operand
    }

    TORCH_CHECK(grad_out.is_cuda() && edge_ptr.is_cuda() && edge_idx.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2-D [N, d]");
    TORCH_CHECK(grad_out.size(0) == edge_ptr.numel() - 1, "grad_out.size(0) must equal N = edge_ptr.numel() - 1");
    TORCH_CHECK(warps_per_block > 0 && warps_per_block <= 32, "warps_per_block must be in [1, 32]");

    // Not zeroed: the non-broadcast path writes every (eid, f) slot once, and
    // the broadcast path reduces each edge across the whole block down to a
    // single store, so no slot is left untouched.
    auto grad_rhs = torch::empty(rhs.sizes(), grad_opts);

    const int64_t num_nodes = grad_out.size(0);
    const int64_t num_edges = edge_idx.numel();
    const int64_t d         = grad_out.size(1);

    std::visit(
        [&](auto idxInfo, auto typeInfo, auto op_c, auto bcast_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t  = typename decltype(typeInfo)::CudaType;

            constexpr BinaryOp BOP = static_cast<BinaryOp>(decltype(op_c)::value);
            constexpr bool BCAST   = decltype(bcast_c)::value;
            using BOps             = BinaryOps<BOP>;

            // copy_u returned above, so it never reaches a launch; the
            // broadcast flag is likewise impossible without an edge operand.
            if constexpr (!BOps::USE_RHS || (BCAST && !BOps::USE_RHS)) {
                return;
            } else {
                cuda_t const *lhs_ptr = nullptr;
                cuda_t const *rhs_ptr = nullptr;
                if constexpr (BOps::GRAD_USES_OPERANDS) {
                    lhs_ptr = reinterpret_cast<cuda_t const *>(lhs.data_ptr<torch_t>());
                    rhs_ptr = reinterpret_cast<cuda_t const *>(rhs.data_ptr<torch_t>());
                }

                const unsigned threads = static_cast<unsigned>(static_cast<size_t>(warps_per_block) * kWarpSize);
                // One warp per run of kGSpMMEdgesPerWarp edges, capped: the
                // kernel grid-strides over whatever does not fit.
                const int64_t runs         = (num_edges + kGSpMMEdgesPerWarp - 1) / kGSpMMEdgesPerWarp;
                const int64_t warps_needed = (runs + static_cast<int64_t>(warps_per_block) - 1) / warps_per_block;
                const unsigned blocks      = static_cast<unsigned>(std::max<int64_t>(std::min<int64_t>(warps_needed, 65535), 1));

                gspmm_backward_edge_kernel<BOP, BCAST, cuda_t, index_t, /*grad_t=*/cuda_t>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    index_ptr<index_t>(edge_ptr),
                    index_ptr<index_t>(edge_idx),
                    reinterpret_cast<cuda_t const *>(grad_out.data_ptr<torch_t>()),
                    lhs_ptr,
                    rhs_ptr,
                    reinterpret_cast<cuda_t *>(grad_rhs.data_ptr<torch_t>()),
                    static_cast<size_t>(num_nodes),
                    static_cast<size_t>(num_edges),
                    static_cast<size_t>(d)
                );
            }
        },
        MakeIndexVariant<int32_t, int64_t>(edge_ptr.scalar_type()),
        MakeTypeVariant<float, at::Half, at::BFloat16>(grad_out.scalar_type()),
        MakeIntVariant<0, 1, 2, 3, 4, 5>(static_cast<int>(bop)),
        MakeBoolVariant<false, true>(deduce_rhs_broadcast(bop, rhs, d))
    );

    CUDA_KERNEL_CHECK();

    return grad_rhs;
}
