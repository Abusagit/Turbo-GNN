#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include "common/row_kernels.cuh"
#include "reduction/reduction_forward.cuh"

// =============================================================================
// Host side of the forward walk (reduction_forward.cuh), shared by reduction_aggr and g-SpMM: the
// slice table of the heavy bucket, the outputs and merge buffers, and the launches -- one over every
// work item, the heavy rows' slices first, being the longest, then the light rows, and, when there
// are slices, one merge folding them into their rows. All work goes to the caller's current stream.
// =============================================================================

namespace turbo_gnn::reduction {

// Signed view of an index tensor, for the torch ops below (they have no unsigned kernels).
inline at::Tensor signed_view(at::Tensor const& t) {
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
inline Slices build_slices(at::Tensor const& row_ptr, at::Tensor const& heavy, int64_t slice, char const *what) {
    const int64_t num_heavy = heavy.numel();
    const at::Tensor ptr    = signed_view(row_ptr);
    const at::Tensor h      = signed_view(heavy).to(at::kLong);
    const at::Tensor deg    = (ptr.index_select(0, h + 1) - ptr.index_select(0, h)).to(at::kLong);
    const at::Tensor counts = (deg + (slice - 1)).div(slice, "floor").clamp_min(1);

    at::Tensor offset = at::zeros({num_heavy + 1}, h.options());
    offset.slice(0, 1).copy_(counts.cumsum(0));
    const int64_t num = offset[num_heavy].item<int64_t>();
    TORCH_CHECK(num <= std::numeric_limits<int32_t>::max(), what, ": too many heavy-row slices (", num, "); raise the slice size");

    // The indices overload (slot k repeated counts[k] times); a plain int64_t would select the one
    // repeating every element of `counts` that many times.
    const at::Tensor chunk_node  = at::repeat_interleave(counts, std::optional<int64_t>(num));
    const at::Tensor chunk_start = (at::arange(num, h.options()) - offset.index_select(0, chunk_node)) * slice;
    return {chunk_node.to(at::kInt), chunk_start.to(at::kInt), offset.to(at::kInt), num};
}

// The given slice table, checked, or one built here when none is given.
inline Slices slices_for(
    at::Tensor const& row_ptr, at::Tensor const& heavy, int64_t slice, at::Tensor const& chunk_node, at::Tensor const& chunk_start,
    at::Tensor const& node_chunk_offset, char const *what
) {
    if (heavy.numel() == 0) {
        return {};
    }
    if (!chunk_node.defined() || chunk_node.numel() == 0) {
        return build_slices(row_ptr, heavy, slice, what);
    }
    for (at::Tensor const *t : {&chunk_node, &chunk_start, &node_chunk_offset}) {
        TORCH_CHECK(
            t->is_cuda() && t->device() == row_ptr.device() && t->is_contiguous() && t->scalar_type() == at::kInt && t->dim() == 1, what,
            ": the slice table must be contiguous 1-D int32 tensors on the CSR's device"
        );
    }
    TORCH_CHECK(chunk_start.numel() == chunk_node.numel(), what, ": chunk_node and chunk_start must have the same length");
    TORCH_CHECK(node_chunk_offset.numel() == heavy.numel() + 1, what, ": node_chunk_offset must be [num_heavy + 1]");
    return {chunk_node, chunk_start, node_chunk_offset, chunk_node.numel()};
}

// One forward launch, planned: its outputs, merge buffers and the build of the walk.
struct ForwardLaunch {
    at::Tensor out;     // [N_dst, D]
    at::Tensor arg;     // [N_dst, D] of the index dtype for min/max, else empty
    at::Tensor part;    // per-slice partial sums, when sum rows need merging
    at::Tensor packed;  // packed min/max candidates, when min/max rows need merging
    Slices slices;
    int64_t D         = 0;
    int64_t num_light = 0;
    int64_t num_heavy = 0;
    bool wide         = false;
    uint32_t nvec     = 0;
    uint32_t passes   = 0;
    bool merge        = false;
    bool long_rows    = false;
    int warps         = 0;
    int64_t items     = 0;  // 0: nothing to launch

    // Fills in the parts of p this plan decides: everything but the operands (X, stride_x, E,
    // edge_map), which the caller sets.
    template <FloatingNum cuda_t, IntegralNum index_t>
    void fill(
        ForwardParams<cuda_t, index_t>& p, at::Tensor const& row_ptr, at::Tensor const& col_idx, at::Tensor const& light,
        at::Tensor const& heavy, int64_t slice_size, bool tracks_arg
    ) const {
        p.D           = static_cast<uint32_t>(D);
        p.nvec        = nvec;
        p.passes      = passes;
        p.row_ptr     = index_ptr<index_t>(row_ptr);
        p.col_idx     = index_ptr<index_t>(col_idx);
        p.num_slices  = slices.num;
        p.num_light   = num_light;
        p.light_nodes = num_light > 0 ? index_ptr<index_t>(light) : nullptr;
        p.out         = reinterpret_cast<cuda_t *>(out.data_ptr());
        p.arg         = tracks_arg ? static_cast<index_t *>(arg.data_ptr()) : nullptr;
        if (slices.num > 0) {
            p.heavy_nodes = index_ptr<index_t>(heavy);
            p.chunk_node  = slices.chunk_node.data_ptr<int>();
            p.chunk_start = slices.chunk_start.data_ptr<int>();
            p.slice_size  = static_cast<uint32_t>(slice_size);
            p.part        = part.defined() ? part.data_ptr<float>() : nullptr;
            p.packed      = packed.defined() ? reinterpret_cast<unsigned long long *>(packed.data_ptr<int64_t>()) : nullptr;
        }
    }
};

// Plans a forward of D-wide rows over the CSR (row_ptr, col_idx). wide: every row the walk reads in
// vectors -- the node rows, and the edge rows of a message -- is 16-byte aligned (out and the
// partials, fresh allocations, then are as well). light_degree_bound: an upper bound on the light
// rows' degrees, or -1 if unknown (see reduction_aggr_forward_cuda).
inline ForwardLaunch plan_forward(
    at::Tensor const& row_ptr, at::Tensor const& col_idx, at::Tensor const& light, at::Tensor const& heavy, Slices slices, int64_t D,
    size_t elem_size, bool wide, ReductionOp op, int warps_per_block, int64_t light_degree_bound, at::TensorOptions values
) {
    ForwardLaunch L;
    const bool tracks_arg = op != ReductionOp::SUM;
    const int64_t n_dst   = row_ptr.numel() - 1;
    L.D                   = D;
    L.slices              = std::move(slices);
    L.num_light           = light.numel();
    L.num_heavy           = heavy.numel();

    // One output row per CSR row. Accumulating reducers have no arg: at [N, D] of the index dtype it
    // would cost as much as the output itself.
    L.out = at::empty({n_dst, D}, values);
    L.arg = tracks_arg ? at::empty({n_dst, D}, row_ptr.options()) : at::empty({0}, row_ptr.options());
    if (n_dst == 0 || D == 0 || L.num_light + L.slices.num == 0) {
        return L;
    }

    // Row layout: 16-byte vectors when every row allows them, single elements otherwise.
    L.wide         = wide;
    const int code = rows::plan_sweep_shape(D, elem_size, wide);
    L.nvec         = static_cast<uint32_t>(D) / (wide ? static_cast<uint32_t>(16 / elem_size) : 1u);
    L.passes       = rows::sweep_passes(code, L.nvec);

    // Heavy rows of more than one slice: per-slice partial sums, or one packed min/max candidate per
    // element (see pack_candidate), starting out empty. A row of one slice is written in place, and
    // when every heavy row is (num_slices == num_heavy) nothing needs merging at all.
    L.merge = L.slices.num > L.num_heavy;
    if (L.merge && tracks_arg) {
        L.packed = op == ReductionOp::MIN ? at::full({L.num_heavy, D}, -1, values.dtype(at::kLong))
                                          : at::zeros({L.num_heavy, D}, values.dtype(at::kLong));
    } else if (L.merge) {
        L.part = at::empty({L.slices.num, D}, values.dtype(at::kFloat));
    }
    L.warps = std::min(warps_per_block, static_cast<int>(kForwardMaxWarps));
    // The build for whole-warp rows: unbounded when the longest light row -- one warp's walk -- is at
    // least 1.5 times each resident warp's share of all the edges, so that the finish waits on long
    // rows rather than on throughput (see kForwardMinBlocks). Measured on real graphs: tolokers-2
    // (ratio 2.2-6.7) and hm-categories at D = 128 (2.5) ran 14-28% faster unbounded; hm-categories
    // at D = 256 (1.2) 3% faster; flickr, ogbn-arxiv, twitch-views, reddit and ogbn-products (0.03-
    // 0.57) 1-9% slower. An unknown bound keeps the bounded build.
    cudaDeviceProp const *const props = at::cuda::getCurrentDeviceProperties();
    const int64_t resident_warps      = int64_t{props->multiProcessorCount} * (props->maxThreadsPerMultiProcessor / kWarpSize);
    L.long_rows = light_degree_bound > 0 && 2 * light_degree_bound * resident_warps >= 3 * col_idx.numel() * static_cast<int64_t>(L.passes);
    L.items     = (L.slices.num + L.num_light) * static_cast<int64_t>(L.passes);
    return L;
}

// Runs a planned forward: the walk over every item with message M, then the slice merge. p carries
// the operands and has been filled by L.fill.
template <ReductionOp OP, class M, FloatingNum cuda_t, IntegralNum index_t>
void run_forward(ForwardLaunch const& L, ForwardParams<cuda_t, index_t> const& p, size_t elem_size, char const *what) {
    if (L.items == 0) {
        return;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rows::with_sweep_shape(L.D, elem_size, L.wide, [&](auto shape_c) {
        constexpr int SHAPE = decltype(shape_c)::value;
        // Items per block: one per lane group.
        const int64_t per_block = static_cast<int64_t>(L.warps) * static_cast<int64_t>(kWarpSize / Shape<SHAPE>::G);
        const int64_t blocks    = ceil_div(L.items, per_block);
        TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(), what, ": too much work for one launch (", L.items, " items)");
        auto launch = [&]<bool OCCUPANCY>() {
            reduction_forward_kernel<OP, SHAPE, cuda_t, index_t, OCCUPANCY, M>
                <<<static_cast<unsigned>(blocks), static_cast<unsigned>(L.warps) * kWarpSize, 0, stream>>>(p);
        };
        if constexpr (kHasIlpTwin<OP, SHAPE, cuda_t, index_t, M>) {
            if (L.long_rows) {
                launch.template operator()<false>();
                return;
            }
        }
        launch.template operator()<true>();
    });

    if (L.merge) {
        int const *const offsets = L.slices.node_chunk_offset.data_ptr<int>();
        if constexpr (OP == ReductionOp::SUM) {
            launch_sum_slices(p, static_cast<int>(L.num_heavy), offsets, L.wide, stream);
        } else {
            launch_finalize_heavy<OP, cuda_t, index_t, M>(p, static_cast<int>(L.num_heavy), offsets, L.wide, stream);
        }
    }
}

}  // namespace turbo_gnn::reduction
