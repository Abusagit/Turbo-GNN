#pragma once

// The dtype axis is fixed by the including shard (gsddmm_launch_<op>_<dtype>.cu): one
// translation unit per op and dtype keeps each nvcc job small enough to finish quickly.
#ifndef GSDDMM_SHARD_TYPE
#error "include gsddmm_dispatch.cuh through a per-op, per-dtype shard"
#endif

#include <ATen/cuda/CUDAEvent.h>

#include <optional>
#include <variant>

#include "common/traits.cuh"
#include "gsddmm/gsddmm.cu"
#include "gsddmm/gsddmm_launch.cuh"

// =============================================================================
// The templated GSDDMM dispatch grid, instantiated once per op shard.
//
// Include this from a gsddmm_launch_<op>_<dtype>.cu only. Every inclusion compiles, per member pair,
// (10 row shapes) x (2 index widths) x (2 warp counts for the node kernels, or the edge-id remap for
// the edge ones) kernels of each of the four families; gsddmm_binding.cu deliberately does not
// include it.
// =============================================================================

namespace gsddmm {

// Internal linkage on purpose. These templates are parameterized only by the op's member pairs, but
// each per-dtype shard compiles a different body (GSDDMM_SHARD_TYPE fixes the dtype axis). With
// external linkage the three instantiations share one mangled name, the linker keeps one of them,
// and the other two dtypes run a body that rejects their operands.
namespace {

// The operand view every kernel takes. Row strides were narrowed to 32 bits by the binding.
template <typename cuda_t, typename torch_t, int SHAPE>
Operands<cuda_t> make_operands(const torch::Tensor& L, const torch::Tensor& R, uint64_t D) {
    constexpr uint32_t kVec = static_cast<uint32_t>(Shape<SHAPE>::template vec<cuda_t>);
    const uint32_t nvec     = static_cast<uint32_t>(D) / kVec;
    return Operands<cuda_t>{
        .L         = reinterpret_cast<cuda_t const *>(L.data_ptr<torch_t>()),
        .R         = reinterpret_cast<cuda_t const *>(R.data_ptr<torch_t>()),
        .stride_l  = static_cast<uint32_t>(L.stride(0) * L.element_size()),
        .stride_r  = static_cast<uint32_t>(R.stride(0) * R.element_size()),
        .D         = static_cast<uint32_t>(D),
        .row_bytes = static_cast<uint32_t>(D * sizeof(cuda_t)),
        .nvec      = nvec,
        .passes    = row_passes<SHAPE>(nvec),
    };
}

// The (x, y, z) grid of the edge kernels for num_warps warps of warps_per_block per block; the linear
// block count is folded into three dimensions to stay within the per-dimension limits.
inline dim3 edge_grid(uint64_t num_warps, uint32_t warps_per_block, const char *what) {
    const uint64_t num_blocks     = ceil_div<uint64_t>(num_warps, warps_per_block);
    constexpr uint64_t kMaxGridX  = (1ull << 31) - 1ull;
    constexpr uint64_t kMaxGridYZ = 65535ull;
    const uint32_t grid_dim_x     = static_cast<uint32_t>(num_blocks < kMaxGridX ? num_blocks : kMaxGridX);
    const uint64_t x_blocks       = ceil_div<uint64_t>(num_blocks, grid_dim_x);
    const uint32_t grid_dim_y     = static_cast<uint32_t>(x_blocks < kMaxGridYZ ? x_blocks : kMaxGridYZ);
    const uint64_t xy_blocks      = ceil_div<uint64_t>(x_blocks, grid_dim_y);
    const uint32_t grid_dim_z     = static_cast<uint32_t>(xy_blocks < kMaxGridYZ ? xy_blocks : kMaxGridYZ);
    TORCH_CHECK(xy_blocks <= kMaxGridYZ, what, ": edge list too large for the launch grid");
    return dim3(grid_dim_x, grid_dim_y, grid_dim_z);
}

// Instantiates gsddmm_forward_node_kernel over Lros x the shape / index / warps grid and launches the
// light and heavy node buckets of args.
template <LRO... Lros>
void gsddmm_dispatch(const GsddmmLaunchArgs& args) {
    auto lro_variant   = MakeEnumVariant<LRO, Lros...>(args.key);
    const int code     = plan_shape(static_cast<int64_t>(args.D), args.L.element_size(), args.wide);
    auto shape_variant = make_shape_variant(code);

    // Launches the kernel for a bucket of nodes (or node chunks, when block_parts != nullptr).
    auto launch_bucket = [&](const torch::Tensor& node_indices, const torch::Tensor *block_parts, uint32_t edges_per_block,
                             const at::cuda::CUDAStream& stream, auto warp_variant) {
        const int64_t num_blocks = node_indices.numel();
        if (num_blocks == 0) [[unlikely]] {
            return;
        }
        std::visit(
            [&](auto lro_c, auto idxInfo, auto typeInfo, auto shape_c, auto warp_c) {
                constexpr GSDDMM_OP OP     = decltype(lro_c)::value.op;
                constexpr GSDDMM_MEMBER LL = decltype(lro_c)::value.l;
                constexpr GSDDMM_MEMBER RR = decltype(lro_c)::value.r;
                using index_t              = typename decltype(idxInfo)::Type;
                using torch_t              = typename decltype(typeInfo)::TorchType;
                using cuda_t               = typename decltype(typeInfo)::CudaType;
                constexpr int SHAPE        = decltype(shape_c)::value;
                constexpr size_t W         = decltype(warp_c)::value;

                const NodeForwardParams<cuda_t, index_t> p{
                    .x               = make_operands<cuda_t, torch_t, SHAPE>(args.L, args.R, args.D),
                    .O               = reinterpret_cast<cuda_t *>(args.O.data_ptr<torch_t>()),
                    .row_ptr         = index_ptr<index_t>(args.row_ptr),
                    .col_idx         = index_ptr<index_t>(args.col_idx),
                    .node_indices    = index_ptr<index_t>(node_indices),
                    .block_part      = block_parts ? index_ptr<index_t>(*block_parts) : nullptr,
                    .edges_per_block = edges_per_block,
                    .N               = static_cast<int64_t>(args.N),
                };
                gsddmm_forward_node_kernel<OP, LL, RR, W, SHAPE, cuda_t, index_t, float>
                    <<<static_cast<unsigned>(num_blocks), static_cast<unsigned>(W * kWarpSize), 0, stream>>>(p);
            },
            lro_variant, MakeIndexWidthVariant(args.row_ptr.scalar_type()), MakeTypeVariant<GSDDMM_SHARD_TYPE>(args.L.scalar_type()),
            shape_variant, warp_variant
        );
    };

    // The two bucket launches are independent (read-only inputs, disjoint output rows). With
    // overlap_buckets the light bucket runs on a pool stream forked from the caller's stream while
    // the heavy bucket stays on it, so the light blocks fill the SMs the heavy launch's tail leaves
    // idle (and vice versa). The fork event orders both after the caller's earlier work; the join
    // event puts the caller's stream behind the light kernel again, so everything the caller does
    // next -- including the allocator's stream-ordered frees of the tensors used here -- sees both
    // buckets complete. No record_stream needed.
    const bool overlap = args.overlap_buckets && args.light_nodes.numel() > 0 && args.heavy_nodes.numel() > 0;
    std::optional<at::cuda::CUDAStream> light_stream;
    at::cuda::CUDAEvent fork, join;
    if (overlap) {
        light_stream.emplace(at::cuda::getStreamFromPool(/*isHighPriority=*/false, args.stream.device_index()));
        fork.record(args.stream);
        fork.block(*light_stream);
    }

    // Heavy nodes: one block per node, or per edges_per_block-wide chunk.
    launch_bucket(
        args.heavy_nodes, args.heavy_block_parts, args.heavy_edges_per_block, args.stream, MakeIntVariant<32>(args.heavy_warps_per_block)
    );
    // Light nodes: always one block per node.
    launch_bucket(args.light_nodes, nullptr, 0, overlap ? *light_stream : args.stream, MakeIntVariant<4>(args.light_warps_per_block));

    if (overlap) {
        join.record(*light_stream);
        join.block(args.stream);
    }
}

// Edge-block version: instantiates gsddmm_forward_edge_kernel over Lros x the shape / index / remap
// grid and launches one grid over ceil(E / chunk) warps. The remap instantiation exists only for
// pairs without a Dst_V operand: the others traverse the destination-grouped list, which is in
// canonical order already.
template <LRO... Lros>
void gsddmm_dispatch_edge_block(const GsddmmLaunchArgsEdge& args) {
    if (args.E == 0) [[unlikely]] {
        return;
    }
    auto lro_variant   = MakeEnumVariant<LRO, Lros...>(args.key);
    const int code     = plan_shape(static_cast<int64_t>(args.D), args.L.element_size(), args.wide);
    auto shape_variant = make_shape_variant(code);

    std::visit(
        [&](auto lro_c, auto idxInfo, auto typeInfo, auto shape_c, auto remap_c) {
            constexpr GSDDMM_OP OP     = decltype(lro_c)::value.op;
            constexpr GSDDMM_MEMBER LL = decltype(lro_c)::value.l;
            constexpr GSDDMM_MEMBER RR = decltype(lro_c)::value.r;
            using index_t              = typename decltype(idxInfo)::Type;
            using torch_t              = typename decltype(typeInfo)::TorchType;
            using cuda_t               = typename decltype(typeInfo)::CudaType;
            constexpr int SHAPE        = decltype(shape_c)::value;
            constexpr bool REMAP_IDS   = decltype(remap_c)::value != 0;

            if constexpr (REMAP_IDS && GsddmmPlan<OP, LL, RR>::READS_DST) {
                TORCH_CHECK(
                    false, "GSDDMM forward (edge blocks): canonical_edge_idx is only supported for operand pairs without a dst member; ",
                    "pass the destination-grouped edge list, which is in canonical order"
                );
            } else {
                const uint32_t chunk = edge_chunk<SHAPE>(args.edges_per_warp);
                const EdgeForwardParams<cuda_t, index_t> p{
                    .x         = make_operands<cuda_t, torch_t, SHAPE>(args.L, args.R, args.D),
                    .O         = reinterpret_cast<cuda_t *>(args.O.data_ptr<torch_t>()),
                    .pairs     = reinterpret_cast<index_pair_t<index_t> const *>(args.edge_nodes_idx),
                    .canonical = reinterpret_cast<index_t const *>(args.canonical_edge_idx),
                    .E         = static_cast<std::make_unsigned_t<index_t>>(args.E),
                    .chunk     = chunk,
                };
                const dim3 grid = edge_grid(ceil_div<uint64_t>(args.E, chunk), args.warps_per_block, "GSDDMM forward (edge blocks)");
                gsddmm_forward_edge_kernel<OP, LL, RR, SHAPE, cuda_t, index_t, float, REMAP_IDS>
                    <<<grid, static_cast<unsigned>(args.warps_per_block * kWarpSize), 0, args.stream>>>(p);
            }
        },
        lro_variant, MakeIndexWidthVariant(args.edge_index_dtype), MakeTypeVariant<GSDDMM_SHARD_TYPE>(args.L.scalar_type()), shape_variant,
        MakeIntVariant<0, 1>(args.canonical_edge_idx != nullptr ? 1 : 0)
    );
}

// Compile-time tag for one backward pass direction.
template <GSDDMM_REDUCE reduce>
using ReduceTag = std::integral_constant<GSDDMM_REDUCE, reduce>;

// Node-parallel backward: instantiates gsddmm_backward_node_kernel over Lros x the shape / index /
// warps grid, for each pass the member pair needs.
template <LRO... Lros>
void gsddmm_backward_dispatch(const GsddmmBackwardLaunchArgs& args) {
    auto lro_variant   = MakeEnumVariant<LRO, Lros...>(args.key);
    const int code     = plan_shape(static_cast<int64_t>(args.D), args.L.element_size(), args.wide);
    auto shape_variant = make_shape_variant(code);

    // One (bucket, direction) launch. warp_variant fixes the block's warp count.
    auto launch_bucket = [&](const torch::Tensor& node_indices, auto reduce_c, auto warp_variant) {
        const int64_t num_blocks = node_indices.numel();
        if (num_blocks == 0) [[unlikely]] {
            return;
        }
        std::visit(
            [&](auto lro_c, auto idxInfo, auto typeInfo, auto shape_c, auto warp_c) {
                constexpr GSDDMM_OP OP         = decltype(lro_c)::value.op;
                constexpr GSDDMM_MEMBER LL     = decltype(lro_c)::value.l;
                constexpr GSDDMM_MEMBER RR     = decltype(lro_c)::value.r;
                constexpr GSDDMM_REDUCE REDUCE = decltype(reduce_c)::value;
                using Plan                     = GsddmmBackwardPlan<OP, LL, RR, REDUCE>;

                // Pairs without an operand on this side have no pass; skipping here keeps the kernel
                // (and its static_assert) uninstantiated.
                if constexpr (Plan::HAS_SELF) {
                    using index_t         = typename decltype(idxInfo)::Type;
                    using torch_t         = typename decltype(typeInfo)::TorchType;
                    using cuda_t          = typename decltype(typeInfo)::CudaType;
                    constexpr int SHAPE   = decltype(shape_c)::value;
                    constexpr size_t W    = decltype(warp_c)::value;
                    constexpr bool IS_SRC = REDUCE == GSDDMM_REDUCE::Src;

                    // This pass reduces one operand and, when the pair has an edge operand, stores the
                    // other's per-edge gradient.
                    torch::Tensor& self_out = Plan::SELF_IS_LHS ? args.dL : args.dR;
                    torch::Tensor& edge_out = Plan::SELF_IS_LHS ? args.dR : args.dL;
                    const NodeBackwardParams<cuda_t, index_t> p{
                        .x            = make_operands<cuda_t, torch_t, SHAPE>(args.L, args.R, args.D),
                        .dO           = reinterpret_cast<cuda_t const *>(args.dO.data_ptr<torch_t>()),
                        .d_self       = reinterpret_cast<cuda_t *>(self_out.data_ptr<torch_t>()),
                        .d_edge       = Plan::WRITE_EDGE_GRAD ? reinterpret_cast<cuda_t *>(edge_out.data_ptr<torch_t>()) : nullptr,
                        .row_ptr      = index_ptr<index_t>(IS_SRC ? args.row_ptr_T : args.row_ptr),
                        .col_idx      = index_ptr<index_t>(IS_SRC ? args.col_idx_T : args.col_idx),
                        .node_indices = index_ptr<index_t>(node_indices),
                        .canonical    = IS_SRC ? reinterpret_cast<index_t const *>(args.canonical_edge_idx) : nullptr,
                        .N            = static_cast<int64_t>(args.N),
                    };
                    auto kernel            = gsddmm_backward_node_kernel<OP, LL, RR, REDUCE, W, SHAPE, cuda_t, index_t, float>;
                    constexpr size_t shmem = backward_node_smem_bytes<W, SHAPE, cuda_t, float>();
                    ensure_dynamic_shmem(kernel, shmem, "GSDDMM backward");
                    kernel<<<static_cast<unsigned>(num_blocks), static_cast<unsigned>(W * kWarpSize), shmem, args.stream>>>(p);
                }
            },
            lro_variant, MakeIndexWidthVariant(args.row_ptr.scalar_type()), MakeTypeVariant<GSDDMM_SHARD_TYPE>(args.L.scalar_type()),
            shape_variant, warp_variant
        );
    };

    // Dst pass over the forward CSR, then Src pass over the backward CSR. Each writes a different
    // operand's gradient, so they are independent.
    launch_bucket(args.heavy_nodes, ReduceTag<GSDDMM_REDUCE::Dst>{}, MakeIntVariant<32>(args.heavy_warps_per_block));
    launch_bucket(args.light_nodes, ReduceTag<GSDDMM_REDUCE::Dst>{}, MakeIntVariant<4>(args.light_warps_per_block));
    launch_bucket(args.heavy_nodes_T, ReduceTag<GSDDMM_REDUCE::Src>{}, MakeIntVariant<32>(args.heavy_warps_per_block));
    launch_bucket(args.light_nodes_T, ReduceTag<GSDDMM_REDUCE::Src>{}, MakeIntVariant<4>(args.light_warps_per_block));
}

// Edge-parallel backward: instantiates gsddmm_backward_edge_kernel over Lros x the shape / index
// grid and launches one grid of ceil(E / chunk) warps per pass. Only the Src pass remaps: the Dst
// pass walks the destination-grouped list, which is in canonical order.
template <LRO... Lros>
void gsddmm_backward_dispatch_edge_block(const GsddmmBackwardLaunchArgsEdge& args) {
    if (args.E == 0) [[unlikely]] {
        return;
    }
    auto lro_variant   = MakeEnumVariant<LRO, Lros...>(args.key);
    const int code     = plan_shape(static_cast<int64_t>(args.D), args.L.element_size(), args.wide);
    auto shape_variant = make_shape_variant(code);

    auto launch_pass = [&](auto reduce_c) {
        constexpr GSDDMM_REDUCE REDUCE = decltype(reduce_c)::value;
        // Grouped by the reduced node: the Src pass walks the source-grouped list and remaps its
        // slots to canonical ids; the Dst pass's list is already in canonical order.
        void const *edges     = REDUCE == GSDDMM_REDUCE::Src ? args.edge_nodes_idx_src : args.edge_nodes_idx_dst;
        void const *canonical = REDUCE == GSDDMM_REDUCE::Src ? args.canonical_edge_idx : nullptr;

        std::visit(
            [&](auto lro_c, auto idxInfo, auto typeInfo, auto shape_c, auto remap_c) {
                constexpr GSDDMM_OP OP     = decltype(lro_c)::value.op;
                constexpr GSDDMM_MEMBER LL = decltype(lro_c)::value.l;
                constexpr GSDDMM_MEMBER RR = decltype(lro_c)::value.r;
                constexpr bool REMAP_IDS   = decltype(remap_c)::value != 0;
                using Plan                 = GsddmmBackwardPlan<OP, LL, RR, REDUCE>;

                if constexpr (Plan::HAS_SELF && (REDUCE == GSDDMM_REDUCE::Src || !REMAP_IDS)) {
                    using index_t       = typename decltype(idxInfo)::Type;
                    using torch_t       = typename decltype(typeInfo)::TorchType;
                    using cuda_t        = typename decltype(typeInfo)::CudaType;
                    constexpr int SHAPE = decltype(shape_c)::value;

                    torch::Tensor& self_f32 = Plan::SELF_IS_LHS ? args.dL_f32 : args.dR_f32;
                    torch::Tensor& edge_out = Plan::SELF_IS_LHS ? args.dR : args.dL;
                    const uint32_t chunk    = edge_chunk<SHAPE>(args.edges_per_warp);
                    const EdgeBackwardParams<cuda_t, index_t, float> p{
                        .x         = make_operands<cuda_t, torch_t, SHAPE>(args.L, args.R, args.D),
                        .dO        = reinterpret_cast<cuda_t const *>(args.dO.data_ptr<torch_t>()),
                        .d_self    = self_f32.data_ptr<float>(),
                        .d_edge    = Plan::WRITE_EDGE_GRAD ? reinterpret_cast<cuda_t *>(edge_out.data_ptr<torch_t>()) : nullptr,
                        .pairs     = reinterpret_cast<index_pair_t<index_t> const *>(edges),
                        .canonical = reinterpret_cast<index_t const *>(canonical),
                        .E         = static_cast<std::make_unsigned_t<index_t>>(args.E),
                        .chunk     = chunk,
                    };
                    auto kernel        = gsddmm_backward_edge_kernel<OP, LL, RR, REDUCE, SHAPE, cuda_t, index_t, float, REMAP_IDS>;
                    const size_t shmem = args.warps_per_block * backward_edge_smem_bytes_per_warp<SHAPE, cuda_t, float>();
                    ensure_dynamic_shmem(kernel, shmem, "GSDDMM backward (edge blocks)");
                    const dim3 grid = edge_grid(ceil_div<uint64_t>(args.E, chunk), args.warps_per_block, "GSDDMM backward (edge blocks)");
                    kernel<<<grid, static_cast<unsigned>(args.warps_per_block * kWarpSize), shmem, args.stream>>>(p);
                }
            },
            lro_variant, MakeIndexWidthVariant(args.edge_index_dtype), MakeTypeVariant<GSDDMM_SHARD_TYPE>(args.L.scalar_type()), shape_variant,
            MakeIntVariant<0, 1>(canonical != nullptr ? 1 : 0)
        );
    };

    launch_pass(ReduceTag<GSDDMM_REDUCE::Dst>{});
    launch_pass(ReduceTag<GSDDMM_REDUCE::Src>{});
}

// The six ordered member pairs with lhs != rhs (same-member ops are dense-data ops, rejected by
// GsddmmPlan's static_assert), for one binary op.
#define GSDDMM_BINARY_LROS(OP)                                      \
    LRO{GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Dst_V, GSDDMM_OP::OP}, \
    LRO{GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::OP},  \
    LRO{GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Src_V, GSDDMM_OP::OP}, \
    LRO{GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::OP},  \
    LRO{GSDDMM_MEMBER::Edge, GSDDMM_MEMBER::Src_V, GSDDMM_OP::OP},  \
    LRO{GSDDMM_MEMBER::Edge, GSDDMM_MEMBER::Dst_V, GSDDMM_OP::OP}

// Copy propagates the lhs to the edges and never reads the rhs, so only the (ignored) edge-indexed
// rhs is instantiated, and the lhs can not be Edge.
#define GSDDMM_COPY_LROS                                             \
    LRO{GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::Copy}, \
    LRO{GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::Copy}

}  // namespace

}  // namespace gsddmm
