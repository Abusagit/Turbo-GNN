#pragma once

#include <ATen/cuda/CUDAEvent.h>

#include <optional>
#include <variant>

#include "common/traits.cuh"
#include "gsddmm/gsddmm.cu"
#include "gsddmm/gsddmm_launch.cuh"

// =============================================================================
// The templated GSDDMM forward dispatch grid, instantiated once per op shard.
//
// Include this from a gsddmm_launch_<op>.cu only -- every inclusion compiles
// (LRO count) x (2 index types) x (3 dtypes) x (4 feature dims) x (2 warp
// counts) x (4 pipeline depths) kernels. gsddmm_binding.cu deliberately does
// not include it.
// =============================================================================

namespace gsddmm {

// Instantiates GSDDMM_forward_normal over Lros x the dtype / index / D / warps /
// stages grid and launches the light and heavy node buckets of args.
template <LRO... Lros>
void gsddmm_dispatch(const GsddmmLaunchArgs& args) {
    auto lro_variant = MakeEnumVariant<LRO, Lros...>(args.key);

    // Lambda to launch the kernel for a bucket of nodes (or node chunks, when
    // block_parts != nullptr) with a given warp count
    auto launch_bucket = [&](const torch::Tensor& node_indices, const torch::Tensor *block_parts, uint32_t edges_per_block,
                             const at::cuda::CUDAStream& stream, auto warp_variant) {
        const int64_t num_blocks = node_indices.numel();
        if (num_blocks == 0) [[unlikely]] {
            return;
        }

        std::visit(
            [&](auto lro_c, auto idxInfo, auto typeInfo, auto d_c, auto warp_c, auto stages_c) {
                constexpr GSDDMM_OP OP     = decltype(lro_c)::value.op;
                constexpr GSDDMM_MEMBER LL = decltype(lro_c)::value.l;
                constexpr GSDDMM_MEMBER RR = decltype(lro_c)::value.r;
                using index_t              = typename decltype(idxInfo)::Type;
                using torch_t              = typename decltype(typeInfo)::TorchType;
                using cuda_t               = typename decltype(typeInfo)::CudaType;
                constexpr size_t DC        = decltype(d_c)::value;
                constexpr size_t W         = decltype(warp_c)::value;
                constexpr int STAGES       = decltype(stages_c)::value;

                cuda_t const *L_ptr = reinterpret_cast<const cuda_t *>(args.L.data_ptr<torch_t>());
                cuda_t const *R_ptr = reinterpret_cast<const cuda_t *>(args.R.data_ptr<torch_t>());
                cuda_t *O_ptr       = reinterpret_cast<cuda_t *>(args.O.data_ptr<torch_t>());

                auto kernel = GSDDMM_forward_normal<OP, LL, RR, W, DC, cuda_t, index_t, float, STAGES>;

                constexpr size_t shmem = gsddmm_forward_shmem_bytes<OP, LL, RR, W, DC, cuda_t, STAGES>();
                ensure_dynamic_shmem(kernel, shmem, "GSDDMM forward");

                dim3 blocks(static_cast<unsigned>(num_blocks));
                dim3 threads(kWarpSize, W);

                kernel<<<blocks, threads, shmem, stream>>>(
                    static_cast<size_t>(args.N), L_ptr, R_ptr, O_ptr, index_ptr<index_t>(args.row_ptr), index_ptr<index_t>(args.col_idx),
                    index_ptr<index_t>(node_indices), block_parts ? index_ptr<index_t>(*block_parts) : nullptr, edges_per_block
                );
            },
            lro_variant, MakeIndexVariant<int32_t, int64_t>(args.row_ptr.scalar_type()),
            MakeTypeVariant<float, at::Half, at::BFloat16>(args.L.scalar_type()), MakeIntVariant<32, 64, 128, 256>(static_cast<int>(args.D)),
            warp_variant, MakeIntVariant<0, 1, 2, 3>(args.pipeline_stages)
        );
    };

    // The two bucket launches are independent (read-only inputs, disjoint output
    // rows). With overlap_buckets the light bucket runs on a pool stream forked
    // from the caller's stream while the heavy bucket stays on it, so the light
    // blocks fill the SMs the heavy launch's tail leaves idle (and vice versa).
    // The fork event orders both after the caller's earlier work; the join event
    // puts the caller's stream behind the light kernel again, so everything the
    // caller does next -- including the allocator's stream-ordered frees of the
    // tensors used here -- sees both buckets complete. No record_stream needed.
    // Heavy is launched first: its 1024-thread blocks need a whole SM's worth of
    // resources, which they get while the 128-thread light blocks trickle in.
    const bool overlap = args.overlap_buckets && args.light_nodes.numel() > 0 && args.heavy_nodes.numel() > 0;
    std::optional<at::cuda::CUDAStream> light_stream;
    at::cuda::CUDAEvent fork, join;
    if (overlap) {
        light_stream.emplace(at::cuda::getStreamFromPool(/*isHighPriority=*/false, args.stream.device_index()));
        fork.record(args.stream);
        fork.block(*light_stream);
    }

    // Heavy nodes: one block per node, or per edges_per_block-wide chunk
    launch_bucket(
        args.heavy_nodes, args.heavy_block_parts, args.heavy_edges_per_block, args.stream, MakeIntVariant<32>(args.heavy_warps_per_block)
    );

    // Light nodes: always one block per node
    launch_bucket(args.light_nodes, nullptr, 0, overlap ? *light_stream : args.stream, MakeIntVariant<4>(args.light_warps_per_block));

    if (overlap) {
        join.record(*light_stream);
        join.block(args.stream);
    }
}

// Edge-block version: instantiates GSDDMM_forward_edge_block over Lros x the
// dtype / D / stages grid (the edge list is always ulonglong2, so there is no
// index-type axis) and launches one grid over ceil(E / edges_per_warp) warps.
template <LRO... Lros>
void gsddmm_dispatch_edge_block(const GsddmmLaunchArgsEdge& args) {
    if (args.E == 0) [[unlikely]] {
        return;
    }

    auto lro_variant = MakeEnumVariant<LRO, Lros...>(args.key);

    std::visit(
        [&](auto lro_c, auto typeInfo, auto d_c, auto stages_c) {
            constexpr GSDDMM_OP OP     = decltype(lro_c)::value.op;
            constexpr GSDDMM_MEMBER LL = decltype(lro_c)::value.l;
            constexpr GSDDMM_MEMBER RR = decltype(lro_c)::value.r;
            using torch_t              = typename decltype(typeInfo)::TorchType;
            using cuda_t               = typename decltype(typeInfo)::CudaType;
            constexpr size_t DC        = decltype(d_c)::value;
            constexpr int STAGES       = decltype(stages_c)::value;

            cuda_t const *L_ptr = reinterpret_cast<const cuda_t *>(args.L.data_ptr<torch_t>());
            cuda_t const *R_ptr = reinterpret_cast<const cuda_t *>(args.R.data_ptr<torch_t>());
            cuda_t *O_ptr       = reinterpret_cast<cuda_t *>(args.O.data_ptr<torch_t>());

            auto kernel = GSDDMM_forward_edge_block<OP, LL, RR, DC, cuda_t, float, STAGES>;

            const size_t shmem = args.warps_per_block * gsddmm_forward_edge_shmem_bytes_per_warp<OP, LL, RR, DC, cuda_t, STAGES>();
            ensure_dynamic_shmem(kernel, shmem, "GSDDMM forward (edge blocks)");

            // Warps own edge chunks; blocks pack warps_per_block of them. The
            // linear block count is folded into (x, y, z) to stay within the
            // per-dimension grid limits for very large edge lists.
            const uint64_t num_warps  = ceil_div<uint64_t>(args.E, args.edges_per_warp);
            const uint64_t num_blocks = ceil_div<uint64_t>(num_warps, args.warps_per_block);

            constexpr uint64_t kMaxGridX  = (1ull << 31) - 1ull;
            constexpr uint64_t kMaxGridYZ = 65535ull;
            const uint32_t grid_dim_x     = static_cast<uint32_t>(num_blocks < kMaxGridX ? num_blocks : kMaxGridX);
            const uint64_t x_blocks       = ceil_div<uint64_t>(num_blocks, grid_dim_x);
            const uint32_t grid_dim_y     = static_cast<uint32_t>(x_blocks < kMaxGridYZ ? x_blocks : kMaxGridYZ);
            const uint64_t xy_blocks      = ceil_div<uint64_t>(x_blocks, grid_dim_y);
            const uint32_t grid_dim_z     = static_cast<uint32_t>(xy_blocks < kMaxGridYZ ? xy_blocks : kMaxGridYZ);
            TORCH_CHECK(xy_blocks <= kMaxGridYZ, "GSDDMM forward (edge blocks): edge list too large for the launch grid");

            const dim3 blocks(grid_dim_x, grid_dim_y, grid_dim_z);
            const dim3 threads(kWarpSize, args.warps_per_block);

            kernel<<<blocks, threads, shmem, args.stream>>>(
                args.E, L_ptr, R_ptr, O_ptr, args.edge_nodes_idx, args.canonical_edge_idx, static_cast<uint32_t>(args.edges_per_warp)
            );
        },
        lro_variant, MakeTypeVariant<float, at::Half, at::BFloat16>(args.L.scalar_type()),
        MakeIntVariant<32, 64, 128, 256>(static_cast<int>(args.D)), MakeIntVariant<0, 1, 2, 3>(args.pipeline_stages)
    );
}

// Compile-time tag for one backward pass direction.
template <GSDDMM_REDUCE reduce>
using ReduceTag = std::integral_constant<GSDDMM_REDUCE, reduce>;

// Node-parallel backward: instantiates GSDDMM_backward_normal over Lros x the
// dtype / index / D / warps grid, for each pass the member pair needs.
//
// There is no pipeline-stage axis here (the backward gathers one row per edge
// with direct loads), which keeps this grid at a quarter of the forward's.
template <LRO... Lros>
void gsddmm_backward_dispatch(const GsddmmBackwardLaunchArgs& args) {
    auto lro_variant = MakeEnumVariant<LRO, Lros...>(args.key);

    // One (bucket, direction) launch. warp_variant fixes the block's warp count.
    auto launch_bucket = [&](const torch::Tensor& node_indices, auto reduce_c, auto warp_variant) {
        const int64_t num_blocks = node_indices.numel();
        if (num_blocks == 0) [[unlikely]] {
            return;
        }

        std::visit(
            [&](auto lro_c, auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                constexpr GSDDMM_OP OP         = decltype(lro_c)::value.op;
                constexpr GSDDMM_MEMBER LL     = decltype(lro_c)::value.l;
                constexpr GSDDMM_MEMBER RR     = decltype(lro_c)::value.r;
                constexpr GSDDMM_REDUCE REDUCE = decltype(reduce_c)::value;
                using Plan                     = GsddmmBackwardPlan<OP, LL, RR, REDUCE>;

                // Pairs without an operand on this side have no pass; skipping
                // here keeps the kernel (and its static_assert) uninstantiated.
                if constexpr (Plan::HAS_SELF) {
                    using index_t         = typename decltype(idxInfo)::Type;
                    using torch_t         = typename decltype(typeInfo)::TorchType;
                    using cuda_t          = typename decltype(typeInfo)::CudaType;
                    constexpr size_t DC   = decltype(d_c)::value;
                    constexpr size_t W    = decltype(warp_c)::value;
                    constexpr bool IS_SRC = REDUCE == GSDDMM_REDUCE::Src;

                    cuda_t const *L_ptr  = reinterpret_cast<const cuda_t *>(args.L.data_ptr<torch_t>());
                    cuda_t const *R_ptr  = reinterpret_cast<const cuda_t *>(args.R.data_ptr<torch_t>());
                    cuda_t const *dO_ptr = reinterpret_cast<const cuda_t *>(args.dO.data_ptr<torch_t>());
                    // This pass reduces one operand and, when the pair has an
                    // edge operand, stores the other's per-edge gradient.
                    torch::Tensor& self_out = Plan::SELF_IS_LHS ? args.dL : args.dR;
                    torch::Tensor& edge_out = Plan::SELF_IS_LHS ? args.dR : args.dL;
                    cuda_t *self_ptr        = reinterpret_cast<cuda_t *>(self_out.data_ptr<torch_t>());
                    cuda_t *edge_ptr        = Plan::WRITE_EDGE_GRAD ? reinterpret_cast<cuda_t *>(edge_out.data_ptr<torch_t>()) : nullptr;

                    auto kernel = GSDDMM_backward_normal<OP, LL, RR, REDUCE, W, DC, cuda_t, index_t, float>;

                    constexpr size_t shmem = gsddmm_backward_shmem_bytes<W, DC, float>();
                    ensure_dynamic_shmem(kernel, shmem, "GSDDMM backward");

                    dim3 blocks(static_cast<unsigned>(num_blocks));
                    dim3 threads(kWarpSize, W);

                    kernel<<<blocks, threads, shmem, args.stream>>>(
                        static_cast<size_t>(args.N), L_ptr, R_ptr, dO_ptr, self_ptr, edge_ptr,
                        index_ptr<index_t>(IS_SRC ? args.row_ptr_T : args.row_ptr), index_ptr<index_t>(IS_SRC ? args.col_idx_T : args.col_idx),
                        index_ptr<index_t>(node_indices), IS_SRC ? args.canonical_edge_idx : nullptr
                    );
                }
            },
            lro_variant, MakeIndexVariant<int32_t, int64_t>(args.row_ptr.scalar_type()),
            MakeTypeVariant<float, at::Half, at::BFloat16>(args.L.scalar_type()), MakeIntVariant<32, 64, 128, 256>(static_cast<int>(args.D)),
            warp_variant
        );
    };

    // Dst pass over the forward CSR, then Src pass over the backward CSR. Each
    // writes a different operand's gradient, so they are independent.
    launch_bucket(args.heavy_nodes, ReduceTag<GSDDMM_REDUCE::Dst>{}, MakeIntVariant<32>(args.heavy_warps_per_block));
    launch_bucket(args.light_nodes, ReduceTag<GSDDMM_REDUCE::Dst>{}, MakeIntVariant<4>(args.light_warps_per_block));
    launch_bucket(args.heavy_nodes_T, ReduceTag<GSDDMM_REDUCE::Src>{}, MakeIntVariant<32>(args.heavy_warps_per_block));
    launch_bucket(args.light_nodes_T, ReduceTag<GSDDMM_REDUCE::Src>{}, MakeIntVariant<4>(args.light_warps_per_block));
}

// Edge-parallel backward: instantiates GSDDMM_backward_edge_block over Lros x the
// dtype / D grid and launches one grid of ceil(E / edges_per_warp) warps per pass.
template <LRO... Lros>
void gsddmm_backward_dispatch_edge_block(const GsddmmBackwardLaunchArgsEdge& args) {
    if (args.E == 0) [[unlikely]] {
        return;
    }

    auto lro_variant = MakeEnumVariant<LRO, Lros...>(args.key);

    auto launch_pass = [&](auto reduce_c) {
        std::visit(
            [&](auto lro_c, auto typeInfo, auto d_c) {
                constexpr GSDDMM_OP OP         = decltype(lro_c)::value.op;
                constexpr GSDDMM_MEMBER LL     = decltype(lro_c)::value.l;
                constexpr GSDDMM_MEMBER RR     = decltype(lro_c)::value.r;
                constexpr GSDDMM_REDUCE REDUCE = decltype(reduce_c)::value;
                using Plan                     = GsddmmBackwardPlan<OP, LL, RR, REDUCE>;

                if constexpr (Plan::HAS_SELF) {
                    using torch_t         = typename decltype(typeInfo)::TorchType;
                    using cuda_t          = typename decltype(typeInfo)::CudaType;
                    constexpr size_t DC   = decltype(d_c)::value;
                    constexpr bool IS_SRC = REDUCE == GSDDMM_REDUCE::Src;

                    cuda_t const *L_ptr     = reinterpret_cast<const cuda_t *>(args.L.data_ptr<torch_t>());
                    cuda_t const *R_ptr     = reinterpret_cast<const cuda_t *>(args.R.data_ptr<torch_t>());
                    cuda_t const *dO_ptr    = reinterpret_cast<const cuda_t *>(args.dO.data_ptr<torch_t>());
                    torch::Tensor& self_f32 = Plan::SELF_IS_LHS ? args.dL_f32 : args.dR_f32;
                    torch::Tensor& edge_out = Plan::SELF_IS_LHS ? args.dR : args.dL;
                    float *self_ptr         = self_f32.data_ptr<float>();
                    cuda_t *edge_ptr        = Plan::WRITE_EDGE_GRAD ? reinterpret_cast<cuda_t *>(edge_out.data_ptr<torch_t>()) : nullptr;

                    auto kernel = GSDDMM_backward_edge_block<OP, LL, RR, REDUCE, DC, cuda_t, float>;

                    // Grouped by the reduced node: the Src pass walks the
                    // source-grouped list and remaps its slots to canonical ids.
                    ulonglong2 const *edges             = IS_SRC ? args.edge_nodes_idx_src : args.edge_nodes_idx_dst;
                    unsigned long long const *canonical = IS_SRC ? args.canonical_edge_idx : nullptr;

                    const uint64_t num_warps  = ceil_div<uint64_t>(args.E, args.edges_per_warp);
                    const uint64_t num_blocks = ceil_div<uint64_t>(num_warps, args.warps_per_block);

                    constexpr uint64_t kMaxGridX  = (1ull << 31) - 1ull;
                    constexpr uint64_t kMaxGridYZ = 65535ull;
                    const uint32_t grid_dim_x     = static_cast<uint32_t>(num_blocks < kMaxGridX ? num_blocks : kMaxGridX);
                    const uint64_t x_blocks       = ceil_div<uint64_t>(num_blocks, grid_dim_x);
                    const uint32_t grid_dim_y     = static_cast<uint32_t>(x_blocks < kMaxGridYZ ? x_blocks : kMaxGridYZ);
                    const uint64_t xy_blocks      = ceil_div<uint64_t>(x_blocks, grid_dim_y);
                    const uint32_t grid_dim_z     = static_cast<uint32_t>(xy_blocks < kMaxGridYZ ? xy_blocks : kMaxGridYZ);
                    TORCH_CHECK(xy_blocks <= kMaxGridYZ, "GSDDMM backward (edge blocks): edge list too large for the launch grid");

                    const dim3 blocks(grid_dim_x, grid_dim_y, grid_dim_z);
                    const dim3 threads(kWarpSize, args.warps_per_block);

                    kernel<<<blocks, threads, 0, args.stream>>>(
                        args.E, L_ptr, R_ptr, dO_ptr, self_ptr, edge_ptr, edges, canonical, static_cast<uint32_t>(args.edges_per_warp)
                    );
                }
            },
            lro_variant, MakeTypeVariant<float, at::Half, at::BFloat16>(args.L.scalar_type()),
            MakeIntVariant<32, 64, 128, 256>(static_cast<int>(args.D))
        );
    };

    launch_pass(ReduceTag<GSDDMM_REDUCE::Dst>{});
    launch_pass(ReduceTag<GSDDMM_REDUCE::Src>{});
}

// The six ordered member pairs with lhs != rhs (same-member ops are dense-data
// ops, rejected by GsddmmPlan's static_assert), for one binary op.
#define GSDDMM_BINARY_LROS(OP)                                      \
    LRO{GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Dst_V, GSDDMM_OP::OP}, \
    LRO{GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::OP},  \
    LRO{GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Src_V, GSDDMM_OP::OP}, \
    LRO{GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::OP},  \
    LRO{GSDDMM_MEMBER::Edge, GSDDMM_MEMBER::Src_V, GSDDMM_OP::OP},  \
    LRO{GSDDMM_MEMBER::Edge, GSDDMM_MEMBER::Dst_V, GSDDMM_OP::OP}

// Copy propagates the lhs to the edges and never reads the rhs, so only the
// (ignored) edge-indexed rhs is instantiated, and the lhs can not be Edge.
#define GSDDMM_COPY_LROS                                             \
    LRO{GSDDMM_MEMBER::Src_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::Copy}, \
    LRO{GSDDMM_MEMBER::Dst_V, GSDDMM_MEMBER::Edge, GSDDMM_OP::Copy}

}  // namespace gsddmm
