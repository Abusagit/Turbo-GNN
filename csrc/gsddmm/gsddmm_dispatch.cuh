#pragma once

#include <string>
#include <variant>

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

// Runtime -> compile-time for the (lhs, rhs, op) triple: returns a variant whose
// active alternative carries the matching LRO as a template argument. Values not
// in the dispatch set are a hard error -- there is no kernel for them.
template <typename EnumT, EnumT... Values>
std::variant<std::integral_constant<EnumT, Values>...> MakeEnumVariant(EnumT value) {
    std::variant<std::integral_constant<EnumT, Values>...> result;
    bool found = false;
    (
        [&] {
            if (value == Values) {
                result.template emplace<std::integral_constant<EnumT, Values>>();
                found = true;
            }
        }(),
        ...);
    TORCH_CHECK(found, "GSDDMM: enum value not in the dispatch set");
    return result;
}

// Instantiates GSDDMM_forward_normal over Lros x the dtype / index / D / warps /
// stages grid and launches the light and heavy node buckets of args.
template <LRO... Lros>
void gsddmm_dispatch(const GsddmmLaunchArgs& args) {
    auto lro_variant = MakeEnumVariant<LRO, Lros...>(args.key);

    // Lambda to launch the kernel for a bucket of nodes with a given warp count
    auto launch_bucket = [&](const torch::Tensor& node_indices, int64_t num_nodes_bucket, auto warp_variant) {
        if (num_nodes_bucket == 0) return;

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

                dim3 blocks(static_cast<unsigned>(num_nodes_bucket));
                dim3 threads(kWarpSize, W);

                kernel<<<blocks, threads, shmem, args.stream>>>(
                    static_cast<size_t>(args.N), L_ptr, R_ptr, O_ptr, index_ptr<index_t>(args.row_ptr), index_ptr<index_t>(args.col_idx),
                    index_ptr<index_t>(node_indices)
                );
            },
            lro_variant, MakeIndexVariant<int32_t, int64_t>(args.row_ptr.scalar_type()),
            MakeTypeVariant<float, at::Half, at::BFloat16>(args.L.scalar_type()), MakeIntVariant<32, 64, 128, 256>(static_cast<int>(args.D)),
            warp_variant, MakeIntVariant<0, 1, 2, 3>(args.pipeline_stages)
        );
    };

    // Light nodes
    launch_bucket(args.light_nodes, args.light_nodes.numel(), MakeIntVariant<4>(args.light_warps_per_block));

    // Heavy nodes
    launch_bucket(args.heavy_nodes, args.heavy_nodes.numel(), MakeIntVariant<32>(args.heavy_warps_per_block));
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

};  // namespace gsddmm
