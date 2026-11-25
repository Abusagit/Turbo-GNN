// Compiled once per dtype through gatv2/gatv2_shard_{f32,f16,bf16}.cu, which define
// ATTN_SHARD_TYPE and ATTN_SHARD_FN. One translation unit instantiating every dtype took
// most of the build; split three ways the shards compile in parallel.
#ifndef ATTN_SHARD_TYPE
#error "compile this file through its per-dtype shards"
#endif

#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common/row_kernels.cuh"
#include "gatv2/gatv2_backward.cu"
#include "gatv2/gatv2_forward.cu"

namespace {

namespace gatv2     = turbo_gnn::gatv2;
namespace rows      = turbo_gnn::rows;
namespace sched_ns  = turbo_gnn::sched;
namespace stream_ns = turbo_gnn::streams;
using gatv2::BackwardMode;

// Only the one-node-per-block schedule is compiled into the attention kernels (build time).
inline constexpr auto kSchedule = sched_ns::ScheduleKind::OneNodePerBlock;

void check_schedule(int schedule) {
    TORCH_CHECK(sched_ns::schedule_from_int(schedule) == kSchedule, "GATv2: only schedule 0 (one_per_block) is compiled, got ", schedule);
}

// Cap on the vectors per lane (S) of the fused shapes: every kernel holds at most three rows and
// two accumulators per lane and follows the default (two below 32 warps per block).
constexpr int max_steps(int warps) { return rows::kMaxSteps(warps); }

bool chunked(int64_t D, at::Tensor const& t, bool wide, int max_steps) {
    return rows::is_chunked(rows::plan_shape(D, t.element_size(), wide, max_steps));
}

void check_features(at::Tensor const& t, int64_t N, int64_t H, int64_t D, at::ScalarType dtype, char const *what) {
    TORCH_CHECK(t.is_cuda(), what, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() == 3 && t.size(0) == N && t.size(1) == H && t.size(2) == D, what, " must be [N, H, D] = [", N, ", ", H, ", ", D, "]");
    TORCH_CHECK(t.scalar_type() == dtype, what, " must have dtype ", dtype, ", got ", t.scalar_type());
    TORCH_CHECK(rows::features_contiguous(t), what, ": the feature dim (D) must have unit stride");
}

void check_csr(at::Tensor const& row_ptr, at::Tensor const& col_idx, int64_t N, char const *what) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), what, " must be CUDA tensors");
    TORCH_CHECK(row_ptr.is_contiguous() && col_idx.is_contiguous(), what, " must be contiguous");
    TORCH_CHECK(is_supported_index_type(row_ptr.scalar_type()), what, ": index tensors must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == row_ptr.scalar_type(), what, ": row_ptr and col_idx must have the same dtype");
    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.size(0) == N + 1, what, ": row_ptr must be [N + 1]");
    TORCH_CHECK(col_idx.dim() == 1, what, ": col_idx must be [E]");
    rows::check_index_range(row_ptr, col_idx, what);
}

// attn_vec [H, D] as an [1, H, D] view, for the row-layout checks.
bool attn_rows_aligned(at::Tensor const& attn_vec) { return rows::rows_16b_aligned(attn_vec.unsqueeze(0)); }

}  // namespace

std::vector<torch::Tensor> ATTN_SHARD_FN(gatv2_forward_cuda)(
    torch::Tensor l,         // [N, H, D] - left (destination) features
    torch::Tensor r,         // [N, H, D] - right (source) features
    torch::Tensor row_ptr,   // [N+1] - CSR row pointers
    torch::Tensor col_idx,   // [E] - CSR column indices
    torch::Tensor attn_vec,  // [H, D] - contiguous attention vector
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice,
    int pipeline_stages,
    // Separate depth for the heavy bucket: its staging buffer is warps * stages * D * sizeof(T),
    // so 8-32 heavy warps exhaust shared memory at a depth the 1-4 light warps absorb.
    int heavy_pipeline_stages
) {
    at::cuda::CUDAGuard device_guard(l.device());
    check_schedule(schedule);

    TORCH_CHECK(l.is_cuda() && l.dim() == 3, "l must be a CUDA tensor [N, H, D]");
    TORCH_CHECK(
        l.dtype() == torch::kFloat32 || l.dtype() == torch::kFloat16 || l.dtype() == torch::kBFloat16, "l must be float32, float16, or bfloat16"
    );
    const int64_t N = l.size(0);
    const int64_t H = l.size(1);
    const int64_t D = l.size(2);
    rows::check_head_dim(D, "GATv2 forward");
    check_features(l, N, H, D, l.scalar_type(), "l");
    check_features(r, N, H, D, l.scalar_type(), "r");
    TORCH_CHECK(attn_vec.is_cuda() && attn_vec.is_contiguous(), "attn_vec must be a contiguous CUDA tensor");
    TORCH_CHECK(attn_vec.dim() == 2 && attn_vec.size(0) == H && attn_vec.size(1) == D, "attn_vec must be [H, D]");
    TORCH_CHECK(attn_vec.scalar_type() == l.scalar_type(), "l, r, and attn_vec must have the same dtype");
    check_csr(row_ptr, col_idx, N, "GATv2 forward CSR");
    rows::check_node_list(light_nodes, row_ptr, "light_nodes");
    rows::check_node_list(heavy_nodes, row_ptr, "heavy_nodes");
    const auto idx_dtype = row_ptr.scalar_type();

    // h_out matches the input dtype, logsumexp is always float32.
    auto f32          = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());
    torch::Tensor O   = torch::empty({N, H, D}, l.options());
    torch::Tensor lse = torch::empty({N, H}, f32);
    if (N == 0 || H == 0) {
        return {O, lse};
    }

    // Row layout. O is a fresh allocation, aligned whenever the inputs are.
    const bool wide       = rows::rows_16b_aligned(l) && rows::rows_16b_aligned(r) && attn_rows_aligned(attn_vec);
    const uint32_t vec    = wide ? static_cast<uint32_t>(16 / l.element_size()) : 1u;
    const uint32_t nvec   = static_cast<uint32_t>(D) / vec;
    const int64_t n_edges = col_idx.numel();
    const bool any_heavy  = heavy_edge_slice > 0 ? chunk_node.numel() > 0 : heavy_nodes.numel() > 0;
    // Each bucket plans its own shape from its warp count; the scratch is needed if either chunks.
    const bool any_chunked = (light_nodes.numel() > 0 && chunked(D, l, wide, max_steps(light_warps_per_block))) ||
                             (any_heavy && chunked(D, l, wide, max_steps(heavy_warps_per_block)));

    // Chunked path: per-edge scores, [H, E].
    torch::Tensor scores = any_chunked ? torch::empty({H, n_edges}, f32) : torch::Tensor();

    auto make_params = [&]<typename cuda_t, typename index_t>() {
        gatv2::ForwardParams<cuda_t, index_t, float> p{};
        p.N              = N;
        p.H              = H;
        p.D              = static_cast<uint32_t>(D);
        p.nvec           = nvec;
        p.L              = reinterpret_cast<cuda_t const *>(l.data_ptr());
        p.R              = reinterpret_cast<cuda_t const *>(r.data_ptr());
        p.stride_l_n     = l.stride(0);
        p.stride_l_h     = l.stride(1);
        p.stride_r_n     = r.stride(0);
        p.stride_r_h     = r.stride(1);
        p.A              = reinterpret_cast<cuda_t const *>(attn_vec.data_ptr());
        p.negative_slope = negative_slope;
        p.row_ptr        = index_ptr<index_t>(row_ptr);
        p.col_idx        = index_ptr<index_t>(col_idx);
        p.num_edges      = n_edges;
        p.O              = reinterpret_cast<cuda_t *>(O.data_ptr());
        p.logsumexp      = lse.data_ptr<float>();
        p.scores         = any_chunked ? scores.data_ptr<float>() : nullptr;
        return p;
    };

    // Shared memory of one variant; must match the kernel's layout.
    auto smem_of = [&]<size_t W, int SHAPE, int STAGES, typename cuda_t, typename index_t>() -> size_t {
        if constexpr (rows::Shape<SHAPE>::kChunked) {
            return gatv2::ForwardChunked<W, rows::Shape<SHAPE>::kWide, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        } else {
            return gatv2::ForwardFused<W, SHAPE, STAGES, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        }
    };

    // Launches `launch<STAGES>` with the deepest pipeline the shape supports and the device's shared
    // memory fits, no deeper than `stages`.
    auto with_stages = [&]<size_t W, int SHAPE, typename cuda_t, typename index_t>(int stages, auto&& launch) {
        rows::with_fitted_stages<SHAPE>(
            stages, l.device(), [&]<int S>() { return smem_of.template operator()<W, SHAPE, S, cuda_t, index_t>(); }, launch
        );
    };

    stream_ns::BucketStreams buckets(stream_ns::bucket_launch_from_int(bucket_launch), l.device());
    buckets.record_all(l, r, attn_vec, O, lse, row_ptr, col_idx, light_nodes, scores);

    auto launch_bucket = [&](torch::Tensor& node_indices, auto warp_variant, at::cuda::CUDAStream stream, int stages) {
        const int count = static_cast<int>(node_indices.numel());
        if (count == 0) return;
        // Guarded so any host-side allocation below lands on this bucket's stream too.
        at::cuda::CUDAStreamGuard guard(stream);
        const int gx = sched_ns::persistent_grid_x(kSchedule, count, blocks_per_sm, static_cast<int>(H), sched_chunk);

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto warp_c) {
                using index_t      = typename decltype(idxInfo)::Type;
                using cuda_t       = typename decltype(typeInfo)::CudaType;
                constexpr size_t W = decltype(warp_c)::value;
                const auto p       = make_params.template operator()<cuda_t, index_t>();
                const auto sp      = sched_ns::make_params<index_t>(
                    kSchedule, index_ptr<index_t>(node_indices), count, {}, static_cast<int>(H), 0, {}, sched_chunk
                );

                rows::with_shape<max_steps(W)>(D, sizeof(cuda_t), wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<W, SHAPE, cuda_t, index_t>(stages, [&]<int STAGES>() {
                        auto kernel        = gatv2::gatv2_forward_kernel<kSchedule, W, SHAPE, STAGES, false, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GATv2 forward");
                        kernel<<<dim3(gx, static_cast<unsigned>(H)), W * kWarpSize, shmem, stream>>>(p, sp);
                    });
                });
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(l.scalar_type()), warp_variant
        );
    };

    // Edge-parallel heavy path: one block per fixed-size slice of a heavy node's edge list rather
    // than one block per node, so the grid is sized by the bucket's edge count instead of its node
    // count and every block gets the same amount of work. Partials are merged by a second kernel.
    // Selected by heavy_edge_slice > 0; 0 keeps the node-per-block path above.
    torch::Tensor part_o, part_ml;
    auto launch_heavy_split = [&](at::cuda::CUDAStream stream) {
        const int num_heavy  = static_cast<int>(heavy_nodes.numel());
        const int num_slices = static_cast<int>(chunk_node.numel());
        if (num_heavy == 0 || num_slices == 0) return;
        at::cuda::CUDAStreamGuard guard(stream);

        part_o  = torch::empty({num_slices, H, D}, f32);
        part_ml = torch::empty({num_slices, H, 2}, f32);
        buckets.record_all(part_o, part_ml, chunk_node, chunk_start, node_chunk_offset, heavy_nodes);

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto warp_c) {
                using index_t      = typename decltype(idxInfo)::Type;
                using cuda_t       = typename decltype(typeInfo)::CudaType;
                constexpr size_t W = decltype(warp_c)::value;

                auto p        = make_params.template operator()<cuda_t, index_t>();
                p.heavy_nodes = index_ptr<index_t>(heavy_nodes);
                p.chunk_node  = chunk_node.data_ptr<int>();
                p.chunk_start = chunk_start.data_ptr<int>();
                p.slice_size  = heavy_edge_slice;
                p.num_slices  = num_slices;
                p.part_o      = part_o.data_ptr<float>();
                p.part_ml     = part_ml.data_ptr<float>();

                rows::with_shape<max_steps(W)>(D, sizeof(cuda_t), wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<W, SHAPE, cuda_t, index_t>(heavy_pipeline_stages, [&]<int STAGES>() {
                        auto kernel        = gatv2::gatv2_forward_kernel<kSchedule, W, SHAPE, STAGES, true, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GATv2 forward (slice)");
                        kernel<<<dim3(num_slices, static_cast<unsigned>(H)), W * kWarpSize, shmem, stream>>>(p, {});
                    });
                });

                const rows::SliceTable<index_t> table{p.heavy_nodes, node_chunk_offset.data_ptr<int>(), num_heavy};
                rows::launch_softmax_merge<cuda_t, index_t>(table, H, p.D, nvec, wide, p.part_o, p.part_ml, p.O, H * D, D, p.logsumexp, stream);
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(l.scalar_type()),
            MakeIntVariant<8, 16, 32>(heavy_warps_per_block)
        );
    };

    stream_ns::run_buckets(
        buckets,
        [&](at::cuda::CUDAStream st) {
            const int warps = rows::light_warps_for(
                light_warps_per_block, D, l.element_size(), wide, light_nodes.numel() * H, max_steps(light_warps_per_block)
            );
            launch_bucket(light_nodes, MakeIntVariant<1, 2, 4>(warps), st, pipeline_stages);
        },
        [&](at::cuda::CUDAStream st) {
            if (heavy_edge_slice > 0) {
                launch_heavy_split(st);
            } else {
                launch_bucket(heavy_nodes, MakeIntVariant<8, 16, 32>(heavy_warps_per_block), st, heavy_pipeline_stages);
            }
        }
    );

    CUDA_KERNEL_CHECK();

    return {O, lse};
}

std::vector<torch::Tensor> ATTN_SHARD_FN(gatv2_backward_cuda)(
    torch::Tensor grad_h,     // [N, H, D] - gradient of the output
    torch::Tensor l,          // [N, H, D] - left features (saved)
    torch::Tensor r,          // [N, H, D] - right features (saved)
    torch::Tensor row_ptr,    // [N+1] - CSR row pointers
    torch::Tensor col_idx,    // [E] - CSR column indices
    torch::Tensor row_ptr_T,  // [N+1] - CSR^T row pointers
    torch::Tensor col_idx_T,  // [E] - CSR^T column indices
    torch::Tensor attn_vec,   // [H, D] - attention vector (saved)
    torch::Tensor logsumexp,  // [N, H] - logsumexp (saved)
    torch::Tensor out,        // [N, H, D] - forward output (saved)
    float negative_slope,
    int /*grad_A_reduce_row_chunk_size: unused since the gradient of `a` is reduced in-kernel*/,
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    // Edge-slice table over the forward CSR's heavy bucket; 0 keeps node-per-block.
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int backward_heavy_edge_slice,
    int pipeline_stages,
    // Separate depth for the heavy bucket; see gatv2_forward_cuda.
    int backward_heavy_pipeline_stages
) {
    at::cuda::CUDAGuard device_guard(l.device());
    check_schedule(schedule);

    TORCH_CHECK(l.is_cuda() && l.dim() == 3, "l must be a CUDA tensor [N, H, D]");
    TORCH_CHECK(
        l.dtype() == torch::kFloat32 || l.dtype() == torch::kFloat16 || l.dtype() == torch::kBFloat16, "l must be float32, float16, or bfloat16"
    );
    const int64_t N    = l.size(0);
    const int64_t H    = l.size(1);
    const int64_t D    = l.size(2);
    const auto dtype   = l.scalar_type();
    const auto options = l.options();
    rows::check_head_dim(D, "GATv2 backward");
    check_features(l, N, H, D, dtype, "l");
    check_features(r, N, H, D, dtype, "r");
    check_features(grad_h, N, H, D, dtype, "grad_h");
    check_features(out, N, H, D, dtype, "out");
    TORCH_CHECK(attn_vec.is_cuda() && attn_vec.is_contiguous(), "attn_vec must be a contiguous CUDA tensor");
    TORCH_CHECK(attn_vec.dim() == 2 && attn_vec.size(0) == H && attn_vec.size(1) == D, "attn_vec must be [H, D]");
    TORCH_CHECK(attn_vec.scalar_type() == dtype, "grad_h, l, r, and attn_vec must have the same dtype");
    TORCH_CHECK(logsumexp.is_cuda() && logsumexp.is_contiguous(), "logsumexp must be a contiguous CUDA tensor");
    TORCH_CHECK(logsumexp.scalar_type() == torch::kFloat32, "logsumexp must be float32");
    TORCH_CHECK(logsumexp.dim() == 2 && logsumexp.size(0) == N && logsumexp.size(1) == H, "logsumexp must be [N, H]");
    check_csr(row_ptr, col_idx, N, "GATv2 backward forward CSR");
    rows::check_node_list(fwd_light_nodes, row_ptr, "fwd_light_nodes");
    rows::check_node_list(fwd_heavy_nodes, row_ptr, "fwd_heavy_nodes");
    if (is_directed) {
        check_csr(row_ptr_T, col_idx_T, N, "GATv2 backward CSR^T");
        TORCH_CHECK(row_ptr_T.scalar_type() == row_ptr.scalar_type(), "CSR^T must have the same index dtype as the forward CSR");
        rows::check_node_list(bwd_light_nodes, row_ptr_T, "bwd_light_nodes");
        rows::check_node_list(bwd_heavy_nodes, row_ptr_T, "bwd_heavy_nodes");
    }
    const auto idx_dtype = row_ptr.scalar_type();

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());

    // Row gradients in the input dtype, each row written once. The gradient of `a` is summed into
    // kGradASlots fp32 rows per head, reduced at the end.
    torch::Tensor dL       = torch::empty({N, H, D}, options);
    torch::Tensor dR       = torch::empty({N, H, D}, options);
    torch::Tensor dA_slots = torch::zeros({gatv2::kGradASlots, H, D}, f32);
    if (N == 0 || H == 0) {
        return {dL, dR, torch::zeros({H, D}, options)};
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(l.device().index());

    // Delta[i, h] = <grad_h[i, h], out[i, h]>, on the caller's stream.
    torch::Tensor delta = torch::empty({N, H}, f32);
    std::visit(
        [&](auto typeInfo) { rows::launch_row_dot<typename decltype(typeInfo)::CudaType>(grad_h, out, delta, stream); },
        MakeTypeVariant<ATTN_SHARD_TYPE>(dtype)
    );

    // Row layout. The gradients are fresh allocations, aligned whenever the inputs are.
    const bool wide = rows::rows_16b_aligned(l) && rows::rows_16b_aligned(r) && rows::rows_16b_aligned(grad_h) && attn_rows_aligned(attn_vec);
    const uint32_t vec    = wide ? static_cast<uint32_t>(16 / l.element_size()) : 1u;
    const uint32_t nvec   = static_cast<uint32_t>(D) / vec;
    const int64_t n_edges = col_idx.numel();

    // The launches: AL over the forward CSR, and R over the CSR^T -- the forward CSR itself when the
    // graph is undirected. Each walks its CSR's light/heavy buckets. The slice table is built over
    // the forward CSR's heavy bucket, so it serves AL always and R on an undirected graph.
    struct Launch {
        BackwardMode mode;
        torch::Tensor const& row_ptr;
        torch::Tensor const& col_idx;
        torch::Tensor& light;
        torch::Tensor& heavy;
        bool sliced;
        torch::Tensor scalars;  // chunked path: per-edge scalars, [kBackwardScalars, H, E]
        torch::Tensor part;     // split-K partials, [num_slices, H, D]
    };
    const bool sliced          = backward_heavy_edge_slice > 0 && chunk_node.numel() > 0 && fwd_heavy_nodes.numel() > 0;
    torch::Tensor const& t_ptr = is_directed ? row_ptr_T : row_ptr;
    torch::Tensor const& t_idx = is_directed ? col_idx_T : col_idx;
    torch::Tensor& t_light     = is_directed ? bwd_light_nodes : fwd_light_nodes;
    torch::Tensor& t_heavy     = is_directed ? bwd_heavy_nodes : fwd_heavy_nodes;
    Launch launches[2]         = {
        {BackwardMode::AL, row_ptr, col_idx, fwd_light_nodes, fwd_heavy_nodes, sliced, {}, {}},
        {BackwardMode::R, t_ptr, t_idx, t_light, t_heavy, sliced && !is_directed, {}, {}},
    };

    // Chunked path: per-edge scalars, [scalars, H, E], one scratch per launch.
    for (Launch& ln : launches) {
        const bool any_heavy = ln.sliced || ln.heavy.numel() > 0;
        if ((ln.light.numel() > 0 && chunked(D, l, wide, max_steps(light_warps_per_block))) ||
            (any_heavy && chunked(D, l, wide, max_steps(heavy_warps_per_block)))) {
            ln.scalars = torch::empty({ln.mode == BackwardMode::AL ? 1 : 2, H, n_edges}, f32);
        }
    }

    auto make_params = [&]<typename cuda_t, typename index_t>(Launch const& ln) {
        gatv2::BackwardParams<cuda_t, index_t, float> p{};
        p.N              = N;
        p.H              = H;
        p.D              = static_cast<uint32_t>(D);
        p.nvec           = nvec;
        p.L              = reinterpret_cast<cuda_t const *>(l.data_ptr());
        p.R              = reinterpret_cast<cuda_t const *>(r.data_ptr());
        p.G              = reinterpret_cast<cuda_t const *>(grad_h.data_ptr());
        p.stride_l_n     = l.stride(0);
        p.stride_l_h     = l.stride(1);
        p.stride_r_n     = r.stride(0);
        p.stride_r_h     = r.stride(1);
        p.stride_g_n     = grad_h.stride(0);
        p.stride_g_h     = grad_h.stride(1);
        p.A              = reinterpret_cast<cuda_t const *>(attn_vec.data_ptr());
        p.negative_slope = negative_slope;
        p.logsumexp      = logsumexp.data_ptr<float>();
        p.delta          = delta.data_ptr<float>();
        p.row_ptr        = index_ptr<index_t>(ln.row_ptr);
        p.col_idx        = index_ptr<index_t>(ln.col_idx);
        p.num_edges      = n_edges;
        p.dL             = reinterpret_cast<cuda_t *>(dL.data_ptr());
        p.dR             = reinterpret_cast<cuda_t *>(dR.data_ptr());
        p.dA_slots       = dA_slots.data_ptr<float>();
        p.edge_scalars   = ln.scalars.defined() ? ln.scalars.data_ptr<float>() : nullptr;
        return p;
    };

    auto smem_of = [&]<BackwardMode MODE, size_t W, int SHAPE, int STAGES, typename cuda_t, typename index_t>() -> size_t {
        if constexpr (rows::Shape<SHAPE>::kChunked) {
            return gatv2::BackwardChunked<MODE, W, rows::Shape<SHAPE>::kWide, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        } else {
            return gatv2::BackwardFused<MODE, W, SHAPE, STAGES, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        }
    };

    auto with_stages = [&]<BackwardMode MODE, size_t W, int SHAPE, typename cuda_t, typename index_t>(int stages, auto&& launch) {
        rows::with_fitted_stages<SHAPE>(
            stages, l.device(), [&]<int S>() { return smem_of.template operator()<MODE, W, SHAPE, S, cuda_t, index_t>(); }, launch
        );
    };

    // One mode's light or heavy bucket, node per block.
    auto launch_bucket = [&](Launch const& ln, torch::Tensor& node_indices, auto warp_variant, at::cuda::CUDAStream bucket_stream, int stages) {
        const int count = static_cast<int>(node_indices.numel());
        if (count == 0) return;
        at::cuda::CUDAStreamGuard guard(bucket_stream);
        const int gx = sched_ns::persistent_grid_x(kSchedule, count, blocks_per_sm, static_cast<int>(H), sched_chunk);

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto warp_c, auto mode_c) {
                using index_t       = typename decltype(idxInfo)::Type;
                using cuda_t        = typename decltype(typeInfo)::CudaType;
                constexpr size_t W  = decltype(warp_c)::value;
                constexpr auto MODE = decltype(mode_c)::value;
                const auto p        = make_params.template operator()<cuda_t, index_t>(ln);
                const auto sp       = sched_ns::make_params<index_t>(
                    kSchedule, index_ptr<index_t>(node_indices), count, {}, static_cast<int>(H), 0, {}, sched_chunk
                );

                rows::with_shape<max_steps(W)>(D, sizeof(cuda_t), wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<MODE, W, SHAPE, cuda_t, index_t>(stages, [&]<int STAGES>() {
                        auto kernel        = gatv2::gatv2_backward_kernel<MODE, kSchedule, W, SHAPE, STAGES, false, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<MODE, W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GATv2 backward");
                        kernel<<<dim3(gx, static_cast<unsigned>(H)), W * kWarpSize, shmem, bucket_stream>>>(p, sp);
                    });
                });
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(dtype), warp_variant,
            MakeEnumVariant<BackwardMode, BackwardMode::AL, BackwardMode::R>(ln.mode)
        );
    };

    // Edge-parallel heavy path. The partials are plain sums over the row's edges; the
    // gradient of `a` goes straight to its slot rows.
    auto launch_heavy_split = [&](Launch& ln, at::cuda::CUDAStream bucket_stream) {
        const int num_heavy  = static_cast<int>(ln.heavy.numel());
        const int num_slices = static_cast<int>(chunk_node.numel());
        if (num_heavy == 0 || num_slices == 0) return;
        at::cuda::CUDAStreamGuard guard(bucket_stream);

        ln.part = torch::empty({num_slices, H, D}, f32);
        std::visit(
            [&](auto idxInfo, auto typeInfo, auto warp_c, auto mode_c) {
                using index_t       = typename decltype(idxInfo)::Type;
                using cuda_t        = typename decltype(typeInfo)::CudaType;
                constexpr size_t W  = decltype(warp_c)::value;
                constexpr auto MODE = decltype(mode_c)::value;

                auto p        = make_params.template operator()<cuda_t, index_t>(ln);
                p.heavy_nodes = index_ptr<index_t>(ln.heavy);
                p.chunk_node  = chunk_node.data_ptr<int>();
                p.chunk_start = chunk_start.data_ptr<int>();
                p.slice_size  = backward_heavy_edge_slice;
                p.num_slices  = num_slices;
                p.part        = ln.part.data_ptr<float>();

                rows::with_shape<max_steps(W)>(D, sizeof(cuda_t), wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<MODE, W, SHAPE, cuda_t, index_t>(backward_heavy_pipeline_stages, [&]<int STAGES>() {
                        auto kernel        = gatv2::gatv2_backward_kernel<MODE, kSchedule, W, SHAPE, STAGES, true, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<MODE, W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GATv2 backward (slice)");
                        kernel<<<dim3(num_slices, static_cast<unsigned>(H)), W * kWarpSize, shmem, bucket_stream>>>(p, {});
                    });
                });

                const rows::SliceTable<index_t> table{p.heavy_nodes, node_chunk_offset.data_ptr<int>(), num_heavy};
                cuda_t *const grad = MODE == BackwardMode::AL ? p.dL : p.dR;
                rows::launch_sum_merge<1, cuda_t, index_t>(table, H, p.D, nvec, wide, {{p.part}, {grad}}, bucket_stream);
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(dtype), MakeIntVariant<8, 16, 32>(heavy_warps_per_block),
            MakeEnumVariant<BackwardMode, BackwardMode::AL, BackwardMode::R>(ln.mode)
        );
    };

    auto run_launch = [&](Launch& ln) {
        stream_ns::BucketStreams buckets(stream_ns::bucket_launch_from_int(bucket_launch), l.device());
        buckets.record_all(l, r, grad_h, attn_vec, logsumexp, delta, ln.row_ptr, ln.col_idx, ln.light, dL, dR, dA_slots, ln.scalars);
        stream_ns::run_buckets(
            buckets,
            [&](at::cuda::CUDAStream st) {
                const int warps = rows::light_warps_for(
                    light_warps_per_block, D, l.element_size(), wide, ln.light.numel() * H, max_steps(light_warps_per_block)
                );
                launch_bucket(ln, ln.light, MakeIntVariant<1, 2, 4>(warps), st, pipeline_stages);
            },
            [&](at::cuda::CUDAStream st) {
                if (ln.sliced) {
                    launch_heavy_split(ln, st);
                } else {
                    launch_bucket(ln, ln.heavy, MakeIntVariant<8, 16, 32>(heavy_warps_per_block), st, backward_heavy_pipeline_stages);
                }
            }
        );
    };

    // AL on the caller's stream, R on a side stream: they share no outputs, and overlapping them lets
    // either fill the device while the other waits on its longest rows. Measured against running
    // them in turn: 1.25-1.57x on an undirected power-law graph without edge slicing, where both
    // launches have a straggler, and within a few percent either way elsewhere. The side stream
    // joins the caller's before anything returns.
    {
        at::cuda::CUDAStream side = at::cuda::getStreamFromPool(/*isHighPriority=*/false, l.device().index());
        at::cuda::CUDAEvent fork;
        fork.record(stream);
        fork.block(side);
        run_launch(launches[0]);
        {
            at::cuda::CUDAStreamGuard guard(side);
            run_launch(launches[1]);
        }
        at::cuda::CUDAEvent done;
        done.record(side);
        done.block(stream);
    }

    CUDA_KERNEL_CHECK();

    // The slots summed in a fixed order, then the input dtype (a no-op for fp32).
    torch::Tensor dA = dA_slots.sum(0).to(dtype);
    return {dL, dR, dA};
}
