// Compiled once per dtype through gt/gt_shard_{f32,f16,bf16}.cu, which define
// ATTN_SHARD_TYPE and ATTN_SHARD_FN. One translation unit instantiating every dtype took
// most of the build; split three ways the shards compile in parallel.
#ifndef ATTN_SHARD_TYPE
#error "compile this file through its per-dtype shards"
#endif

#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>

#include "common/row_kernels.cuh"
#include "gt/gt_backward.cu"
#include "gt/gt_forward.cu"

namespace {

namespace gt        = turbo_gnn::gt;
namespace rows      = turbo_gnn::rows;
namespace sched_ns  = turbo_gnn::sched;
namespace stream_ns = turbo_gnn::streams;

// Only the one-node-per-block schedule is compiled into the attention kernels (build time).
inline constexpr auto kSchedule = sched_ns::ScheduleKind::OneNodePerBlock;

void check_schedule(int schedule) {
    TORCH_CHECK(
        sched_ns::schedule_from_int(schedule) == kSchedule, "graph transformer: only schedule 0 (one_per_block) is compiled, got ", schedule
    );
}

// Whether a launch with `warps` warps per block takes the chunked path (see rows::kMaxSteps).
bool chunked_for(int64_t D, at::Tensor const& t, bool wide, int warps) {
    return rows::is_chunked(rows::plan_shape(D, t.element_size(), wide, rows::kMaxSteps(warps)));
}

// Runs fn(shape_c) for the row shape of a W-warp launch, shape_c an integral_constant.
template <size_t W, typename Fn>
void with_shape(int64_t D, at::Tensor const& t, bool wide, Fn&& fn) {
    rows::with_shape<rows::kMaxSteps(static_cast<int>(W))>(D, t.element_size(), wide, std::forward<Fn>(fn));
}

// Light-bucket warps; see rows::light_warps_for.
int light_warps_for(int requested, int64_t D, at::Tensor const& t, bool wide, int64_t blocks) {
    return rows::light_warps_for(requested, D, t.element_size(), wide, blocks, rows::kMaxSteps(requested));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> ATTN_SHARD_FN(graph_attention_forward_csr_mh_cuda)(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
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
    // Separate depth for the heavy bucket. The staging buffer is
    // warps*stages*D*sizeof(T), so 8-32 heavy warps exhaust shared memory at a depth the
    // 1-4 light warps absorb.
    int heavy_pipeline_stages
) {
    at::cuda::CUDAGuard device_guard(Q.device());
    check_schedule(schedule);

    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "CSR indices must be CUDA");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be [N, H, D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V sizes must match");
    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "Q, K, V must have the same dtype");
    TORCH_CHECK(
        Q.dtype() == torch::kFloat32 || Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16, "Q must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(rows::features_contiguous(Q) && rows::features_contiguous(K) && rows::features_contiguous(V), "Feature dim must be contiguous");

    const auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");
    rows::check_index_range(row_ptr, col_idx, "GT forward");
    rows::check_node_list(light_nodes, row_ptr, "light_nodes");
    rows::check_node_list(heavy_nodes, row_ptr, "heavy_nodes");

    const int64_t N = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t D = Q.size(2);
    rows::check_head_dim(D, "GT forward");

    // O matches the input dtype, logsumexp is always float32.
    auto f32          = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    torch::Tensor O   = torch::empty({N, H, D}, Q.options());
    torch::Tensor lse = torch::empty({N, H}, f32);
    if (N == 0 || H == 0) {
        return std::make_tuple(O, lse);
    }

    // Row layout. O is a fresh allocation: its rows are aligned whenever Q's are.
    const bool wide       = rows::rows_16b_aligned(Q) && rows::rows_16b_aligned(K) && rows::rows_16b_aligned(V);
    const uint32_t vec    = wide ? static_cast<uint32_t>(16 / Q.element_size()) : 1u;
    const uint32_t nvec   = static_cast<uint32_t>(D) / vec;
    const int64_t n_edges = col_idx.numel();
    const bool any_heavy  = heavy_edge_slice > 0 ? chunk_node.numel() > 0 : heavy_nodes.numel() > 0;
    // Each bucket plans its own shape from its warp count; the scratch is needed if either chunks.
    const bool chunked = (light_nodes.numel() > 0 && chunked_for(D, Q, wide, light_warps_per_block)) ||
                         (any_heavy && chunked_for(D, Q, wide, heavy_warps_per_block));

    // Chunked path: per-edge scores, [H, E].
    torch::Tensor scores = chunked ? torch::empty({H, n_edges}, f32) : torch::Tensor();

    auto make_params = [&]<typename cuda_t, typename index_t>() {
        gt::ForwardParams<cuda_t, index_t, float> p{};
        p.N          = N;
        p.H          = H;
        p.D          = static_cast<uint32_t>(D);
        p.nvec       = nvec;
        p.Q          = reinterpret_cast<cuda_t const *>(Q.data_ptr());
        p.K          = reinterpret_cast<cuda_t const *>(K.data_ptr());
        p.V          = reinterpret_cast<cuda_t const *>(V.data_ptr());
        p.stride_q_n = Q.stride(0);
        p.stride_q_h = Q.stride(1);
        p.stride_k_n = K.stride(0);
        p.stride_k_h = K.stride(1);
        p.stride_v_n = V.stride(0);
        p.stride_v_h = V.stride(1);
        p.row_ptr    = index_ptr<index_t>(row_ptr);
        p.col_idx    = index_ptr<index_t>(col_idx);
        p.num_edges  = n_edges;
        p.scale      = scale;
        p.O          = reinterpret_cast<cuda_t *>(O.data_ptr());
        p.stride_o_n = O.stride(0);
        p.stride_o_h = O.stride(1);
        p.logsumexp  = lse.data_ptr<float>();
        p.scores     = chunked ? scores.data_ptr<float>() : nullptr;
        return p;
    };

    // Shared memory of one variant; must match the kernel's layout.
    auto smem_of = [&]<size_t W, int SHAPE, int STAGES, typename cuda_t, typename index_t>() -> size_t {
        if constexpr (gt::Shape<SHAPE>::kChunked) {
            return gt::ForwardChunked<W, gt::Shape<SHAPE>::kWide, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        } else {
            return gt::ForwardFused<W, SHAPE, STAGES, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        }
    };

    // Launches `launch<STAGES>` with the deepest pipeline the shape supports and the device's
    // shared memory fits, no deeper than `stages`.
    auto with_stages = [&]<size_t W, int SHAPE, typename cuda_t, typename index_t>(int stages, auto&& launch) {
        rows::with_fitted_stages<SHAPE>(
            stages, Q.device(), [&]<int S>() { return smem_of.template operator()<W, SHAPE, S, cuda_t, index_t>(); }, launch
        );
    };

    stream_ns::BucketStreams buckets(stream_ns::bucket_launch_from_int(bucket_launch), Q.device());
    buckets.record_all(Q, K, V, O, lse, row_ptr, col_idx, light_nodes, scores);

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

                with_shape<W>(D, Q, wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<W, SHAPE, cuda_t, index_t>(stages, [&]<int STAGES>() {
                        auto kernel        = gt::gt_forward_kernel<kSchedule, W, SHAPE, STAGES, false, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GT forward");
                        kernel<<<dim3(gx, static_cast<unsigned>(H)), W * kWarpSize, shmem, stream>>>(p, sp);
                    });
                });
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(Q.scalar_type()), warp_variant
        );
    };

    // Edge-parallel heavy path. One block per fixed-size slice of a heavy node's edge list
    // rather than one block per node, so the grid is sized by the bucket's edge count instead of
    // its node count and every block gets the same amount of work. Partials are merged by a
    // second kernel. Selected by heavy_edge_slice > 0; 0 keeps the node-per-block path above.
    torch::Tensor part_o, part_ml;
    auto launch_heavy_split = [&](at::cuda::CUDAStream stream) {
        const int num_heavy  = static_cast<int>(heavy_nodes.numel());
        const int num_slices = static_cast<int>(chunk_node.numel());
        if (num_heavy == 0 || num_slices == 0) return;
        at::cuda::CUDAStreamGuard guard(stream);

        part_o  = torch::empty({num_slices, H, D}, f32);
        part_ml = torch::empty({num_slices, H, 2}, f32);
        // Allocated on this stream, so the allocator must be told before the join.
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

                with_shape<W>(D, Q, wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<W, SHAPE, cuda_t, index_t>(heavy_pipeline_stages, [&]<int STAGES>() {
                        auto kernel        = gt::gt_forward_kernel<kSchedule, W, SHAPE, STAGES, true, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GT forward (slice)");
                        kernel<<<dim3(num_slices, static_cast<unsigned>(H)), W * kWarpSize, shmem, stream>>>(p, {});
                    });
                });

                const rows::SliceTable<index_t> table{p.heavy_nodes, node_chunk_offset.data_ptr<int>(), num_heavy};
                rows::launch_softmax_merge<cuda_t, index_t>(
                    table, H, p.D, nvec, wide, p.part_o, p.part_ml, p.O, p.stride_o_n, p.stride_o_h, p.logsumexp, stream
                );
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(Q.scalar_type()),
            MakeIntVariant<8, 16, 32>(heavy_warps_per_block)
        );
    };

    stream_ns::run_buckets(
        buckets,
        [&](at::cuda::CUDAStream st) {
            const int warps = light_warps_for(light_warps_per_block, D, Q, wide, light_nodes.numel() * H);
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

    return std::make_tuple(O, lse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ATTN_SHARD_FN(graph_attention_backward_csr_mh_cuda)(
    torch::Tensor row_ptr,    // [N+1], forward CSR,
    torch::Tensor col_idx,    // [E],   forward CSR,
    torch::Tensor row_ptr_T,  // [N+1], CSR^T (backward)
    torch::Tensor col_idx_T,  // [E],   CSR^T (backward)
    torch::Tensor Q,          // [N, H, D],
    torch::Tensor K,          // [N, H, D],
    torch::Tensor V,          // [N, H, D],
    torch::Tensor O,          // [N, H, D] (forward output)
    torch::Tensor dO,         // [N, H, D],
    torch::Tensor logsumexp,  // [N, H],   float32,
    float scale,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed,
    int schedule,
    int blocks_per_sm,
    int sched_chunk,
    int bucket_launch,
    torch::Tensor chunk_node,
    torch::Tensor chunk_start,
    torch::Tensor node_chunk_offset,
    int heavy_edge_slice,
    int pipeline_stages,
    // Separate depth for the heavy bucket. The staging buffer is
    // warps*stages*D*sizeof(T), so 8-32 heavy warps exhaust shared memory at a depth the
    // 1-4 light warps absorb.
    int backward_heavy_pipeline_stages
) {
    at::cuda::CUDAGuard device_guard(Q.device());
    check_schedule(schedule);

    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda(), "Forward CSR indices must be CUDA");
    TORCH_CHECK(row_ptr_T.is_cuda() && col_idx_T.is_cuda(), "CSR^T indices must be CUDA");
    TORCH_CHECK(
        Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda() && dO.is_cuda() && logsumexp.is_cuda(),
        "Q, K, V, O, dO, logsumexp must be CUDA"
    );
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3 && O.dim() == 3 && dO.dim() == 3, "Q, K, V, O, dO must be [N, H, D]");
    TORCH_CHECK(
        Q.sizes() == K.sizes() && Q.sizes() == V.sizes() && Q.sizes() == O.sizes() && Q.sizes() == dO.sizes(),
        "Q, K, V, O, dO sizes must match [N, H, D]"
    );
    TORCH_CHECK(
        Q.dtype() == K.dtype() && Q.dtype() == V.dtype() && Q.dtype() == O.dtype() && Q.dtype() == dO.dtype(),
        "Q, K, V, O, dO must have the same dtype"
    );
    TORCH_CHECK(
        Q.dtype() == torch::kFloat32 || Q.dtype() == torch::kFloat16 || Q.dtype() == torch::kBFloat16, "Q must be float32, float16, or bfloat16"
    );

    const auto idx_dtype = row_ptr_T.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype), "row_ptr_T must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx_T.scalar_type() == idx_dtype, "col_idx_T must have same dtype as row_ptr_T");
    TORCH_CHECK(row_ptr.scalar_type() == idx_dtype && col_idx.scalar_type() == idx_dtype, "forward CSR must have the same dtype as CSR^T");
    rows::check_index_range(row_ptr_T, col_idx_T, "GT backward");
    rows::check_node_list(light_nodes, row_ptr_T, "light_nodes");
    rows::check_node_list(heavy_nodes, row_ptr_T, "heavy_nodes");

    TORCH_CHECK(logsumexp.dtype() == torch::kFloat32, "logsumexp must be float32");
    TORCH_CHECK(logsumexp.dim() == 2, "logsumexp must be [N, H]");

    const int64_t N = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t D = Q.size(2);
    rows::check_head_dim(D, "GT backward");

    TORCH_CHECK(row_ptr_T.dim() == 1 && row_ptr_T.size(0) == N + 1, "row_ptr_T must be [N+1]");
    TORCH_CHECK(col_idx_T.dim() == 1, "col_idx_T must be [E]");
    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.size(0) == N + 1, "row_ptr must be [N+1]");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must be [E]");
    TORCH_CHECK(logsumexp.size(0) == N && logsumexp.size(1) == H, "logsumexp must be [N, H]");

    TORCH_CHECK(
        rows::features_contiguous(Q) && rows::features_contiguous(K) && rows::features_contiguous(V) && rows::features_contiguous(O),
        "feature dim (D) must be contiguous (stride(2) == 1) for Q, K, V, O"
    );
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous [N, H, D]");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous [N, H]");
    TORCH_CHECK(row_ptr_T.is_contiguous() && col_idx_T.is_contiguous(), "CSR^T arrays must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous() && col_idx.is_contiguous(), "Forward CSR arrays must be contiguous");

    auto f32_options   = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto typed_options = Q.options();

    // Directed: dK is scattered with fp32 atomics; undirected: written once per row, no atomics.
    torch::Tensor dQ = torch::empty({N, H, D}, typed_options);
    torch::Tensor dV = torch::empty({N, H, D}, typed_options);
    torch::Tensor dK_f32, dK_typed;
    if (is_directed) {
        dK_f32 = torch::zeros({N, H, D}, f32_options);
    } else {
        dK_typed = torch::empty({N, H, D}, typed_options);
    }
    if (N == 0 || H == 0) {
        return std::make_tuple(dQ, is_directed ? dK_f32.to(Q.scalar_type()) : dK_typed, dV);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(Q.device().index());

    // Row layout. dO is contiguous; the gradients are fresh allocations, aligned whenever dO is.
    const bool wide     = rows::rows_16b_aligned(Q) && rows::rows_16b_aligned(K) && rows::rows_16b_aligned(V) && rows::rows_16b_aligned(dO);
    const uint32_t vec  = wide ? static_cast<uint32_t>(16 / Q.element_size()) : 1u;
    const uint32_t nvec = static_cast<uint32_t>(D) / vec;

    // Delta[i,h] = <O[i,h,:], dO[i,h,:]>, on the caller's stream. O and dO have their own layout: Q
    // may be an offset view while they are not.
    torch::Tensor Delta = torch::empty({N, H}, f32_options);
    std::visit(
        [&](auto typeInfo) { rows::launch_row_dot<typename decltype(typeInfo)::CudaType>(dO, O, Delta, stream); },
        MakeTypeVariant<ATTN_SHARD_TYPE>(Q.scalar_type())
    );

    // The CSR a launch walks: the transpose when directed, the forward CSR -- its own transpose --
    // when undirected. The light/heavy buckets and slice tables passed in are the backward ones,
    // built from the same CSR.
    const gt::BackwardMode mode   = is_directed ? gt::BackwardMode::Directed : gt::BackwardMode::Undirected;
    torch::Tensor const& walk_ptr = is_directed ? row_ptr_T : row_ptr;
    torch::Tensor const& walk_idx = is_directed ? col_idx_T : col_idx;
    const int64_t n_edges         = walk_idx.numel();
    const bool any_heavy          = heavy_edge_slice > 0 ? chunk_node.numel() > 0 : heavy_nodes.numel() > 0;
    const bool chunked            = (light_nodes.numel() > 0 && chunked_for(D, Q, wide, light_warps_per_block)) ||
                                    (any_heavy && chunked_for(D, Q, wide, heavy_warps_per_block));

    // Chunked path: per-edge scalars, [scalars, H, E].
    torch::Tensor edge_scalars = chunked ? torch::empty({is_directed ? 2 : 3, H, n_edges}, f32_options) : torch::Tensor();

    auto make_params = [&]<typename cuda_t, typename index_t>() {
        gt::BackwardParams<cuda_t, index_t, float> p{};
        p.N            = N;
        p.H            = H;
        p.D            = static_cast<uint32_t>(D);
        p.nvec         = nvec;
        p.Q            = reinterpret_cast<cuda_t const *>(Q.data_ptr());
        p.K            = reinterpret_cast<cuda_t const *>(K.data_ptr());
        p.V            = reinterpret_cast<cuda_t const *>(V.data_ptr());
        p.dO           = reinterpret_cast<cuda_t const *>(dO.data_ptr());
        p.stride_q_n   = Q.stride(0);
        p.stride_q_h   = Q.stride(1);
        p.stride_k_n   = K.stride(0);
        p.stride_k_h   = K.stride(1);
        p.stride_v_n   = V.stride(0);
        p.stride_v_h   = V.stride(1);
        p.logsumexp    = logsumexp.data_ptr<float>();
        p.delta        = Delta.data_ptr<float>();
        p.row_ptr      = index_ptr<index_t>(walk_ptr);
        p.col_idx      = index_ptr<index_t>(walk_idx);
        p.num_edges    = n_edges;
        p.scale        = scale;
        p.dQ           = reinterpret_cast<cuda_t *>(dQ.data_ptr());
        p.dV           = reinterpret_cast<cuda_t *>(dV.data_ptr());
        p.dK           = is_directed ? nullptr : reinterpret_cast<cuda_t *>(dK_typed.data_ptr());
        p.dK_atomic    = is_directed ? dK_f32.data_ptr<float>() : nullptr;
        p.edge_scalars = chunked ? edge_scalars.data_ptr<float>() : nullptr;
        return p;
    };

    auto smem_of = [&]<gt::BackwardMode MODE, size_t W, int SHAPE, int STAGES, typename cuda_t, typename index_t>() -> size_t {
        if constexpr (gt::Shape<SHAPE>::kChunked) {
            return gt::BackwardChunked<MODE, W, gt::Shape<SHAPE>::kWide, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        } else {
            return gt::BackwardFused<MODE, W, SHAPE, STAGES, cuda_t, index_t, float>::smem_bytes(static_cast<uint32_t>(D));
        }
    };

    auto with_stages = [&]<gt::BackwardMode MODE, size_t W, int SHAPE, typename cuda_t, typename index_t>(int stages, auto&& launch) {
        rows::with_fitted_stages<SHAPE>(
            stages, Q.device(), [&]<int S>() { return smem_of.template operator()<MODE, W, SHAPE, S, cuda_t, index_t>(); }, launch
        );
    };

    auto mode_variant = [&] { return MakeEnumVariant<gt::BackwardMode, gt::BackwardMode::Directed, gt::BackwardMode::Undirected>(mode); };

    stream_ns::BucketStreams buckets(stream_ns::bucket_launch_from_int(bucket_launch), Q.device());
    buckets.record_all(Q, K, V, dO, logsumexp, Delta, walk_ptr, walk_idx, light_nodes, dQ, dV, dK_f32, dK_typed, edge_scalars);

    auto launch_bucket = [&](torch::Tensor& node_indices, auto warp_variant, at::cuda::CUDAStream bucket_stream, int stages) {
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
                const auto p        = make_params.template operator()<cuda_t, index_t>();
                const auto sp       = sched_ns::make_params<index_t>(
                    kSchedule, index_ptr<index_t>(node_indices), count, {}, static_cast<int>(H), 0, {}, sched_chunk
                );

                with_shape<W>(D, Q, wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<MODE, W, SHAPE, cuda_t, index_t>(stages, [&]<int STAGES>() {
                        auto kernel        = gt::gt_backward_kernel<MODE, kSchedule, W, SHAPE, STAGES, false, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<MODE, W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GT backward");
                        kernel<<<dim3(gx, static_cast<unsigned>(H)), W * kWarpSize, shmem, bucket_stream>>>(p, sp);
                    });
                });
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(Q.scalar_type()), warp_variant, mode_variant()
        );
    };

    // Edge-parallel heavy path. The partials are plain sums over the row's edges; the directed
    // dK scatter is written directly by the slice kernel and needs no merge.
    torch::Tensor part_q, part_v, part_k;
    auto launch_heavy_split = [&](at::cuda::CUDAStream bucket_stream) {
        const int num_heavy  = static_cast<int>(heavy_nodes.numel());
        const int num_slices = static_cast<int>(chunk_node.numel());
        if (num_heavy == 0 || num_slices == 0) return;
        at::cuda::CUDAStreamGuard guard(bucket_stream);

        part_q = torch::empty({num_slices, H, D}, f32_options);
        part_v = torch::empty({num_slices, H, D}, f32_options);
        if (!is_directed) {
            part_k = torch::empty({num_slices, H, D}, f32_options);
        }
        buckets.record_all(part_q, part_v, part_k, chunk_node, chunk_start, node_chunk_offset, heavy_nodes);

        std::visit(
            [&](auto idxInfo, auto typeInfo, auto warp_c, auto mode_c) {
                using index_t       = typename decltype(idxInfo)::Type;
                using cuda_t        = typename decltype(typeInfo)::CudaType;
                constexpr size_t W  = decltype(warp_c)::value;
                constexpr auto MODE = decltype(mode_c)::value;

                auto p        = make_params.template operator()<cuda_t, index_t>();
                p.heavy_nodes = index_ptr<index_t>(heavy_nodes);
                p.chunk_node  = chunk_node.data_ptr<int>();
                p.chunk_start = chunk_start.data_ptr<int>();
                p.slice_size  = heavy_edge_slice;
                p.num_slices  = num_slices;
                p.part_q      = part_q.data_ptr<float>();
                p.part_v      = part_v.data_ptr<float>();
                p.part_k      = part_k.defined() ? part_k.data_ptr<float>() : nullptr;

                with_shape<W>(D, Q, wide, [&](auto shape_c) {
                    constexpr int SHAPE = decltype(shape_c)::value;
                    with_stages.template operator()<MODE, W, SHAPE, cuda_t, index_t>(backward_heavy_pipeline_stages, [&]<int STAGES>() {
                        auto kernel        = gt::gt_backward_kernel<MODE, kSchedule, W, SHAPE, STAGES, true, cuda_t, index_t, float>;
                        const size_t shmem = smem_of.template operator()<MODE, W, SHAPE, STAGES, cuda_t, index_t>();
                        ensure_dynamic_shmem(kernel, shmem, "GT backward (slice)");
                        kernel<<<dim3(num_slices, static_cast<unsigned>(H)), W * kWarpSize, shmem, bucket_stream>>>(p, {});
                    });
                });

                constexpr size_t kGrads = gt::kBackwardGrads<MODE>;
                const rows::SliceTable<index_t> table{p.heavy_nodes, node_chunk_offset.data_ptr<int>(), num_heavy};
                rows::SumSlices<kGrads, cuda_t, float> io{};
                cuda_t *const outs[3]       = {p.dQ, p.dV, p.dK};
                float const *const parts[3] = {p.part_q, p.part_v, p.part_k};
                for (size_t k = 0; k < kGrads; ++k) {
                    io.parts[k] = parts[k];
                    io.outs[k]  = outs[k];
                }
                rows::launch_sum_merge<kGrads, cuda_t, index_t>(table, H, p.D, nvec, wide, io, bucket_stream);
            },
            MakeIndexWidthVariant(idx_dtype), MakeTypeVariant<ATTN_SHARD_TYPE>(Q.scalar_type()),
            MakeIntVariant<8, 16, 32>(heavy_warps_per_block), mode_variant()
        );
    };

    stream_ns::run_buckets(
        buckets,
        [&](at::cuda::CUDAStream st) {
            const int warps = light_warps_for(light_warps_per_block, D, Q, wide, light_nodes.numel() * H);
            launch_bucket(light_nodes, MakeIntVariant<1, 2, 4>(warps), st, pipeline_stages);
        },
        [&](at::cuda::CUDAStream st) {
            if (heavy_edge_slice > 0) {
                launch_heavy_split(st);
            } else {
                launch_bucket(heavy_nodes, MakeIntVariant<8, 16, 32>(heavy_warps_per_block), st, backward_heavy_pipeline_stages);
            }
        }
    );

    CUDA_KERNEL_CHECK();

    // The fp32 scatter target back to the input dtype (a no-op for fp32).
    torch::Tensor dK = is_directed ? dK_f32.to(Q.scalar_type()) : dK_typed;
    return std::make_tuple(dQ, dK, dV);
}
