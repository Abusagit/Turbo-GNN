// pybind bridge for exercising csrc/common/scheduler.cuh from pytest.
//
// The scheduler decides which thread block processes which node, so the property that
// matters is coverage: across all blocks, every work item must be visited exactly once, for
// every policy and at any grid size. This bridge runs a kernel whose entire body is that
// bookkeeping, so a failure points at the scheduler and nothing else.
//
// It deliberately includes an isolated-node `continue` path. That is the case that motivated
// putting the inter-iteration barrier inside `next()` rather than at the bottom of the loop
// body, and it needs to stay exercised.

#include <torch/extension.h>

#include "common.cuh"

using namespace turbo_gnn::sched;

namespace {

/// Records one visit per (work item, head), and how many items each block handled.
///
/// `skip_every` > 0 makes every n-th work item take a `continue`, mimicking a degree-0 node.
/// Those items are still counted in `visits` before the skip, so coverage stays checkable.
template <ScheduleKind SK, typename index_t, bool SyncBlock>
__global__ void scheduler_probe_kernel(
    SchedulerParams<index_t> sched_params, int *__restrict__ visits, int *__restrict__ per_block,
    int *__restrict__ first_idx, int heads, int skip_every
) {
    using Sched = NodeScheduler<SK, index_t, SyncBlock>;
    __shared__ typename Sched::SharedStorage smem;
    Sched sched(sched_params, smem);

    const int head       = static_cast<int>(blockIdx.y);
    const int flat_block = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) +
                           static_cast<int>(blockIdx.x);
    int handled = 0;
    bool first  = true;

    for (auto w = sched.first(); sched.valid(w); w = sched.next(w)) {
        const index_t node_i = sched.node(w);

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            atomicAdd(&visits[static_cast<int>(node_i) * heads + head], 1);
            if (first) {
                first_idx[flat_block] = sched.slot(w);
            }
        }
        first = false;
        ++handled;

        // Isolated-node path: `continue` skips the rest of the body. If the fence lived at
        // the bottom of this loop instead of inside next(), this is where a block would
        // split into threads that hit the barrier and threads that did not.
        if (skip_every > 0 && (sched.slot(w) % skip_every) == 0) {
            continue;
        }

        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        per_block[flat_block] = handled;
    }
}

template <ScheduleKind SK, bool SyncBlock>
void launch_probe(
    at::Tensor const &nodes, at::Tensor const &block_offsets, at::Tensor const &counters, at::Tensor &visits,
    at::Tensor &per_block, at::Tensor &first_idx, int count, int heads, int grid_x, int threads, int skip_every,
    int chunk
) {
    using index_t = int32_t;
    auto stream   = at::cuda::getCurrentCUDAStream();

    auto params = make_params<index_t>(
        SK, nodes.defined() ? nodes.data_ptr<index_t>() : nullptr, count, counters, heads,
        /*launch_index=*/0, block_offsets, chunk
    );

    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(heads));
    scheduler_probe_kernel<SK, index_t, SyncBlock><<<grid, threads, 0, stream>>>(
        params, visits.data_ptr<int>(), per_block.data_ptr<int>(), first_idx.data_ptr<int>(), heads, skip_every
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

/// Run one policy and report coverage.
///
/// Returns (visits[count, heads], per_block[heads, grid_x], first_idx[heads, grid_x], grid_x).
std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t> run_scheduler(
    int64_t schedule, int64_t count, int64_t heads, int64_t blocks_per_sm, int64_t threads,
    c10::optional<at::Tensor> nodes_opt, c10::optional<at::Tensor> block_offsets_opt, int64_t skip_every,
    c10::optional<int64_t> force_grid_x, int64_t chunk
) {
    const ScheduleKind kind = schedule_from_int(static_cast<int>(schedule));
    TORCH_CHECK(count >= 0, "count must be non-negative");
    TORCH_CHECK(heads >= 1, "heads must be >= 1");

    at::Tensor nodes = nodes_opt.has_value() ? nodes_opt.value() : at::Tensor{};
    at::Tensor offs  = block_offsets_opt.has_value() ? block_offsets_opt.value() : at::Tensor{};
    const auto device = nodes.defined() ? nodes.device() : at::Device(at::kCUDA, 0);

    int grid_x = force_grid_x.has_value()
                     ? static_cast<int>(force_grid_x.value())
                     : persistent_grid_x(kind, static_cast<int>(count), static_cast<int>(blocks_per_sm),
                                         static_cast<int>(heads), static_cast<int>(chunk));
    // PrecomputedList's assignment fixes the block count.
    if (kind == ScheduleKind::PrecomputedList) {
        TORCH_CHECK(offs.defined(), "PrecomputedList needs block_offsets");
        grid_x = static_cast<int>(offs.numel()) - 1;
    }
    grid_x = std::max(1, grid_x);

    auto i32 = at::TensorOptions().dtype(at::kInt).device(device);
    at::Tensor visits    = at::zeros({count, heads}, i32);
    at::Tensor per_block = at::zeros({heads, grid_x}, i32);
    at::Tensor first_idx = at::full({heads, grid_x}, -1, i32);
    at::Tensor counters  = make_counters(kind, static_cast<int>(heads), /*num_launches=*/1, device);

    const int c  = static_cast<int>(count);
    const int h  = static_cast<int>(heads);
    const int th = static_cast<int>(threads);
    const int se = static_cast<int>(skip_every);
    const int ch = std::max(1, static_cast<int>(chunk));

#define TGNN_DISPATCH_KIND(K)                                                                            \
    case K:                                                                                              \
        launch_probe<K, true>(nodes, offs, counters, visits, per_block, first_idx, c, h, grid_x, th, se, ch); \
        break;

    switch (kind) {
        TGNN_DISPATCH_KIND(ScheduleKind::OneNodePerBlock)
        TGNN_DISPATCH_KIND(ScheduleKind::GridStride)
        TGNN_DISPATCH_KIND(ScheduleKind::PrecomputedList)
        TGNN_DISPATCH_KIND(ScheduleKind::DynamicQueue)
    }
#undef TGNN_DISPATCH_KIND

    return {visits, per_block, first_idx, static_cast<int64_t>(grid_x)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_scheduler", &run_scheduler, "Exercise one scheduler policy and report coverage",
          py::arg("schedule"), py::arg("count"), py::arg("heads"), py::arg("blocks_per_sm"), py::arg("threads"),
          py::arg("nodes") = c10::nullopt, py::arg("block_offsets") = c10::nullopt, py::arg("skip_every") = 0,
          py::arg("force_grid_x") = c10::nullopt, py::arg("chunk") = 1);
}
