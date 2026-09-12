// One of the six per-op shards of the GSDDMM forward dispatch grid; see
// gsddmm_launch.cuh for why the grid is split across translation units.
#include "gsddmm/gsddmm_dispatch.cuh"

namespace gsddmm {

void gsddmm_forward_launch_sub(const GsddmmLaunchArgs& args) { gsddmm_dispatch<GSDDMM_BINARY_LROS(Sub)>(args); }

void gsddmm_forward_edge_launch_sub(const GsddmmLaunchArgsEdge& args) { gsddmm_dispatch_edge_block<GSDDMM_BINARY_LROS(Sub)>(args); }

void gsddmm_backward_launch_sub(const GsddmmBackwardLaunchArgs& args) { gsddmm_backward_dispatch<GSDDMM_BINARY_LROS(Sub)>(args); }

void gsddmm_backward_edge_launch_sub(const GsddmmBackwardLaunchArgsEdge& args) {
    gsddmm_backward_dispatch_edge_block<GSDDMM_BINARY_LROS(Sub)>(args);
}

};  // namespace gsddmm
