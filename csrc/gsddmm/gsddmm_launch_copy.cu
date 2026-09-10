// One of the six per-op shards of the GSDDMM forward dispatch grid; see
// gsddmm_launch.cuh for why the grid is split across translation units.
#include "gsddmm/gsddmm_dispatch.cuh"

namespace gsddmm {

void gsddmm_forward_launch_copy(const GsddmmLaunchArgs& args) { gsddmm_dispatch<GSDDMM_COPY_LROS>(args); }

void gsddmm_forward_edge_launch_copy(const GsddmmLaunchArgsEdge& args) { gsddmm_dispatch_edge_block<GSDDMM_COPY_LROS>(args); }

};  // namespace gsddmm
