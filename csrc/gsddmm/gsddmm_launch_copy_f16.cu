// One of the per-op, per-dtype shards of the GSDDMM dispatch grid; see
// gsddmm_launch.cuh for why the grid is split across translation units.
// Copy propagates the lhs to the edges and never reads the rhs, so its LRO
// set is GSDDMM_COPY_LROS rather than the default binary pairs.
#define GSDDMM_SHARD_TYPE at::Half
#include "gsddmm/gsddmm_dispatch.cuh"

namespace gsddmm {

#define GSDDMM_SHARD_LROS GSDDMM_COPY_LROS
#define GSDDMM_SHARD_ENUM Copy
#define GSDDMM_SHARD_NAME copy
#define GSDDMM_SHARD_SUFFIX _f16
#include "gsddmm/gsddmm_launch_shard.inc"

}  // namespace gsddmm
