// One of the six per-op shards of the GSDDMM dispatch grid; see
// gsddmm_launch.cuh for why the grid is split across translation units.
#include "gsddmm/gsddmm_dispatch.cuh"

namespace gsddmm {

#define GSDDMM_SHARD_ENUM Mul
#define GSDDMM_SHARD_NAME mul
#include "gsddmm/gsddmm_launch_shard.inc"

}  // namespace gsddmm
