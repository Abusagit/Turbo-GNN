// One of the per-op, per-dtype shards of the GSDDMM dispatch grid; see
// gsddmm_launch.cuh for why the grid is split across translation units.
#define GSDDMM_SHARD_TYPE at::BFloat16
#include "gsddmm/gsddmm_dispatch.cuh"

namespace gsddmm {

#define GSDDMM_SHARD_ENUM Dot
#define GSDDMM_SHARD_NAME dot
#define GSDDMM_SHARD_SUFFIX _bf16
#include "gsddmm/gsddmm_launch_shard.inc"

}  // namespace gsddmm
