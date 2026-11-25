// One of the per-op, per-dtype shards of the g-SpMM dispatch grid; see gspmm_launch.h.
#define GSPMM_SHARD_OP MUL
#define GSPMM_SHARD_NAME mul
#define GSPMM_SHARD_TYPE at::Half
#define GSPMM_SHARD_SUFFIX f16
#include "spmm/gspmm_shard.inc"
