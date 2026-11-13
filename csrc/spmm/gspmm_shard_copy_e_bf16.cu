// One of the per-op, per-dtype shards of the g-SpMM dispatch grid; see gspmm_launch.h.
#define GSPMM_SHARD_OP COPY_E
#define GSPMM_SHARD_NAME copy_e
#define GSPMM_SHARD_TYPE at::BFloat16
#define GSPMM_SHARD_SUFFIX bf16
#include "spmm/gspmm_shard.inc"
