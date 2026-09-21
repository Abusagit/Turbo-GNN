#define ATTN_SHARD_TYPE at::Half
#define ATTN_SHARD_FN(name) name##_f16
#include "gt/graph_transformer.cu"
