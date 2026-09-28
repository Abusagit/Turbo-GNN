#define ATTN_SHARD_TYPE float
#define ATTN_SHARD_FN(name) name##_f32
#include "gt/graph_transformer.cu"
