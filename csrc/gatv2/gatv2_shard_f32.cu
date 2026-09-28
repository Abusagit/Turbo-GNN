#define ATTN_SHARD_TYPE float
#define ATTN_SHARD_FN(name) name##_f32
#include "gatv2/gatv2_kernel.cu"
