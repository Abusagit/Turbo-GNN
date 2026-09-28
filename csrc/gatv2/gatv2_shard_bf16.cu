#define ATTN_SHARD_TYPE at::BFloat16
#define ATTN_SHARD_FN(name) name##_bf16
#include "gatv2/gatv2_kernel.cu"
