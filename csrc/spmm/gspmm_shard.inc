// Body of a per-op, per-dtype g-SpMM shard: the two entry points of one op on one dtype, from
// GSPMM_SHARD_OP (the BinaryOp enumerator), GSPMM_SHARD_NAME (its lowercase name), GSPMM_SHARD_TYPE
// (the torch dtype) and GSPMM_SHARD_SUFFIX (f32 / f16 / bf16), all defined by the including shard.
//
// No include guard on purpose: one include per shard, and every macro it uses is #undef'd below.
#include "spmm/gspmm_dispatch.cuh"

#define GSPMM_CONCAT_(a, b, c) a##b##_##c
#define GSPMM_CONCAT(a, b, c) GSPMM_CONCAT_(a, b, c)

void GSPMM_CONCAT(gspmm_forward_, GSPMM_SHARD_NAME, GSPMM_SHARD_SUFFIX)(turbo_gnn::gspmm::ForwardArgs const& args) {
    turbo_gnn::gspmm::forward_shard<BinaryOp::GSPMM_SHARD_OP, GSPMM_SHARD_TYPE>(args);
}

void GSPMM_CONCAT(gspmm_edge_grad_, GSPMM_SHARD_NAME, GSPMM_SHARD_SUFFIX)(turbo_gnn::gspmm::EdgeGradArgs const& args) {
    turbo_gnn::gspmm::edge_grad_shard<BinaryOp::GSPMM_SHARD_OP, GSPMM_SHARD_TYPE>(args);
}

#undef GSPMM_CONCAT
#undef GSPMM_CONCAT_
#undef GSPMM_SHARD_OP
#undef GSPMM_SHARD_NAME
#undef GSPMM_SHARD_TYPE
#undef GSPMM_SHARD_SUFFIX
