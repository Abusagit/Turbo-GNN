// Body of a per-op launcher translation unit: the four family entry points
// for one op, derived entirely from GSDDMM_SHARD_ENUM (the bare enumerator)
// GSDDMM_SHARD_NAME (the lowercase name the entry points are suffixed
// with) and GSDDMM_SHARD_SUFFIX (the dtype, e.g. _f16), all defined by the including shard before this file. A shard that
// needs a nonstandard LRO set (Copy) defines GSDDMM_SHARD_LROS itself first;
// the default is the six mixed member pairs of GSDDMM_BINARY_LROS.
//
// No include guard on purpose: one include per shard, and every macro this
// file touches is #undef'd again below.
#ifndef GSDDMM_SHARD_LROS
#define GSDDMM_SHARD_LROS GSDDMM_BINARY_LROS(GSDDMM_SHARD_ENUM)
#endif

#define GSDDMM_CONCAT_(a, b) a##b
#define GSDDMM_CONCAT(a, b) GSDDMM_CONCAT_(a, b)

void GSDDMM_CONCAT(GSDDMM_CONCAT(gsddmm_forward_launch_, GSDDMM_SHARD_NAME), GSDDMM_SHARD_SUFFIX)(const GsddmmLaunchArgs& args) {
    gsddmm_dispatch<GSDDMM_SHARD_LROS>(args);
}

void GSDDMM_CONCAT(GSDDMM_CONCAT(gsddmm_forward_edge_launch_, GSDDMM_SHARD_NAME), GSDDMM_SHARD_SUFFIX)(const GsddmmLaunchArgsEdge& args) {
    gsddmm_dispatch_edge_block<GSDDMM_SHARD_LROS>(args);
}

void GSDDMM_CONCAT(GSDDMM_CONCAT(gsddmm_backward_launch_, GSDDMM_SHARD_NAME), GSDDMM_SHARD_SUFFIX)(const GsddmmBackwardLaunchArgs& args) {
    gsddmm_backward_dispatch<GSDDMM_SHARD_LROS>(args);
}

void GSDDMM_CONCAT(GSDDMM_CONCAT(gsddmm_backward_edge_launch_, GSDDMM_SHARD_NAME), GSDDMM_SHARD_SUFFIX)(const GsddmmBackwardLaunchArgsEdge& args) {
    gsddmm_backward_dispatch_edge_block<GSDDMM_SHARD_LROS>(args);
}

#undef GSDDMM_SHARD_LROS
#undef GSDDMM_SHARD_ENUM
#undef GSDDMM_SHARD_NAME
#undef GSDDMM_SHARD_SUFFIX
#undef GSDDMM_CONCAT
#undef GSDDMM_CONCAT_
