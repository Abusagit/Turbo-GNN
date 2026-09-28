// The GSDDMM op set -- the single source of truth. Every place that must
// enumerate the ops (the per-op launcher declarations in gsddmm_launch.cuh,
// the binding's op parsing and routing switches, and the per-op shard
// translation units) includes this file with XX defined to emit what it
// needs. No include guard on purpose: the whole point is to include it
// repeatedly with different XX definitions.
//
// XX(Enumerator, lowercase_name)
XX(Add, add)
XX(Sub, sub)
XX(Mul, mul)
XX(Div, div)
XX(Dot, dot)
XX(Copy, copy)
