TODO -- for reduced grad a - use cuda_t for grad_a unreduced -- thus load larger amount and reduce inside the kernel


NOTE - maybe its incorrect to use float4 reinterpret for the output in GATv2 when we are operating with halves


register usage can be very large for low precision accumulators... Mayve use shared memory? Deal with bank conflicts here, use static shared memory maybe?

__hfma2 for half and bf16

grad_a let be cuda_t instead of being float. That's how we can add more values inside the kernel using vectoriaed operations
if constexpr (DC <= 64) -- we need to check not only that but also the data type because we load twice as much with halfs


fuse r kernel for gatv2 backward for undirected graphs
