import triton
import triton.language as tl

@triton.jit
def spgemm_csr_dense_kernel(
    B_crow_ptr, B_col_idx_ptr, B_val_ptr, B_nnz,
    H_ptr,
    C_ptr,
    n, d
):
    block_id = tl.program_id(0)

    # all blocks load:
    # - B entirely
    # - corresponding column of H
    # offset = block_id * n
    # H_block_ptr = H_ptr + offset

    B_idx = 0
    cur_col, cur_row = tl.load(B_col_idx_ptr), 0
    while B_idx < B_nnz:
        # load value and column index
        cur_col = tl.load(B_col_idx_ptr + B_idx)

        while cur_row < n - 1 and tl.load(B_crow_ptr + cur_row + 1) == B_idx:
            cur_row += 1
        C_delta_val = tl.load(B_val_ptr + B_idx) * tl.load(H_ptr + block_id * n + cur_col)
        C_update_val = tl.load(C_ptr + block_id * n + cur_row) + C_delta_val
        tl.store(C_ptr + block_id * n + cur_row, C_update_val)

        B_idx += 1

import torch

"""

class TritonMatMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, H):
        pass
        return ...

    @staticmethod
    def backward(ctx, d_out):
        pass
        return ...
"""

device = "cuda"

n, d = 512, 256

def generate_B_HT():
    # build random sparse tensors
    global n, d
    density = 1e-3
    B = torch.randn((n, n), device=device)
    B[B.abs() < 1.0] = 0 # ~71%
    #B = torch.zeros((n, n), device=device)
    #B[0][1] = torch.randn((1,), device=device).item()
    B = B.to_sparse_csr()

    HT = torch.randn((d, n), device=device)

    B.requires_grad=True
    HT.requires_grad=True
    return B, HT

def calculate_gt(B, H):
    res_sparse = torch.matmul(B, H)
    res = res_sparse.to_dense()
    # res.backward()
    return res, None, None # B.grad, H.grad

B_csr, HT_dense = generate_B_HT()
gt_C, gt_grad_B, gt_grad_H = calculate_gt(B_csr, HT_dense.T)

# extract arrays
B_crow = B_csr.crow_indices().to(device)
B_col  = B_csr.col_indices().to(device)
B_val  = B_csr.values().to(device)

HT_dense = HT_dense.to(device)

# allocate dense output
CT = torch.zeros((d, n), dtype=torch.float32, device=device)

# TODO: check that H is stored and passed colunmn-major
# launch
grid = (d, )
spgemm_csr_dense_kernel[grid](
    B_crow, B_col, B_val, B_val.shape[0],
    HT_dense,
    CT,
    n, d
)
torch.cuda.synchronize()

C = CT.T

# print(f"{C = }")
# print(f"{gt_C = }")

print(f"{abs(gt_C.sum() - C.sum()) = }")
print(f"{torch.max(torch.abs(gt_C - C)) = }")
# assert abs(gt_C.sum() - C.sum()) < 1e-7 * n * d
assert torch.max(torch.abs(gt_C - C)) < 1e-4
# assert torch.allclose(gt_C, C)

print("C.shape =", C.shape)
print("Done.")
