import triton
import triton.language as tl


# ----------------------- Forward Kernel ----------------------- #

@triton.jit
def spgemm_csr_dense_kernel_row_parallel(
    BT_crow_ptr, BT_col_idx_ptr, BT_val_ptr,
    H_ptr,
    C_ptr,
    n, d,
    BLOCK_SIZE_D: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block_id = tl.program_id(1)

    col_start = col_block_id * BLOCK_SIZE_D
    cols = col_start + tl.arange(0, BLOCK_SIZE_D)
    col_mask = cols < d

    acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    row_begin = tl.load(BT_crow_ptr + row_idx)
    row_end = tl.load(BT_crow_ptr + row_idx + 1)

    for nz_idx in range(row_begin, row_end):
        BT_col = tl.load(BT_col_idx_ptr + nz_idx)
        BT_val = tl.load(BT_val_ptr + nz_idx)

        h_offsets = BT_col * d + cols
        h_vals = tl.load(H_ptr + h_offsets, mask=col_mask, other=0.0)

        acc += BT_val * h_vals

    c_offsets = row_idx * d + cols
    tl.store(C_ptr + c_offsets, acc, mask=col_mask)


# ----------------------- Backward Kernel ----------------------- #

@triton.jit
def spgemm_csr_dense_backward_H_row_parallel(
    BT_crow_ptr, BT_col_idx_ptr, BT_val_ptr,
    dC_ptr,
    dH_ptr,
    n, d,
    BLOCK_SIZE_D: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block_id = tl.program_id(1)

    col_start = col_block_id * BLOCK_SIZE_D
    cols = col_start + tl.arange(0, BLOCK_SIZE_D)
    col_mask = cols < d

    dc_offsets = row_idx * d + cols
    dC_vals = tl.load(dC_ptr + dc_offsets, mask=col_mask, other=0.0)

    row_begin = tl.load(BT_crow_ptr + row_idx)
    row_end = tl.load(BT_crow_ptr + row_idx + 1)

    for nz_idx in range(row_begin, row_end):
        BT_col = tl.load(BT_col_idx_ptr + nz_idx)
        BT_val = tl.load(BT_val_ptr + nz_idx)

        contrib = BT_val * dC_vals
        h_offsets = BT_col * d + cols

        tl.atomic_add(dH_ptr + h_offsets, contrib, mask=col_mask)

# --- autograd wrapper ---

import torch

class SpGemmCSR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, BT_crow, BT_col, BT_val, H, n, d, block_size):
        C = torch.empty((n, d), device=H.device, dtype=H.dtype)

        grid = (n, triton.cdiv(d, block_size))

        spgemm_csr_dense_kernel_row_parallel[grid](
            BT_crow, BT_col, BT_val,
            H,
            C,
            n, d,
            BLOCK_SIZE_D=block_size
        )

        ctx.save_for_backward(BT_crow, BT_col, BT_val, H)
        ctx.n, ctx.d, ctx.block_size = n, d, block_size
        return C

    @staticmethod
    def backward(ctx, dC):
        BT_crow, BT_col, BT_val, H = ctx.saved_tensors
        n, d, block_size = ctx.n, ctx.d, ctx.block_size

        dH = torch.zeros_like(H)

        grid = (n, triton.cdiv(d, block_size))

        spgemm_csr_dense_backward_H_row_parallel[grid](
            BT_crow, BT_col, BT_val,
            dC,
            dH,
            n, d,
            BLOCK_SIZE_D=block_size
        )

        # grads: B is not differentiable for now (only H has grad)
        return None, None, None, dH, None, None, None


def spgemm_csr(BT_crow, BT_col, BT_val, H, block_size=128):
    n = BT_crow.numel() - 1
    d = H.shape[1]
    return SpGemmCSR.apply(BT_crow, BT_col, BT_val, H, n, d, block_size)

# --- test correctness ---

def test_correctness():
    torch.manual_seed(0)

    # Sparse dimensions
    n = 20
    d = 32

    # Build random CSR matrix
    BT = torch.randn((n, n), device='cuda')
    BT[BT.abs() < 2.] = 0
    BT.requires_grad = True
    BT_csr =  BT.to_sparse_csr()
    BT_crow = BT_csr.crow_indices()
    BT_col = BT_csr.col_indices()
    BT_val = BT_csr.values()

    H = torch.randn(n, d, device='cuda', requires_grad=True)
    H_ref = H.clone().detach().requires_grad_(True)

    # Forward
    C_triton = spgemm_csr(BT_crow, BT_col, BT_val, H)
    C_ref = BT @ H_ref

    print("Forward max diff:", (C_triton - C_ref).abs().max().item())

    # Backward check
    grad_output = torch.randn_like(C_triton)
    C_triton.backward(grad_output)
    C_ref.backward(grad_output)

    print("Backward max diff (dH):", (H.grad - H_ref.grad).abs().max().item())


# Run test
test_correctness()

