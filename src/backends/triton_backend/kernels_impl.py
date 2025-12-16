import math

import torch
import triton
import triton.language as tl

from src.data.converters import WSBFormat

from .triton_constants import ROW_WINDOW_SIZE, TCB_SIZE, TCB_WIDTH

#####################################################
################# GraphConv Kernels #################
#####################################################


@triton.jit
def wsb_spmm_kernel_tc(
    tcb_row_offset_ptr,
    col_idx_ptr,
    weights_ptr,
    X_ptr,
    Y_ptr,
    N,
    F,
    stride_xn,
    stride_xf,
    stride_yn,
    stride_yf,
    BLOCK_F: tl.constexpr,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TCB_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
):
    row_window_idx = tl.program_id(0)
    f_block = tl.program_id(1)

    row_start = row_window_idx * ROW_WINDOW_SIZE
    f_start = f_block * BLOCK_F

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)
    f_offs = tl.arange(0, BLOCK_F)
    k_offs = tl.arange(0, TILE_K)

    global_rows = row_start + row_offs
    global_f = f_start + f_offs

    row_mask = global_rows < N
    f_mask = global_f < F

    tcb_start = tl.load(tcb_row_offset_ptr + row_window_idx)
    tcb_end = tl.load(tcb_row_offset_ptr + row_window_idx + 1)
    num_tcbs = tcb_end - tcb_start

    # skip empty row windows
    if num_tcbs == 0:
        return

    # fp32 accumulator
    acc = tl.zeros((ROW_WINDOW_SIZE, BLOCK_F), dtype=tl.float32)

    num_pairs = (num_tcbs + 1) // 2  # we need to construct 16x16 tiles for WMMA from two 16x8 tiles

    for pair_idx in range(num_pairs):
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        # build weight matrix [16, 16] from 2 TCBs
        w_row_idx = row_offs[:, None]
        w_col_idx = k_offs[None, :]

        # build mask to check from which TCB (second of first) the columns come from
        tcb_select = w_col_idx >= TCB_WIDTH

        # local column within TCB (0-7)
        # local_col = [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7]
        local_col = tl.where(tcb_select, w_col_idx - TCB_WIDTH, w_col_idx)

        # which tcb index to use to load columns:
        tcb_idx = tl.where(tcb_select, tcb_idx_1, tcb_idx_0)

        # valid mask in case where second TCB might note exist
        valid_tcb = tcb_idx < tcb_end

        # compute weight address
        w_ptr = weights_ptr + tcb_idx * TCB_SIZE + w_row_idx * TCB_WIDTH + local_col
        W_full = tl.load(w_ptr, mask=valid_tcb, other=0.0).to(tl.float16)

        # build column indices [16]
        col_idx_local = k_offs % TCB_WIDTH  # [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7]
        tcb_for_col = k_offs // TCB_WIDTH  # [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1]
        tcb_idx_for_col = tl.where(tcb_for_col == 0, tcb_idx_0, tcb_idx_1)
        valid_col = tcb_idx_for_col < tcb_end

        col_ptr = col_idx_ptr + tcb_idx_for_col * TCB_WIDTH + col_idx_local
        cols_full = tl.load(col_ptr, mask=valid_col, other=0)

        # gather X - this is the expensive part
        X_tile = tl.load(
            X_ptr + cols_full[:, None] * stride_xn + global_f[None, :] * stride_xf,
            mask=valid_col[:, None] & f_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        # tensor core matmul
        # acc[16, BLOCK_F] += W_full[16, 16] @ X_tile[16, BLOCK_F]
        acc = tl.dot(W_full, X_tile, acc, out_dtype=tl.float32)

    y_ptrs = Y_ptr + global_rows[:, None] * stride_yn + global_f[None, :] * stride_yf
    tl.store(y_ptrs, acc, mask=row_mask[:, None] & f_mask[None, :])


def wsb_spmm_tc_forward(wsb: WSBFormat, X: torch.Tensor) -> torch.Tensor:
    """SpMM with tensor cores using Weighted Block Sparse Format"""
    assert X.shape[0] == wsb.num_nodes
    assert X.is_contiguous()

    N, F = X.shape
    device = X.device

    Y = torch.zeros_like(X)  # NOTE for now it's fp32

    # use fp16 here
    weights = wsb.weights.half()
    X_fp16 = X.half()

    BLOCK_F = 128 if F >= 128 else max(16, triton.next_power_of_2(F))  # NOTE this is heuristic, we can autotune it btw

    grid = (wsb.num_row_windows, triton.cdiv(F, BLOCK_F))

    wsb_spmm_kernel_tc[grid](
        wsb.tcb_row_offset,
        wsb.col_idx,
        weights,
        X_fp16,
        Y,
        N,
        F,
        X_fp16.stride(0),
        X_fp16.stride(1),
        Y.stride(0),
        Y.stride(1),
        BLOCK_F=BLOCK_F,
        ROW_WINDOW_SIZE=ROW_WINDOW_SIZE,
        TCB_WIDTH=TCB_WIDTH,
        TCB_SIZE=TCB_SIZE,
        TILE_K=16,
    )

    return Y


@triton.jit
def wsb_spmm_backward_kernel_tc(
    tcb_row_offset_ptr,
    col_idx_ptr,
    weights_ptr,
    G_ptr,
    dX_ptr,
    N,
    F,
    stride_gn,
    stride_gf,
    stride_dxn,
    stride_dxf,
    BLOCK_F: tl.constexpr,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TCB_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """
    Backward kernel with tensor cores. # NOTE very slow
    """

    rw = tl.program_id(0)
    f_block = tl.program_id(1)

    row_start = rw * ROW_WINDOW_SIZE
    f_start = f_block * BLOCK_F

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)
    f_offs = tl.arange(0, BLOCK_F)
    k_offs = tl.arange(0, TILE_K)

    global_rows = row_start + row_offs
    global_f = f_start + f_offs

    row_mask = global_rows < N
    f_mask = global_f < F

    tcb_start = tl.load(tcb_row_offset_ptr + rw)
    tcb_end = tl.load(tcb_row_offset_ptr + rw + 1)
    num_tcbs = tcb_end - tcb_start

    if num_tcbs == 0:
        return

    # load G for this row window: [16, BLOCK_F]
    G_tile = tl.load(
        G_ptr + global_rows[:, None] * stride_gn + global_f[None, :] * stride_gf,
        mask=row_mask[:, None] & f_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    num_pairs = (num_tcbs + 1) // 2

    for pair_idx in range(num_pairs):
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        out_col_idx = k_offs[:, None]
        in_row_idx = row_offs[None, :]

        tcb_select_t = out_col_idx >= TCB_WIDTH
        local_col_t = tl.where(tcb_select_t, out_col_idx - TCB_WIDTH, out_col_idx)
        tcb_idx_for_t = tl.where(tcb_select_t, tcb_idx_1, tcb_idx_0)
        valid_tcb_t = tcb_idx_for_t < tcb_end

        w_t_ptr = weights_ptr + tcb_idx_for_t * TCB_SIZE + in_row_idx * TCB_WIDTH + local_col_t
        W_T = tl.load(w_t_ptr, mask=valid_tcb_t, other=0.0).to(tl.float16)

        # W_T[16, 16] @ G[16, BLOCK_F] -> [16, BLOCK_F]
        contrib = tl.dot(W_T, G_tile, out_dtype=tl.float32)

        # first 8 columns from TCB 0
        for k in tl.static_range(TCB_WIDTH):
            # create mask for row k: [TILE_K] -> broadcast to [TILE_K, BLOCK_F]
            row_select = (k_offs == k)[:, None]
            contrib_row = tl.sum(tl.where(row_select, contrib, 0.0), axis=0)

            col_k = tl.load(col_idx_ptr + tcb_idx_0 * TCB_WIDTH + k)
            tl.atomic_add(dX_ptr + col_k * stride_dxn + global_f * stride_dxf, contrib_row, mask=f_mask)

        # second 8 columns from TCB 1 (if exists)
        second_valid = tcb_idx_1 < tcb_end
        for k in tl.static_range(TCB_WIDTH):
            row_select = (k_offs == (k + TCB_WIDTH))[:, None]
            contrib_row = tl.sum(tl.where(row_select, contrib, 0.0), axis=0)

            col_k = tl.load(col_idx_ptr + tcb_idx_1 * TCB_WIDTH + k, mask=second_valid, other=0)
            # zero out contribution if second TCB is invalid
            contrib_row_safe = tl.where(second_valid, contrib_row, 0.0)
            tl.atomic_add(dX_ptr + col_k * stride_dxn + global_f * stride_dxf, contrib_row_safe, mask=f_mask)


def wsb_spmm_backward_tc(wsb: WSBFormat, grad_output: torch.Tensor) -> torch.Tensor:
    """
    Backward pass with tensor cores: dX = D^T @ G
    """
    assert grad_output.shape[0] == wsb.num_nodes
    assert grad_output.is_contiguous()

    N, F = grad_output.shape

    grad_input = torch.zeros_like(grad_output)

    weights = wsb.weights.half()
    G = grad_output.half()

    BLOCK_F = 128 if F >= 128 else max(16, triton.next_power_of_2(F))
    grid = (wsb.num_row_windows, triton.cdiv(F, BLOCK_F))

    wsb_spmm_backward_kernel_tc[grid](
        wsb.tcb_row_offset,
        wsb.col_idx,
        weights,
        G,
        grad_input,
        N,
        F,
        G.stride(0),
        G.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
        BLOCK_F=BLOCK_F,
        ROW_WINDOW_SIZE=ROW_WINDOW_SIZE,
        TCB_WIDTH=TCB_WIDTH,
        TCB_SIZE=TCB_SIZE,
        TILE_K=16,
    )

    return grad_input


def wsb_spmm_backward_cusparse(adj_mat_csr_backward: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Compute gradient with respect to inputs using torch.spmm which is faster for precomputed transposed matrix

    Args:
        adj_mat_csr_backward (torch.Tensor): transposed adjacency matrix
        grad_output (torch.Tensor): gradient with respect to outputs

    Returns:
        torch.Tensor: gradient with respect to inputs
    """

    return torch.mm(adj_mat_csr_backward, grad_output)


class WSBSpMM(torch.autograd.Function):
    """Autograd function for WSB SpMM"""

    @staticmethod
    def forward(ctx, X: torch.Tensor, wsb: WSBFormat) -> torch.Tensor:
        ctx.wsb = wsb
        ctx.save_for_backward(wsb.adjacency_matrices_meta.adj_mat_csr_backward)
        return wsb_spmm_tc_forward(wsb, X)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        adj_mat_csr_backward = ctx.saved_tensors
        grad_input = wsb_spmm_backward_cusparse(adj_mat_csr_backward, grad_output)
        return grad_input, None


#####################################################
################# Graph Transformer Kernels #########
#####################################################


@triton.jit
def wsb_flashattn_tc_kernel(
    tcb_row_offset_ptr,  # int32 [num_row_windows + 1]
    col_idx_ptr,  # int32 [num_tcbs * 8]
    bitmap_ptr,  # int64 [num_tcbs * 2]
    Q_ptr,  # fp16 [N, D]
    K_ptr,  # fp16 [N, D]
    V_ptr,  # fp16 [N, D]
    O_ptr,  # fp32 [N, D]
    num_nodes,
    D,
    stride_qn,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_on,
    stride_od,
    scale,
    BLOCK_F: tl.constexpr,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """
    Tensor core accelerated sparse attention.

    Pairs adjacent TCBs to create 16x16 tiles for tensor cores
    """

    rw_id = tl.program_id(0)
    d_block = tl.program_id(1)

    row_start = rw_id * ROW_WINDOW_SIZE
    rows = row_start + tl.arange(0, ROW_WINDOW_SIZE)
    row_mask = rows < num_nodes

    d_start = d_block * BLOCK_F
    d_offs = d_start + tl.arange(0, BLOCK_F)
    d_mask = d_offs < D

    # load Q block [16, BLOCK_F]
    q_ptrs = Q_ptr + rows[:, None] * stride_qn + d_offs[None, :] * stride_qd
    Q_block = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float16)

    # online softmax state
    m_i = tl.full((ROW_WINDOW_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((ROW_WINDOW_SIZE,), dtype=tl.float32)
    acc = tl.zeros((ROW_WINDOW_SIZE, BLOCK_F), dtype=tl.float32)

    tcb_start = tl.load(tcb_row_offset_ptr + rw_id)
    tcb_end = tl.load(tcb_row_offset_ptr + rw_id + 1)
    n_tcb = tcb_end - tcb_start

    n_pairs = (n_tcb + 1) // 2

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)
    k_offs = tl.arange(0, TILE_K)

    for pair_idx in range(n_pairs):
        # TCB indices
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        # validity checks
        has_tcb_0 = tcb_idx_0 < tcb_end
        has_tcb_1 = tcb_idx_1 < tcb_end

        # safe indices for memory access (clamp to valid range)
        # use tcb_start as fallback (guaranteed valid if n_tcb > 0)
        safe_tcb_0 = tl.where(has_tcb_0, tcb_idx_0, tcb_start)
        safe_tcb_1 = tl.where(has_tcb_1, tcb_idx_1, tcb_start)

        # build column indices [16]
        in_second_half = k_offs >= TCB_WIDTH
        local_col = k_offs % TCB_WIDTH

        # load columns from both TCBs
        cols_0 = tl.load(col_idx_ptr + safe_tcb_0 * TCB_WIDTH + local_col)
        cols_1 = tl.load(col_idx_ptr + safe_tcb_1 * TCB_WIDTH + local_col)

        # combine columns
        cols = tl.where(in_second_half, cols_1, cols_0)

        # first half valid if has_tcb_0, second half valid if has_tcb_1
        col_valid = tl.where(in_second_half, has_tcb_1, has_tcb_0)
        valid_mask = col_valid[:, None] & d_mask[None, :]

        # load K and V [16, BLOCK_D]
        k_ptrs = K_ptr + cols[:, None] * stride_kn + d_offs[None, :] * stride_kd
        v_ptrs = V_ptr + cols[:, None] * stride_vn + d_offs[None, :] * stride_vd

        K_block = tl.load(k_ptrs, mask=valid_mask, other=0.0)  # .to(tl.float16)
        V_block = tl.load(v_ptrs, mask=valid_mask, other=0.0)  # .to(tl.float16)

        # SDDMM: Q @ K^T [16, 16]
        logits = tl.dot(Q_block, tl.trans(K_block)) * scale  # .to(tl.float32) * scale

        # lolad bitmaps for both TCBs
        bm_lo_0 = tl.load(bitmap_ptr + safe_tcb_0 * 2 + 0)
        bm_hi_0 = tl.load(bitmap_ptr + safe_tcb_0 * 2 + 1)
        bm_lo_1 = tl.load(bitmap_ptr + safe_tcb_1 * 2 + 0)
        bm_hi_1 = tl.load(bitmap_ptr + safe_tcb_1 * 2 + 1)

        # build bitmap mask for [16, 16] logits
        row_idx = row_offs[:, None]  # [16, 1]
        col_idx_mat = k_offs[None, :]  # [1, 16]

        # bitmap half selection (rows 0-7 use lo, rows 8-15 use hi)
        use_hi_bm = row_idx >= 8
        row_in_half = row_idx % 8

        # TCB selection (columns 0-7 from TCB0, columns 8-15 from TCB1)
        use_tcb_1 = col_idx_mat >= TCB_WIDTH
        col_in_tcb = col_idx_mat % TCB_WIDTH

        # bit position: row_in_half * 8 + col_in_tcb
        bit_pos = row_in_half * TCB_WIDTH + col_in_tcb

        # select appropriate bitmap
        bm_val = tl.where(use_tcb_1, tl.where(use_hi_bm, bm_hi_1, bm_lo_1), tl.where(use_hi_bm, bm_hi_0, bm_lo_0))

        # check if edge exists
        edge_exists = ((bm_val >> bit_pos) & 1) == 1

        # build full mask: edge exists AND column valid AND row valid
        # broadcast col_valid to [16, 16]
        col_valid_2d = col_valid[None, :] | ~col_valid[None, :]  # trick to broadcast
        col_valid_2d = tl.where(use_tcb_1, has_tcb_1, has_tcb_0)

        full_mask = edge_exists & col_valid_2d & row_mask[:, None]

        # apply mask
        logits = tl.where(full_mask, logits, -float("inf"))

        # online softmax
        m_block = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_block)

        # Safe exp computation (handle -inf)
        exp_scale = tl.exp(m_i - m_new)
        exp_scale = tl.where(m_i > -float("inf"), exp_scale, 0.0)

        exp_logits = tl.exp(logits - m_new[:, None])
        l_block = tl.sum(exp_logits, axis=1)
        l_new = l_i * exp_scale + l_block

        # SpMM: exp(logits) @ V [16, 16] @ [16, BLOCK_D]
        attn = exp_logits.to(tl.float16)
        acc *= exp_scale[:, None]
        acc = tl.dot(attn, V_block, acc=acc)

        m_i = m_new
        l_i = l_new

    acc = acc / l_i[:, None]
    acc = tl.where(l_i[:, None] > 0, acc, 0.0)

    out_ptrs = O_ptr + rows[:, None] * stride_on + d_offs[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & d_mask[None, :])


def wsb_flashattn_tc_forward(wsb, Q, K, V, scale=None, block_f=64):
    """
    Tensor core accelerated FlashAttention on WSB layout.

    Args:
        wsb: WSBFormat object with tcb_row_offset, col_idx, bitmap
        Q: [N, D] fp16 query matrix
        K: [N, D] fp16 key matrix
        V: [N, D] fp16 value matrix
        scale: Attention scaling factor (default: 1/sqrt(D))
        block_f: Block size for D dimension (should be 16, 32, or 64 for tensor cores)

    Returns:
        O: [N, D] fp32 output matrix
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == torch.float16, "Q must be fp16 for tensor cores"
    assert block_f >= 16, "block_f must be >= 16 for tensor cores"

    N, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output = torch.zeros((N, D), device=Q.device, dtype=torch.float32)

    grid = (wsb.num_row_windows, triton.cdiv(D, block_f))

    wsb_flashattn_tc_kernel[grid](
        wsb.tcb_row_offset,
        wsb.col_idx,
        wsb.bitmap,
        Q,
        K,
        V,
        output,
        N,
        D,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        output.stride(0),
        output.stride(1),
        scale,
        BLOCK_F=block_f,
        ROW_WINDOW_SIZE=ROW_WINDOW_SIZE,
        TCB_WIDTH=TCB_WIDTH,
        TILE_K=16,
    )

    return output


class WSBGraphTransformer(torch.autograd.Function):
    """Autograd function for WSB Graph Transformer"""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, wsb: WSBFormat) -> torch.Tensor:
        ctx.wsb = wsb
        ctx.save_for_backward(Q, K, V)
        output = wsb_flashattn_tc_forward(wsb, Q, K, V, scale=Q.shape[-1])  # TODO add logsumexp

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO
        raise NotImplementedError("TODO")


#####################################################
################# GATv2 kernels #####################
#####################################################

# TODO
