import math

import torch
import triton
import triton.language as tl

from src.data.converters import WSBFormat
from src.utils.triton_constants import ROW_WINDOW_SIZE, TCB_SIZE, TCB_WIDTH

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
    F: tl.constexpr,
    stride_xn,
    stride_xf,
    stride_yn,
    stride_yf,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TCB_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
):
    row_window_idx = tl.program_id(0)

    row_start = row_window_idx * ROW_WINDOW_SIZE

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)
    k_offs = tl.arange(0, TILE_K)

    global_rows = row_start + row_offs
    global_f = tl.arange(0, F)

    row_mask = global_rows < N

    tcb_start = tl.load(tcb_row_offset_ptr + row_window_idx)
    tcb_end = tl.load(tcb_row_offset_ptr + row_window_idx + 1)
    num_tcbs = tcb_end - tcb_start

    # skip empty row windows
    if num_tcbs == 0:
        return

    # fp32 accumulator
    acc = tl.zeros((ROW_WINDOW_SIZE, F), dtype=tl.float32)

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
            mask=valid_col[:, None],
            other=0.0,
        ).to(tl.float16)

        # tensor core matmul
        # acc[16, F] += W_full[16, 16] @ X_tile[16, F]
        acc = tl.dot(W_full, X_tile, acc, out_dtype=tl.float32)

    y_ptrs = Y_ptr + global_rows[:, None] * stride_yn + global_f[None, :] * stride_yf
    tl.store(y_ptrs, acc, mask=row_mask[:, None])


def wsb_spmm_tc_forward(wsb, X: torch.Tensor) -> torch.Tensor:
    """SpMM with tensor cores using Weighted Block Sparse Format"""
    assert X.shape[0] == wsb.num_nodes
    assert X.is_contiguous()

    N, F = X.shape

    Y = torch.empty_like(X)  # NOTE for now it's fp32

    # use fp16 here
    weights = wsb.weights.half()
    X_fp16 = X.half()

    grid = (wsb.num_row_windows,)

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
    F: tl.constexpr,
    stride_gn,
    stride_gf,
    stride_dxn,
    stride_dxf,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TCB_SIZE: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """
    Backward kernel with tensor cores. # NOTE very slow
    """

    rw = tl.program_id(0)

    row_start = rw * ROW_WINDOW_SIZE

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)
    f_offs = tl.arange(0, F)
    k_offs = tl.arange(0, TILE_K)

    global_rows = row_start + row_offs
    global_f = f_offs

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

        # W_T[16, 16] @ G[16, F] -> [16, BLOCK_F]
        contrib = tl.dot(W_T, G_tile, out_dtype=tl.float32)

        # first 8 columns from TCB 0
        for k in tl.static_range(TCB_WIDTH):
            # create mask for row k: [TILE_K] -> broadcast to [TILE_K, F]
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


def wsb_spmm_backward_tc(wsb, grad_output: torch.Tensor) -> torch.Tensor:
    """
    Backward pass with tensor cores: dX = D^T @ G
    """
    assert grad_output.shape[0] == wsb.num_nodes
    assert grad_output.is_contiguous()

    N, F = grad_output.shape

    grad_input = torch.empty_like(grad_output)

    weights = wsb.weights.half()
    G = grad_output.half()

    grid = (wsb.num_row_windows,)

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
        (adj_mat_csr_backward,) = ctx.saved_tensors
        grad_input = wsb_spmm_backward_cusparse(adj_mat_csr_backward, grad_output)
        return grad_input, None


#####################################################
################# Graph Transformer Kernels #########
#####################################################
@triton.jit
def wsb_flashattn_tc_forward_kernel(
    tcb_row_offset_ptr,  # int32 [num_row_windows + 1]
    col_idx_ptr,  # int32 [num_tcbs * 8]
    bitmap_ptr,  # int64 [num_tcbs * 2]
    Q_ptr,  # fp16 [N, D]
    K_ptr,  # fp16 [N, D]
    V_ptr,  # fp16 [N, D]
    O_ptr,  # fp32 [N, D]
    L_ptr,  # fp32 [N]
    num_nodes,
    num_heads,
    D: tl.constexpr,
    # Q strides
    stride_qn,
    stride_qh,
    stride_qd,
    # K strides
    stride_kn,
    stride_kh,
    stride_kd,
    # V strides
    stride_vn,
    stride_vh,
    stride_vd,
    # O strides
    stride_on,
    stride_oh,
    stride_od,
    # L strides
    stride_ln,
    stride_lh,
    scale,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TILE_K: tl.constexpr,
):
    rw_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # rows in this row-window
    row_start = rw_id * ROW_WINDOW_SIZE
    rows = row_start + tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    row_mask = rows < num_nodes

    d_offs = tl.arange(0, D)

    # load Q block [16, D] (fp16)
    q_ptrs = Q_ptr + rows[:, None] * stride_qn + head_id * stride_qh + d_offs[None, :] * stride_qd

    Q_block = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # online softmax state
    m_i = tl.full((ROW_WINDOW_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((ROW_WINDOW_SIZE,), dtype=tl.float32)
    acc = tl.zeros((ROW_WINDOW_SIZE, D), dtype=tl.float32)

    # TCB range for this row-window
    tcb_start = tl.load(tcb_row_offset_ptr + rw_id)
    tcb_end = tl.load(tcb_row_offset_ptr + rw_id + 1)
    n_tcb = tcb_end - tcb_start
    n_pairs = (n_tcb + 1) // 2

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    k_offs = tl.arange(0, TILE_K)  # [16]  (2 * TCB_WIDTH)

    for pair_idx in tl.range(n_pairs, num_stages=2, warp_specialize=True):
        # two TCBs in this pair
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        has_tcb_0 = tcb_idx_0 < tcb_end
        has_tcb_1 = tcb_idx_1 < tcb_end

        # safe indices for out-of-range (point to tcb_start)
        safe_tcb_0 = tl.where(has_tcb_0, tcb_idx_0, tcb_start)
        safe_tcb_1 = tl.where(has_tcb_1, tcb_idx_1, tcb_start)

        # construct 16 columns: 0..7 from tcb_0, 8..15 from tcb_1
        in_second_half = k_offs >= TCB_WIDTH
        local_col = k_offs % TCB_WIDTH

        cols_0 = tl.load(col_idx_ptr + safe_tcb_0 * TCB_WIDTH + local_col)
        cols_1 = tl.load(col_idx_ptr + safe_tcb_1 * TCB_WIDTH + local_col)
        cols = tl.where(in_second_half, cols_1, cols_0)  # [16]

        # validity of each column (second half only if has_tcb_1) for this head
        col_valid = tl.where(in_second_half, has_tcb_1, has_tcb_0)  # [16]

        # K, V loads [16, D], unmasked (cols are always valid indices; padded cols are 0)
        k_ptrs = K_ptr + cols[:, None] * stride_kn + head_id * stride_kh + d_offs[None, :] * stride_kd

        v_ptrs = V_ptr + cols[:, None] * stride_vn + head_id * stride_vh + d_offs[None, :] * stride_vd

        K_block = tl.load(k_ptrs).to(tl.float16)
        V_block = tl.load(v_ptrs).to(tl.float16)

        # QK^T -> logits [16, 16], fp32
        logits = tl.dot(Q_block, tl.trans(K_block)) * scale

        # load bitmaps for both TCBs
        bm_lo_0 = tl.load(bitmap_ptr + safe_tcb_0 * 2 + 0)
        bm_hi_0 = tl.load(bitmap_ptr + safe_tcb_0 * 2 + 1)
        bm_lo_1 = tl.load(bitmap_ptr + safe_tcb_1 * 2 + 0)
        bm_hi_1 = tl.load(bitmap_ptr + safe_tcb_1 * 2 + 1)

        # bitmap indexing
        row_idx = row_offs[:, None]  # [16, 1]
        col_idx_mat = k_offs[None, :]  # [1, 16]

        use_hi_bm = row_idx >= 8
        row_in_half = row_idx % 8

        use_tcb_1 = col_idx_mat >= TCB_WIDTH
        col_in_tcb = col_idx_mat % TCB_WIDTH

        bit_pos = row_in_half * TCB_WIDTH + col_in_tcb  # [16, 16]

        # pick correct bitmap (TCB0/1, low/high)
        bm_val = tl.where(
            use_tcb_1,
            tl.where(use_hi_bm, bm_hi_1, bm_lo_1),
            tl.where(use_hi_bm, bm_hi_0, bm_lo_0),
        )

        # edge mask from bitmap
        edge_exists = ((bm_val >> bit_pos) & 1) == 1

        # column validity broadcasted to [16, 16]
        col_valid_2d = col_valid[None, :]  # [1, 16] -> broadcast with [16,16]

        # full mask: edge exists, column valid, row valid
        full_mask = edge_exists & col_valid_2d & row_mask[:, None]

        # mask logits (invalid edges -> -inf)
        logits = tl.where(full_mask, logits, -float("inf"))

        # online softmax
        m_block = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_block)

        # exp scaling factor for previous accumulator
        exp_scale = tl.exp(m_i - m_new)
        exp_scale = tl.where(m_i > -float("inf"), exp_scale, 0.0)

        # exp(logits - m_new)
        exp_logits = tl.exp(logits - m_new[:, None])
        l_block = tl.sum(exp_logits, axis=1)

        # update l_i
        l_new = l_i * exp_scale + l_block

        # update acc = exp_scale * acc + exp_logits @ V
        acc *= exp_scale[:, None]
        acc = tl.dot(exp_logits.to(tl.float16), V_block, acc=acc)

        m_i = m_new
        l_i = l_new

    # normalization
    acc = acc / l_i[:, None]
    acc = tl.where(l_i[:, None] > 0, acc, 0.0)

    # store O [N, H, D]
    out_ptrs = O_ptr + rows[:, None] * stride_on + head_id * stride_oh + d_offs[None, :] * stride_od

    tl.store(out_ptrs, acc, mask=row_mask[:, None])

    # store logsumexp
    logsumexp = m_i + tl.log(l_i)
    logsumexp = tl.where(l_i > 0, logsumexp, -float("inf"))
    l_out_ptrs = L_ptr + rows * stride_ln + head_id * stride_lh
    tl.store(l_out_ptrs, logsumexp, mask=row_mask)


def wsb_flashattn_tc_forward(wsb, Q, K, V, scale):
    """
    Tensor core accelerated FlashAttention on WSB layout.

    Args:
        wsb: WSBFormat object with tcb_row_offset, col_idx, bitmap
        Q: [N, H, D] fp16 query tensor
        K: [N, H, D] fp16 key tensor
        V: [N, H, D] fp16 value tensor
        scale: Attention scaling factor (default: 1/sqrt(D))

    Returns:
        O: [N, D] fp32 output
        L: [N, H] fp32 logsumexp for backward pass
    """
    assert Q.ndim == 3, f"Q must be [N, H, D], got shape {Q.shape}"
    assert K.ndim == 3, f"K must be [N, H, D], got shape {K.shape}"
    assert V.ndim == 3, f"C must be [N, H, D], got shape {V.shape}"

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == torch.float16, "Q must be fp16 for tensor cores"

    N, H, D = Q.shape
    assert D in {16, 32, 64, 128, 256, 512}, f"HEAD_DIM must be power-of-2 ≤ 512, got {D}"

    output = torch.empty((N, H, D), device=Q.device, dtype=torch.float32)
    logsumexp = torch.full((N, H), -float("inf"), device=Q.device, dtype=torch.float32)

    grid = (wsb.num_row_windows, H)

    wsb_flashattn_tc_forward_kernel[grid](
        wsb.tcb_row_offset,
        wsb.col_idx,
        wsb.bitmap,
        Q,
        K,
        V,
        output,
        logsumexp,
        N,
        H,
        D,
        # Q strides
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        # K strides
        K.stride(0),
        K.stride(1),
        K.stride(2),
        # V strides
        V.stride(0),
        V.stride(1),
        V.stride(2),
        # O strides
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # L strides
        logsumexp.stride(0),
        logsumexp.stride(1),
        scale,
        ROW_WINDOW_SIZE=ROW_WINDOW_SIZE,
        TCB_WIDTH=TCB_WIDTH,
        TILE_K=16,
    )

    return output, logsumexp


@triton.jit
def wsb_flashattn_tc_backward_kernel(
    tcb_row_offset_ptr,
    col_idx_ptr,
    bitmap_ptr,
    Q_ptr,  # fp16 [N, H, D]
    K_ptr,  # fp16 [N, H, D]
    V_ptr,  # fp16 [N, H, D]
    O_ptr,  # fp32 [N, H, D]
    L_ptr,  # fp32 [N, H]
    dO_ptr,  # fp32 [N, H, D]
    dQ_ptr,  # fp32 [N, H, D]
    dK_ptr,  # fp32 [N, H, D]
    dV_ptr,  # fp32 [N, H, D]
    num_nodes,
    num_heads,
    D: tl.constexpr,
    # Q strides
    stride_qn,
    stride_qh,
    stride_qd,
    # K strides
    stride_kn,
    stride_kh,
    stride_kd,
    # V strides
    stride_vn,
    stride_vh,
    stride_vd,
    # O strides
    stride_on,
    stride_oh,
    stride_od,
    # L strides
    stride_ln,
    stride_lh,
    # dO strides
    stride_don,
    stride_doh,
    stride_dod,
    # dQ strides
    stride_dqn,
    stride_dqh,
    stride_dqd,
    # dK strides
    stride_dkn,
    stride_dkh,
    stride_dkd,
    # dV strides
    stride_dvn,
    stride_dvh,
    stride_dvd,
    scale,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TILE_K: tl.constexpr,
):
    rw_id = tl.program_id(0)
    head_id = tl.program_id(1)

    row_start = rw_id * ROW_WINDOW_SIZE
    rows = row_start + tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    row_mask = rows < num_nodes

    d_offs = tl.arange(0, D)

    # Q [16, D] fp16
    q_ptrs = Q_ptr + rows[:, None] * stride_qn + head_id * stride_qh + d_offs[None, :] * stride_qd

    Q_block = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # O, dO [16, D] fp32

    o_ptrs = O_ptr + rows[:, None] * stride_on + head_id * stride_oh + d_offs[None, :] * stride_od

    do_ptrs = dO_ptr + rows[:, None] * stride_don + head_id * stride_doh + d_offs[None, :] * stride_dod

    O_block = tl.load(o_ptrs, mask=row_mask[:, None], other=0.0)
    dO_block = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)

    # L [16]
    l_ptrs = L_ptr + rows * stride_ln + head_id * stride_lh
    L_vec = tl.load(l_ptrs, mask=row_mask, other=-float("inf"))

    # D_vec = sum_j dO_ij * O_ij [16]
    D_vec = tl.sum(dO_block * O_block, axis=1)

    # dQ accumulator [16, D]
    dQ_acc = tl.zeros((ROW_WINDOW_SIZE, D), dtype=tl.float32)

    # TCB range for this row-window
    tcb_start = tl.load(tcb_row_offset_ptr + rw_id)
    tcb_end = tl.load(tcb_row_offset_ptr + rw_id + 1)

    n_tcb = tcb_end - tcb_start
    n_pairs = (n_tcb + 1) // 2

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    k_offs = tl.arange(0, TILE_K)  # [16]

    for pair_idx in tl.range(n_pairs, num_stages=2, warp_specialize=True):
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        has_tcb_0 = tcb_idx_0 < tcb_end
        has_tcb_1 = tcb_idx_1 < tcb_end

        safe_tcb_0 = tl.where(has_tcb_0, tcb_idx_0, tcb_start)
        safe_tcb_1 = tl.where(has_tcb_1, tcb_idx_1, tcb_start)

        in_second_half = k_offs >= TCB_WIDTH
        local_col = k_offs % TCB_WIDTH

        cols_0 = tl.load(col_idx_ptr + safe_tcb_0 * TCB_WIDTH + local_col)
        cols_1 = tl.load(col_idx_ptr + safe_tcb_1 * TCB_WIDTH + local_col)
        cols = tl.where(in_second_half, cols_1, cols_0)  # [16]

        col_valid = tl.where(in_second_half, has_tcb_1, has_tcb_0)  # [16]

        # K, V [16, D], unmasked
        k_ptrs = K_ptr + cols[:, None] * stride_kn + head_id * stride_kh + d_offs[None, :] * stride_kd

        v_ptrs = V_ptr + cols[:, None] * stride_vn + head_id * stride_vh + d_offs[None, :] * stride_vd

        K_block = tl.load(k_ptrs).to(tl.float16)
        V_block = tl.load(v_ptrs).to(tl.float16)

        # S = Q K^T [16,16] fp32
        S_block = tl.dot(Q_block, tl.trans(K_block)) * scale

        # bitmaps
        bm_lo_0 = tl.load(bitmap_ptr + safe_tcb_0 * 2 + 0)
        bm_hi_0 = tl.load(bitmap_ptr + safe_tcb_0 * 2 + 1)
        bm_lo_1 = tl.load(bitmap_ptr + safe_tcb_1 * 2 + 0)
        bm_hi_1 = tl.load(bitmap_ptr + safe_tcb_1 * 2 + 1)

        row_idx = row_offs[:, None]
        col_idx_mat = k_offs[None, :]

        use_hi_bm = row_idx >= 8
        row_in_half = row_idx % 8
        use_tcb_1 = col_idx_mat >= TCB_WIDTH
        col_in_tcb = col_idx_mat % TCB_WIDTH

        bit_pos = row_in_half * TCB_WIDTH + col_in_tcb

        bm_val = tl.where(
            use_tcb_1,
            tl.where(use_hi_bm, bm_hi_1, bm_lo_1),
            tl.where(use_hi_bm, bm_hi_0, bm_lo_0),
        )

        edge_exists = ((bm_val >> bit_pos) & 1) == 1

        col_valid_2d = col_valid[None, :]  # [1,16] -> broadcast
        full_mask = edge_exists & col_valid_2d & row_mask[:, None]

        # mask S for invalid edges
        S_block = tl.where(full_mask, S_block, -float("inf"))

        # P = softmax(S) = exp(S - L) with L from forward
        P_block = tl.exp(S_block - L_vec[:, None])
        P_block = tl.where(full_mask, P_block, 0.0)

        # dV = P^T @ dO  [16, D]
        dV_block = tl.dot(tl.trans(P_block).to(tl.float16), dO_block.to(tl.float16)).to(tl.float32)

        # atomically add dV
        dv_ptrs = dV_ptr + cols[:, None] * stride_dvn + head_id * stride_dvh + d_offs[None, :] * stride_dvd

        atomic_mask_dv = col_valid[:, None]  # [16,1] broadcast with D
        tl.atomic_add(dv_ptrs, dV_block, mask=atomic_mask_dv)

        # dP = dO @ V^T [16, 16]
        dP_block = tl.dot(dO_block.to(tl.float16), tl.trans(V_block)).to(tl.float32)

        # softmax backward: dS = P * (dP - D_vec[:,None])
        dS_block = P_block * (dP_block - D_vec[:, None])
        dS_block = tl.where(full_mask, dS_block, 0.0)

        # dQ += dS @ K [16, D]
        dQ_acc = tl.dot(dS_block.to(tl.float16), K_block, acc=dQ_acc)

        # dK = dS^T @ Q [16, D]
        dK_block = tl.dot(tl.trans(dS_block).to(tl.float16), Q_block).to(tl.float32)

        # atomically add dK
        dk_ptrs = dK_ptr + cols[:, None] * stride_dkn + head_id * stride_dkh + d_offs[None, :] * stride_dkd

        atomic_mask_dk = col_valid[:, None]
        tl.atomic_add(dk_ptrs, dK_block, mask=atomic_mask_dk)

    # write dQ (no atomics needed, each row window owns its rows)
    dq_ptrs = dQ_ptr + rows[:, None] * stride_dqn + head_id * stride_dqh + d_offs[None, :] * stride_dqd

    tl.store(dq_ptrs, dQ_acc, mask=row_mask[:, None])


def wsb_flashattn_tc_backward(wsb, Q, K, V, output, L, dO, scale):
    """
    Backward pass computing dQ, dK, dV for multi-head case.

    All tensors Q, K, V, output, dO are [N, H, D].

        L is [N, H].


    Args:
        L: Logsumexp from forward pass
        dO: Gradient of output

    Returns:
        dQ, dK, dV: Gradients
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda and output.is_cuda
    assert dO.is_cuda and L.is_cuda

    assert Q.shape == K.shape == V.shape == output.shape == dO.shape
    assert Q.ndim == 3, f"Q must be [N, H, D], got {Q.shape}"

    N, H, D = Q.shape

    assert L.shape == (N, H), f"L must be [N, H], got {L.shape}"
    assert D in {16, 32, 64, 128, 256, 512}, f"HEAD_DIM must be power-of-2 ≤ 512, got {D}"

    dQ = torch.empty_like(Q, dtype=torch.float32)
    dK = torch.empty_like(K, dtype=torch.float32)
    dV = torch.empty_like(V, dtype=torch.float32)

    grid = (wsb.num_row_windows, H)

    wsb_flashattn_tc_backward_kernel[grid](
        wsb.tcb_row_offset,
        wsb.col_idx,
        wsb.bitmap,
        Q,
        K,
        V,
        output,
        L,
        dO,
        dQ,
        dK,
        dV,
        N,
        H,
        D,
        # Q strides
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        # K strides
        K.stride(0),
        K.stride(1),
        K.stride(2),
        # V strides
        V.stride(0),
        V.stride(1),
        V.stride(2),
        # O strides
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # L strides
        L.stride(0),
        L.stride(1),
        # dO strides
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        # dQ strides
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
        # dK strides
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        # dV strides
        dV.stride(0),
        dV.stride(1),
        dV.stride(2),
        scale,
        ROW_WINDOW_SIZE=ROW_WINDOW_SIZE,
        TCB_WIDTH=TCB_WIDTH,
        TILE_K=16,
    )

    return dQ, dK, dV


class WSBGraphTransformer(torch.autograd.Function):
    """Autograd function for WSB Graph Transformer"""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, wsb: WSBFormat, scale: float) -> torch.Tensor:
        Q = Q.half()
        K = K.half()
        V = V.half()

        ctx.wsb = wsb
        ctx.scale = scale

        output, logsumexp = wsb_flashattn_tc_forward(wsb, Q, K, V, scale=ctx.scale)
        ctx.save_for_backward(Q, K, V, logsumexp, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        Q, K, V, logsumexp, output = ctx.saved_tensors
        head_dim = Q.shape[2]
        num_heads = Q.shape[1]
        grad_output = grad_output.view(-1, num_heads, head_dim)

        dQ, dK, dV = wsb_flashattn_tc_backward(ctx.wsb, Q, K, V, output, logsumexp, grad_output, scale=ctx.scale)
        return dQ, dK, dV, None, None
