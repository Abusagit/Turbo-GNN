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
def wsb_flashattn_tc_forward_kernel(
    tcb_row_offset_ptr,  # int32 [num_row_windows + 1]
    col_idx_ptr,  # int32 [num_tcbs * 8]
    bitmap_ptr,  # int64 [num_tcbs * 2]
    Q_ptr,  # fp16 [N, D]
    K_ptr,  # fp16 [N, D]
    V_ptr,  # fp16 [N, D]
    O_ptr,  # fp32 [N, D]
    L_ptr,  # fp32 [N] - logsumexp output
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

    for pair_idx in tl.range(n_pairs, num_stages=2, warp_specialize=True):
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

        K_block = tl.load(k_ptrs, mask=valid_mask, other=0.0)
        V_block = tl.load(v_ptrs, mask=valid_mask, other=0.0)

        # SDDMM: Q @ K^T [16, 16]
        logits = tl.dot(Q_block, tl.trans(K_block)).to(tl.float32) * scale

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
        bm_val = tl.where(
            use_tcb_1,
            tl.where(use_hi_bm, bm_hi_1, bm_lo_1),
            tl.where(use_hi_bm, bm_hi_0, bm_lo_0),
        )

        # check if edge exists
        edge_exists = ((bm_val >> bit_pos) & 1) == 1

        # build full mask: edge exists AND column valid AND row valid
        # broadcast col_valid to [16, 16]
        col_valid_2d = col_valid[None, :] | ~col_valid[None, :]  # trick to broadcast
        col_valid_2d = tl.where(use_tcb_1, has_tcb_1, has_tcb_0)

        full_mask = edge_exists & col_valid_2d & row_mask[:, None]

        # apply mask
        logits = tl.where(full_mask, logits, -float("inf"))

        # online softmax update (keep output unnormalized)
        m_block = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_block)

        # compute exp scaling factor
        exp_scale = tl.exp(m_i - m_new)
        exp_scale = tl.where(m_i > -float("inf"), exp_scale, 0.0)

        # compute exponential of logits with new max
        exp_logits = tl.exp(logits - m_new[:, None])
        l_block = tl.sum(exp_logits, axis=1)

        # update sum of exponentials
        l_new = l_i * exp_scale + l_block

        # SpMM: exp(logits) @ V [16, 16] @ [16, BLOCK_D]
        acc *= exp_scale[:, None]
        acc = tl.dot(exp_logits.to(tl.float16), V_block, acc=acc)

        # update statistics
        m_i = m_new
        l_i = l_new

    # normalize output in the end
    acc = acc / l_i[:, None]
    acc = tl.where(l_i[:, None] > 0, acc, 0.0)

    # write output
    out_ptrs = O_ptr + rows[:, None] * stride_on + d_offs[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & d_mask[None, :])

    # save logsumexp L = m + log(l) for backward pass
    # !!!!! only the first `d_block` should write (to avoid race conditions)
    if d_block == 0:
        logsumexp = m_i + tl.log(l_i)
        logsumexp = tl.where(l_i > 0, logsumexp, -float("inf"))
        l_out_ptrs = L_ptr + rows
        tl.store(l_out_ptrs, logsumexp, mask=row_mask)


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

    # NOTE TODO Current implementation supports only single head,  need to expand it to arbitrary number of heads

    N, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output = torch.zeros((N, D), device=Q.device, dtype=torch.float32)
    logsumexp = torch.full((N,), -float("inf"), device=Q.device, dtype=torch.float32)

    grid = (wsb.num_row_windows, triton.cdiv(D, block_f))

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

    return output, logsumexp


@triton.jit
def wsb_flashattn_tc_backward_kernel(
    tcb_row_offset_ptr,  # int32 [num_row_windows + 1]
    col_idx_ptr,  # int32 [num_tcbs * 8]
    bitmap_ptr,  # int64 [num_tcbs * 2]
    Q_ptr,  # fp16 [N, D]
    K_ptr,  # fp16 [N, D]
    V_ptr,  # fp16 [N, D]
    O_ptr,  # fp32 [N, D]
    L_ptr,  # fp32 [N] - logsumexp from forward
    dO_ptr,  # fp32 [N, D] - gradient of output
    dQ_ptr,  # fp32 [N, D] - gradient of Q (output)
    dK_ptr,  # fp32 [N, D] - gradient of K (output)
    dV_ptr,  # fp32 [N, D] - gradient of V (output)
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
    stride_don,
    stride_dod,
    stride_dqn,
    stride_dqd,
    stride_dkn,
    stride_dkd,
    stride_dvn,
    stride_dvd,
    scale,
    BLOCK_F: tl.constexpr,
    ROW_WINDOW_SIZE: tl.constexpr,
    TCB_WIDTH: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """
    Backward pass parallelized over row windows
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

    # load O, dO, L for this row window
    o_ptrs = O_ptr + rows[:, None] * stride_on + d_offs[None, :] * stride_od
    do_ptrs = dO_ptr + rows[:, None] * stride_don + d_offs[None, :] * stride_dod
    O_block = tl.load(o_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)
    dO_block = tl.load(do_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    l_ptrs = L_ptr + rows
    L_vec = tl.load(l_ptrs, mask=row_mask, other=-float("inf"))  # [16]

    # compute D = rowsum(dO * O) [16]
    D_vec = tl.sum(dO_block * O_block, axis=1)

    # initialize dQ accumulator
    dQ_acc = tl.zeros((ROW_WINDOW_SIZE, BLOCK_F), dtype=tl.float32)

    # get TCB range for this row window
    tcb_start = tl.load(tcb_row_offset_ptr + rw_id)
    tcb_end = tl.load(tcb_row_offset_ptr + rw_id + 1)

    n_tcb = tcb_end - tcb_start
    n_pairs = (n_tcb + 1) // 2

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)
    k_offs = tl.arange(0, TILE_K)

    # loop over TCB pairs in current row window
    for pair_idx in tl.range(n_pairs, num_stages=2, warp_specialize=True):
        # TCB indices
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        # validity checks
        has_tcb_0 = tcb_idx_0 < tcb_end
        has_tcb_1 = tcb_idx_1 < tcb_end

        safe_tcb_0 = tl.where(has_tcb_0, tcb_idx_0, tcb_start)
        safe_tcb_1 = tl.where(has_tcb_1, tcb_idx_1, tcb_start)

        # build column indices [16]
        in_second_half = k_offs >= TCB_WIDTH
        local_col = k_offs % TCB_WIDTH

        cols_0 = tl.load(col_idx_ptr + safe_tcb_0 * TCB_WIDTH + local_col)
        cols_1 = tl.load(col_idx_ptr + safe_tcb_1 * TCB_WIDTH + local_col)
        cols = tl.where(in_second_half, cols_1, cols_0)

        col_valid = tl.where(in_second_half, has_tcb_1, has_tcb_0)
        valid_mask = col_valid[:, None] & d_mask[None, :]

        # load K and V [16, BLOCK_F]
        k_ptrs = K_ptr + cols[:, None] * stride_kn + d_offs[None, :] * stride_kd
        v_ptrs = V_ptr + cols[:, None] * stride_vn + d_offs[None, :] * stride_vd

        K_block = tl.load(k_ptrs, mask=valid_mask, other=0.0)
        V_block = tl.load(v_ptrs, mask=valid_mask, other=0.0)

        # recompute attention: S = Q @ K^T [16, 16]
        S_block = tl.dot(Q_block, tl.trans(K_block)).to(tl.float32) * scale

        # load bitmaps and apply mask (same as forward)
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
        bm_val = tl.where(use_tcb_1, tl.where(use_hi_bm, bm_hi_1, bm_lo_1), tl.where(use_hi_bm, bm_hi_0, bm_lo_0))

        edge_exists = ((bm_val >> bit_pos) & 1) == 1
        col_valid_2d = tl.where(use_tcb_1, has_tcb_1, has_tcb_0)
        full_mask = edge_exists & col_valid_2d & row_mask[:, None]

        S_block = tl.where(full_mask, S_block, -float("inf"))

        # recompute attention weights: P = exp(S - L) [16, 16]
        P_block = tl.exp(S_block - L_vec[:, None])
        P_block = tl.where(full_mask, P_block, 0.0)

        # dV = P^T @ dO [16, BLOCK_F]
        dV_block = tl.dot(tl.trans(P_block).to(tl.float16), dO_block.to(tl.float16)).to(tl.float32)

        # atomica add dV to global memory
        dv_ptrs = dV_ptr + cols[:, None] * stride_dvn + d_offs[None, :] * stride_dvd
        atomic_mask_dv = col_valid[:, None] & d_mask[None, :]
        tl.atomic_add(dv_ptrs, dV_block, mask=atomic_mask_dv)

        # dP = dO @ V^T [16, 16]
        dP_block = tl.dot(dO_block.to(tl.float16), tl.trans(V_block)).to(tl.float32)

        # Softmax backward: dS = P * (dP - D) [16, 16]
        dS_block = P_block * (dP_block - D_vec[:, None])
        dS_block = tl.where(full_mask, dS_block, 0.0)

        # dQ += dS @ K [16, BLOCK_F]
        dQ_acc = tl.dot(dS_block.to(tl.float16), K_block, acc=dQ_acc).to(tl.float32)

        # dK = dS^T @ Q [16, BLOCK_F]
        dK_block = tl.dot(tl.trans(dS_block).to(tl.float16), Q_block).to(tl.float32)

        # we need atomic add dK to global memory
        dk_ptrs = dK_ptr + cols[:, None] * stride_dkn + d_offs[None, :] * stride_dkd
        atomic_mask_dk = col_valid[:, None] & d_mask[None, :]
        tl.atomic_add(dk_ptrs, dK_block, mask=atomic_mask_dk)

    # write dQ for this row window (no atomics needed b.c. each row window owns its rows)
    dq_ptrs = dQ_ptr + rows[:, None] * stride_dqn + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_acc, mask=row_mask[:, None] & d_mask[None, :])


def wsb_flashattn_tc_backward(wsb, Q, K, V, output, L, dO, scale=None, block_f=64):
    """
    Backward pass computing dQ, dK, dV.

    Args:
        L: Logsumexp [N] from forward pass
        dO: Gradient of output [N, D]

    Returns:
        dQ, dK, dV: Gradients
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda and output.is_cuda
    assert dO.is_cuda and L.is_cuda

    N, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK = torch.zeros_like(K, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)

    grid = (wsb.num_row_windows, triton.cdiv(D, block_f))

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
        D,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        output.stride(0),
        output.stride(1),
        dO.stride(0),
        dO.stride(1),
        dQ.stride(0),
        dQ.stride(1),
        dK.stride(0),
        dK.stride(1),
        dV.stride(0),
        dV.stride(1),
        scale,
        BLOCK_F=block_f,
        ROW_WINDOW_SIZE=16,
        TCB_WIDTH=8,
        TILE_K=16,
    )

    return dQ, dK, dV


class WSBGraphTransformer(torch.autograd.Function):
    """Autograd function for WSB Graph Transformer"""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, wsb: WSBFormat) -> torch.Tensor:
        Q = Q.half()
        K = K.half()
        V = V.half()

        ctx.wsb = wsb
        ctx.scale = Q.shape[-1]

        output, logsumexp = wsb_flashattn_tc_forward(wsb, Q, K, V, scale=ctx.scale)
        ctx.save_for_backward(Q, K, V, logsumexp, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        Q, K, V, logsumexp, output = ctx.saved_tensors

        dQ, dK, dV = wsb_flashattn_tc_backward(ctx.wsb, Q, K, V, output, logsumexp, grad_output, scale=ctx.scale)
        return dQ, dK, dV, None


#####################################################
################# GATv2 kernels #####################
#####################################################

# TODO
