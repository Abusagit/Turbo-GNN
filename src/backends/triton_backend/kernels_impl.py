import math
import os

import torch
import triton
import triton.language as tl

from src.data.converters import WSBFormat
from src.utils.triton_constants import ROW_WINDOW_SIZE, TCB_SIZE, TCB_WIDTH

_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_AUTOTUNE_DISABLED = os.environ.get("TRITON_AUTOTUNE_DISABLED", "0") == "1"


def _low_precision(t: torch.Tensor) -> torch.Tensor:
    """fp16/bf16 pass through; fp32 defaults to fp16."""
    if t.dtype in (torch.float16, torch.bfloat16):
        return t
    return t.half()


def _triton_dtype(t: torch.Tensor):
    """Map a PyTorch 16-bit dtype to the Triton equivalent."""
    return _TORCH_TO_TRITON_DTYPE[t.dtype]


# --- Triton autotune config spaces ------------------------------------------
_LOOP_CONFIGS = [(1, False), (2, True), (3, True)]  # (LOOP_NUM_STAGES, WARP_SPECIALIZE)

SPMM_AUTOTUNE_CONFIGS = [
    triton.Config({"LOOP_NUM_STAGES": ls, "WARP_SPECIALIZE": ws}, num_warps=w, num_stages=s)
    for w in [1, 2, 4, 8]
    for s in [1, 2, 3]
    for ls, ws in _LOOP_CONFIGS
]

FLASHATTN_AUTOTUNE_CONFIGS = [
    triton.Config({"LOOP_NUM_STAGES": ls, "WARP_SPECIALIZE": ws}, num_warps=w, num_stages=s)
    for w in [2, 4, 8]
    for s in [2, 3]
    for ls, ws in _LOOP_CONFIGS
]

_SAFE_CONFIG = [triton.Config({"LOOP_NUM_STAGES": 1, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=1)]

_SPMM_ACTIVE_CONFIGS = _SAFE_CONFIG if _AUTOTUNE_DISABLED else SPMM_AUTOTUNE_CONFIGS
_FLASHATTN_ACTIVE_CONFIGS = _SAFE_CONFIG if _AUTOTUNE_DISABLED else FLASHATTN_AUTOTUNE_CONFIGS

#####################################################
################# GraphConv Kernels #################
#####################################################


@triton.autotune(
    configs=_SPMM_ACTIVE_CONFIGS,
    key=["N", "F"],
)
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
    COMPUTE_DTYPE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
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
        y_ptrs = Y_ptr + global_rows[:, None] * stride_yn + global_f[None, :] * stride_yf
        tl.store(y_ptrs, 0.0, mask=row_mask[:, None])
        return

    # fp32 accumulator
    acc = tl.zeros((ROW_WINDOW_SIZE, F), dtype=tl.float32)

    num_pairs = (num_tcbs + 1) // 2  # we need to construct 16x16 tiles for WMMA from two 16x8 tiles

    for pair_idx in tl.range(num_pairs, num_stages=LOOP_NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
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
        W_full = tl.load(w_ptr, mask=valid_tcb, other=0.0).to(COMPUTE_DTYPE)

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
        ).to(COMPUTE_DTYPE)

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

    X_lp = _low_precision(X)
    weights = wsb.weights.to(X_lp.dtype)

    grid = (wsb.num_row_windows,)

    wsb_spmm_kernel_tc[grid](
        wsb.tcb_row_offset,
        wsb.col_idx,
        weights,
        X_lp,
        Y,
        N,
        F,
        X_lp.stride(0),
        X_lp.stride(1),
        Y.stride(0),
        Y.stride(1),
        ROW_WINDOW_SIZE=ROW_WINDOW_SIZE,
        TCB_WIDTH=TCB_WIDTH,
        TCB_SIZE=TCB_SIZE,
        TILE_K=16,
        COMPUTE_DTYPE=_triton_dtype(X_lp),
    )

    return Y


@triton.autotune(
    configs=_SPMM_ACTIVE_CONFIGS,
    key=["N", "F"],
)
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
    COMPUTE_DTYPE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
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
    ).to(COMPUTE_DTYPE)

    num_pairs = (num_tcbs + 1) // 2

    for pair_idx in tl.range(num_pairs, num_stages=LOOP_NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        tcb_idx_0 = tcb_start + pair_idx * 2
        tcb_idx_1 = tcb_idx_0 + 1

        out_col_idx = k_offs[:, None]
        in_row_idx = row_offs[None, :]

        tcb_select_t = out_col_idx >= TCB_WIDTH
        local_col_t = tl.where(tcb_select_t, out_col_idx - TCB_WIDTH, out_col_idx)
        tcb_idx_for_t = tl.where(tcb_select_t, tcb_idx_1, tcb_idx_0)
        valid_tcb_t = tcb_idx_for_t < tcb_end

        w_t_ptr = weights_ptr + tcb_idx_for_t * TCB_SIZE + in_row_idx * TCB_WIDTH + local_col_t
        W_T = tl.load(w_t_ptr, mask=valid_tcb_t, other=0.0).to(COMPUTE_DTYPE)

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

    grad_input = torch.zeros_like(grad_output)

    G = _low_precision(grad_output)
    weights = wsb.weights.to(G.dtype)

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
        COMPUTE_DTYPE=_triton_dtype(G),
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
@triton.autotune(
    configs=_FLASHATTN_ACTIVE_CONFIGS,
    key=["num_nodes", "D"],
)
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
    COMPUTE_DTYPE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    rw_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # rows in this row-window
    row_start = rw_id * ROW_WINDOW_SIZE
    rows = row_start + tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    row_mask = rows < num_nodes

    d_offs = tl.arange(0, D)

    # load K block [16, D] at row positions (row = aggregation target = "dst")
    k_ptrs = K_ptr + rows[:, None] * stride_kn + head_id * stride_kh + d_offs[None, :] * stride_kd

    K_block = tl.load(k_ptrs, mask=row_mask[:, None], other=0.0).to(COMPUTE_DTYPE)

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

    for pair_idx in tl.range(n_pairs, num_stages=LOOP_NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
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

        # Q, V loads [16, D] at column positions (col = neighbor = "src")
        q_ptrs = Q_ptr + cols[:, None] * stride_qn + head_id * stride_qh + d_offs[None, :] * stride_qd

        v_ptrs = V_ptr + cols[:, None] * stride_vn + head_id * stride_vh + d_offs[None, :] * stride_vd

        Q_block = tl.load(q_ptrs).to(COMPUTE_DTYPE)
        V_block = tl.load(v_ptrs).to(COMPUTE_DTYPE)

        # K[row] @ Q[col]^T -> logits [16, 16], fp32
        # = K[dst] · Q[src] which matches DGL's q[src] · k[dst]
        logits = tl.dot(K_block, tl.trans(Q_block)) * scale

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

        # exp(logits - m_new); guard NaN from exp(-inf - (-inf))
        exp_logits = tl.exp(logits - m_new[:, None])
        exp_logits = tl.where(full_mask, exp_logits, 0.0)
        l_block = tl.sum(exp_logits, axis=1)

        # update l_i
        l_new = l_i * exp_scale + l_block

        # update acc = exp_scale * acc + exp_logits @ V
        acc *= exp_scale[:, None]
        acc = tl.dot(exp_logits.to(COMPUTE_DTYPE), V_block, acc=acc)

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
    assert Q.dtype in (torch.float16, torch.bfloat16), "Q must be fp16 or bf16 for tensor cores"

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
        COMPUTE_DTYPE=_triton_dtype(Q),
    )

    return output, logsumexp


@triton.autotune(
    configs=_FLASHATTN_ACTIVE_CONFIGS,
    key=["num_nodes", "D"],
)
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
    COMPUTE_DTYPE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    IS_UNDIRECTED: tl.constexpr = False,
):
    """Backward kernel with K at rows (aggregation target) and Q, V at cols (neighbors).

    Forward was: O[row] = softmax(K[row] @ Q[col].T * scale) @ V[col]
    Backward computes: dK[row] (local accum), dQ[col] and dV[col] (atomic scatter).

    When IS_UNDIRECTED=True, also computes reverse direction locally:
    dQ[row] and dV[row] are accumulated without atomics by exploiting symmetric adjacency.
    """
    rw_id = tl.program_id(0)
    head_id = tl.program_id(1)

    row_start = rw_id * ROW_WINDOW_SIZE
    rows = row_start + tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    row_mask = rows < num_nodes

    d_offs = tl.arange(0, D)

    # K [16, D] at row positions (matches forward: K at rows)
    k_ptrs = K_ptr + rows[:, None] * stride_kn + head_id * stride_kh + d_offs[None, :] * stride_kd
    K_block = tl.load(k_ptrs, mask=row_mask[:, None], other=0.0).to(COMPUTE_DTYPE)

    # O, dO [16, D] fp32
    o_ptrs = O_ptr + rows[:, None] * stride_on + head_id * stride_oh + d_offs[None, :] * stride_od
    do_ptrs = dO_ptr + rows[:, None] * stride_don + head_id * stride_doh + d_offs[None, :] * stride_dod

    O_block = tl.load(o_ptrs, mask=row_mask[:, None], other=0.0)
    dO_block = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)

    # L [16]
    l_ptrs = L_ptr + rows * stride_ln + head_id * stride_lh
    L_vec = tl.load(l_ptrs, mask=row_mask, other=-float("inf"))

    # D_vec[i] = sum_d dO[i,d] * O[i,d]
    D_vec = tl.sum(dO_block * O_block, axis=1)

    # dK accumulator [16, D] — accumulated locally at rows
    dK_acc = tl.zeros((ROW_WINDOW_SIZE, D), dtype=tl.float32)

    # Undirected: additional row loads and accumulators for reverse direction
    if IS_UNDIRECTED:
        # Q, V at row positions (for reverse direction attention)
        q_row_ptrs = Q_ptr + rows[:, None] * stride_qn + head_id * stride_qh + d_offs[None, :] * stride_qd
        Q_rows = tl.load(q_row_ptrs, mask=row_mask[:, None], other=0.0).to(COMPUTE_DTYPE)

        v_row_ptrs = V_ptr + rows[:, None] * stride_vn + head_id * stride_vh + d_offs[None, :] * stride_vd
        V_rows = tl.load(v_row_ptrs, mask=row_mask[:, None], other=0.0).to(COMPUTE_DTYPE)

        # Local accumulators for dQ and dV (no atomics needed)
        dQ_acc = tl.zeros((ROW_WINDOW_SIZE, D), dtype=tl.float32)
        dV_acc = tl.zeros((ROW_WINDOW_SIZE, D), dtype=tl.float32)

    # TCB range for this row-window
    tcb_start = tl.load(tcb_row_offset_ptr + rw_id)
    tcb_end = tl.load(tcb_row_offset_ptr + rw_id + 1)

    n_tcb = tcb_end - tcb_start
    n_pairs = (n_tcb + 1) // 2

    row_offs = tl.arange(0, ROW_WINDOW_SIZE)  # [16]
    k_offs = tl.arange(0, TILE_K)  # [16]

    for pair_idx in tl.range(n_pairs, num_stages=LOOP_NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
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

        # Q, V [16, D] at column positions (col = neighbor = "src")
        q_ptrs = Q_ptr + cols[:, None] * stride_qn + head_id * stride_qh + d_offs[None, :] * stride_qd
        v_ptrs = V_ptr + cols[:, None] * stride_vn + head_id * stride_vh + d_offs[None, :] * stride_vd

        Q_block = tl.load(q_ptrs).to(COMPUTE_DTYPE)
        V_block = tl.load(v_ptrs).to(COMPUTE_DTYPE)

        # S = K[row] @ Q[col]^T [16,16] fp32 (matches forward convention)
        S_block = tl.dot(K_block, tl.trans(Q_block)) * scale

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
        if not IS_UNDIRECTED:
            # Directed path: scatter dV to column nodes via atomics
            dV_block = tl.dot(tl.trans(P_block).to(COMPUTE_DTYPE), dO_block.to(COMPUTE_DTYPE)).to(tl.float32)
            dv_ptrs = dV_ptr + cols[:, None] * stride_dvn + head_id * stride_dvh + d_offs[None, :] * stride_dvd
            atomic_mask_dv = col_valid[:, None]
            tl.atomic_add(dv_ptrs, dV_block, mask=atomic_mask_dv)

        # dP = dO @ V^T [16, 16]
        dP_block = tl.dot(dO_block.to(COMPUTE_DTYPE), tl.trans(V_block)).to(tl.float32)

        # softmax backward: dS = P * (dP - D_vec[:,None])
        dS_block = P_block * (dP_block - D_vec[:, None])
        dS_block = tl.where(full_mask, dS_block, 0.0)

        # chain rule through S = K@Q.T * scale: d(K@Q.T) = dS * scale
        dS_scaled = dS_block * scale

        # dK[row] += dS_scaled @ Q[col] [16, D] — local accumulation at rows
        dK_acc = tl.dot(dS_scaled.to(COMPUTE_DTYPE), Q_block, acc=dK_acc)

        if not IS_UNDIRECTED:
            # Directed path: scatter dQ to column nodes via atomics
            dQ_block = tl.dot(tl.trans(dS_scaled).to(COMPUTE_DTYPE), K_block).to(tl.float32)
            dq_ptrs = dQ_ptr + cols[:, None] * stride_dqn + head_id * stride_dqh + d_offs[None, :] * stride_dqd
            atomic_mask_dq = col_valid[:, None]
            tl.atomic_add(dq_ptrs, dQ_block, mask=atomic_mask_dq)

        if IS_UNDIRECTED:
            # ── Reverse direction: dQ[row], dV[row] accumulated locally ──
            # For undirected graphs, edge (row, col) implies edge (col, row).
            # P_rev[row, col] = exp(Q[row] @ K[col]^T * scale - L[col])

            # Load additional column data for reverse direction
            k_col_ptrs = K_ptr + cols[:, None] * stride_kn + head_id * stride_kh + d_offs[None, :] * stride_kd
            K_cols = tl.load(k_col_ptrs).to(COMPUTE_DTYPE)

            do_col_ptrs = dO_ptr + cols[:, None] * stride_don + head_id * stride_doh + d_offs[None, :] * stride_dod
            dO_cols = tl.load(do_col_ptrs)

            o_col_ptrs = O_ptr + cols[:, None] * stride_on + head_id * stride_oh + d_offs[None, :] * stride_od
            O_cols = tl.load(o_col_ptrs)

            l_col_ptrs = L_ptr + cols * stride_ln + head_id * stride_lh
            L_cols = tl.load(l_col_ptrs)

            D_cols = tl.sum(dO_cols * O_cols, axis=1)  # [16]

            # S_rev = Q[rows] @ K[cols]^T * scale  [16, 16]
            S_rev = tl.dot(Q_rows, tl.trans(K_cols)) * scale
            S_rev = tl.where(full_mask, S_rev, -float("inf"))

            # P_rev = exp(S_rev - L_cols)  -- L at COLUMNS, broadcast across rows
            P_rev = tl.exp(S_rev - L_cols[None, :])
            P_rev = tl.where(full_mask, P_rev, 0.0)

            # dV[rows] += P_rev @ dO_cols  [16, D]
            dV_acc = tl.dot(P_rev.to(COMPUTE_DTYPE), dO_cols.to(COMPUTE_DTYPE), acc=dV_acc)

            # dP_rev = V[rows] @ dO_cols^T  [16, 16]
            dP_rev = tl.dot(V_rows, tl.trans(dO_cols.to(COMPUTE_DTYPE))).to(tl.float32)

            # dS_rev = P_rev * (dP_rev - D_cols)
            dS_rev = P_rev * (dP_rev - D_cols[None, :])
            dS_rev = tl.where(full_mask, dS_rev, 0.0)
            dS_rev_scaled = dS_rev * scale

            # dQ[rows] += dS_rev_scaled @ K_cols  [16, D]
            dQ_acc = tl.dot(dS_rev_scaled.to(COMPUTE_DTYPE), K_cols, acc=dQ_acc)

    # write dK at rows (no atomics needed, each row window owns its rows)
    dk_ptrs = dK_ptr + rows[:, None] * stride_dkn + head_id * stride_dkh + d_offs[None, :] * stride_dkd
    tl.store(dk_ptrs, dK_acc, mask=row_mask[:, None])

    if IS_UNDIRECTED:
        # write dQ, dV at rows (no atomics — accumulated locally)
        dq_row_ptrs = dQ_ptr + rows[:, None] * stride_dqn + head_id * stride_dqh + d_offs[None, :] * stride_dqd
        tl.store(dq_row_ptrs, dQ_acc, mask=row_mask[:, None])

        dv_row_ptrs = dV_ptr + rows[:, None] * stride_dvn + head_id * stride_dvh + d_offs[None, :] * stride_dvd
        tl.store(dv_row_ptrs, dV_acc, mask=row_mask[:, None])


def wsb_flashattn_tc_backward(wsb, Q, K, V, output, L, dO, scale, is_undirected=False):
    """
    Backward pass computing dQ, dK, dV for multi-head case.

    All tensors Q, K, V, output, dO are [N, H, D].
    L is [N, H].

    Args:
        L: Logsumexp from forward pass
        dO: Gradient of output
        is_undirected: If True, use atomic-free reverse direction computation.

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

    if is_undirected:
        # Undirected: all gradients accumulated locally, no atomics needed
        dQ = torch.empty_like(Q, dtype=torch.float32)
        dV = torch.empty_like(V, dtype=torch.float32)
    else:
        # Directed: dQ/dV scattered with atomics -> must be zero-initialized
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)

    # dK is accumulated locally at rows -> no atomics in either mode
    dK = torch.empty_like(K, dtype=torch.float32)

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
        COMPUTE_DTYPE=_triton_dtype(Q),
        IS_UNDIRECTED=is_undirected,
    )

    return dQ, dK, dV


class WSBGraphTransformer(torch.autograd.Function):
    """Autograd function for WSB Graph Transformer"""

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        wsb: WSBFormat,
        scale: float,
        is_undirected: bool = False,
    ) -> torch.Tensor:
        Q = _low_precision(Q)
        K = K.to(Q.dtype)
        V = V.to(Q.dtype)

        ctx.wsb = wsb
        ctx.scale = scale
        ctx.is_undirected = is_undirected

        output, logsumexp = wsb_flashattn_tc_forward(wsb, Q, K, V, scale=ctx.scale)
        ctx.save_for_backward(Q, K, V, logsumexp, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        Q, K, V, logsumexp, output = ctx.saved_tensors
        head_dim = Q.shape[2]
        num_heads = Q.shape[1]
        grad_output = grad_output.view(-1, num_heads, head_dim)

        dQ, dK, dV = wsb_flashattn_tc_backward(
            ctx.wsb,
            Q,
            K,
            V,
            output,
            logsumexp,
            grad_output,
            scale=ctx.scale,
            is_undirected=ctx.is_undirected,
        )
        return dQ, dK, dV, None, None, None
