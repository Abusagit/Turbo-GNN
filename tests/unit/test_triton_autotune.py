"""Tests for Triton kernel @triton.autotune decorator configurations.

Verifies decorator metadata (configs, keys, arg_names) via Python
introspection — no GPU required.
"""

from __future__ import annotations

import pytest

# pytestmark = pytest.mark.skip(reason="Requires triton>=3.1.0 for warp_specialize support")
from triton.runtime.autotuner import Autotuner

from src.backends.triton_backend.kernels_impl import (
    FLASHATTN_AUTOTUNE_CONFIGS,
    SPMM_AUTOTUNE_CONFIGS,
    wsb_flashattn_tc_backward_kernel,
    wsb_flashattn_tc_forward_kernel,
    wsb_spmm_backward_kernel_tc,
    wsb_spmm_kernel_tc,
)  # type: ignore

# ===================================================================
# Tests — Autotuner wrapping
# ===================================================================


class TestTritonAutotuneDecorators:
    def test_spmm_forward_is_autotuned(self):
        assert isinstance(wsb_spmm_kernel_tc, Autotuner)

    def test_spmm_backward_is_autotuned(self):
        assert isinstance(wsb_spmm_backward_kernel_tc, Autotuner)

    def test_flashattn_forward_is_autotuned(self):
        assert isinstance(wsb_flashattn_tc_forward_kernel, Autotuner)

    def test_flashattn_backward_is_autotuned(self):
        assert isinstance(wsb_flashattn_tc_backward_kernel, Autotuner)


# ===================================================================
# Tests — SpMM config spaces
# ===================================================================


class TestSpMMAutotuneConfigs:
    def test_spmm_forward_config_count(self):
        assert len(SPMM_AUTOTUNE_CONFIGS) == 36

    def test_spmm_backward_config_count(self):
        assert len(SPMM_AUTOTUNE_CONFIGS) == 36

    def test_spmm_forward_num_warps_values(self):
        warps = {c.num_warps for c in SPMM_AUTOTUNE_CONFIGS}
        assert warps == {1, 2, 4, 8}

    def test_spmm_forward_num_stages_values(self):
        stages = {c.num_stages for c in SPMM_AUTOTUNE_CONFIGS}
        assert stages == {1, 2, 3}

    def test_spmm_forward_loop_num_stages_values(self):
        loop_stages = {c.kwargs["LOOP_NUM_STAGES"] for c in SPMM_AUTOTUNE_CONFIGS}
        assert loop_stages == {1, 2, 3}

    def test_spmm_forward_warp_specialize_values(self):
        ws = {c.kwargs["WARP_SPECIALIZE"] for c in SPMM_AUTOTUNE_CONFIGS}
        assert ws == {True, False}

    def test_spmm_forward_all_configs_have_loop_params(self):
        for c in SPMM_AUTOTUNE_CONFIGS:
            assert "LOOP_NUM_STAGES" in c.kwargs
            assert "WARP_SPECIALIZE" in c.kwargs


# ===================================================================
# Tests — FlashAttention config spaces
# ===================================================================


class TestFlashAttnAutotuneConfigs:
    def test_flashattn_forward_config_count(self):
        assert len(FLASHATTN_AUTOTUNE_CONFIGS) == 18

    def test_flashattn_backward_config_count(self):
        assert len(FLASHATTN_AUTOTUNE_CONFIGS) == 18

    def test_flashattn_forward_num_warps_values(self):
        warps = {c.num_warps for c in FLASHATTN_AUTOTUNE_CONFIGS}
        assert warps == {2, 4, 8}

    def test_flashattn_forward_num_stages_values(self):
        stages = {c.num_stages for c in FLASHATTN_AUTOTUNE_CONFIGS}
        assert stages == {2, 3}

    def test_flashattn_forward_loop_num_stages_values(self):
        loop_stages = {c.kwargs["LOOP_NUM_STAGES"] for c in FLASHATTN_AUTOTUNE_CONFIGS}
        assert loop_stages == {1, 2, 3}

    def test_flashattn_forward_warp_specialize_values(self):
        ws = {c.kwargs["WARP_SPECIALIZE"] for c in FLASHATTN_AUTOTUNE_CONFIGS}
        assert ws == {True, False}

    def test_flashattn_forward_all_configs_have_loop_params(self):
        for c in FLASHATTN_AUTOTUNE_CONFIGS:
            assert "LOOP_NUM_STAGES" in c.kwargs
            assert "WARP_SPECIALIZE" in c.kwargs


# ===================================================================
# Tests — Autotune keys
# ===================================================================


class TestAutotuneKeys:
    def test_spmm_forward_keys(self):
        names = list(wsb_spmm_kernel_tc.keys)
        assert set(names) == {"N", "F"}

    def test_spmm_backward_keys(self):
        names = list(wsb_spmm_backward_kernel_tc.keys)
        assert set(names) == {"N", "F"}

    def test_flashattn_forward_keys(self):
        names = list(wsb_flashattn_tc_forward_kernel.keys)
        assert set(names) == {"num_nodes", "D"}

    def test_flashattn_backward_keys(self):
        names = list(wsb_flashattn_tc_backward_kernel.keys)
        assert set(names) == {"num_nodes", "D"}


# ===================================================================
# Tests — Warp specialize constraints
# ===================================================================


class TestWarpSpecializeConstraints:
    def test_spmm_warp_specialize_requires_loop_stages_ge_2(self):
        for c in SPMM_AUTOTUNE_CONFIGS:
            if c.kwargs["WARP_SPECIALIZE"] is True:
                assert c.kwargs["LOOP_NUM_STAGES"] >= 2

    def test_flashattn_warp_specialize_requires_loop_stages_ge_2(self):
        for c in FLASHATTN_AUTOTUNE_CONFIGS:
            if c.kwargs["WARP_SPECIALIZE"] is True:
                assert c.kwargs["LOOP_NUM_STAGES"] >= 2


# ===================================================================
# Tests — Kernel signature params
# ===================================================================


class TestKernelSignatureParams:
    def test_spmm_forward_has_loop_constexprs(self):
        assert "LOOP_NUM_STAGES" in wsb_spmm_kernel_tc.arg_names
        assert "WARP_SPECIALIZE" in wsb_spmm_kernel_tc.arg_names

    def test_spmm_backward_has_loop_constexprs(self):
        assert "LOOP_NUM_STAGES" in wsb_spmm_backward_kernel_tc.arg_names
        assert "WARP_SPECIALIZE" in wsb_spmm_backward_kernel_tc.arg_names

    def test_flashattn_forward_has_loop_constexprs(self):
        assert "LOOP_NUM_STAGES" in wsb_flashattn_tc_forward_kernel.arg_names
        assert "WARP_SPECIALIZE" in wsb_flashattn_tc_forward_kernel.arg_names

    def test_flashattn_backward_has_loop_constexprs(self):
        assert "LOOP_NUM_STAGES" in wsb_flashattn_tc_backward_kernel.arg_names
        assert "WARP_SPECIALIZE" in wsb_flashattn_tc_backward_kernel.arg_names
