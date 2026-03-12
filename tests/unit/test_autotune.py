"""Comprehensive tests for the autotuning engine.

Tests cover:
- TunableParam / AutotuneConfig dataclasses
- AutotuneCache (compute_trial_key, load_trial, save_trial, clear_cache)
- _build_combinations helper
- _apply_best_config helper
- run_autotune (grid search, cache hit/miss, backward tuning)
- BaseConvolution autotune methods (configure, autotune, enable_autotune)
- _autotune_forward_pre_hook (lazy autotuning)
- End-to-end flows
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.backends.autotune import (
    AutotuneCache,
    _apply_best_config,
    _build_combinations,
    run_autotune,
    run_autotune_kernel,
)
from src.backends.base import (
    AutotuneConfig,
    BaseConvolution,
    TunableKernel,
    TunableParam,
    _autotune_forward_pre_hook,
    _InlineAutotuneCache,
    with_autotune,
)
from src.data.converters import (
    AdjacencyForwardBackwardWithNodeBuckets,
    _bucket_nodes_by_degree,
)

# ---------------------------------------------------------------------------
# Dummy convolutions
# ---------------------------------------------------------------------------


class DummyConv(BaseConvolution):
    """Minimal concrete convolution — no tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward_warps_per_block = 8
        self.forward_tile_size = 32

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x


class TunableDummyConv(BaseConvolution):
    """Convolution with both kernel and graph tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward_warps_per_block = 8
        self.forward_tile_size = 32

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_warps_per_block", [4, 8, 16], default=8),
            TunableParam("forward_tile_size", [16, 32], default=32),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.99], default=-1),
        ]


class KernelOnlyConv(BaseConvolution):
    """Convolution with only kernel-level tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward_block_size = 128

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("forward_block_size", [64, 128, 256], default=128)]


class GraphOnlyConv(BaseConvolution):
    """Convolution with only graph-level tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9], default=-1)]


class BackwardOnlyConv(BaseConvolution):
    """Convolution with only backward kernel tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backward_grad_chunk = 512

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("backward_grad_chunk", [128, 256, 512], default=512)]


class MixedFwdBwdConv(BaseConvolution):
    """Convolution with both forward and backward kernel + graph params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward_warps_per_block = 8
        self.backward_grad_chunk = 512

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("forward_warps_per_block", [4, 8], default=8)]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.99], default=-1)]

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("backward_grad_chunk", [128, 256, 512], default=512)]

    def get_tunable_backward_graph_params(self) -> list[TunableParam]:
        return [TunableParam("backward_huge_degree_threshold_quantile", [-1, 0.9], default=-1)]


# ---------------------------------------------------------------------------
# Fake MicrobenchResult (matches the real dataclass interface)
# ---------------------------------------------------------------------------


@dataclass
class FakeMicrobenchResult:
    iters: int
    ms_per_iter: float
    device: str = "cpu"
    std_ms: float | None = None
    memory_allocated: float | None = None


def _make_time_callable_mock(results_ms: list[float]):
    """Return (fake_time_callable, call_counter_dict)."""
    counter = {"n": 0}

    def fake(fn, warmup=10, iters=50, do_memory_profile=False):
        idx = counter["n"] % len(results_ms)
        counter["n"] += 1
        return FakeMicrobenchResult(iters=iters, ms_per_iter=results_ms[idx])

    return fake, counter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_graph_sample_mock(num_nodes=100, num_edges=500):
    gs = MagicMock()
    gs.num_nodes = num_nodes
    gs.num_edges = num_edges
    gs.kernel_related_kwargs = {}
    gs.graph_repr = "mock_graph_repr"
    gs.update_graph_repr_with_new_hyperparameters = MagicMock(return_value=gs)
    return gs


@pytest.fixture
def graph_sample():
    return _make_graph_sample_mock()


@pytest.fixture
def x_tensor():
    return torch.randn(100, 16)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return str(tmp_path / "autotune_cache")


# ===================================================================
# Tests — TunableParam
# ===================================================================


class TestTunableParam:
    def test_creation(self):
        p = TunableParam("warps_per_block", [1, 2, 4], default=2)
        assert p.name == "warps_per_block"
        assert p.values == [1, 2, 4]
        assert p.default == 2

    def test_different_value_types(self):
        assert TunableParam("s", [32, 64, 128], default=64).values == [32, 64, 128]
        assert TunableParam("q", [0.9, 0.95, 0.99], default=0.95).default == 0.95
        assert TunableParam("t", [-1, 0.9], default=-1).default == -1

    def test_single_value(self):
        assert len(TunableParam("x", [42], default=42).values) == 1

    def test_empty_values(self):
        assert TunableParam("x", [], default=0).values == []


# ===================================================================
# Tests — AutotuneConfig
# ===================================================================


class TestAutotuneConfig:
    def test_defaults(self):
        cfg = AutotuneConfig()
        assert cfg.warmup == 10
        assert cfg.iters == 50
        assert cfg.tune_backward is False
        assert cfg.cache_dir is None
        assert cfg.use_cache is True

    def test_custom_values(self):
        cfg = AutotuneConfig(
            warmup=5,
            iters=100,
            tune_backward=True,
            cache_dir="/tmp/cache",
            use_cache=False,
        )
        assert cfg.warmup == 5
        assert cfg.iters == 100
        assert cfg.tune_backward is True
        assert cfg.cache_dir == "/tmp/cache"
        assert cfg.use_cache is False

    def test_partial_override(self):
        cfg = AutotuneConfig(warmup=20)
        assert cfg.warmup == 20
        assert cfg.iters == 50  # kept default


# ===================================================================
# Tests — AutotuneCache
# ===================================================================


class TestAutotuneCache:
    # --- compute_trial_key ---

    def test_trial_key_deterministic(self):
        cfg = {"forward_block_size": 128}
        k1 = AutotuneCache.compute_trial_key("MyConv", 16, 100, 500, "cpu", cfg)
        k2 = AutotuneCache.compute_trial_key("MyConv", 16, 100, 500, "cpu", cfg)
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex digest

    @pytest.mark.parametrize(
        "field,args_a,args_b",
        [
            ("conv_class", ("ConvA", 16, 100, 500, "cpu"), ("ConvB", 16, 100, 500, "cpu")),
            ("feature_dim", ("C", 16, 100, 500, "cpu"), ("C", 32, 100, 500, "cpu")),
            ("num_nodes", ("C", 16, 100, 500, "cpu"), ("C", 16, 200, 500, "cpu")),
            ("num_edges", ("C", 16, 100, 500, "cpu"), ("C", 16, 100, 600, "cpu")),
            ("gpu_name", ("C", 16, 100, 500, "cpu"), ("C", 16, 100, 500, "NVIDIA H100")),
        ],
    )
    def test_trial_key_differs_on_field(self, field, args_a, args_b):
        cfg = {"a": 1}
        assert AutotuneCache.compute_trial_key(*args_a, cfg) != AutotuneCache.compute_trial_key(*args_b, cfg)

    def test_trial_key_differs_on_config(self):
        k1 = AutotuneCache.compute_trial_key("C", 16, 100, 500, "cpu", {"a": 1})
        k2 = AutotuneCache.compute_trial_key("C", 16, 100, 500, "cpu", {"a": 2})
        assert k1 != k2

    def test_trial_key_sorted_keys_consistent(self):
        """Key order in trial_config should not matter (json sort_keys=True)."""
        k1 = AutotuneCache.compute_trial_key("C", 16, 100, 500, "cpu", {"b": 2, "a": 1})
        k2 = AutotuneCache.compute_trial_key("C", 16, 100, 500, "cpu", {"a": 1, "b": 2})
        assert k1 == k2

    # --- load_trial / save_trial ---

    def test_load_trial_nonexistent_returns_none(self, tmp_cache_dir):
        assert AutotuneCache.load_trial(tmp_cache_dir, "X", "missing") is None

    def test_save_and_load_trial(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "MyConv", "k1", 5.0)
        assert AutotuneCache.load_trial(tmp_cache_dir, "MyConv", "k1") == 5.0

    def test_save_multiple_trials(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "C", "k1", 3.0)
        AutotuneCache.save_trial(tmp_cache_dir, "C", "k2", 7.0)
        assert AutotuneCache.load_trial(tmp_cache_dir, "C", "k1") == 3.0
        assert AutotuneCache.load_trial(tmp_cache_dir, "C", "k2") == 7.0

    def test_save_trial_overwrites(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "C", "k1", 3.0)
        AutotuneCache.save_trial(tmp_cache_dir, "C", "k1", 99.0)
        assert AutotuneCache.load_trial(tmp_cache_dir, "C", "k1") == 99.0

    def test_save_creates_nested_directory(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c")
        AutotuneCache.save_trial(deep, "C", "k", 1.0)
        assert AutotuneCache.load_trial(deep, "C", "k") == 1.0

    def test_separate_subdirs_per_conv_class(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "A", "k", 1.0)
        AutotuneCache.save_trial(tmp_cache_dir, "B", "k", 2.0)
        assert AutotuneCache.load_trial(tmp_cache_dir, "A", "k") == 1.0
        assert AutotuneCache.load_trial(tmp_cache_dir, "B", "k") == 2.0
        # Verify subdirectory structure
        assert (Path(tmp_cache_dir) / "A" / "k.json").exists()
        assert (Path(tmp_cache_dir) / "B" / "k.json").exists()

    def test_load_corrupted_json_returns_none(self, tmp_cache_dir):
        p = Path(tmp_cache_dir) / "C"
        p.mkdir(parents=True, exist_ok=True)
        (p / "k.json").write_text("{bad json")
        assert AutotuneCache.load_trial(tmp_cache_dir, "C", "k") is None

    def test_saved_file_is_valid_json(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "C", "k1", 5.0)
        path = Path(tmp_cache_dir) / "C" / "k1.json"
        data = json.loads(path.read_text())
        assert data == {"ms_per_iter": 5.0}

    # --- clear_cache ---

    def test_clear_specific_class(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "A", "k", 1.0)
        AutotuneCache.save_trial(tmp_cache_dir, "B", "k", 2.0)
        assert AutotuneCache.clear_cache(tmp_cache_dir, "A") == 1
        assert AutotuneCache.load_trial(tmp_cache_dir, "A", "k") is None
        assert AutotuneCache.load_trial(tmp_cache_dir, "B", "k") == 2.0

    def test_clear_all(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "A", "k", 1.0)
        AutotuneCache.save_trial(tmp_cache_dir, "B", "k", 2.0)
        assert AutotuneCache.clear_cache(tmp_cache_dir) == 2

    def test_clear_nonexistent_dir(self):
        assert AutotuneCache.clear_cache("/nonexistent/path") == 0

    def test_clear_nonexistent_class(self, tmp_cache_dir):
        AutotuneCache.save_trial(tmp_cache_dir, "A", "k", 1.0)
        assert AutotuneCache.clear_cache(tmp_cache_dir, "X") == 0


# ===================================================================
# Tests — _build_combinations
# ===================================================================


class TestBuildCombinations:
    def test_empty_params(self):
        assert _build_combinations([]) == [{}]

    def test_single_param(self):
        params = [TunableParam("a", [1, 2, 3], default=1)]
        assert _build_combinations(params) == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_two_params_cartesian(self):
        params = [
            TunableParam("a", [1, 2], default=1),
            TunableParam("b", [10, 20], default=10),
        ]
        assert _build_combinations(params) == [
            {"a": 1, "b": 10},
            {"a": 1, "b": 20},
            {"a": 2, "b": 10},
            {"a": 2, "b": 20},
        ]

    def test_combination_count(self):
        params = [
            TunableParam("a", [1, 2, 3], default=1),
            TunableParam("b", [10, 20], default=10),
            TunableParam("c", list(range(4)), default=0),
        ]
        assert len(_build_combinations(params)) == 3 * 2 * 4


# ===================================================================
# Tests — _apply_best_config
# ===================================================================


class TestApplyBestConfig:
    def test_kernel_only(self, graph_sample):
        conv = TunableDummyConv()
        _apply_best_config(conv, graph_sample, {"forward_warps_per_block": 16, "forward_tile_size": 64}, [])
        assert conv.forward_warps_per_block == 16
        assert conv.forward_tile_size == 64
        graph_sample.update_graph_repr_with_new_hyperparameters.assert_not_called()

    def test_graph_only(self, graph_sample):
        conv = TunableDummyConv()
        gp = [TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.99], default=-1)]
        graph_sample.kernel_related_kwargs = {"existing": 42}

        _apply_best_config(conv, graph_sample, {"forward_huge_degree_threshold_quantile": 0.99}, gp)

        graph_sample.update_graph_repr_with_new_hyperparameters.assert_called_once_with(
            {"existing": 42, "forward_huge_degree_threshold_quantile": 0.99},
        )

    def test_mixed(self, graph_sample):
        conv = TunableDummyConv()
        gp = [TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.99], default=-1)]
        graph_sample.kernel_related_kwargs = {}

        _apply_best_config(
            conv,
            graph_sample,
            {"forward_warps_per_block": 4, "forward_huge_degree_threshold_quantile": 0.99},
            gp,
        )
        assert conv.forward_warps_per_block == 4
        graph_sample.update_graph_repr_with_new_hyperparameters.assert_called_once()

    def test_empty_config_is_noop(self, graph_sample):
        conv = TunableDummyConv()
        orig = conv.forward_warps_per_block
        _apply_best_config(conv, graph_sample, {}, [])
        assert conv.forward_warps_per_block == orig
        graph_sample.update_graph_repr_with_new_hyperparameters.assert_not_called()


# ===================================================================
# Tests — BaseConvolution autotune interface
# ===================================================================


class TestBaseConvolutionAutotune:
    def test_default_tunable_params_empty(self):
        conv = DummyConv()
        assert conv.get_tunable_forward_kernel_params() == []
        assert conv.get_tunable_forward_graph_params() == []
        assert conv.get_tunable_backward_kernel_params() == []
        assert conv.get_tunable_backward_graph_params() == []

    def test_configure_sets_attrs(self):
        conv = DummyConv()
        conv.configure(forward_warps_per_block=16, forward_tile_size=64)
        assert conv.forward_warps_per_block == 16
        assert conv.forward_tile_size == 64

    def test_configure_creates_new_attrs(self):
        conv = DummyConv()
        conv.configure(new_param=42)
        assert conv.new_param == 42

    def test_initial_state(self):
        conv = DummyConv()
        assert conv._autotune_enabled is False
        assert conv._is_tuned is False
        assert conv._is_autotuning is False
        assert isinstance(conv._autotune_config, AutotuneConfig)
        assert conv._graph_sample_ref is None

    def test_enable_autotune_flag(self):
        conv = DummyConv()
        conv.enable_autotune()
        assert conv._autotune_enabled is True

    def test_enable_autotune_stores_config(self):
        conv = DummyConv()
        cfg = AutotuneConfig(warmup=3, iters=10)
        conv.enable_autotune(config=cfg)
        assert conv._autotune_config is cfg

    def test_enable_autotune_stores_graph_sample(self, graph_sample):
        conv = DummyConv()
        conv.enable_autotune(graph_sample=graph_sample)
        assert conv._graph_sample_ref is graph_sample

    def test_enable_autotune_registers_hook(self):
        conv = DummyConv()
        n_before = len(conv._forward_pre_hooks)
        conv.enable_autotune()
        assert len(conv._forward_pre_hooks) == n_before + 1

    @patch("src.backends.autotune.run_autotune")
    def test_autotune_delegates_to_run_autotune(self, mock_run, graph_sample, x_tensor):
        mock_run.return_value = {"warps_per_block": 16}
        conv = DummyConv()
        result = conv.autotune(x_tensor, graph_sample)

        mock_run.assert_called_once_with(conv, x_tensor, graph_sample, conv._autotune_config)
        assert result == {"warps_per_block": 16}
        assert conv._is_tuned is True

    @patch("src.backends.autotune.run_autotune")
    def test_autotune_config_override(self, mock_run, graph_sample, x_tensor):
        mock_run.return_value = {}
        conv = DummyConv()
        cfg = AutotuneConfig(warmup=1, iters=5)
        conv.autotune(x_tensor, graph_sample, config=cfg)

        assert conv._autotune_config is cfg
        mock_run.assert_called_once_with(conv, x_tensor, graph_sample, cfg)

    @patch("src.backends.autotune.run_autotune", side_effect=RuntimeError("boom"))
    def test_autotune_resets_flag_on_error(self, mock_run, graph_sample, x_tensor):
        conv = DummyConv()
        with pytest.raises(RuntimeError, match="boom"):
            conv.autotune(x_tensor, graph_sample)
        assert conv._is_autotuning is False
        assert conv._is_tuned is False  # should NOT be marked tuned

    @patch("src.backends.autotune.run_autotune")
    def test_is_autotuning_true_during_run(self, mock_run, graph_sample, x_tensor):
        captured = {}

        def side_effect(conv, x, gs, cfg):
            captured["during"] = conv._is_autotuning
            return {}

        mock_run.side_effect = side_effect
        conv = DummyConv()
        conv.autotune(x_tensor, graph_sample)
        assert captured["during"] is True
        assert conv._is_autotuning is False


# ===================================================================
# Tests — _autotune_forward_pre_hook
# ===================================================================


class TestForwardPreHook:
    def test_noop_when_disabled(self, x_tensor):
        conv = DummyConv()  # _autotune_enabled = False
        assert _autotune_forward_pre_hook(conv, (x_tensor,)) is None

    def test_noop_when_already_tuned(self, x_tensor):
        conv = DummyConv()
        conv._autotune_enabled = True
        conv._is_tuned = True
        assert _autotune_forward_pre_hook(conv, (x_tensor,)) is None

    def test_noop_when_autotuning_in_progress(self, x_tensor):
        conv = DummyConv()
        conv._autotune_enabled = True
        conv._is_autotuning = True
        assert _autotune_forward_pre_hook(conv, (x_tensor,)) is None

    def test_warns_without_graph_sample(self, x_tensor, caplog):
        conv = DummyConv()
        conv._autotune_enabled = True
        conv._graph_sample_ref = None

        with caplog.at_level("WARNING"):
            _autotune_forward_pre_hook(conv, (x_tensor,))

        assert conv._is_tuned is True
        assert "no GraphSample available" in caplog.text

    @patch("src.backends.autotune.run_autotune", return_value={})
    def test_triggers_on_first_forward(self, mock_run, graph_sample, x_tensor):
        conv = TunableDummyConv()
        conv.enable_autotune(graph_sample=graph_sample)
        conv(x_tensor, graph_sample.graph_repr)

        mock_run.assert_called_once()
        assert conv._is_tuned is True

    @patch("src.backends.autotune.run_autotune", return_value={})
    def test_does_not_retrigger(self, mock_run, graph_sample, x_tensor):
        conv = TunableDummyConv()
        conv.enable_autotune(graph_sample=graph_sample)
        conv(x_tensor, graph_sample.graph_repr)
        conv(x_tensor, graph_sample.graph_repr)
        assert mock_run.call_count == 1

    def test_detects_graph_sample_as_second_arg(self, x_tensor):
        """Hook picks up GraphSample passed as the graph argument."""
        from src.data.datasets import GraphSample

        mock_gs = MagicMock(spec=GraphSample)
        mock_gs.num_nodes = 10
        mock_gs.num_edges = 20
        mock_gs.kernel_related_kwargs = {}
        mock_gs.graph_repr = "mock"

        conv = DummyConv()
        conv._autotune_enabled = True
        conv._graph_sample_ref = None

        with patch.object(conv, "autotune") as mock_at:
            _autotune_forward_pre_hook(conv, (x_tensor, mock_gs))
            mock_at.assert_called_once_with(x_tensor, mock_gs)


# ===================================================================
# Tests — run_autotune
# ===================================================================


class TestRunAutotune:
    @patch("src.backends.autotune._microbench.time_callable")
    def test_no_tunable_params_early_return(self, mock_tc, graph_sample, x_tensor):
        result = run_autotune(DummyConv(), x_tensor, graph_sample, AutotuneConfig())
        assert result == {}
        mock_tc.assert_not_called()

    @patch("src.backends.autotune._microbench")
    def test_kernel_only_grid_search(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([5.0, 2.0, 8.0])

        conv = KernelOnlyConv()
        result = run_autotune(conv, x_tensor, graph_sample, AutotuneConfig())

        assert ctr["n"] == 3
        assert result == {"forward_block_size": 128}  # 2.0 ms is min
        assert conv.forward_block_size == 128

    @patch("src.backends.autotune._microbench")
    def test_graph_only_grid_search(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([3.0, 7.0])

        conv = GraphOnlyConv()
        result = run_autotune(conv, x_tensor, graph_sample, AutotuneConfig())

        assert ctr["n"] == 2
        assert result == {"forward_huge_degree_threshold_quantile": -1}

    @patch("src.backends.autotune._microbench")
    def test_mixed_grid_search(self, mock_mb, graph_sample, x_tensor):
        # TunableDummyConv: 2 graph x (3 kernel_a x 2 kernel_b) = 12 trials
        ms = [10.0] * 12
        ms[4] = 1.0  # trial 4 is fastest
        mock_mb.time_callable, ctr = _make_time_callable_mock(ms)

        conv = TunableDummyConv()
        result = run_autotune(conv, x_tensor, graph_sample, AutotuneConfig())

        assert ctr["n"] == 12
        # Trial 4: graph={hdtq:-1}, kernel={wpb:16, ts:16}
        assert result == {
            "forward_huge_degree_threshold_quantile": -1,
            "forward_warps_per_block": 16,
            "forward_tile_size": 16,
        }

    @patch("src.backends.autotune._microbench")
    def test_applies_best_config_to_conv(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, _ = _make_time_callable_mock([10.0, 1.0, 10.0])

        conv = KernelOnlyConv()
        run_autotune(conv, x_tensor, graph_sample, AutotuneConfig())
        assert conv.forward_block_size == 128

    @patch("src.backends.autotune._microbench")
    def test_graph_param_triggers_rebuild(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, _ = _make_time_callable_mock([3.0, 7.0])

        run_autotune(GraphOnlyConv(), x_tensor, graph_sample, AutotuneConfig())
        assert graph_sample.update_graph_repr_with_new_hyperparameters.call_count >= 1

    @patch("src.backends.autotune._microbench")
    def test_passes_warmup_iters_to_timer(self, mock_mb, graph_sample, x_tensor):
        captured = []

        def fake(fn, warmup=10, iters=50, do_memory_profile=False):
            captured.append((warmup, iters))
            return FakeMicrobenchResult(iters=iters, ms_per_iter=1.0)

        mock_mb.time_callable = fake

        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, AutotuneConfig(warmup=3, iters=7))
        for w, i in captured:
            assert w == 3
            assert i == 7

    @patch("src.backends.autotune._microbench")
    def test_1d_input_feature_dim(self, mock_mb, graph_sample):
        mock_mb.time_callable, _ = _make_time_callable_mock([1.0, 2.0, 3.0])

        result = run_autotune(
            KernelOnlyConv(),
            torch.randn(100),
            graph_sample,
            AutotuneConfig(),
        )
        assert result == {"forward_block_size": 64}  # first is fastest

    @patch("src.backends.autotune._microbench")
    def test_equal_times_picks_first(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, _ = _make_time_callable_mock([5.0])

        result = run_autotune(KernelOnlyConv(), x_tensor, graph_sample, AutotuneConfig())
        assert result == {"forward_block_size": 64}

    @patch("src.backends.autotune._microbench")
    def test_backward_tuning_creates_callables(self, mock_mb, graph_sample, x_tensor):
        fns = []

        def fake(fn, **kw):
            fns.append(fn)
            return FakeMicrobenchResult(iters=50, ms_per_iter=5.0)

        mock_mb.time_callable = fake

        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, AutotuneConfig(tune_backward=True))
        assert len(fns) == 3
        assert all(callable(f) for f in fns)

    # --- caching ---

    @patch("src.backends.autotune._microbench")
    def test_cache_save(self, mock_mb, graph_sample, x_tensor, tmp_cache_dir):
        mock_mb.time_callable, _ = _make_time_callable_mock([5.0, 2.0, 8.0])

        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, AutotuneConfig(cache_dir=tmp_cache_dir))

        trial_dir = Path(tmp_cache_dir) / "KernelOnlyConv"
        assert trial_dir.is_dir()
        files = list(trial_dir.glob("*.json"))
        assert len(files) == 3  # one per trial

    @patch("src.backends.autotune._microbench")
    def test_cache_hit_skips_search(self, mock_mb, graph_sample, x_tensor, tmp_cache_dir):
        mock_mb.time_callable, ctr = _make_time_callable_mock([5.0, 2.0, 8.0])

        cfg = AutotuneConfig(cache_dir=tmp_cache_dir)
        r1 = run_autotune(KernelOnlyConv(), x_tensor, graph_sample, cfg)

        ctr["n"] = 0
        r2 = run_autotune(KernelOnlyConv(), x_tensor, graph_sample, cfg)

        assert ctr["n"] == 0  # no timing on cache hit
        assert r1 == r2

    @patch("src.backends.autotune._microbench")
    def test_use_cache_false_skips_load(self, mock_mb, graph_sample, x_tensor, tmp_cache_dir):
        mock_mb.time_callable, ctr = _make_time_callable_mock([5.0, 2.0, 8.0])

        cfg = AutotuneConfig(cache_dir=tmp_cache_dir, use_cache=False)
        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, cfg)
        n1 = ctr["n"]

        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, cfg)
        assert ctr["n"] == n1 * 2  # ran grid search both times

    @patch("src.backends.autotune._microbench")
    def test_no_cache_dir_means_no_caching(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([5.0, 2.0, 8.0])

        cfg = AutotuneConfig(cache_dir=None)
        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, cfg)
        n1 = ctr["n"]

        run_autotune(KernelOnlyConv(), x_tensor, graph_sample, cfg)
        assert ctr["n"] == n1 * 2

    # --- backward-specific tests ---

    @patch("src.backends.autotune._microbench")
    def test_backward_separate_grid_search(self, mock_mb, graph_sample, x_tensor):
        """Forward and backward run independent grid searches."""
        fns = []

        def fake(fn, **kw):
            fns.append(fn)
            return FakeMicrobenchResult(iters=50, ms_per_iter=5.0)

        mock_mb.time_callable = fake

        # MixedFwdBwdConv:
        #   fwd: 2 kernel (warps_per_block) x 2 graph (hdtq) = 4 trials
        #   bwd: 3 kernel (grad_chunk) x 2 graph (bwd_hdtq) = 6 trials
        conv = MixedFwdBwdConv()
        run_autotune(conv, x_tensor, graph_sample, AutotuneConfig(tune_backward=True))

        assert len(fns) == 4 + 6  # 4 forward + 6 backward

    @patch("src.backends.autotune._microbench")
    def test_backward_params_merged_with_forward(self, mock_mb, graph_sample, x_tensor):
        """Combined result dict has both forward and backward params."""
        # fwd trials: 2 kernel x 2 graph = 4, bwd trials: 3 kernel x 2 graph = 6
        fwd_ms = [10.0, 10.0, 1.0, 10.0]  # best at trial 2 (wpb=8, hdtq=-1 -> wait, let me think)
        bwd_ms = [10.0, 10.0, 10.0, 2.0, 10.0, 10.0]  # best at trial 3
        mock_mb.time_callable, ctr = _make_time_callable_mock(fwd_ms + bwd_ms)

        conv = MixedFwdBwdConv()
        result = run_autotune(conv, x_tensor, graph_sample, AutotuneConfig(tune_backward=True))

        # result must contain keys from both fwd and bwd
        assert "forward_warps_per_block" in result
        assert "forward_huge_degree_threshold_quantile" in result
        assert "backward_grad_chunk" in result
        assert "backward_huge_degree_threshold_quantile" in result
        assert ctr["n"] == 10

    @patch("src.backends.autotune._microbench")
    def test_no_backward_params_skips_backward_search(self, mock_mb, graph_sample, x_tensor):
        """tune_backward=True but no bwd params → only forward search runs."""
        fns = []

        def fake(fn, **kw):
            fns.append(fn)
            return FakeMicrobenchResult(iters=50, ms_per_iter=5.0)

        mock_mb.time_callable = fake

        # KernelOnlyConv has no backward params
        conv = KernelOnlyConv()
        run_autotune(conv, x_tensor, graph_sample, AutotuneConfig(tune_backward=True))

        # Only forward search: 3 kernel combos
        assert len(fns) == 3

    @patch("src.backends.autotune._microbench")
    def test_backward_only_conv(self, mock_mb, graph_sample, x_tensor):
        """Conv with only backward params, tune_backward=True."""
        mock_mb.time_callable, ctr = _make_time_callable_mock([10.0, 2.0, 8.0])

        conv = BackwardOnlyConv()
        result = run_autotune(conv, x_tensor, graph_sample, AutotuneConfig(tune_backward=True))

        assert ctr["n"] == 3  # 3 backward kernel combos
        assert result == {"backward_grad_chunk": 256}  # 2.0 ms is min (index 1 → value 256)
        assert conv.backward_grad_chunk == 256


# ===================================================================
# Tests — end-to-end flows
# ===================================================================


class TestEndToEnd:
    @patch("src.backends.autotune._microbench")
    def test_enable_then_forward_triggers_autotune(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([10.0, 1.0, 5.0])

        conv = KernelOnlyConv()
        conv.enable_autotune(graph_sample=graph_sample)

        assert not conv._is_tuned
        conv(x_tensor, graph_sample.graph_repr)
        assert conv._is_tuned
        assert ctr["n"] == 3
        assert conv.forward_block_size == 128

    @patch("src.backends.autotune._microbench")
    def test_explicit_autotune_prevents_hook_retrigger(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([10.0, 1.0, 5.0])

        conv = KernelOnlyConv()
        conv.enable_autotune(graph_sample=graph_sample)
        conv.autotune(x_tensor, graph_sample)
        n_after = ctr["n"]

        conv(x_tensor, graph_sample.graph_repr)  # hook must not re-trigger
        assert ctr["n"] == n_after

    @patch("src.backends.autotune._microbench")
    def test_cache_round_trip(self, mock_mb, graph_sample, x_tensor, tmp_cache_dir):
        mock_mb.time_callable, ctr = _make_time_callable_mock([10.0, 1.0, 5.0])
        cfg = AutotuneConfig(cache_dir=tmp_cache_dir)

        conv1 = KernelOnlyConv()
        r1 = conv1.autotune(x_tensor, graph_sample, config=cfg)

        ctr["n"] = 0
        conv2 = KernelOnlyConv()
        r2 = conv2.autotune(x_tensor, graph_sample, config=cfg)

        assert ctr["n"] == 0
        assert r1 == r2
        assert conv2.forward_block_size == conv1.forward_block_size


# ===================================================================
# Dummy TunableKernel subclasses
# ===================================================================


class DummyKernel(TunableKernel):
    """Minimal kernel with no tunable params."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def _execute(self, *args, **kwargs):
        self.call_count += 1
        return torch.zeros(4)


class TunableDummyKernel(TunableKernel):
    """Kernel with forward kernel + graph tunable params."""

    def __init__(self):
        super().__init__()
        self.forward_warps_per_block = 8
        self.forward_tile_size = 32

    def _execute(self, *args, **kwargs):
        return torch.zeros(4)

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_warps_per_block", [4, 8, 16], default=8),
            TunableParam("forward_tile_size", [16, 32], default=32),
        ]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.99], default=-1),
        ]

    def make_forward_bench_fn(self, x, graph_repr, **kwargs):
        def _bench():
            return self._execute(graph_repr, x)

        return _bench


class BackwardTunableKernel(TunableKernel):
    """Kernel with backward kernel params."""

    def __init__(self):
        super().__init__()
        self.backward_grad_chunk = 512

    def _execute(self, *args, **kwargs):
        return torch.zeros(4)

    def get_tunable_backward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("backward_grad_chunk", [128, 256, 512], default=512)]

    def make_backward_bench_fn(self, x, graph_repr, **kwargs):
        def _bench():
            return self._execute(graph_repr, x)

        return _bench


class KernelOnlyTunableKernel(TunableKernel):
    """Kernel with only forward kernel params."""

    def __init__(self):
        super().__init__()
        self.forward_block_size = 128

    def _execute(self, *args, **kwargs):
        return torch.zeros(4)

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("forward_block_size", [64, 128, 256], default=128)]


# ===================================================================
# Tests — TunableKernel ABC
# ===================================================================


class TestTunableKernel:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            TunableKernel()

    def test_default_tunable_params_empty(self):
        k = DummyKernel()
        assert k.get_tunable_forward_kernel_params() == []
        assert k.get_tunable_forward_graph_params() == []
        assert k.get_tunable_backward_kernel_params() == []
        assert k.get_tunable_backward_graph_params() == []

    def test_configure_sets_attrs(self):
        k = TunableDummyKernel()
        k.configure(forward_warps_per_block=16, forward_tile_size=64)
        assert k.forward_warps_per_block == 16
        assert k.forward_tile_size == 64

    def test_configure_creates_new_attrs(self):
        k = DummyKernel()
        k.configure(new_param=42)
        assert k.new_param == 42

    def test_name_property(self):
        assert DummyKernel().name == "DummyKernel"
        assert TunableDummyKernel().name == "TunableDummyKernel"

    def test_initial_state(self):
        k = DummyKernel()
        assert k._autotune_enabled is False
        assert k._is_tuned is False
        assert k._is_autotuning is False
        assert isinstance(k._autotune_config, AutotuneConfig)

    def test_callable(self):
        k = DummyKernel()
        result = k()  # __call__ -> _execute
        assert k.call_count == 1
        assert result.shape == (4,)

    def test_tunable_params_returned(self):
        k = TunableDummyKernel()
        fwd_k = k.get_tunable_forward_kernel_params()
        fwd_g = k.get_tunable_forward_graph_params()
        assert len(fwd_k) == 2
        assert len(fwd_g) == 1
        assert fwd_k[0].name == "forward_warps_per_block"

    def test_make_forward_bench_fn_default_delegates_to_execute(self):
        k = DummyKernel()
        bench_fn = k.make_forward_bench_fn(torch.zeros(4), None)
        result = bench_fn()
        assert result.shape == (4,)
        assert k.call_count == 1


# ===================================================================
# Tests — run_autotune_kernel
# ===================================================================


class TestRunAutotuneKernel:
    @patch("src.backends.autotune._microbench.time_callable")
    def test_no_tunable_params_early_return(self, mock_tc, graph_sample, x_tensor):
        result = run_autotune_kernel(DummyKernel(), x_tensor, graph_sample, AutotuneConfig())
        assert result == {}
        mock_tc.assert_not_called()

    @patch("src.backends.autotune._microbench")
    def test_kernel_only_grid_search(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([5.0, 2.0, 8.0])

        kernel = KernelOnlyTunableKernel()
        result = run_autotune_kernel(kernel, x_tensor, graph_sample, AutotuneConfig())

        assert ctr["n"] == 3
        assert result == {"forward_block_size": 128}
        assert kernel.forward_block_size == 128

    @patch("src.backends.autotune._microbench")
    def test_mixed_grid_search(self, mock_mb, graph_sample, x_tensor):
        # TunableDummyKernel: 2 graph x (3 kernel_a x 2 kernel_b) = 12 trials
        ms = [10.0] * 12
        ms[4] = 1.0  # trial 4 is fastest
        mock_mb.time_callable, ctr = _make_time_callable_mock(ms)

        kernel = TunableDummyKernel()
        result = run_autotune_kernel(kernel, x_tensor, graph_sample, AutotuneConfig())

        assert ctr["n"] == 12
        assert result == {
            "forward_huge_degree_threshold_quantile": -1,
            "forward_warps_per_block": 16,
            "forward_tile_size": 16,
        }

    @patch("src.backends.autotune._microbench")
    def test_applies_best_config(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, _ = _make_time_callable_mock([10.0, 1.0, 10.0])

        kernel = KernelOnlyTunableKernel()
        run_autotune_kernel(kernel, x_tensor, graph_sample, AutotuneConfig())
        assert kernel.forward_block_size == 128

    @patch("src.backends.autotune._microbench")
    def test_backward_tuning(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([10.0, 2.0, 8.0])

        kernel = BackwardTunableKernel()
        result = run_autotune_kernel(kernel, x_tensor, graph_sample, AutotuneConfig(tune_backward=True))

        assert ctr["n"] == 3
        assert result == {"backward_grad_chunk": 256}
        assert kernel.backward_grad_chunk == 256

    @patch("src.backends.autotune._microbench")
    def test_cache_save(self, mock_mb, graph_sample, x_tensor, tmp_cache_dir):
        mock_mb.time_callable, _ = _make_time_callable_mock([5.0, 2.0, 8.0])

        run_autotune_kernel(
            KernelOnlyTunableKernel(),
            x_tensor,
            graph_sample,
            AutotuneConfig(cache_dir=tmp_cache_dir),
        )

        trial_dir = Path(tmp_cache_dir) / "KernelOnlyTunableKernel"
        assert trial_dir.is_dir()
        files = list(trial_dir.glob("*.json"))
        assert len(files) == 3

    @patch("src.backends.autotune._microbench")
    def test_cache_hit_skips_search(self, mock_mb, graph_sample, x_tensor, tmp_cache_dir):
        mock_mb.time_callable, ctr = _make_time_callable_mock([5.0, 2.0, 8.0])

        cfg = AutotuneConfig(cache_dir=tmp_cache_dir)
        r1 = run_autotune_kernel(KernelOnlyTunableKernel(), x_tensor, graph_sample, cfg)

        ctr["n"] = 0
        r2 = run_autotune_kernel(KernelOnlyTunableKernel(), x_tensor, graph_sample, cfg)

        assert ctr["n"] == 0
        assert r1 == r2


# ===================================================================
# Tests — TunableKernel.autotune() method
# ===================================================================


class TestTunableKernelAutotune:
    @patch("src.backends.autotune.run_autotune_kernel")
    def test_delegates_to_run_autotune_kernel(self, mock_run, graph_sample, x_tensor):
        mock_run.return_value = {"forward_block_size": 64}
        kernel = KernelOnlyTunableKernel()
        result = kernel.autotune(x_tensor, graph_sample)

        mock_run.assert_called_once_with(kernel, x_tensor, graph_sample, kernel._autotune_config)
        assert result == {"forward_block_size": 64}
        assert kernel._is_tuned is True

    @patch("src.backends.autotune.run_autotune_kernel")
    def test_config_override(self, mock_run, graph_sample, x_tensor):
        mock_run.return_value = {}
        kernel = KernelOnlyTunableKernel()
        cfg = AutotuneConfig(warmup=1, iters=5)
        kernel.autotune(x_tensor, graph_sample, config=cfg)

        assert kernel._autotune_config is cfg
        mock_run.assert_called_once_with(kernel, x_tensor, graph_sample, cfg)

    @patch("src.backends.autotune.run_autotune_kernel", side_effect=RuntimeError("boom"))
    def test_resets_flag_on_error(self, mock_run, graph_sample, x_tensor):
        kernel = KernelOnlyTunableKernel()
        with pytest.raises(RuntimeError, match="boom"):
            kernel.autotune(x_tensor, graph_sample)
        assert kernel._is_autotuning is False
        assert kernel._is_tuned is False

    @patch("src.backends.autotune.run_autotune_kernel")
    def test_is_autotuning_true_during_run(self, mock_run, graph_sample, x_tensor):
        captured = {}

        def side_effect(k, x, gs, cfg):
            captured["during"] = k._is_autotuning
            return {}

        mock_run.side_effect = side_effect
        kernel = KernelOnlyTunableKernel()
        kernel.autotune(x_tensor, graph_sample)
        assert captured["during"] is True
        assert kernel._is_autotuning is False


# ===================================================================
# Tests — BaseConvolution delegation to kernel callables
# ===================================================================


class DelegatingConv(BaseConvolution):
    """Conv that registers a TunableDummyKernel."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel = TunableDummyKernel()
        self.register_kernel(self.kernel)

    def forward(self, x, graph, **kwargs):
        return x


class MultiKernelConv(BaseConvolution):
    """Conv that registers multiple kernels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k1 = KernelOnlyTunableKernel()
        self.k2 = BackwardTunableKernel()
        self.register_kernel(self.k1)
        self.register_kernel(self.k2)

    def forward(self, x, graph, **kwargs):
        return x


class TestBaseConvolutionDelegation:
    def test_register_kernel(self):
        conv = DelegatingConv()
        assert len(conv._kernel_callables) == 1
        assert conv._kernel_callables[0] is conv.kernel

    def test_delegates_forward_kernel_params(self):
        conv = DelegatingConv()
        params = conv.get_tunable_forward_kernel_params()
        assert len(params) == 2
        assert params[0].name == "forward_warps_per_block"
        assert params[1].name == "forward_tile_size"

    def test_delegates_forward_graph_params(self):
        conv = DelegatingConv()
        params = conv.get_tunable_forward_graph_params()
        assert len(params) == 1
        assert params[0].name == "forward_huge_degree_threshold_quantile"

    def test_delegates_backward_params(self):
        conv = DelegatingConv()
        assert conv.get_tunable_backward_kernel_params() == []
        assert conv.get_tunable_backward_graph_params() == []

    def test_configure_routes_to_kernel(self):
        conv = DelegatingConv()
        conv.configure(forward_warps_per_block=16, forward_tile_size=64)
        assert conv.kernel.forward_warps_per_block == 16
        assert conv.kernel.forward_tile_size == 64

    def test_configure_unknown_param_on_self(self):
        conv = DelegatingConv()
        conv.configure(forward_warps_per_block=16, custom_param=42)
        assert conv.kernel.forward_warps_per_block == 16
        assert conv.custom_param == 42

    def test_multi_kernel_aggregates_params(self):
        conv = MultiKernelConv()
        fwd_k = conv.get_tunable_forward_kernel_params()
        bwd_k = conv.get_tunable_backward_kernel_params()
        assert len(fwd_k) == 1  # from KernelOnlyTunableKernel
        assert fwd_k[0].name == "forward_block_size"
        assert len(bwd_k) == 1  # from BackwardTunableKernel
        assert bwd_k[0].name == "backward_grad_chunk"

    def test_multi_kernel_configure_routes_correctly(self):
        conv = MultiKernelConv()
        conv.configure(forward_block_size=256, backward_grad_chunk=128)
        assert conv.k1.forward_block_size == 256
        assert conv.k2.backward_grad_chunk == 128

    def test_no_kernel_callables_configure_fallback(self):
        """Conv with no registered kernels uses setattr fallback."""
        conv = DummyConv()
        conv.configure(forward_warps_per_block=16)
        assert conv.forward_warps_per_block == 16

    @patch("src.backends.autotune._microbench")
    def test_autotune_via_delegation(self, mock_mb, graph_sample, x_tensor):
        """end-to-end: autotune on a conv with registered kernel."""
        # DelegatingConv uses TunableDummyKernel: 2 graph x 6 kernel = 12 trials
        ms = [10.0] * 12
        ms[4] = 1.0
        mock_mb.time_callable, ctr = _make_time_callable_mock(ms)

        conv = DelegatingConv()
        result = conv.autotune(x_tensor, graph_sample)

        assert ctr["n"] == 12
        assert "forward_warps_per_block" in result
        assert conv._is_tuned is True

    @patch("src.backends.autotune._microbench")
    def test_enable_autotune_triggers_on_forward(self, mock_mb, graph_sample, x_tensor):
        ms = [10.0] * 12
        ms[4] = 1.0
        mock_mb.time_callable, ctr = _make_time_callable_mock(ms)

        conv = DelegatingConv()
        conv.enable_autotune(graph_sample=graph_sample)
        assert not conv._is_tuned
        conv(x_tensor, graph_sample.graph_repr)
        assert conv._is_tuned


# ===================================================================
# Helper: create a real AdjacencyForwardBackwardWithNodeBuckets
# ===================================================================


def _make_graph_repr(num_nodes=100, avg_degree=5):
    """Create a simple graph repr for testing."""
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    degrees = torch.randint(1, avg_degree * 2, (num_nodes,))
    indptr[1:] = degrees.cumsum(0)
    num_edges = indptr[-1].item()
    indices = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int32)

    all_nodes = torch.arange(num_nodes, dtype=torch.int32)
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=indptr,
        forward_indices=indices,
        backward_indptr=indptr.clone(),
        backward_indices=indices.clone(),
        forward_light_nodes=all_nodes,
        forward_heavy_nodes=torch.tensor([], dtype=torch.int32),
        backward_light_nodes=all_nodes,
        backward_heavy_nodes=torch.tensor([], dtype=torch.int32),
    )


@pytest.fixture
def graph_repr():
    return _make_graph_repr()


# ===================================================================
# Tests — repartition
# ===================================================================


class TestRepartition:
    def test_repartition_creates_new_instance(self, graph_repr):
        new_graph = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.9)
        assert new_graph is not graph_repr

    def test_repartition_shares_csr_tensors(self, graph_repr):
        new_graph = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.9)
        assert new_graph.forward_indptr is graph_repr.forward_indptr
        assert new_graph.forward_indices is graph_repr.forward_indices
        assert new_graph.backward_indptr is graph_repr.backward_indptr
        assert new_graph.backward_indices is graph_repr.backward_indices

    def test_repartition_changes_forward_buckets(self, graph_repr):
        new_graph = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.5)
        # With 50th percentile, there should be some heavy nodes
        assert new_graph.forward_heavy_nodes.numel() > 0
        total = new_graph.forward_light_nodes.numel() + new_graph.forward_heavy_nodes.numel()
        assert total == graph_repr.forward_indptr.numel() - 1

    def test_repartition_changes_backward_buckets(self, graph_repr):
        new_graph = graph_repr.repartition(backward_huge_degree_threshold_quantile=0.5)
        assert new_graph.backward_heavy_nodes.numel() > 0
        total = new_graph.backward_light_nodes.numel() + new_graph.backward_heavy_nodes.numel()
        assert total == graph_repr.backward_indptr.numel() - 1

    def test_repartition_no_kwargs_unchanged(self, graph_repr):
        new_graph = graph_repr.repartition()
        assert new_graph.forward_light_nodes is graph_repr.forward_light_nodes
        assert new_graph.forward_heavy_nodes is graph_repr.forward_heavy_nodes

    def test_repartition_preserves_max_degree(self, graph_repr):
        new_graph = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.9)
        assert new_graph.max_degree == graph_repr.max_degree


# ===================================================================
# Tests — _InlineAutotuneCache
# ===================================================================


class TestInlineAutotuneCache:
    def test_lookup_empty_returns_none(self, graph_repr):
        cache = _InlineAutotuneCache()
        assert cache.lookup(graph_repr, 16) is None

    def test_store_and_lookup(self, graph_repr):
        cache = _InlineAutotuneCache()
        result = {"kernel_config": {"warps": 8}, "graph_repr": graph_repr}
        cache.store(graph_repr, 16, result)
        cached = cache.lookup(graph_repr, 16)
        assert cached is result

    def test_different_feat_dim_miss(self, graph_repr):
        cache = _InlineAutotuneCache()
        result = {"kernel_config": {}, "graph_repr": graph_repr}
        cache.store(graph_repr, 16, result)
        assert cache.lookup(graph_repr, 32) is None

    def test_different_graph_miss(self):
        cache = _InlineAutotuneCache()
        g1 = _make_graph_repr(50)
        g2 = _make_graph_repr(60)
        result = {"kernel_config": {}, "graph_repr": g1}
        cache.store(g1, 16, result)
        assert cache.lookup(g2, 16) is None

    def test_cached_value_includes_graph_repr(self, graph_repr):
        cache = _InlineAutotuneCache()
        repartitioned = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.9)
        result = {"kernel_config": {"warps": 4}, "graph_repr": repartitioned}
        cache.store(graph_repr, 16, result)
        cached = cache.lookup(graph_repr, 16)
        assert cached["graph_repr"] is repartitioned


# ===================================================================
# Tests — with_autotune decorator
# ===================================================================


class _TestKernel(TunableKernel):
    """Test kernel for decorator tests."""

    def __init__(self, reduce="sum", **kwargs):
        super().__init__()
        self.reduce = reduce
        self.forward_block_size = 128
        self.execute_calls = []

    def _execute(self, graph, x, **kwargs):
        self.execute_calls.append((graph, x, kwargs))
        return x * 2

    def get_tunable_forward_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("forward_block_size", [64, 128], default=128)]

    def get_tunable_forward_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("forward_huge_degree_threshold_quantile", [-1, 0.9], default=-1),
        ]


class TestWithAutotuneDecorator:
    def setup_method(self):
        # Clear singleton cache between tests
        TunableKernel._shared_instances.clear()

    def test_decorator_accepts_autotune_kwarg(self, graph_repr):
        """The wrapper accepts autotune and autotune_config kwargs."""
        called = []

        @with_autotune(_TestKernel, init_params=("reduce",))
        def my_fn(graph, x, block_size=128, reduce="sum"):
            called.append(True)
            return x

        x = torch.randn(100, 16)
        # Without autotune — should call original
        my_fn(graph_repr, x, reduce="sum", autotune=False)
        assert len(called) == 1

        # autotune_config is also accepted without error
        my_fn(graph_repr, x, reduce="sum", autotune=False, autotune_config=None)
        assert len(called) == 2

    def test_non_autotune_passthrough(self, graph_repr):
        called = []

        @with_autotune(_TestKernel, init_params=("reduce",))
        def my_fn(graph, x, reduce="sum"):
            called.append(True)
            return x + 1

        x = torch.randn(100, 16)
        result = my_fn(graph_repr, x, reduce="sum")
        assert len(called) == 1
        assert torch.equal(result, x + 1)

    def test_non_autotune_default(self, graph_repr):
        """Without autotune kwarg, calls original function."""
        called = []

        @with_autotune(_TestKernel)
        def my_fn(graph, x):
            called.append(True)
            return x

        x = torch.randn(100, 16)
        my_fn(graph_repr, x)
        assert len(called) == 1

    @patch("src.backends.base._InlineAutotuneCache.lookup", return_value=None)
    @patch("src.backends.base.TunableKernel._inline_autotune")
    def test_autotune_routes_to_kernel(self, mock_autotune, mock_lookup, graph_repr):
        mock_autotune.return_value = {"kernel_config": {}, "graph_repr": graph_repr}

        @with_autotune(_TestKernel, init_params=("reduce",))
        def my_fn(graph, x, reduce="sum"):
            return x

        x = torch.randn(100, 16)
        my_fn(graph_repr, x, reduce="sum", autotune=True)
        mock_autotune.assert_called_once()

    def test_preserves_function_name(self):
        @with_autotune(_TestKernel)
        def my_special_fn(graph, x):
            return x

        assert my_special_fn.__name__ == "my_special_fn"


# ===================================================================
# Tests — _get_or_create singleton
# ===================================================================


class TestGetOrCreate:
    def setup_method(self):
        TunableKernel._shared_instances.clear()

    def test_same_kwargs_returns_same_instance(self):
        k1 = _TestKernel._get_or_create(reduce="sum")
        k2 = _TestKernel._get_or_create(reduce="sum")
        assert k1 is k2

    def test_different_kwargs_returns_different_instances(self):
        k1 = _TestKernel._get_or_create(reduce="sum")
        k2 = _TestKernel._get_or_create(reduce="min")
        assert k1 is not k2

    def test_no_kwargs_returns_same_instance(self):
        k1 = _TestKernel._get_or_create()
        k2 = _TestKernel._get_or_create()
        assert k1 is k2

    def test_different_classes_return_different_instances(self):
        k1 = _TestKernel._get_or_create()
        k2 = DummyKernel._get_or_create()
        assert k1 is not k2


# ===================================================================
# Tests — per-kernel caching
# ===================================================================


class TestPerKernelCaching:
    def setup_method(self):
        TunableKernel._shared_instances.clear()

    def test_different_kernels_cache_independently(self, graph_repr):
        k1 = _TestKernel._get_or_create(reduce="sum")
        k2 = _TestKernel._get_or_create(reduce="min")

        r1_graph = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.9)
        r2_graph = graph_repr.repartition(forward_huge_degree_threshold_quantile=0.5)

        k1._inline_cache.store(graph_repr, 16, {"kernel_config": {"forward_block_size": 64}, "graph_repr": r1_graph})
        k2._inline_cache.store(graph_repr, 16, {"kernel_config": {"forward_block_size": 128}, "graph_repr": r2_graph})

        c1 = k1._inline_cache.lookup(graph_repr, 16)
        c2 = k2._inline_cache.lookup(graph_repr, 16)

        assert c1["kernel_config"]["forward_block_size"] == 64
        assert c2["kernel_config"]["forward_block_size"] == 128
        assert c1["graph_repr"] is r1_graph
        assert c2["graph_repr"] is r2_graph


# ===================================================================
# Tests — inline autotune
# ===================================================================


class TestInlineAutotune:
    def setup_method(self):
        TunableKernel._shared_instances.clear()

    @patch("src.benchmarking.microbench.time_callable")
    def test_inline_autotune_tunes_kernel_and_graph(self, mock_tc, graph_repr):
        # _TestKernel has 2 graph combos x 2 kernel combos = 4 trials
        ms_values = [10.0, 5.0, 3.0, 8.0]
        counter = {"n": 0}

        def fake(fn, warmup=10, iters=50, do_memory_profile=False):
            idx = counter["n"] % len(ms_values)
            counter["n"] += 1
            return FakeMicrobenchResult(iters=iters, ms_per_iter=ms_values[idx])

        mock_tc.side_effect = fake

        kernel = _TestKernel(reduce="sum")
        result = kernel._inline_autotune(torch.randn(100, 16), graph_repr)

        assert counter["n"] == 4
        assert "kernel_config" in result
        assert "graph_repr" in result

    @patch("src.benchmarking.microbench.time_callable")
    def test_inline_autotune_returns_best(self, mock_tc, graph_repr):
        # Best is trial 2 (graph combo 1, kernel combo 0) → 1.0 ms
        ms_values = [10.0, 10.0, 1.0, 10.0]
        counter = {"n": 0}

        def fake(fn, warmup=10, iters=50, do_memory_profile=False):
            idx = counter["n"] % len(ms_values)
            counter["n"] += 1
            return FakeMicrobenchResult(iters=iters, ms_per_iter=ms_values[idx])

        mock_tc.side_effect = fake

        kernel = _TestKernel(reduce="sum")
        result = kernel._inline_autotune(torch.randn(100, 16), graph_repr)

        # Trial 2: graph combo index=1 (q=0.9), kernel combo index=0 (block_size=64)
        assert result["kernel_config"] == {"forward_block_size": 64}

    @patch("src.benchmarking.microbench.time_callable")
    def test_inline_autotune_no_params(self, mock_tc, graph_repr):
        kernel = DummyKernel()  # no tunable params
        result = kernel._inline_autotune(torch.randn(100, 16), graph_repr)
        assert result == {"kernel_config": {}, "graph_repr": graph_repr}
        mock_tc.assert_not_called()


# ===================================================================
# Tests — TunableKernel.__call__ autotune path
# ===================================================================


class TestTunableKernelCallAutotune:
    def setup_method(self):
        TunableKernel._shared_instances.clear()

    @patch("src.benchmarking.microbench.time_callable")
    def test_call_with_autotune_true(self, mock_tc, graph_repr):
        ms_values = [5.0, 3.0, 8.0, 10.0]
        counter = {"n": 0}

        def fake(fn, warmup=10, iters=50, do_memory_profile=False):
            idx = counter["n"] % len(ms_values)
            counter["n"] += 1
            return FakeMicrobenchResult(iters=iters, ms_per_iter=ms_values[idx])

        mock_tc.side_effect = fake

        kernel = _TestKernel(reduce="sum")
        x = torch.randn(100, 16)
        result = kernel(graph_repr, x, autotune=True)

        # Should have autotuned and produced a result
        assert result is not None
        assert counter["n"] == 4  # 2 graph x 2 kernel combos

    @patch("src.benchmarking.microbench.time_callable")
    def test_call_with_autotune_caches(self, mock_tc, graph_repr):
        ms_values = [5.0, 3.0, 8.0, 10.0]
        counter = {"n": 0}

        def fake(fn, warmup=10, iters=50, do_memory_profile=False):
            idx = counter["n"] % len(ms_values)
            counter["n"] += 1
            return FakeMicrobenchResult(iters=iters, ms_per_iter=ms_values[idx])

        mock_tc.side_effect = fake

        kernel = _TestKernel(reduce="sum")
        x = torch.randn(100, 16)
        kernel(graph_repr, x, autotune=True)  # first call: autotunes
        n_after_first = counter["n"]
        kernel(graph_repr, x, autotune=True)  # second call: cached
        assert counter["n"] == n_after_first  # no additional benchmark calls

    def test_call_without_autotune(self, graph_repr):
        kernel = _TestKernel(reduce="sum")
        x = torch.randn(100, 16)
        result = kernel(graph_repr, x)
        assert len(kernel.execute_calls) == 1
        assert torch.equal(result, x * 2)
