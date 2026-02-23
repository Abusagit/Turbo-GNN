"""Comprehensive tests for the autotuning engine.

Tests cover:
- TunableParam / AutotuneConfig dataclasses
- AutotuneCache (compute_cache_key, load, save, clear_cache)
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
)
from src.backends.base import (
    AutotuneConfig,
    BaseConvolution,
    TunableParam,
    _autotune_forward_pre_hook,
)

# ---------------------------------------------------------------------------
# Dummy convolutions
# ---------------------------------------------------------------------------


class DummyConv(BaseConvolution):
    """Minimal concrete convolution — no tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warps_per_block = 8
        self.tile_size = 32

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x


class TunableDummyConv(BaseConvolution):
    """Convolution with both kernel and graph tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warps_per_block = 8
        self.tile_size = 32

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_kernel_params(self) -> list[TunableParam]:
        return [
            TunableParam("warps_per_block", [4, 8, 16], default=8),
            TunableParam("tile_size", [16, 32], default=32),
        ]

    def get_tunable_graph_params(self) -> list[TunableParam]:
        return [
            TunableParam("huge_degree_threshold_quantile", [-1, 0.99], default=-1),
        ]

    def configure(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class KernelOnlyConv(BaseConvolution):
    """Convolution with only kernel-level tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block_size = 128

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_kernel_params(self) -> list[TunableParam]:
        return [TunableParam("block_size", [64, 128, 256], default=128)]


class GraphOnlyConv(BaseConvolution):
    """Convolution with only graph-level tunable params."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, graph: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_tunable_graph_params(self) -> list[TunableParam]:
        return [TunableParam("huge_degree_threshold_quantile", [-1, 0.9], default=-1)]


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
    # --- compute_cache_key ---

    def test_cache_key_deterministic(self):
        params = [TunableParam("a", [1, 2], default=1)]
        k1 = AutotuneCache.compute_cache_key("MyConv", 16, 100, 500, "cpu", params)
        k2 = AutotuneCache.compute_cache_key("MyConv", 16, 100, 500, "cpu", params)
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex digest

    @pytest.mark.parametrize(
        "field,args_a,args_b",
        [
            ("conv_class", ("ConvA", 16, 100, 500, "cpu"), ("ConvB", 16, 100, 500, "cpu")),
            ("feature_dim", ("C", 16, 100, 500, "cpu"), ("C", 32, 100, 500, "cpu")),
            ("num_nodes", ("C", 16, 100, 500, "cpu"), ("C", 16, 200, 500, "cpu")),
            ("num_edges", ("C", 16, 100, 500, "cpu"), ("C", 16, 100, 600, "cpu")),
            ("device", ("C", 16, 100, 500, "cpu"), ("C", 16, 100, 500, "cuda:0")),
        ],
    )
    def test_cache_key_differs_on_field(self, field, args_a, args_b):
        p = [TunableParam("a", [1], default=1)]
        assert AutotuneCache.compute_cache_key(*args_a, p) != AutotuneCache.compute_cache_key(*args_b, p)

    def test_cache_key_differs_on_param_space(self):
        p1 = [TunableParam("a", [1, 2], default=1)]
        p2 = [TunableParam("a", [1, 2, 3], default=1)]
        k1 = AutotuneCache.compute_cache_key("C", 16, 100, 500, "cpu", p1)
        k2 = AutotuneCache.compute_cache_key("C", 16, 100, 500, "cpu", p2)
        assert k1 != k2

    # --- load / save ---

    def test_load_nonexistent_returns_none(self, tmp_cache_dir):
        assert AutotuneCache.load(tmp_cache_dir, "X", "missing") is None

    def test_save_and_load(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "MyConv", "k1", {"warps": 4})
        assert AutotuneCache.load(tmp_cache_dir, "MyConv", "k1") == {"warps": 4}

    def test_save_multiple_keys(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "C", "k1", {"a": 1})
        AutotuneCache.save(tmp_cache_dir, "C", "k2", {"b": 2})
        assert AutotuneCache.load(tmp_cache_dir, "C", "k1") == {"a": 1}
        assert AutotuneCache.load(tmp_cache_dir, "C", "k2") == {"b": 2}

    def test_save_overwrites_key(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "C", "k1", {"a": 1})
        AutotuneCache.save(tmp_cache_dir, "C", "k1", {"a": 99})
        assert AutotuneCache.load(tmp_cache_dir, "C", "k1") == {"a": 99}

    def test_save_creates_nested_directory(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c")
        AutotuneCache.save(deep, "C", "k", {"x": 1})
        assert AutotuneCache.load(deep, "C", "k") == {"x": 1}

    def test_separate_files_per_conv_class(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "A", "k", {"a": 1})
        AutotuneCache.save(tmp_cache_dir, "B", "k", {"b": 2})
        assert AutotuneCache.load(tmp_cache_dir, "A", "k") == {"a": 1}
        assert AutotuneCache.load(tmp_cache_dir, "B", "k") == {"b": 2}

    def test_load_corrupted_json_returns_none(self, tmp_cache_dir):
        p = Path(tmp_cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "C_k_autotune_cache.json").write_text("{bad json")
        assert AutotuneCache.load(tmp_cache_dir, "C", "k") is None

    def test_saved_file_is_valid_json(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "C", "k1", {"a": 1})
        path = Path(tmp_cache_dir) / "C_k1_autotune_cache.json"
        data = json.loads(path.read_text())
        assert data == {"a": 1}

    # --- clear_cache ---

    def test_clear_specific_class(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "A", "k", {"a": 1})
        AutotuneCache.save(tmp_cache_dir, "B", "k", {"b": 2})
        assert AutotuneCache.clear_cache(tmp_cache_dir, "A") == 1
        assert AutotuneCache.load(tmp_cache_dir, "A", "k") is None
        assert AutotuneCache.load(tmp_cache_dir, "B", "k") == {"b": 2}

    def test_clear_all(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "A", "k", {"a": 1})
        AutotuneCache.save(tmp_cache_dir, "B", "k", {"b": 2})
        assert AutotuneCache.clear_cache(tmp_cache_dir) == 2

    def test_clear_nonexistent_dir(self):
        assert AutotuneCache.clear_cache("/nonexistent/path") == 0

    def test_clear_nonexistent_class(self, tmp_cache_dir):
        AutotuneCache.save(tmp_cache_dir, "A", "k", {"a": 1})
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
        _apply_best_config(conv, graph_sample, {"warps_per_block": 16, "tile_size": 64}, [])
        assert conv.warps_per_block == 16
        assert conv.tile_size == 64
        graph_sample.update_graph_repr_with_new_hyperparameters.assert_not_called()

    def test_graph_only(self, graph_sample):
        conv = TunableDummyConv()
        gp = [TunableParam("huge_degree_threshold_quantile", [-1, 0.99], default=-1)]
        graph_sample.kernel_related_kwargs = {"existing": 42}

        _apply_best_config(conv, graph_sample, {"huge_degree_threshold_quantile": 0.99}, gp)

        graph_sample.update_graph_repr_with_new_hyperparameters.assert_called_once_with(
            {"existing": 42, "huge_degree_threshold_quantile": 0.99},
        )

    def test_mixed(self, graph_sample):
        conv = TunableDummyConv()
        gp = [TunableParam("huge_degree_threshold_quantile", [-1, 0.99], default=-1)]
        graph_sample.kernel_related_kwargs = {}

        _apply_best_config(
            conv,
            graph_sample,
            {"warps_per_block": 4, "huge_degree_threshold_quantile": 0.99},
            gp,
        )
        assert conv.warps_per_block == 4
        graph_sample.update_graph_repr_with_new_hyperparameters.assert_called_once()

    def test_empty_config_is_noop(self, graph_sample):
        conv = TunableDummyConv()
        orig = conv.warps_per_block
        _apply_best_config(conv, graph_sample, {}, [])
        assert conv.warps_per_block == orig
        graph_sample.update_graph_repr_with_new_hyperparameters.assert_not_called()


# ===================================================================
# Tests — BaseConvolution autotune interface
# ===================================================================


class TestBaseConvolutionAutotune:
    def test_default_tunable_params_empty(self):
        conv = DummyConv()
        assert conv.get_tunable_kernel_params() == []
        assert conv.get_tunable_graph_params() == []

    def test_configure_sets_attrs(self):
        conv = DummyConv()
        conv.configure(warps_per_block=16, tile_size=64)
        assert conv.warps_per_block == 16
        assert conv.tile_size == 64

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
        assert result == {"block_size": 128}  # 2.0 ms is min
        assert conv.block_size == 128

    @patch("src.backends.autotune._microbench")
    def test_graph_only_grid_search(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, ctr = _make_time_callable_mock([3.0, 7.0])

        conv = GraphOnlyConv()
        result = run_autotune(conv, x_tensor, graph_sample, AutotuneConfig())

        assert ctr["n"] == 2
        assert result == {"huge_degree_threshold_quantile": -1}

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
            "huge_degree_threshold_quantile": -1,
            "warps_per_block": 16,
            "tile_size": 16,
        }

    @patch("src.backends.autotune._microbench")
    def test_applies_best_config_to_conv(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, _ = _make_time_callable_mock([10.0, 1.0, 10.0])

        conv = KernelOnlyConv()
        run_autotune(conv, x_tensor, graph_sample, AutotuneConfig())
        assert conv.block_size == 128

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
        assert result == {"block_size": 64}  # first is fastest

    @patch("src.backends.autotune._microbench")
    def test_equal_times_picks_first(self, mock_mb, graph_sample, x_tensor):
        mock_mb.time_callable, _ = _make_time_callable_mock([5.0])

        result = run_autotune(KernelOnlyConv(), x_tensor, graph_sample, AutotuneConfig())
        assert result == {"block_size": 64}

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

        files = list(Path(tmp_cache_dir).glob("*_autotune_cache.json"))
        assert len(files) == 1
        assert "KernelOnlyConv" in files[0].name

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
        assert conv.block_size == 128

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
        assert conv2.block_size == conv1.block_size
