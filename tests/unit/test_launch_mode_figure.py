import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts" / "ablation"))

from plot_launch_modes import bin_columns  # noqa: E402


def test_short_runs_are_left_alone():
    utilisation = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert bin_columns(utilisation, 6) is utilisation
    assert bin_columns(utilisation, 100) is utilisation


def test_binning_averages_rather_than_samples():
    # A stall that occupies half of a bin must still darken that bin. Plain subsampling would
    # drop it entirely and draw a run that never stalled.
    utilisation = np.array([[1.0], [0.0], [1.0], [0.0]])
    assert bin_columns(utilisation, 2).ravel().tolist() == [0.5, 0.5]


def test_binning_hits_the_requested_width_and_keeps_the_sm_axis():
    utilisation = np.random.default_rng(0).random((1000, 132))
    assert bin_columns(utilisation, 37).shape == (37, 132)
    # Equal-width bins conserve the mean; uneven ones only approximately, so check the exact case.
    assert bin_columns(utilisation, 40).mean() == pytest.approx(utilisation.mean(), rel=1e-9)
