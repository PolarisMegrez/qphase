"""Tests for physical-to-recorded mode mapping."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.dist import DistAnalyzer
from qphase_sde.analyser.polar_dist import PolarDistAnalyzer
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.result import SDEResult
from qphase_sde.state import TrajectorySet
from qphase_sde.utils import resolve_mode_columns

BACKEND = NumpyBackend()


def _trajectory() -> TrajectorySet:
    t = np.arange(256) * 0.1
    data = ((1.0 + 0.1 * np.cos(0.07 * t)) * np.exp(-0.4j * t))[None, :, None]
    return TrajectorySet(
        data=data,
        t0=0.0,
        dt=0.1,
        meta={"mode_indices": [3]},
    )


def test_analyzers_use_physical_mode_mapping():
    trajectory = _trajectory()

    psd = PsdAnalyzer(kind="complex", modes=[3]).analyze(trajectory, BACKEND)
    dist = DistAnalyzer(modes=[3], bins=16).analyze(trajectory, BACKEND)
    polar = PolarDistAnalyzer(modes=[3], bins=16).analyze(trajectory, BACKEND)

    assert psd.data_dict["modes"] == [3]
    assert 3 in dist.data_dict["distributions"]
    assert 3 in polar.data_dict["distributions"]


def test_missing_recorded_mode_is_rejected():
    with pytest.raises(ValueError, match="were not recorded"):
        resolve_mode_columns(_trajectory(), [2])


def test_result_round_trip_preserves_trajectory_mode_mapping(tmp_path):
    path = tmp_path / "result.npz"
    SDEResult(trajectory=_trajectory()).save(path)

    loaded = SDEResult.load(path)

    assert loaded.trajectory.meta["mode_indices"] == [3]
