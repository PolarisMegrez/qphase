"""Tests for physical-to-recorded mode mapping."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import load_bundle
from qphase_sde.analyser.base import resolve_mode_columns
from qphase_sde.analyser.dist import DistAnalyser
from qphase_sde.analyser.polar_dist import PolarDistAnalyser
from qphase_sde.analyser.psd import PsdAnalyser
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.result import SDEDataBundle, trajectory_product
from qphase_sde.state import TrajectorySet

pytestmark = pytest.mark.integration

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
    product = trajectory_product(trajectory, scan_size=None, n_traj_per_point=None)

    psd = PsdAnalyser(
        measurement={"complex_field": {"channels": [3], "representation": "complex"}}
    ).execute({"signal": product}, backend=BACKEND, label="psd")["psd"]
    dist = DistAnalyser(modes=[3], bins=16).execute(
        {"trajectories": product}, backend=BACKEND, label="dist"
    )["dist"]
    polar = PolarDistAnalyser(modes=[3], bins=16).execute(
        {"trajectories": product}, backend=BACKEND, label="polar"
    )["polar"]

    np.testing.assert_array_equal(psd.handle("modes").materialize(), [3])
    assert "distributions.3.hist" in {item.name for item in dist.variables}
    assert "distributions.3.hist" in {item.name for item in polar.variables}


def test_missing_recorded_mode_is_rejected():
    with pytest.raises(ValueError, match="were not recorded"):
        resolve_mode_columns(_trajectory(), [2])


def test_result_round_trip_preserves_trajectory_mode_mapping(tmp_path):
    trajectory = trajectory_product(
        _trajectory(), scan_size=None, n_traj_per_point=None
    )
    bundle = SDEDataBundle(
        {"trajectories": trajectory},
        provenance=SDEProvenance(dt=0.1),
    )
    bundle.save(tmp_path)

    loaded = load_bundle(tmp_path)

    assert loaded.products["trajectories"].attributes["mode_indices"] == [3]
