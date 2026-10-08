"""Golden-value baselines for the ``pdist`` analyser (Wave 5B equivalence).

The current ``PolarDistAnalyzer`` numeric outputs (histogram, edges, bin
centers, mean, std) are not pinned by any existing CPU test. These tests
capture the present NumPy outputs on a fixed-seed fixture so the Wave 5B
typed-producer rewrite can be checked for exact numerical equivalence.

Tolerance policy: magnitude ``abs`` + ``np.histogram`` + mean/std are pure
NumPy IEEE-deterministic operations, so ``rtol=1e-10`` is platform-safe.
Golden values captured on NumPy 2.x with seed 20260907.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.polar_dist import PolarDistAnalyzer, PolarDistAnalyzerConfig
from qphase_sde.result import _trajectory_product
from qphase_sde.state import TrajectorySet

RTOL = 1e-10


def _execute(analyzer: PolarDistAnalyzer, trajectory: TrajectorySet) -> dict:
    product = analyzer.execute(
        {
            "trajectories": _trajectory_product(
                trajectory, scan_size=None, n_traj_per_point=None
            )
        },
        backend=NumpyBackend(),
        label="pdist",
    )["pdist"]
    payload: dict = {}
    leaves = dict(product.attributes["payload_meta"])
    leaves.update(
        {
            variable.name: product.handle(variable.name).materialize()
            for variable in product.variables
        }
    )
    for name, value in leaves.items():
        target = payload
        parts = name.split(".")
        for part in parts[:-1]:
            key = int(part) if part.isdigit() else part
            target = target.setdefault(key, {})
        key = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
        target[key] = (
            value.item()
            if isinstance(value, np.ndarray) and value.ndim == 0
            else value
        )
    return payload


def _complex_fixture() -> TrajectorySet:
    rng = np.random.default_rng(20260907)
    n_traj, n_time = 3, 64
    mode0 = rng.standard_normal((n_traj, n_time)) + 1j * rng.standard_normal(
        (n_traj, n_time)
    )
    mode1 = (
        0.5
        + 0.25j
        + 0.5
        * (
            rng.standard_normal((n_traj, n_time))
            + 1j * rng.standard_normal((n_traj, n_time))
        )
    )
    data = np.stack([mode0, mode1], axis=-1)
    return TrajectorySet(data=data, dt=0.1, meta={"mode_indices": [0, 1]})


def test_pdist_fixed_bins_baseline():
    out = _execute(
        PolarDistAnalyzer(
            config=PolarDistAnalyzerConfig(modes=[0, 1], bins=8, density=True)
        ),
        _complex_fixture(),
    )

    mode0 = out["distributions"][0]
    np.testing.assert_allclose(
        mode0["hist"],
        np.array(
            [
                0.27047221659811793,
                0.6311018387289418,
                0.5296747575046475,
                0.34935994643923557,
                0.21412383814017663,
                0.1239664326074708,
                0.03380902707476473,
                0.011269675691588243,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode0["edges"],
        np.array(
            [
                0.05841807575703755,
                0.5205727531290638,
                0.9827274305010901,
                1.4448821078731164,
                1.9070367852451429,
                2.3691914626171693,
                2.8313461399891953,
                3.2935008173612217,
                3.755655494733248,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode0["bin_centers"],
        np.array(
            [
                0.28949541444305066,
                0.751650091815077,
                1.2138047691871032,
                1.6759594465591297,
                2.138114123931156,
                2.6002688013031823,
                3.0624234786752087,
                3.5245781560472347,
            ]
        ),
        rtol=RTOL,
    )
    assert mode0["mean"] == pytest.approx(1.238057515609091, rel=RTOL)
    assert mode0["std"] == pytest.approx(0.7072570841261056, rel=RTOL)

    mode1 = out["distributions"][1]
    np.testing.assert_allclose(
        mode1["hist"],
        np.array(
            [
                0.3795912160004047,
                0.7380940311118981,
                1.2020388506679478,
                0.8646244364453669,
                0.5061216213338731,
                0.23197240977802494,
                0.04217680177782276,
                0.08435360355564551,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode1["edges"],
        np.array(
            [
                0.04315033987620841,
                0.29012655020005335,
                0.5371027605238983,
                0.7840789708477433,
                1.031055181171588,
                1.2780313914954329,
                1.525007601819278,
                1.7719838121431228,
                2.0189600224669677,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode1["bin_centers"],
        np.array(
            [
                0.16663844503813088,
                0.41361465536197584,
                0.6605908656858208,
                0.9075670760096657,
                1.1545432863335106,
                1.4015194966573554,
                1.6484957069812003,
                1.8954719173050454,
            ]
        ),
        rtol=RTOL,
    )
    assert mode1["mean"] == pytest.approx(0.7599774774908049, rel=RTOL)
    assert mode1["std"] == pytest.approx(0.3730394308789979, rel=RTOL)

    np.testing.assert_array_equal(out["modes"], [0, 1])
    assert out["bins_config"] == 8
    assert out["density"] is True


def test_pdist_auto_bins_baseline():
    out = _execute(
        PolarDistAnalyzer(
            config=PolarDistAnalyzerConfig(modes=[0], bins="auto", density=True)
        ),
        _complex_fixture(),
    )

    mode0 = out["distributions"][0]
    assert len(mode0["hist"]) == 11
    np.testing.assert_allclose(
        mode0["hist"],
        np.array(
            [
                0.23243706113900756,
                0.6973111834170228,
                0.5268573385817503,
                0.44937831820208135,
                0.41838671005021366,
                0.17045384483527223,
                0.2634286692908752,
                0.13946223668340454,
                0.06198321630373535,
                0.0,
                0.015495804075933838,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode0["edges"],
        np.array(
            [
                0.05841807575703755,
                0.3945305683912385,
                0.7306430610254394,
                1.0667555536596405,
                1.4028680462938414,
                1.7389805389280424,
                2.0750930315622433,
                2.4112055241964443,
                2.7473180168306452,
                3.083430509464846,
                3.419543002099047,
                3.755655494733248,
            ]
        ),
        rtol=RTOL,
    )
    assert mode0["mean"] == pytest.approx(1.238057515609091, rel=RTOL)
    assert mode0["std"] == pytest.approx(0.7072570841261056, rel=RTOL)
    # Current implementation resolves the string estimator and reports the
    # concrete bin count, not the original "auto" string.
    assert out["bins_config"] == 11
