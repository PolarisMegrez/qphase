"""Golden-value baselines for the ``dist`` analyser (Wave 5B equivalence).

The current ``DistAnalyzer`` numeric outputs are not pinned by any existing
test (only the CuPy-vs-NumPy cross-backend equality is, and it requires a
GPU). These tests capture the present CPU NumPy outputs on fixed-seed
fixtures so the Wave 5B typed-producer rewrite can be checked for exact
numerical equivalence.

Tolerance policy: ``np.histogram``/``np.histogram2d`` are pure NumPy integer
binning plus one IEEE-deterministic normalization pass, so ``rtol=1e-10`` is
platform-safe. Golden values captured on NumPy 2.x with seed 20260907/8.
"""

from __future__ import annotations

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.dist import DistAnalyzer, DistAnalyzerConfig
from qphase_sde.state import TrajectorySet

RTOL = 1e-10


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


def test_dist_2d_complex_histogram_baseline():
    out = (
        DistAnalyzer(config=DistAnalyzerConfig(modes=[0, 1], bins=8, density=True))
        .analyze(_complex_fixture(), NumpyBackend())
        .data_dict
    )

    assert out["distributions"][0]["type"] == "2d_complex"
    assert out["distributions"][1]["type"] == "2d_complex"

    np.testing.assert_allclose(
        out["distributions"][0]["hist"],
        np.array(
            [
                [
                    0.01022180373323749,
                    0.01022180373323749,
                    0.020443607466474976,
                    0.030665411199712483,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.02044360746647498,
                    0.030665411199712472,
                    0.020443607466474976,
                    0.04088721493294997,
                    0.020443607466474986,
                    0.020443607466474962,
                    0.0,
                    0.0,
                ],
                [
                    0.030665411199712472,
                    0.061330822399424945,
                    0.0715526261326624,
                    0.10221803733237493,
                    0.07155262613266246,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.030665411199712472,
                    0.061330822399424945,
                    0.1533270559985623,
                    0.13288344853208742,
                    0.08177442986589994,
                    0.020443607466474962,
                    0.020443607466474986,
                    0.0,
                ],
                [
                    0.0,
                    0.06133082239942495,
                    0.13288344853208736,
                    0.17377066346503744,
                    0.08177442986589996,
                    0.02044360746647497,
                    0.02044360746647499,
                    0.010221803733237502,
                ],
                [
                    0.0,
                    0.03066541119971246,
                    0.07155262613266239,
                    0.12266164479884988,
                    0.020443607466474976,
                    0.020443607466474955,
                    0.010221803733237488,
                    0.0,
                ],
                [
                    0.0,
                    0.010221803733237484,
                    0.030665411199712448,
                    0.010221803733237488,
                    0.020443607466474976,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.020443607466474986,
                    0.01022180373323749,
                    0.02044360746647499,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][0]["xedges"],
        np.array(
            [
                -2.4893194554792877,
                -1.8463516601666081,
                -1.2033838648539286,
                -0.560416069541249,
                0.08255172577143055,
                0.7255195210841099,
                1.3684873163967897,
                2.0114551117094694,
                2.654422907022149,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][0]["yedges"],
        np.array(
            [
                -2.591680672326236,
                -1.7992121644624246,
                -1.0067436565986134,
                -0.21427514873480202,
                0.5781933591290089,
                1.3706618669928199,
                2.1631303748566317,
                2.9555988827204427,
                3.748067390584253,
            ]
        ),
        rtol=RTOL,
    )

    np.testing.assert_allclose(
        out["distributions"][1]["hist"],
        np.array(
            [
                [
                    0.0,
                    0.05296826808056678,
                    0.05296826808056678,
                    0.05296826808056678,
                    0.15890480424170028,
                    0.0,
                    0.05296826808056681,
                    0.052968268080566726,
                ],
                [
                    0.0,
                    0.10593653616113356,
                    0.2648413404028339,
                    0.15890480424170034,
                    0.15890480424170028,
                    0.10593653616113362,
                    0.0,
                    0.052968268080566726,
                ],
                [
                    0.0,
                    0.2648413404028339,
                    0.3707778765639675,
                    0.42374614464453425,
                    0.5296826808056677,
                    0.264841340402834,
                    0.21187307232226724,
                    0.052968268080566726,
                ],
                [
                    0.15890480424170034,
                    0.21187307232226713,
                    0.2648413404028339,
                    0.47671441272510107,
                    0.688587485047368,
                    0.4767144127251013,
                    0.10593653616113362,
                    0.15890480424170017,
                ],
                [
                    0.05296826808056678,
                    0.05296826808056678,
                    0.6885874850473682,
                    0.5826509488862347,
                    0.3707778765639674,
                    0.5826509488862349,
                    0.10593653616113362,
                    0.0,
                ],
                [
                    0.05296826808056678,
                    0.05296826808056678,
                    0.0,
                    0.42374614464453425,
                    0.4767144127251009,
                    0.21187307232226724,
                    0.05296826808056681,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.05296826808056678,
                    0.0,
                    0.15890480424170028,
                    0.10593653616113362,
                    0.05296826808056681,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.052968268080566795,
                    0.0,
                    0.10593653616113369,
                    0.0,
                ],
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][1]["xedges"],
        np.array(
            [
                -0.7233655062036748,
                -0.39746122791714456,
                -0.07155694963061432,
                0.2543473286559159,
                0.5802516069424462,
                0.9061558852289764,
                1.2320601635155066,
                1.5579644418020369,
                1.8838687200885669,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][1]["yedges"],
        np.array(
            [
                -1.0199257015415366,
                -0.7182134563153296,
                -0.41650121108912264,
                -0.11478896586291565,
                0.18692327936329134,
                0.48863552458949844,
                0.7903477698157053,
                1.0920600150419122,
                1.3937722602681195,
            ]
        ),
        rtol=RTOL,
    )


def test_dist_1d_real_histogram_baseline():
    rng = np.random.default_rng(20260908)
    ts = TrajectorySet(
        data=rng.standard_normal((2, 48, 1)), dt=0.1, meta={"mode_indices": [0]}
    )

    out = (
        DistAnalyzer(config=DistAnalyzerConfig(modes=[0], bins=6, density=False))
        .analyze(ts, NumpyBackend())
        .data_dict
    )

    assert out["distributions"][0]["type"] == "1d_real"
    np.testing.assert_allclose(
        out["distributions"][0]["hist"],
        np.array([2.0, 7.0, 31.0, 38.0, 14.0, 4.0]),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][0]["edges"],
        np.array(
            [
                -3.0746133434559964,
                -2.0988887832590475,
                -1.1231642230620986,
                -0.14743966286514976,
                0.8282848973317991,
                1.804009457528748,
                2.779734017725697,
            ]
        ),
        rtol=RTOL,
    )


def test_dist_explicit_range_baseline():
    out = (
        DistAnalyzer(
            config=DistAnalyzerConfig(
                modes=[0, 1], bins=6, density=True, range=[[-1.5, 1.5], [-1.0, 2.0]]
            )
        )
        .analyze(_complex_fixture(), NumpyBackend())
        .data_dict
    )

    np.testing.assert_allclose(
        out["distributions"][0]["hist"],
        np.array(
            [
                [
                    0.0547945205479452,
                    0.0273972602739726,
                    0.0547945205479452,
                    0.0547945205479452,
                    0.0547945205479452,
                    0.0821917808219178,
                ],
                [
                    0.136986301369863,
                    0.1643835616438356,
                    0.1095890410958904,
                    0.1917808219178082,
                    0.1095890410958904,
                    0.0547945205479452,
                ],
                [
                    0.136986301369863,
                    0.0821917808219178,
                    0.3013698630136986,
                    0.1643835616438356,
                    0.1643835616438356,
                    0.0273972602739726,
                ],
                [
                    0.0821917808219178,
                    0.1643835616438356,
                    0.1917808219178082,
                    0.273972602739726,
                    0.1095890410958904,
                    0.0821917808219178,
                ],
                [
                    0.0821917808219178,
                    0.1095890410958904,
                    0.1917808219178082,
                    0.2191780821917808,
                    0.0821917808219178,
                    0.0821917808219178,
                ],
                [
                    0.0547945205479452,
                    0.0821917808219178,
                    0.0547945205479452,
                    0.1095890410958904,
                    0.0547945205479452,
                    0.0,
                ],
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][0]["xedges"],
        np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        out["distributions"][1]["hist"],
        np.array(
            [
                [
                    0.0,
                    0.041884816753926704,
                    0.06282722513089005,
                    0.0,
                    0.020942408376963352,
                    0.0,
                ],
                [
                    0.041884816753926704,
                    0.16753926701570682,
                    0.2094240837696335,
                    0.08376963350785341,
                    0.041884816753926704,
                    0.0,
                ],
                [
                    0.14659685863874344,
                    0.4397905759162304,
                    0.4397905759162304,
                    0.33507853403141363,
                    0.08376963350785341,
                    0.0,
                ],
                [
                    0.041884816753926704,
                    0.39790575916230364,
                    0.6701570680628273,
                    0.31413612565445026,
                    0.041884816753926704,
                    0.0,
                ],
                [
                    0.041884816753926704,
                    0.020942408376963352,
                    0.18848167539267016,
                    0.08376963350785341,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.020942408376963352,
                    0.041884816753926704,
                    0.020942408376963352,
                    0.0,
                ],
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        out["distributions"][1]["yedges"],
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
        rtol=0,
        atol=0,
    )
    assert out["modes"] == [0, 1]
    assert out["bins"] == 6
    assert out["density"] is True
