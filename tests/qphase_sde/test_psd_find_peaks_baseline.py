"""Golden-value baselines for the standalone PSD peak analyser (Wave 5B equivalence).

Migration row: PSD inline peak finding -> ``analyser.peaks`` +
``peak_finder.scipy``. The spectrum remains produced by ``PsdAnalyzer`` and
these tests pin the numerical output of the standalone finder on that spectrum.
Tolerance policy: peak ``indices`` are exact integers. ``frequencies`` and
``values`` involve pocketfft + savgol (deterministic per environment, but
last-ulp drift across NumPy/SciPy builds is possible), so they are pinned
with ``rtol=1e-6``. Golden values captured with NumPy/SciPy from uv.lock,
seed 20260911.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.peak_finder import ScipyPeakFinder, ScipyPeakFinderConfig
from qphase_sde.result import SDEResult
from qphase_sde.state import TrajectorySet

RTOL = 1e-6

DT = 0.05
N_TIME = 256
N_TRAJ = 64
DF = 2.0 * np.pi / (N_TIME * DT)


def _tone_fixture() -> TrajectorySet:
    rng = np.random.default_rng(20260911)
    t = np.arange(N_TIME) * DT
    noise = 0.05 * (
        rng.standard_normal((N_TRAJ, N_TIME, 2))
        + 1j * rng.standard_normal((N_TRAJ, N_TIME, 2))
    )
    data = np.empty((N_TRAJ, N_TIME, 2), dtype=complex)
    data[:, :, 0] = np.exp(-1j * 5 * DF * t) + 0.3 * np.exp(-1j * 23 * DF * t)
    data[:, :, 1] = 0.6 * np.exp(-1j * 9 * DF * t)
    return TrajectorySet(data=data + noise, dt=DT, meta={"mode_indices": [0, 1]})


def _baseline_peaks() -> dict[str, dict[int, dict[str, object]]]:
    spectrum = (
        PsdAnalyzer(modes=[0, 1], kind="complex")
        .analyze(_tone_fixture(), NumpyBackend())
        .data_dict
    )
    finder = ScipyPeakFinder(
        ScipyPeakFinderConfig(
            distance=5,
            smooth_window=5,
            noise_threshold=None,
            prominence=None,
        )
    )
    axis = np.asarray(spectrum["axis"])
    matrix = np.asarray(spectrum["psd"])
    return {
        "peaks": {
            mode: finder.find_peaks(axis, matrix[:, index]).model_dump()
            for index, mode in enumerate((0, 1))
        }
    }


def test_psd_find_peaks_false_emits_empty_payload():
    out = (
        PsdAnalyzer(modes=[0, 1], kind="complex")
        .analyze(_tone_fixture(), NumpyBackend())
        .data_dict
    )

    assert "peaks" not in out


def test_psd_find_peaks_true_payload_baseline():
    out = _baseline_peaks()

    mode0 = out["peaks"][0]
    assert mode0["properties"] == {}
    assert mode0["indices"] == [
        3,
        12,
        19,
        29,
        35,
        42,
        49,
        55,
        64,
        70,
        77,
        86,
        93,
        101,
        116,
        126,
        132,
        137,
        142,
        150,
        159,
        167,
        177,
        186,
        195,
        201,
        212,
        220,
        225,
        231,
        237,
        243,
        248,
    ]
    np.testing.assert_allclose(
        mode0["frequencies"],
        np.array(
            [
                -60.6921101512001,
                -56.60672670521255,
                -52.97974027499447,
                -48.10221356120544,
                -45.174176964182756,
                -41.49221399196194,
                -38.10420872517877,
                -35.15570152907517,
                -31.07634734503942,
                -27.855063383909016,
                -24.543315165118685,
                -20.153060477377466,
                -16.63769599059342,
                -12.727903205156919,
                -5.4965459513535935,
                -0.24841356799945183,
                2.465455139755878,
                4.953778480135435,
                7.328763303826403,
                11.285100820237158,
                15.671972889911725,
                19.804313617985805,
                24.491369945361363,
                28.76120096388302,
                33.50507024689797,
                36.40129854135714,
                41.56504478741306,
                45.63348755907489,
                48.093479932691324,
                51.11253898992904,
                54.18866392406092,
                57.146804776001986,
                59.37757861616876,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode0["values"],
        np.array(
            [
                3.734639200263237e-05,
                4.639586562912478e-05,
                4.394676699035597e-05,
                4.287594132092467e-05,
                4.512105966534906e-05,
                3.920236370962209e-05,
                4.4483518938959506e-05,
                4.172691709409934e-05,
                4.3626082435369056e-05,
                4.133649017677094e-05,
                5.089674365507544e-05,
                4.7568936098372695e-05,
                4.809632493553664e-05,
                4.291760849666372e-05,
                4.726427960763475e-05,
                4.4436161693710416e-05,
                0.007234426244320985,
                3.572903132814008e-05,
                4.3325843728539945e-05,
                0.002441697566499805,
                4.368430455297085e-05,
                4.778484798256447e-05,
                4.552885685095701e-05,
                3.926660782414352e-05,
                4.5926404143816036e-05,
                4.418804236431451e-05,
                3.784999094407944e-05,
                4.193389141326812e-05,
                4.621837596156487e-05,
                4.489071735360402e-05,
                3.878133303471842e-05,
                4.138711292748338e-05,
                5.040913747986365e-05,
            ]
        ),
        rtol=RTOL,
    )

    mode1 = out["peaks"][1]
    assert mode1["properties"] == {}
    assert mode1["indices"] == [
        2,
        9,
        20,
        25,
        35,
        43,
        49,
        56,
        64,
        72,
        77,
        84,
        93,
        100,
        109,
        117,
        125,
        130,
        136,
        145,
        155,
        163,
        170,
        175,
        182,
        190,
        196,
        202,
        207,
        212,
        217,
        231,
        236,
        248,
    ]
    np.testing.assert_allclose(
        mode1["frequencies"],
        np.array(
            [
                -61.4933553572175,
                -57.76108185949951,
                -52.466404903364676,
                -50.17761743945154,
                -44.99232524355234,
                -41.28113754934869,
                -38.179850730733534,
                -34.763701711760554,
                -30.814191880822044,
                -27.00823163534186,
                -24.462055823936215,
                -20.986002832111353,
                -16.628586248922435,
                -13.183337137054979,
                -9.072290812219562,
                -4.70283385500073,
                -0.8270255893102771,
                1.3877819017540942,
                4.413527660135938,
                8.878613322255848,
                13.877938632711592,
                17.437384624524444,
                20.944088872619297,
                23.67474325594471,
                27.23654267514958,
                30.87114725852974,
                33.70839501241449,
                36.98351738316412,
                39.355292091820225,
                41.818928004038504,
                43.96906496021485,
                51.08588287736282,
                53.49591832959671,
                59.25379043531512,
            ]
        ),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        mode1["values"],
        np.array(
            [
                4.441547301988163e-05,
                4.81230562023757e-05,
                4.744667791180103e-05,
                4.4154035369897574e-05,
                3.9939250642632535e-05,
                4.1239741996026974e-05,
                4.3860233528299424e-05,
                4.201423082972245e-05,
                4.265145533360209e-05,
                4.282070995768227e-05,
                4.5521367550568885e-05,
                4.531304235219715e-05,
                4.802560333239495e-05,
                4.22458332765184e-05,
                3.840324454038015e-05,
                4.371450751150214e-05,
                4.334780096710555e-05,
                4.574694400865668e-05,
                0.004795357750593226,
                4.8705877130928526e-05,
                4.580968423747334e-05,
                4.893892490855821e-05,
                4.247957471612788e-05,
                4.343054675191586e-05,
                4.232252368245889e-05,
                3.9457759319615e-05,
                4.462801899829962e-05,
                3.618137416976758e-05,
                4.4851511933342426e-05,
                4.5640085611106566e-05,
                4.274150069486282e-05,
                4.7403045542050145e-05,
                4.496532692174925e-05,
                4.4244629905799794e-05,
            ]
        ),
        rtol=RTOL,
    )


def test_standalone_peak_analyser_uses_baseline_spectrum():
    spectrum = (
        PsdAnalyzer(modes=[0, 1], kind="complex")
        .analyze(_tone_fixture(), NumpyBackend())
        .data_dict
    )
    analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(
            scan_param="unused",
        ),
        subplugins={
            "finder": ScipyPeakFinder(
                ScipyPeakFinderConfig(
                    distance=5,
                    smooth_window=5,
                    noise_threshold=None,
                    prominence=None,
                )
            )
        },
    )

    payload = analyzer.analyze(
        SDEResult(analysis={"psd": spectrum}), NumpyBackend()
    ).data_dict

    assert len(payload["candidate_tables"]) == 1
    assert payload["modes"] == [0, 1]
    assert payload["candidate_tables"][0]["columns"]["location"][0] == pytest.approx(
        -60.6921101512001, rel=RTOL
    )


def test_psd_find_peaks_carrier_anchor_baseline():
    """The dominant detected peak per mode must sit on the injected tones."""
    out = _baseline_peaks()

    top0 = int(np.argmax(out["peaks"][0]["values"]))
    assert out["peaks"][0]["indices"][top0] == 132
    assert out["peaks"][0]["frequencies"][top0] == pytest.approx(
        2.465455139755878, rel=RTOL
    )
    assert out["peaks"][0]["values"][top0] == pytest.approx(
        0.007234426244320985, rel=RTOL
    )

    top1 = int(np.argmax(out["peaks"][1]["values"]))
    assert out["peaks"][1]["indices"][top1] == 136
    assert out["peaks"][1]["frequencies"][top1] == pytest.approx(
        4.413527660135938, rel=RTOL
    )
    assert out["peaks"][1]["values"][top1] == pytest.approx(
        0.004795357750593226, rel=RTOL
    )

    secondary = int(
        np.argmax(np.isclose(out["peaks"][0]["frequencies"], 23 * DF, atol=DF))
    )
    assert out["peaks"][0]["values"][secondary] == pytest.approx(
        0.002441697566499805, rel=RTOL
    )
