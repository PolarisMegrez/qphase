"""The package's public surface: typed products, with history behind the importer.

The pre-2.0 result types are historical payloads. They stay importable from their own
modules and from :mod:`qphase_cam.data.legacy`, which is the one-time import boundary,
but they are not part of the package's runtime API: a consumer that needs a candidate
resolves the typed product into a
:class:`~qphase_cam.postprocessor.candidates.CandidateView`.
"""

from __future__ import annotations

import numpy as np
import pytest
import qphase_cam
from qphase_cam.bifurcation_result import CAMBifurcationResult
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.data import legacy
from qphase_cam.postprocessor.candidates import (
    candidate_parameters,
    candidate_state,
    candidate_view,
)

LEGACY_EXPORTS = ("CAMResult", "CAMBifurcationResult", "CAMBifurcationScanResult")
RUNTIME_EXPORTS = {
    "BifurcationCapabilityError",
    "CAMBifurcationModel",
    "CAMBogoliubovModel",
    "CAMModel",
    "CAMSymbolicSpec",
    "CAMVectorModel",
    "JacobianUnavailableError",
    "QPSGenCompatibilityError",
    "SolutionCapacityError",
}


class _Model:
    """Minimal parameter container; the parameter map only reads ``params``."""

    name = "public_api_stub"
    params = {"omega": 0.0, "gamma": 1.0}


def _legacy_candidate_result() -> CAMBifurcationResult:
    """Return one historical candidate row the importer can consume."""
    matrices = np.asarray([np.eye(2)], dtype=complex)
    signature = {
        "state_order": 3,
        "perturbation_order": 1,
        "coupling_state_order": 0,
        "exponent_numerator": 1,
        "exponent_denominator": 3,
        "sublinear": True,
        "branches": (
            {
                "epsilon_side": 1,
                "amplitude": 1.0,
                "leading_state_coefficient": [
                    [
                        {"real": float(value.real), "imag": float(value.imag)}
                        for value in row
                    ]
                    for row in np.eye(2, dtype=complex)
                ],
            },
        ),
    }
    signatures = np.empty(1, dtype=object)
    signatures[0] = [signature]
    return CAMBifurcationResult(
        states=matrices,
        state_vectors=np.asarray([matrix_to_vector(matrices[0])], dtype=float),
        control_values=np.full((1, 1), -0.5),
        control_names=("gamma",),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=np.asarray(["verified"]),
        method=np.asarray(["test"]),
        verification_digits=np.full(1, 50, dtype=np.int64),
        verification_status=np.asarray(["verified"]),
        diagnostics={"scaling_signatures": signatures},
        meta={},
    )


@pytest.mark.parametrize("name", LEGACY_EXPORTS)
def test_the_root_does_not_export_a_historical_result_type(name):
    assert not hasattr(qphase_cam, name)
    assert name not in qphase_cam.__all__


def test_the_root_declares_exactly_its_runtime_api():
    assert set(qphase_cam.__all__) == RUNTIME_EXPORTS
    for name in qphase_cam.__all__:
        assert hasattr(qphase_cam, name), name
    # The compatibility error type is a normal error, not a historical result.
    assert issubclass(qphase_cam.QPSGenCompatibilityError, Exception)


def test_the_historical_types_stay_importable_where_they_belong():
    from qphase_cam.bifurcation_result import CAMBifurcationScanResult
    from qphase_cam.result import CAMResult

    assert CAMResult.__module__ == "qphase_cam.result"
    assert CAMBifurcationResult.__module__ == "qphase_cam.bifurcation_result"
    assert CAMBifurcationScanResult.__module__ == "qphase_cam.bifurcation_result"
    for name in ("import_cam_result", "import_bifurcation_result"):
        assert callable(getattr(legacy, name)), name
    assert legacy.import_legacy_artifact in {
        value for value in vars(legacy).values() if callable(value)
    }


def test_the_typed_view_is_what_the_runtime_services_take():
    result = _legacy_candidate_result()
    product = legacy.import_bifurcation_result(result).require("candidates")
    view = candidate_view(product)
    np.testing.assert_array_equal(
        candidate_state(view, 0).chart_vector, matrix_to_vector(np.eye(2))
    )
    assert candidate_parameters(view, 0, _Model())["gamma"] == -0.5
    # The historical type is not a view, and the services say so instead of guessing.
    with pytest.raises(TypeError, match="CandidateView"):
        candidate_parameters(result, 0, _Model())
    with pytest.raises(TypeError, match="CandidateView"):
        candidate_state(result, 0)
