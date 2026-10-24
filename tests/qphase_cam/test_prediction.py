"""Phase 10F3b: the immutable prediction contract and its postprocessor.

The numeric fixture is the 10F3a one-mode toy (state matrix = canonical vector,
``J(v) = -1 - 3v²``, unit diffusion, ``v = 0.5``), so the prediction numbers can be
checked against the reconstruction the service already verified.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.data import load_bundle
from qphase_cam.data import build_state_product, cam_bundle
from qphase_cam.data.evidence import SourceArtifactRef
from qphase_cam.nonlinear import (
    ScalarReconstructionConfig,
    ou_intensities,
    reconstructed_intensity,
    response_prediction,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.nonlinear_prediction import (
    NonlinearPrediction,
    NonlinearPredictionConfig,
)
from qphase_cam.prediction import (
    CAMPrediction,
    build_prediction_product,
    prediction_identity,
)

X0 = 0.5
IDENTITY = ObservableIdentity(
    kind=ObservableKind.STATE,
    sector=ChargeSector.CHARGE_ZERO,
    name="x",
    measurement_mapping="x",
)


class _Adapter:
    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray(vector, dtype=float).reshape(1, 1).astype(complex)

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * value**2]])

    def parameter_jacobian(self, vector, params=None):
        del params
        return np.asarray([[1.7]])

    def parameter_direction(self, name, scale=1.0):
        assert name == "mu"
        return np.asarray([scale])


class _Model:
    n_modes = 1
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        return np.ones_like(np.asarray(matrix), dtype=complex)

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


class _BoundedModel(_Model):
    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        value = float(np.real(np.asarray(matrix).reshape(-1)[0]))
        return np.asarray([[1.0 - value]], dtype=complex)


class _ConfiningAdapter(_Adapter):
    """Confining toy so the reconstruction reaches a valid domain."""

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * value**2]])


def _observable(*, gain: float = 1.0):
    return FunctionalObservable(
        identity=IDENTITY,
        value_fn=lambda adapter, state, params: gain
        * float(np.asarray(state, dtype=float).reshape(-1)[0]),
        gradient_fn=lambda adapter, state, params: [gain],
    )


def _confining_manifold():
    from qphase_cam.nonlinear import NonlinearManifold

    return NonlinearManifold(
        order=3,
        soft_rate=1.0,
        critical_eigenvalue_real=-1.0,
        critical_eigenvalue_imag=0.0,
        noncritical_spectral_gap=None,
        right=(1.0,),
        left=(1.0,),
        embeddings=((1.0,),),
        drift=(0.0, -1.5, -0.5, -1.25),
        maximum_bordered_condition=10.0,
        confining=True,
        method={"service": "test fixture"},
    )


def _prediction(**overrides) -> CAMPrediction:
    payload = {
        "prediction_id": "pending",
        "target_identity": "cam:solution:test",
        "observable_name": "x",
        "observable_kind": "state",
        "charge_sector": "charge_zero",
        "measurement_mapping": "x",
        "approximation": "nonlinear_manifold",
        "order": 3,
        "status": CAMOutcomeStatus.VALID,
        "source_artifacts": (SourceArtifactRef(artifact_id="artifact-a"),),
        "allan_intensity_reconstructed": 0.25,
    }
    payload.update(overrides)
    draft = CAMPrediction(**payload)
    return draft.model_copy(update={"prediction_id": prediction_identity(draft)})


# --- contract --------------------------------------------------------------


def test_the_prediction_contract_is_frozen_and_content_addressed():
    prediction = _prediction()
    assert prediction.prediction_id.startswith("cam:prediction:")
    assert len(prediction.prediction_id.split(":")[-1]) == 16
    assert prediction.prediction_id == prediction_identity(prediction)
    with pytest.raises(ValidationError):
        prediction.order = 5  # frozen model: no setters
    changed = _prediction(allan_intensity_reconstructed=0.26)
    assert changed.prediction_id != prediction.prediction_id
    other_source = _prediction(
        source_artifacts=(SourceArtifactRef(artifact_id="artifact-b"),)
    )
    assert other_source.prediction_id != prediction.prediction_id


def test_a_complete_prediction_needs_a_source_and_an_intermediate_a_reason():
    with pytest.raises(ValueError, match="needs at least one source artifact"):
        CAMPrediction(
            prediction_id="pending",
            target_identity="cam:solution:test",
            observable_name="x",
            observable_kind="state",
            charge_sector="charge_zero",
            measurement_mapping="x",
            approximation="nonlinear_manifold",
            order=3,
            status=CAMOutcomeStatus.VALID,
        )
    intermediate = _prediction(
        completeness="intermediate",
        source_artifacts=(),
        reason="in-memory state",
    )
    assert intermediate.completeness == "intermediate"
    with pytest.raises(ValueError, match="must state its reason"):
        _prediction(completeness="intermediate", source_artifacts=())
    with pytest.raises(ValueError, match="must not contain"):
        SourceArtifactRef(artifact_id="run:7")


def test_the_product_is_typed_and_keeps_both_completeness_markers():
    complete = _prediction()
    intermediate = _prediction(
        completeness="intermediate",
        source_artifacts=(),
        reason="in-memory state",
        order=5,
    )
    product = build_prediction_product([complete, intermediate])
    assert product.attributes["product"] == "prediction"
    assert product.attributes["prediction_count"] == 2
    assert product.attributes["complete_count"] == 1
    assert product.attributes["intermediate_count"] == 1
    assert list(np.asarray(product.handle("completeness").materialize())) == [
        "complete",
        "intermediate",
    ]
    assert list(np.asarray(product.handle("approximation").materialize())) == [
        "nonlinear_manifold",
        "nonlinear_manifold",
    ]
    assert np.asarray(product.handle("allan_intensity_reconstructed").materialize())[
        1
    ] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="needs at least one prediction"):
        build_prediction_product([])
    with pytest.raises(ValueError, match="ids must be unique"):
        build_prediction_product([complete, complete])


def test_the_product_round_trips_in_both_layouts(tmp_path):
    product = build_prediction_product([_prediction()])
    for layout in ("single", "sharded"):
        root = tmp_path / layout
        bundle = cam_bundle({"prediction": product})
        bundle.save_dataset(root, layout=layout, shard_target_bytes=4096)
        restored = load_bundle(root).require("prediction")
        assert restored.attributes["prediction_count"] == 1
        assert list(np.asarray(restored.handle("completeness").materialize())) == [
            "complete"
        ]
        sources = str(np.asarray(restored.handle("source_artifacts").materialize())[0])
        assert "artifact-a" in sources
        assert int(np.asarray(restored.handle("order").materialize())[0]) == 3


# --- response service ------------------------------------------------------


def test_the_response_split_matches_the_reference_semantics():
    adapter = _Adapter()
    response = response_prediction(
        adapter,
        _observable(),
        state=[X0],
        params={"mu": 0.0},
        parameter="mu",
    )
    # rate = 1.75, projection = 1, drive = 1.7, left.drive = 1.7
    assert response.rate == pytest.approx(1.75, rel=1e-12)
    assert response.projection == pytest.approx(1.0, rel=1e-12)
    assert response.projected_drive == pytest.approx(1.7, rel=1e-12)
    assert response.soft_response == pytest.approx(1.7 / 1.75, rel=1e-12)
    assert response.explicit_response == pytest.approx(0.0, abs=1e-9)
    assert response.state_response[0] == pytest.approx(1.7 / 1.75, rel=1e-12)
    assert response.full_response == pytest.approx(1.7 / 1.75, rel=1e-9)
    # regular = full - soft; the readout is linear, so the explicit term vanishes.
    assert response.regular_response == pytest.approx(
        response.full_response - response.soft_response, rel=1e-12
    )
    assert response.derivative.method == "central_difference"
    assert response.derivative.metadata["quantity"] == "parameter_direction_response"


# --- postprocessor ---------------------------------------------------------


class _Patch:
    def __init__(self, monkeypatch, adapter):
        monkeypatch.setattr(
            "qphase_cam.postprocessor.nonlinear_prediction."
            "QPSGenDynamicsAdapter.from_model",
            lambda model: adapter,
        )


def _product(values=(X0,), *, success=None):
    states = np.zeros((len(values), 1, 1, 1), dtype=complex)
    states[:, 0, 0, 0] = np.asarray(values, dtype=complex)
    count = len(values)
    return build_state_product(
        states=states,
        residuals=np.zeros((count, 1)),
        success=(
            np.ones((count, 1), dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(count, 1)
        ),
        valid_mask=np.ones((count, 1), dtype=bool),
        solution_count=np.ones(count, dtype=np.int64),
        parameters={"mu": np.asarray(values, dtype=float)},
    )


def _context(product, *, artifact_id="artifact-search-1", model=None):
    metadata = {} if artifact_id is None else {"input_artifact_id": artifact_id}
    return CAMAnnotationContext(
        model=model or _Model(),
        metadata=metadata,
        products={"state": product},
        source_role="state",
    )


def test_the_postprocessor_publishes_complete_predictions(monkeypatch):
    _Patch(monkeypatch, _Adapter())
    product = _product()
    processor = NonlinearPrediction(
        orders=(3,),
        perturbation_parameter="mu",
        reconstruction=ScalarReconstructionConfig(grid_points=401),
    )
    prediction_product = processor.process(product, _context(product))
    assert prediction_product.attributes["product"] == "prediction"
    assert list(
        np.asarray(prediction_product.handle("completeness").materialize())
    ) == ["complete"]
    # The one-mode toy cannot confine its truncation, which the prediction states.
    assert str(np.asarray(prediction_product.handle("status").materialize())[0]) == (
        "uncontrolled_approximation"
    )
    assert (
        str(np.asarray(prediction_product.handle("approximation").materialize())[0])
        == "nonlinear_manifold"
    )
    soft = np.asarray(prediction_product.handle("allan_intensity_soft").materialize())[
        0
    ]
    full = np.asarray(prediction_product.handle("allan_intensity_full").materialize())[
        0
    ]
    reconstructed = np.asarray(
        prediction_product.handle("allan_intensity_reconstructed").materialize()
    )[0]
    assert soft == pytest.approx(0.32653061224489793, rel=1e-9)
    assert full == pytest.approx(soft, rel=1e-9)
    assert np.isfinite(reconstructed)
    # The regular response is the full-minus-soft remainder and the information
    # rate divides by the reconstructed intensity.
    regular = np.asarray(
        prediction_product.handle("expected_frequency_response_regular").materialize()
    )[0]
    full_response = np.asarray(
        prediction_product.handle("expected_frequency_response_full").materialize()
    )[0]
    assert np.isfinite(regular) and np.isfinite(full_response)
    information = np.asarray(
        prediction_product.handle("full_information_rate").materialize()
    )[0]
    assert information == pytest.approx(full_response**2 / reconstructed, rel=1e-9)
    sources = str(
        np.asarray(prediction_product.handle("source_artifacts").materialize())[0]
    )
    assert "artifact-search-1" in sources


def test_the_postprocessor_marks_a_sourceless_prediction_intermediate(monkeypatch):
    _Patch(monkeypatch, _Adapter())
    product = _product()
    prediction_product = NonlinearPrediction(
        orders=(3,),
        perturbation_parameter="mu",
        reconstruction=ScalarReconstructionConfig(grid_points=201),
    ).process(product, _context(product, artifact_id=None))
    assert list(
        np.asarray(prediction_product.handle("completeness").materialize())
    ) == ["intermediate"]
    reason = str(np.asarray(prediction_product.handle("reason").materialize())[0])
    assert "no published source artifact" in reason


def test_an_unconverged_target_carries_no_numbers(monkeypatch):
    _Patch(monkeypatch, _Adapter())
    product = _product(success=[False])
    prediction_product = NonlinearPrediction(
        orders=(3,),
        perturbation_parameter="mu",
        reconstruction=ScalarReconstructionConfig(grid_points=201),
    ).process(product, _context(product))
    assert str(np.asarray(prediction_product.handle("status").materialize())[0]) == (
        "not_converged"
    )
    assert np.isnan(
        np.asarray(prediction_product.handle("allan_intensity_soft").materialize())[0]
    )


def test_the_full_information_rate_is_unavailable_without_a_density(monkeypatch):
    _Patch(monkeypatch, _Adapter())
    product = _product()

    class Blank(_Model):
        @staticmethod
        def cam_diffusion(matrix, params=None):
            del params
            return np.full_like(np.asarray(matrix), np.nan, dtype=complex)

    prediction_product = NonlinearPrediction(
        orders=(3,),
        perturbation_parameter="mu",
        reconstruction=ScalarReconstructionConfig(grid_points=201),
    ).process(product, _context(product, model=Blank()))
    assert np.isnan(
        np.asarray(
            prediction_product.handle("allan_intensity_reconstructed").materialize()
        )[0]
    )
    assert np.isnan(
        np.asarray(prediction_product.handle("full_information_rate").materialize())[0]
    )
    # Every noise-dependent number is unavailable, while the response (which does
    # not need the covariance) is still reported.
    assert np.isnan(
        np.asarray(prediction_product.handle("allan_intensity_full").materialize())[0]
    )
    assert str(np.asarray(prediction_product.handle("status").materialize())[0]) == (
        "stochastic_invalid"
    )
    assert np.isfinite(
        np.asarray(
            prediction_product.handle("expected_frequency_response_full").materialize()
        )[0]
    )


def test_a_confining_prediction_reports_a_valid_domain(monkeypatch):
    _Patch(monkeypatch, _Adapter())
    product = _product()
    prediction_product = NonlinearPrediction(
        orders=(3,),
        perturbation_parameter="mu",
        reconstruction=ScalarReconstructionConfig(grid_points=401),
    ).process(product, _context(product, model=_BoundedModel()))
    # The one-mode truncation still is not confining, but the domain is bounded and
    # the numbers are reported with their status.
    validity = str(
        np.asarray(prediction_product.handle("validity_domain").materialize())[0]
    )
    assert "not_confining" in validity
    assert np.isfinite(
        np.asarray(
            prediction_product.handle("allan_intensity_reconstructed").materialize()
        )[0]
    )


def test_the_scalar_and_gaussian_terms_combine_as_documented():
    adapter = _Adapter()
    manifold = slow_manifold(adapter, [X0], {}, order=3)
    expected = scalar_reconstruction(
        adapter,
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=_confining_manifold(),
        config=ScalarReconstructionConfig(grid_points=401),
    )
    gaussian = ou_intensities(
        adapter, _observable(), state=[X0], params={}, model=_BoundedModel()
    )
    reconstructed = reconstructed_intensity(
        expected.scalar_intensity,
        full_intensity=gaussian.full_intensity,
        soft_intensity=gaussian.soft_intensity,
    )
    assert reconstructed == pytest.approx(
        expected.scalar_intensity + gaussian.full_intensity - gaussian.soft_intensity,
        rel=1e-12,
    )
    assert manifold.order == 3


def test_the_schema_declares_the_prediction_product():
    from qphase_cam.data import CAM_PREDICTION_PRODUCT
    from qphase_cam.manifest import MANIFEST

    declared = {product.name: product for product in MANIFEST.data_products}
    assert "prediction" in declared
    assert declared["prediction"].schema_ref.endswith("CAM_PREDICTION_PRODUCT")
    assert CAM_PREDICTION_PRODUCT.attributes["product"] == "prediction"
    assert "nonlinear_manifold" in CAM_PREDICTION_PRODUCT.attributes["approximations"]


def test_the_postprocessor_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        NonlinearPredictionConfig(perturbation_parameter="mu", unknown_option=1)
