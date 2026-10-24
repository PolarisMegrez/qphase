"""Solver-owned identities: content, stability and the two identity layers.

These tests use one real ``TwoModeFold`` auto run (module-scoped, because it is
the slowest part) plus the pure identity helpers. The scientific expectations of
Phase 10A are asserted elsewhere and are not repeated here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.engine import Engine
from qphase_cam.solver.bifurcation import ControlRange
from qphase_cam.solver.identity import (
    CANDIDATE_IDENTITY_BASIS,
    IDENTITY_SCHEME,
    candidate_content,
    candidate_identity,
    case_identity,
    digest_payload,
    search_identity,
    search_scope,
    start_identity,
)

from .test_bifurcation import TwoModeFoldModel, _solver


class _Control:
    """Minimal control spec for the pure payload helpers."""

    def __init__(self, low: float, high: float, sampling: str = "linear") -> None:
        self.min = low
        self.max = high
        self.sampling = sampling


@pytest.fixture(scope="module")
def auto_run():
    """One auto-strategy TwoModeFold solve (accepted 2, deduplicated to 1)."""
    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    return output


def _scope(model=None, *, strategy="auto", low=-1.0, high=0.0):
    model = model or TwoModeFoldModel()
    return search_scope(
        model,
        target="equilibrium_multiplicity",
        order=2,
        controls={"gamma": _Control(low, high)},
        perturbation={"parameter": "gamma", "scale": 1.0, "side": "both"},
        strategy=strategy,
    )


# --- digests ---------------------------------------------------------------


def test_digest_is_deterministic_and_scheme_tagged():
    payload = {"scheme": IDENTITY_SCHEME, "value": [1, 2.5, "x"]}
    assert digest_payload(payload) == digest_payload(
        dict(reversed(list(payload.items())))
    )
    assert len(digest_payload(payload)) == 16
    assert digest_payload(payload) != digest_payload({**payload, "value": [1, 2.5]})
    # nan/inf never reach canonical JSON as-is, so hashing cannot fail on them.
    assert digest_payload({"value": float("nan")})


def test_search_identity_ignores_the_swept_control_values():
    """Every point of one campaign shares the search identity."""
    scope = _scope()
    assert search_identity(scope) == search_identity(_scope())
    assert search_identity(scope).startswith("cam:search:")
    # The swept parameter is removed from the search scope on purpose.
    assert "gamma" not in scope["params"]
    assert "gamma" in scope["controls"]


def test_search_identity_tracks_the_design():
    base = search_identity(_scope())
    assert base != search_identity(_scope(low=-2.0))
    assert base != search_identity(_scope(strategy="full"))
    other = TwoModeFoldModel()
    other.params = {**other.params, "Gamma": 2.0}
    assert base != search_identity(_scope(other))


def test_case_identity_tracks_the_control_values():
    search = search_identity(_scope())
    left = case_identity(search, {"gamma": -1.0})
    right = case_identity(search, {"gamma": -0.5})
    assert left == case_identity(search, {"gamma": -1.0})
    assert left != right
    assert left.startswith("cam:case:")
    # A different search gives a different case even with the same control value.
    assert left != case_identity(search_identity(_scope(low=-2.0)), {"gamma": -1.0})


def test_candidate_identity_is_search_and_case_free():
    content, basis = candidate_content(
        model_name="two_mode_fold",
        target="equilibrium_multiplicity",
        order=2,
        state_vector=[0.5, 0.25, 0.0, 0.0],
        controls={"gamma": -0.5},
    )
    assert basis == "state_vector_and_controls"
    identity = candidate_identity(content)
    assert identity == candidate_identity(dict(content))
    assert identity.startswith("cam:candidate:")
    # Nothing in the payload refers to a case, a search or a slot.
    assert "case" not in content
    assert "search" not in content
    assert set(CANDIDATE_IDENTITY_BASIS) == {
        "verified_full_state_decimal_values",
        "state_vector_and_controls",
    }


def test_candidate_identity_prefers_verified_decimals():
    common = {
        "model_name": "two_mode_fold",
        "target": "equilibrium_multiplicity",
        "order": 2,
        "state_vector": [0.5, 0.25, 0.0, 0.0],
        "controls": {"gamma": -0.5},
    }
    decimal_content, decimal_basis = candidate_content(
        **common, verified_full_state_decimal_values=("0.500000", "0.250000")
    )
    vector_content, vector_basis = candidate_content(**common)
    assert decimal_basis == "verified_full_state_decimal_values"
    assert vector_basis == "state_vector_and_controls"
    assert candidate_identity(decimal_content) != candidate_identity(vector_content)


def test_start_identity_separates_case_and_method():
    content, _ = candidate_content(
        model_name="two_mode_fold",
        target="equilibrium_multiplicity",
        order=2,
        state_vector=[0.5, 0.25, 0.0, 0.0],
        controls={"gamma": -0.5},
    )
    base = start_identity(case="cam:case:a", method="bordered_full", content=content)
    assert base == start_identity(
        case="cam:case:a", method="bordered_full", content=content
    )
    assert base.startswith("cam:start:")
    assert base != start_identity(
        case="cam:case:b", method="bordered_full", content=content
    )
    assert base != start_identity(
        case="cam:case:a", method="reduced_fraction_free", content=content
    )


def test_identity_is_independent_of_the_python_hash_seed():
    """No ``hash()`` enters an identity, even across interpreter runs."""
    script = """
from qphase_cam.solver.identity import (
    candidate_content, candidate_identity, case_identity, search_identity, search_scope,
)


class C:
    def __init__(self, low, high, sampling="linear"):
        self.min = low
        self.max = high
        self.sampling = sampling


class M:
    name = "two_mode_fold"
    params = {"gamma": -1.0, "Gamma": 1.0, "kappa": 1.0}


scope = search_scope(
    M(),
    target="equilibrium_multiplicity",
    order=2,
    controls={"gamma": C(-1.0, 0.0)},
    perturbation={"parameter": "gamma", "scale": 1.0, "side": "both"},
    strategy="auto",
)
content, basis = candidate_content(
    model_name="two_mode_fold",
    target="equilibrium_multiplicity",
    order=2,
    state_vector=[0.5, 0.25, 0.0, 0.0],
    controls={"gamma": -0.5},
)
print(search_identity(scope))
print(case_identity(search_identity(scope), {"gamma": -1.0}))
print(candidate_identity(content))
print(basis)
"""
    outputs = []
    for seed in ("0", "1"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "", "SYSTEMROOT": ""},
            cwd=Path(__file__).resolve().parents[2],
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    assert outputs[0].splitlines()[0].startswith("cam:search:")


# --- the solver's two identity layers -------------------------------------


def test_solver_emits_both_identity_layers(auto_run):
    totals = auto_run.metadata["audit"]["totals"]
    assert totals["accepted_count"] == 2
    assert totals["unique_candidate_count"] == 1
    starts = auto_run.metadata["accepted_start_identities"]
    finals = auto_run.metadata["accepted_candidate_identities"]
    assert len(starts) == totals["accepted_count"] == 2
    assert len(finals) == totals["unique_candidate_count"] == 1
    assert len(set(starts)) == 2
    assert all(value.startswith("cam:start:") for value in starts)
    assert all(value.startswith("cam:candidate:") for value in finals)
    assert auto_run.metadata["case_identity"].startswith("cam:case:")
    assert auto_run.metadata["search_identity"].startswith("cam:search:")


def test_solver_attaches_starts_to_the_surviving_candidate(auto_run):
    candidates = auto_run.candidates
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (
        candidate.metadata["candidate_identity"]
        in (auto_run.metadata["accepted_candidate_identities"])
    )
    assert candidate.metadata["identity_basis"] in CANDIDATE_IDENTITY_BASIS
    # Both accepted starts are kept on the survivor: dedup loses no evidence.
    assert (
        candidate.metadata["accepted_start_identities"]
        == (auto_run.metadata["accepted_start_identities"])
    )
    assert candidate.metadata["discovery_methods"] == (
        "reduced_fraction_free",
        "bordered_full",
    )


def test_identities_survive_packing_into_the_result(auto_run):
    result = Engine._pack_bifurcation(auto_run, TwoModeFoldModel())
    assert result.meta["search_identity"] == auto_run.metadata["search_identity"]
    assert result.meta["case_identity"] == auto_run.metadata["case_identity"]
    assert (
        result.meta["accepted_start_identities"]
        == (auto_run.metadata["accepted_start_identities"])
    )
    assert list(result.diagnostics["candidate_identity"]) == list(
        auto_run.metadata["accepted_candidate_identities"]
    )
    assert list(result.diagnostics["identity_basis"]) == [
        auto_run.candidates[0].metadata["identity_basis"]
    ]
    # Both start identities are visible per candidate in the diagnostics.
    starts = np.asarray(result.diagnostics["accepted_start_identities"])
    assert starts.shape == (1, 2)


def test_identities_are_reproducible_across_solves(auto_run):
    model = TwoModeFoldModel()
    repeat = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    for key in (
        "search_identity",
        "case_identity",
        "accepted_start_identities",
        "accepted_candidate_identities",
    ):
        assert repeat.metadata[key] == auto_run.metadata[key], key
    assert (
        repeat.candidates[0].metadata["candidate_identity"]
        == (auto_run.candidates[0].metadata["candidate_identity"])
    )


def test_a_different_case_reuses_the_candidate_identity(auto_run):
    """The candidate identity carries no case scope, the start identities do."""
    other_model = TwoModeFoldModel()
    other_model.params = {**other_model.params, "gamma": -0.5}
    other = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        other_model, NumpyBackend()
    )
    assert other.metadata["search_identity"] == auto_run.metadata["search_identity"]
    assert other.metadata["case_identity"] != auto_run.metadata["case_identity"]
    # Same physics found from another case: same candidate identity.
    assert (
        other.metadata["accepted_candidate_identities"]
        == (auto_run.metadata["accepted_candidate_identities"])
    )
    # Accepted starts belong to their case, so the two sets never collide.
    assert set(other.metadata["accepted_start_identities"]).isdisjoint(
        auto_run.metadata["accepted_start_identities"]
    )
