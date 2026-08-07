"""Executable contract between qphase_cam and fpgen's public API."""

from __future__ import annotations

import inspect

import fpgen
import numpy as np
from qphase_cam.core.fpgen import FPGenDynamicsAdapter, validate_fpgen_runtime

from models.vdp_2mode import VDP2ModeModel

REQUIRED_EXPORTS = {
    "CovarianceDynamics",
    "MomentDynamicsSpec",
    "CompiledMomentDynamics",
    "LinearReductionCandidate",
    "ReductionSearchResult",
    "LinearReductionPlan",
    "MaterializedReduction",
    "MaterializationFailure",
    "RejectedPartition",
}


def _parameter_names(callable_object) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def test_fpgen_public_api_versions_and_signatures():
    validate_fpgen_runtime()
    assert REQUIRED_EXPORTS <= set(fpgen.__all__)
    assert _parameter_names(fpgen.CovarianceDynamics.search_linear_reductions) == (
        "self",
        "retained_dimension",
        "retained_ids",
        "equation_partitions",
        "partition_limit",
        "materialization_limit",
        "return_limit",
    )
    assert _parameter_names(fpgen.CovarianceDynamics.linear_reduce) == (
        "self",
        "candidate",
        "order_parameters",
    )
    assert _parameter_names(fpgen.MomentDynamicsSpec.compile_numpy) == ("self",)
    assert _parameter_names(fpgen.ReductionSearchResult.manifest) == ("self",)
    assert _parameter_names(fpgen.LinearReductionPlan.materialize) == (
        "self",
        "method",
    )


def test_fpgen_adapter_numerical_and_reduction_contract():
    model = VDP2ModeModel(
        omega_a=0.1,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.8,
        Gamma=0.0001,
        g=0.5,
    )
    adapter = FPGenDynamicsAdapter.from_model(model)
    assert not hasattr(adapter, "dynamics")
    assert adapter.state_ids == (
        "r_diag_0",
        "r_diag_1",
        "r_re_0_1",
        "r_im_0_1",
    )
    assert adapter.parameter_names == tuple(model.params)

    state = np.asarray([2.0, 1.0, 0.2, -0.1])
    assert adapter.rhs(state).shape == (4,)
    assert adapter.jacobian(state).shape == (4, 4)
    assert adapter.parameter_jacobian(state).shape == (4, len(model.params))
    assert adapter.state_matrix(state).shape == (2, 2)

    search = adapter.search_linear_reductions(
        retained_dimension=1,
        equation_partitions="all",
        return_limit=1,
    )
    manifest = search.manifest()
    assert manifest["api_version"] == "1.1"
    assert manifest["regular_branches_only"] is True
    assert manifest["candidate_count"] >= len(search.candidates) == 1
    assert "coverage" in manifest
    assert "truncation_reasons" in manifest
    assert len(search.rejected_partitions) == manifest["rejected_partition_count"]
    assert {entry.reason for entry in search.rejected_partitions} <= {
        "non_affine_eliminated_block",
        "structurally_rank_deficient",
    }
    assert manifest["materialization_skipped_oversized"] >= 0
    assert manifest["materialization_failure_count"] == len(
        search.materialization_failures
    )

    candidate = search.candidates[0]
    assert candidate.chart_id == (
        f"ret:{','.join(candidate.retained_ids)}"
        f"|eq:{','.join(map(str, candidate.retained_equations))}"
    )

    plan = adapter.linear_reduction(candidate=search.candidates[0])
    materialized = adapter.materialized_linear_reduction(candidate=search.candidates[0])
    assert len(plan.retained_symbols) == 1
    assert materialized.reduced_residual.rows == 1
