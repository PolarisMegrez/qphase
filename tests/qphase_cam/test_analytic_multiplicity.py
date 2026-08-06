"""Tests for the Phase A analytic multiplicity prototype.

Covers the test matrix of reports/analytic_multiplicity_recovery_plan.md §4:
synthetic triple/quadruple roots, the D > q path (with the old shortcut
formula as a counterexample), degree-drop strata, positive-dimensional
systems, chart-singular vs regular classification, nonnegative domain
boundaries, real-model end-to-end cases (Kerr, cross-Kerr, VDP,
pair-hopping with a known numerical triple root), and the subprocess
runner's timeout / JSONL resume behaviour.
"""

from __future__ import annotations

import json
import sys
import time

import pytest
import sympy as sp

from qphase_cam.solver.analytic_multiplicity import (
    ALGEBRAIC_STATUSES,
    STATUS_CHART_SINGULAR,
    STATUS_DEGREE_DROP,
    STATUS_POSITIVE_DIMENSIONAL,
    STATUS_SOLVED,
    STATUS_TIMEOUT,
    MultiplicityCase,
    deduplicate_solutions,
    enumerate_scalar_charts,
    run_case,
    run_cases,
    solve_scalar_multiplicity,
)

x, c0, c1, c2, c3 = sp.symbols("x c0 c1 c2 c3", real=True)


def _solutions(result, level=0):
    return result["strata"][level]["solutions"]


def _filtered(result, level=0):
    return result["strata"][level]["filtered"]


# ---------------------------------------------------------------------------
# Synthetic polynomials: the solving core.
# ---------------------------------------------------------------------------


def test_cubic_triple_root():
    """p = x^3 + c1 x + c2 has a triple root only at c1 = c2 = 0, x = 0."""
    result = solve_scalar_multiplicity(x**3 + c1 * x + c2, x, 3, (c1, c2))
    assert result["status"] == STATUS_SOLVED
    solutions = _solutions(result)
    assert len(solutions) == 1
    assert solutions[0]["exact"] == {"x": "0", "c1": "0", "c2": "0"}
    assert solutions[0]["checks"]["exact_multiplicity"] is True


def test_quartic_quadruple_root():
    """p = x^4 + c1 x^2 + c2 x + c3 has a quadruple root only at the origin."""
    result = solve_scalar_multiplicity(
        x**4 + c1 * x**2 + c2 * x + c3, x, 4, (c1, c2, c3)
    )
    assert result["status"] == STATUS_SOLVED
    solutions = _solutions(result)
    assert len(solutions) == 1
    assert solutions[0]["exact"] == {"x": "0", "c1": "0", "c2": "0", "c3": "0"}


def test_quartic_triple_root_d_greater_than_q():
    """D=4, q=3: the general path finds both triple roots of the quartic family.

    p = x^4 - 6 x^2 + c1 x + c2 has triple roots exactly at
    (x, c1, c2) = (1, 8, -3) and (-1, -8, -3), where p = (x-1)^3 (x+3)
    and p = (x+1)^3 (x-3) respectively.
    """
    p = x**4 - 6 * x**2 + c1 * x + c2
    result = solve_scalar_multiplicity(p, x, 3, (c1, c2))
    assert result["status"] == STATUS_SOLVED
    found = {tuple(sorted(sol["exact"].items())) for sol in _solutions(result)}
    expected = {
        tuple(sorted({"x": "1", "c1": "8", "c2": "-3"}.items())),
        tuple(sorted({"x": "-1", "c1": "-8", "c2": "-3"}.items())),
    }
    assert found == expected

    # The old prototype's shortcut r0 = -b/(q*a) is only valid when
    # degree(p) == q.  Here a=1, b=0 give r0 = 0, and p''(0) = -12 != 0:
    # the shortcut candidate is not even a triple root.
    a, b = sp.Poly(p, x).all_coeffs()[:2]
    r0 = sp.simplify(-b / (3 * a))
    assert r0 == 0
    assert sp.expand(sp.diff(p, x, 2).subs(x, r0)) != 0
    assert all(sol["values"]["x"] != 0.0 for sol in _solutions(result))


def test_resultant_path_quintic_two_controls():
    """D=5 > q=3 with two controls: the resultant elimination path solves it.

    p = (x-1)^3 (x^2+x+1) + (c1-1) x^4 + (c2-1) x has an exact triple root
    at (x, c1, c2) = (1, 1, 1); the deformed coefficients make the
    Groebner-based route unnecessary -- resultants plus quotient-field gcds
    must recover the point with an exact certificate.
    """
    p = sp.expand((x - 1) ** 3 * (x**2 + x + 1)) + (c1 - 1) * x**4 + (c2 - 1) * x
    result = solve_scalar_multiplicity(p, x, 3, (c1, c2))
    assert result["status"] == STATUS_SOLVED
    stratum = result["strata"][0]
    assert stratum["method"] == "resultant_elimination"
    found = {tuple(sorted(sol["exact"].items())) for sol in stratum["solutions"]}
    assert tuple(sorted({"x": "1", "c1": "1", "c2": "1"}.items())) in found
    for sol in stratum["solutions"]:
        assert sol["checks"]["exact_multiplicity"] is True


def test_resultant_fallback_on_structural_common_factor():
    """p with a built-in triple factor: resultants vanish -> honest fallback.

    p = (x-(c1+c2))^3 (x-c1)(x-c2) has a triple root at x = c1+c2 for every
    generic (c1, c2), so the solution set is two-dimensional.  The resultant
    path must detect the structural common factor and fall back; the
    saturated dimension decision then reports positive_dimensional.
    """
    s = c1 + c2
    p = sp.expand((x - s) ** 3 * (x - c1) * (x - c2))
    result = solve_scalar_multiplicity(p, x, 3, (c1, c2))
    assert result["status"] == STATUS_POSITIVE_DIMENSIONAL
    stratum = result["strata"][0]
    assert "resultant_fallback" in stratum
    assert "vanishes identically" in stratum["resultant_fallback"]


def test_resultant_path_on_degree_drop_stratum():
    """The resultant path also solves deeper strata (drop conditions certified).

    p = c0 x^5 + (x^4 - 6 x^2 + c1 x - 3): the c0 = 0 stratum has effective
    degree 4 > q = 3 and must recover the quartic's two triple roots with the
    drop condition c0 = 0 pinned rationally and certified per family.
    """
    p = c0 * x**5 + (x**4 - 6 * x**2 + c1 * x - 3)
    result = solve_scalar_multiplicity(p, x, 3, (c0, c1))
    assert result["status"] == STATUS_SOLVED
    level1 = result["strata"][1]
    assert level1["method"] == "resultant_elimination"
    assert level1["status"] == STATUS_SOLVED
    found = {tuple(sorted(sol["exact"].items())) for sol in level1["solutions"]}
    assert tuple(sorted({"x": "1", "c0": "0", "c1": "8"}.items())) in found
    assert tuple(sorted({"x": "-1", "c0": "0", "c1": "-8"}.items())) in found


def test_degree_drop_stratum_is_solved_not_excluded():
    """p = c0 x^4 + (x-1)^3 + c1 x: the c0 = 0 stratum is solved explicitly."""
    p = c0 * x**4 + (x - 1) ** 3 + c1 * x
    result = solve_scalar_multiplicity(p, x, 3, (c0, c1))
    assert result["status"] == STATUS_SOLVED
    level1 = result["strata"][1]
    assert level1["level"] == 1
    assert level1["effective_degree"] == 3
    assert level1["status"] == STATUS_SOLVED
    found = {tuple(sorted(sol["exact"].items())) for sol in level1["solutions"]}
    assert tuple(sorted({"x": "1", "c0": "0", "c1": "0"}.items())) in found


def test_degree_drop_status_when_fixed_values_kill_leading_coefficient():
    """A substitution that drops the nominal degree is reported, not hidden."""
    result = solve_scalar_multiplicity(
        x**3 + c1 * x + c2, x, 3, (c1, c2), nominal_degree=4
    )
    assert result["status"] == STATUS_DEGREE_DROP
    assert result["strata"][0]["status"] == "empty_stratum"
    level1 = result["strata"][1]
    assert level1["status"] == STATUS_SOLVED
    assert [sol["exact"] for sol in level1["solutions"]] == [
        {"x": "0", "c1": "0", "c2": "0"}
    ]


def test_positive_dimensional_system():
    """(x + c1/2)^2 has a double root for every c1: a positive-dim set."""
    result = solve_scalar_multiplicity(x**2 + c1 * x + c1**2 / 4, x, 2, (c1,))
    assert result["status"] == STATUS_POSITIVE_DIMENSIONAL
    assert _solutions(result) == []


def test_chart_singular_solution_is_classified_not_dropped():
    """A solution on the chart denominator is chart_singular; on another
    chart (different denominator) the same solution is regular."""
    p = x**3 + c1 * x + c2
    singular = solve_scalar_multiplicity(p, x, 3, (c1, c2), denominators=(x,))
    assert singular["status"] == STATUS_CHART_SINGULAR
    solution = _solutions(singular)[0]
    assert solution["classification"] == "chart_singular"
    assert solution["vanishing_chart_factors"] == ["x"]
    assert solution["checks"]["chart_factors_nonzero"] is False

    regular = solve_scalar_multiplicity(p, x, 3, (c1, c2), denominators=(x - 5,))
    assert regular["status"] == STATUS_SOLVED
    assert _solutions(regular)[0]["classification"] == "regular"


def test_nonnegative_domain_allows_zero_boundary():
    """nonnegative parameters may be exactly zero (the old script excluded them)."""
    result = solve_scalar_multiplicity(
        x**3 + c1 * x + c2,
        x,
        3,
        (c1, c2),
        parameter_domains={"c1": "nonnegative", "c2": "nonnegative"},
    )
    assert result["status"] == STATUS_SOLVED
    assert len(_solutions(result)) == 1
    assert _solutions(result)[0]["classification"] == "regular"


def test_nonnegative_domain_filters_negative_controls():
    """p = x^3 + c1 x + 2 has a double root at c1 = -3, x = 1."""
    p = x**3 + c1 * x + 2
    filtered = solve_scalar_multiplicity(
        p, x, 2, (c1,), parameter_domains={"c1": "nonnegative"}
    )
    assert filtered["status"] == STATUS_SOLVED
    assert _solutions(filtered) == []
    reasons = {entry["reason"] for entry in _filtered(filtered)}
    assert "domain_violation" in reasons

    kept = solve_scalar_multiplicity(p, x, 2, (c1,), parameter_domains={"c1": "real"})
    assert [sol["exact"]["c1"] for sol in _solutions(kept)] == ["-3"]


def test_case_validation():
    with pytest.raises(ValueError, match="controls"):
        MultiplicityCase("m:M", "chart", 3, ("a",), {"b": "0"})
    with pytest.raises(ValueError, match="also fixed"):
        MultiplicityCase("m:M", "chart", 2, ("a",), {"a": "0"})


def test_deduplicate_solutions_keeps_all_provenance():
    def record(case_key, chart_id, stratum=0):
        return {
            "record_type": "case",
            "case_key": case_key,
            "case": {"model": "m:M", "chart_id": chart_id},
            "strata": [
                {
                    "level": stratum,
                    "solutions": [
                        {
                            "classification": "regular",
                            "state": [1.0, 2.0, 0.5, -0.25],
                            "values": {"x": 1.0, "gamma_a": 0.5},
                        },
                        {"classification": "chart_singular", "values": {"x": 0.0}},
                    ],
                }
            ],
        }

    groups = deduplicate_solutions([record("k1", "chart:a"), record("k2", "chart:b")])
    assert len(groups) == 1
    provenance = groups[0]["provenance"]
    assert {p["case_key"] for p in provenance} == {"k1", "k2"}
    assert {p["chart_id"] for p in provenance} == {"chart:a", "chart:b"}


# ---------------------------------------------------------------------------
# Real models (end-to-end, in-process).
# ---------------------------------------------------------------------------


def _print_elapsed(label, started, record):
    elapsed = time.perf_counter() - started
    print(f"\n[timing] {label}: {elapsed:.1f}s status={record['algebraic_status']}")


def test_enumerate_scalar_charts_kerr():
    from qphase_cam.core.fpgen import FPGenDynamicsAdapter

    from models.kerr_2mode import Kerr2ModeModel

    adapter = FPGenDynamicsAdapter.from_model(
        Kerr2ModeModel(**{name: 1.0 for name in Kerr2ModeModel.config_schema.model_fields})
    )
    coverage = enumerate_scalar_charts(adapter)
    chart_ids = {chart["chart_id"] for chart in coverage["scalar_charts"]}
    # All equation partitions are kept and addressed by chart_id.
    assert "ret:r_diag_0|eq:0" in chart_ids
    assert len(chart_ids) == len(coverage["scalar_charts"]) >= 2
    for chart in coverage["scalar_charts"]:
        assert chart["materialized"] is True
        assert chart["numerator_degrees"] == [3]
        assert chart["denominators"]
    assert coverage["rejected_partitions"]  # non-affine blocks are recorded


def test_kerr_2mode_triple_root_end_to_end():
    started = time.perf_counter()
    case = MultiplicityCase(
        "models.kerr_2mode:Kerr2ModeModel",
        "ret:r_diag_0|eq:0",
        3,
        ("gamma_a", "gamma_b"),
        {"omega_a": "0", "omega_b": "0", "chi": "1", "g": "1"},
    )
    record = run_case(case)
    _print_elapsed("kerr_2mode D=3 q=3", started, record)
    assert record["algebraic_status"] in ALGEBRAIC_STATUSES - {STATUS_TIMEOUT}
    assert record["strata"]
    for stratum in record["strata"]:
        for solution in stratum["solutions"]:
            assert solution["checks"]["exact_multiplicity"] is True
            if solution["classification"] == "regular":
                assert "min_state_eigenvalue" in solution
                assert "physical" in solution
                assert "full_residual" in solution


def test_crosskerr_2mode_triple_root_end_to_end():
    started = time.perf_counter()
    case = MultiplicityCase(
        "models.crosskerr_2mode:CrossKerr2ModeModel",
        "ret:r_re_0_1|eq:2",
        3,
        ("gamma_a", "gamma_b"),
        {"omega_a": "0", "omega_b": "0", "chi": "1", "g": "1"},
    )
    record = run_case(case)
    _print_elapsed("crosskerr_2mode D=3 q=3", started, record)
    assert record["algebraic_status"] in ALGEBRAIC_STATUSES - {STATUS_TIMEOUT}
    assert record["strata"]


def test_vdp_2mode_quadruple_root_end_to_end():
    started = time.perf_counter()
    case = MultiplicityCase(
        "models.vdp_2mode:VDP2ModeModel",
        "ret:r_diag_0|eq:0",
        4,
        ("gamma_a", "gamma_b", "omega_b"),
        {"omega_a": "0", "Gamma": "1", "g": "1"},
    )
    record = run_case(case)
    _print_elapsed("vdp_2mode D=4 q=4", started, record)
    assert record["algebraic_status"] in ALGEBRAIC_STATUSES - {STATUS_TIMEOUT}
    assert record["strata"]


def test_pair_hopping_2mode_recovers_known_triple_root():
    """Cross-validation against the numerical (3,1,0) candidate of
    reports/cam_bifurcation_model_assessment.md §4:
    omega_a=0, g=1, k=1e-3, omega_b=0.1 fixed, controls (gamma_a, gamma_b),
    gamma_a ≈ 1.29412527, gamma_b ≈ 1.54903018."""
    started = time.perf_counter()
    case = MultiplicityCase(
        "models.pair_hopping_2mode:PairHopping2ModeModel",
        "ret:r_re_0_1|eq:2",
        3,
        ("gamma_a", "gamma_b"),
        {"omega_a": "0", "omega_b": "1/10", "g": "1", "k": "1/1000"},
    )
    record = run_case(case)
    _print_elapsed("pair_hopping_2mode D=5 q=3", started, record)
    assert record["algebraic_status"] == STATUS_SOLVED
    regular = [
        solution
        for stratum in record["strata"]
        for solution in stratum["solutions"]
        if solution["classification"] == "regular"
    ]

    def matches(solution):
        return (
            abs(solution["values"]["gamma_a"] - 1.29412527) < 1e-6
            and abs(solution["values"]["gamma_b"] - 1.54903018) < 1e-6
        )

    assert any(matches(solution) for solution in regular), json.dumps(
        [solution["values"] for solution in regular], indent=1
    )
    for solution in regular:
        assert solution["checks"]["exact_multiplicity"] is True
        assert "physical" in solution and "full_residual" in solution


# ---------------------------------------------------------------------------
# Subprocess runner: isolation, timeout, JSONL resume.
# ---------------------------------------------------------------------------

KERR_CASE = MultiplicityCase(
    "models.kerr_2mode:Kerr2ModeModel",
    "ret:r_diag_0|eq:0",
    3,
    ("gamma_a", "gamma_b"),
    {"omega_a": "0", "omega_b": "0", "chi": "1", "g": "1"},
)

CANNED_COMMAND = [
    sys.executable,
    "-c",
    "import json; print(json.dumps({'algebraic_status': 'solved_zero_dimensional'}))",
]


def test_runner_jsonl_resume_skips_completed_cases(tmp_path):
    out = tmp_path / "results.jsonl"
    cases = [
        KERR_CASE,
        MultiplicityCase(
            "models.crosskerr_2mode:CrossKerr2ModeModel",
            "ret:r_re_0_1|eq:2",
            3,
            ("gamma_a", "gamma_b"),
            {"omega_a": "0", "omega_b": "0", "chi": "1", "g": "1"},
        ),
    ]
    records = run_cases(cases, out, command=CANNED_COMMAND)
    assert len(records) == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["case_key"] for line in lines} == {
        case.key() for case in cases
    }

    # Interrupt recovery: a rerun skips every recorded case key.
    assert run_cases(cases, out, command=CANNED_COMMAND) == []
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2

    # A partially written file (first case only) resumes with the second.
    partial = tmp_path / "partial.jsonl"
    partial.write_text(lines[0] + "\n", encoding="utf-8")
    resumed = run_cases(cases, partial, command=CANNED_COMMAND)
    assert [record["case_key"] for record in resumed] == [cases[1].key()]


def test_runner_records_subprocess_timeout(tmp_path):
    out = tmp_path / "results.jsonl"
    sleep_command = [sys.executable, "-c", "import time; time.sleep(30)"]
    started = time.perf_counter()
    records = run_cases([KERR_CASE], out, timeout_seconds=1.0, command=sleep_command)
    assert time.perf_counter() - started < 15
    assert [record["algebraic_status"] for record in records] == [STATUS_TIMEOUT]
    # The timeout record is part of the resumable state.
    assert run_cases([KERR_CASE], out, timeout_seconds=1.0, command=sleep_command) == []


def test_runner_records_solver_error_on_bad_exit(tmp_path):
    out = tmp_path / "results.jsonl"
    failing = [sys.executable, "-c", "import sys; sys.exit(3)"]
    records = run_cases([KERR_CASE], out, command=failing)
    assert [record["algebraic_status"] for record in records] == ["solver_error"]


def test_runner_real_subprocess_module_entry(tmp_path):
    """Full isolation path: python -m qphase_cam.solver.analytic_multiplicity."""
    out = tmp_path / "results.jsonl"
    started = time.perf_counter()
    records = run_cases([KERR_CASE], out, timeout_seconds=300.0)
    _print_elapsed("kerr_2mode D=3 q=3 (subprocess)", started, records[0])
    assert records[0]["algebraic_status"] in ALGEBRAIC_STATUSES - {STATUS_TIMEOUT}
    assert records[0]["strata"]
    # The recorded case key matches the library key function.
    assert records[0]["case_key"] == KERR_CASE.key()
