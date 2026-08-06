"""Exact algebraic multiplicity solver for CAM scalar reductions (Phase A).

Research prototype for the analytic multiplicity recovery plan
(``reports/analytic_multiplicity_recovery_plan.md`` §4 Phase A).  This is a
pure library module: no plugin registration, no parameter sweeps, no campaign
runner.

Problem definition (plan §3): for a regular scalar chart with reduced
polynomial ``p(x; c, lambda)`` and target multiplicity ``q``, keep the order
parameter ``x`` unknown and solve the simultaneous system

    p = d_x p = ... = d_x^(q-1) p = 0

in the ``q`` unknowns ``(x, c_1, ..., c_{q-1})`` where the ``c_i`` are control
parameters and every other parameter is fixed to an explicit rational value.
A solution is an exact multiplicity-``q`` point only when ``d_x^q p != 0``
there (checked exactly, never via floating-point residuals).

Strata (plan §3): the regular fixed-degree stratum requires the chart
denominators and the leading coefficient to be nonzero.  When the leading
coefficient vanishes the problem is rebuilt explicitly on the degree-drop
stratum (drop the leading term, add the vanishing condition, resolve); it is
never silently excluded.  Solutions where the chart denominators or cleared
factors vanish are classified ``chart_singular`` instead of being dropped.

Solving: the derivative system in ``x`` is first reduced by pseudo-remainder
(prem) chains, which for ``degree(p) == q`` degenerates to the proven
elimination ``x = -b/(q*a)`` and for ``degree(p) > q`` shrinks the
``x``-degrees before the Groebner dimension check.  Pseudo-remainder
reduction never loses common zeros of the original system, and every
candidate is verified exactly against the original derivative equations, so
extraneous factors introduced by leading-coefficient powers are filtered
honestly.  If the reduced system is positive-dimensional the dimension is
re-decided on the original system saturated with the stratum leading
coefficient (Rabinowitsch trick), keeping ``positive_dimensional`` strictly
distinct from solver failures.

For ``degree(p) > q`` with exactly two controls (``q == 3``) a resultant path
replaces the Groebner step, which is prohibitively slow on the prem-reduced
3-variable system (pair-hopping exceeds 600 s).  The elimination ideal in the
controls is generated through
``r_i = resultant_x(p'', p^(i))`` (``i = 0, 1``), stripped of the
leading-coefficient factors of ``p''`` (the degree-drop boundary, covered by
deeper strata).  The square control system ``{r_0, r_1}`` is solved by
factoring the univariate elimination ``U(c_1) = resultant_{c_2}(r_0, r_1)``
over ``QQ`` and, per irreducible factor ``h``, computing
``gcd(r_0, r_1)`` over the quotient field ``QQ[c_1]/(h)`` with exact formal
arithmetic (:func:`_qf_gcd`), then recovering ``x`` by the same quotient gcd
of ``{p, p', p''}``.  All acceptance checks (derivative equations, stratum
degree-drop conditions, stratum leading coefficient, ``d_x^q p != 0``, chart
regularity factors) are certified once per factor by formal reduction modulo
``h`` — sound for every root of ``h`` because ``QQ[c_1]/(h)`` is a field —
while reality and parameter-domain checks are per root via exact ``CRootOf``
properties.
Factors of ``U`` above ``MAX_EXACT_FACTOR_DEGREE`` are not processed exactly;
they are screened numerically at one root (valid for all conjugate roots,
since the gcd degree is constant across embeddings of an irreducible factor)
and recorded under ``skipped_factors`` with the screen verdict, so the
stratum is reported with ``completeness == "partial"`` instead of a silent
loss.  Degenerate inputs (vanishing resultant, common control component)
fall back to the prem-chain + saturated Groebner path.

Isolation: :func:`run_cases` executes each case in a fresh subprocess with a
wall-clock timeout and appends one JSON line per finished case, so an
interrupted run resumes by skipping the case keys already present in the
JSONL file.  Windows-safe: nothing tries to interrupt SymPy from a thread.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

#: JSONL record schema marker.
SCHEMA = "cam_analytic_multiplicity/1"

#: Algebraic status values (plan §3).  Never collapse these into a None.
STATUS_SOLVED = "solved_zero_dimensional"
STATUS_POSITIVE_DIMENSIONAL = "positive_dimensional"
STATUS_DEGREE_DROP = "degree_drop"
STATUS_CHART_SINGULAR = "chart_singular"
STATUS_TIMEOUT = "timeout"
STATUS_UNSUPPORTED = "unsupported"
STATUS_SOLVER_ERROR = "solver_error"

ALGEBRAIC_STATUSES = frozenset(
    {
        STATUS_SOLVED,
        STATUS_POSITIVE_DIMENSIONAL,
        STATUS_DEGREE_DROP,
        STATUS_CHART_SINGULAR,
        STATUS_TIMEOUT,
        STATUS_UNSUPPORTED,
        STATUS_SOLVER_ERROR,
    }
)

#: Per-solution classification values.
CLASSIFICATION_REGULAR = "regular"
CLASSIFICATION_CHART_SINGULAR = "chart_singular"

#: Default wall-clock timeout per case subprocess, in seconds.
DEFAULT_TIMEOUT_SECONDS = 300.0


class ExactCheckError(RuntimeError):
    """An exact (symbolic) zero/reality check could not be decided."""


def _is_exactly_zero(expr: Any) -> bool:
    """Decide ``expr == 0`` exactly; raise :class:`ExactCheckError` if not decidable."""
    try:
        expanded = sp.expand(expr)
        if expanded == 0:
            return True
        return bool(sp.simplify(expanded) == 0)
    except ExactCheckError:
        raise
    except Exception as exc:  # noqa: BLE001 - any SymPy failure means "undecided"
        raise ExactCheckError(f"exact zero check failed for {expr!r}: {exc}") from exc


def _real_value(value: Any) -> tuple[float, str] | None:
    """Return ``(float_value, method)`` if ``value`` is real, else ``None``.

    ``method`` is ``"exact"`` when reality is proven symbolically and
    ``"numeric"`` when it falls back to a high-precision imaginary-part
    check (recorded honestly in the solution record).  Expressions containing
    ``RootOf`` atoms skip the ``simplify`` attempt: SymPy's simplifier can
    grind for minutes on large algebraic-number expressions, while the
    high-precision numeric check decides them immediately.
    """
    if value.free_symbols:
        raise ExactCheckError(f"undetermined value with free symbols: {value!r}")
    candidate = value
    if candidate.is_real is None and not candidate.atoms(sp.RootOf):
        simplified = sp.simplify(candidate)
        if simplified is not None:
            candidate = simplified
    if candidate.is_real:
        return float(sp.N(candidate, 40)), "exact"
    if candidate.is_real is False:
        return None
    numeric = complex(sp.N(candidate, 60))
    if abs(numeric.imag) <= 1e-45 * max(1.0, abs(numeric.real)):
        return numeric.real, "numeric"
    return None


def _is_negative(value: Any) -> bool:
    """Decide whether an exactly-real value is strictly negative.

    Same ``RootOf`` guard as :func:`_real_value`: for algebraic-number
    expressions the sign is decided by 60-digit evaluation instead of
    ``simplify`` (reality of the value is established before this is called).
    """
    if value.is_negative is not None:
        return bool(value.is_negative)
    if value.is_nonnegative:
        return False
    if value.atoms(sp.RootOf):
        return complex(sp.N(value, 60)).real < 0.0
    simplified = sp.simplify(value)
    if simplified.is_negative is not None:
        return bool(simplified.is_negative)
    if simplified.is_nonnegative:
        return False
    return float(sp.N(simplified, 60)) < 0.0


@dataclass(frozen=True)
class MultiplicityCase:
    """One analytic multiplicity case.

    ``model`` is an import path (``"pkg.module:ClassName"`` or
    ``"pkg.module.ClassName"``) resolvable inside a fresh subprocess.
    ``controls`` are the ``order - 1`` control parameter names; every other
    model parameter must appear in ``fixed_parameters`` with an explicit
    rational value (given as a string such as ``"1/1000"`` or ``"0"``).
    Gauge choices belong in ``fixed_parameters``; nothing is fixed implicitly.
    """

    model: str
    chart_id: str
    order: int
    controls: tuple[str, ...]
    fixed_parameters: dict[str, str]

    def __post_init__(self) -> None:
        """Normalize containers and validate the case definition."""
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(
            self,
            "fixed_parameters",
            {k: str(v) for k, v in self.fixed_parameters.items()},
        )
        if self.order < 2:
            raise ValueError("order must be >= 2")
        if len(self.controls) != self.order - 1:
            raise ValueError(
                f"need exactly order-1={self.order - 1} controls, got {self.controls}"
            )
        overlap = set(self.controls) & set(self.fixed_parameters)
        if overlap:
            raise ValueError(f"controls also fixed: {sorted(overlap)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "chart_id": self.chart_id,
            "order": self.order,
            "controls": list(self.controls),
            "fixed_parameters": dict(sorted(self.fixed_parameters.items())),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiplicityCase:
        return cls(
            model=payload["model"],
            chart_id=payload["chart_id"],
            order=int(payload["order"]),
            controls=tuple(payload["controls"]),
            fixed_parameters=dict(payload["fixed_parameters"]),
        )

    def key(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_model(model_spec: str) -> Any:
    """Resolve ``module:Class`` (or ``module.Class``) and build an instance."""
    module_name, sep, class_name = model_spec.partition(":")
    if not sep:
        module_name, _, class_name = model_spec.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"cannot parse model spec {model_spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    fields = getattr(getattr(cls, "config_schema", None), "model_fields", None)
    if fields:
        return cls(**{name: 1.0 for name in fields})
    return cls()


def enumerate_scalar_charts(adapter: Any) -> dict[str, Any]:
    """Enumerate every scalar (retained-dimension-1) chart of a model.

    Keeps all equation partitions, addressed by ``chart_id``; records the
    regularity factors, reduced degree and provenance of each chart, plus the
    partitions fpgen rejected with their reasons.  Charts that cannot be
    materialized (e.g. oversized elimination blocks) are recorded with a
    skip reason instead of being dropped silently.
    """
    search = adapter.search_linear_reductions(
        retained_dimension=1,
        retained_ids=None,
        equation_partitions="all",
        return_limit=None,
        partition_limit=None,
        materialization_limit=None,
    )
    charts: list[dict[str, Any]] = []
    for candidate in search.candidates:
        info: dict[str, Any] = {
            "chart_id": candidate.chart_id,
            "retained_ids": list(candidate.retained_ids),
            "retained_equations": list(candidate.retained_equations),
            "eliminated_ids": list(candidate.eliminated_ids),
            "reduced_degree": candidate.reduced_degree,
        }
        try:
            materialized = adapter.materialized_linear_reduction(candidate=candidate)
        except Exception as exc:  # noqa: BLE001 - record, never drop silently
            info["materialized"] = False
            info["skip_reason"] = f"materialization_failed: {type(exc).__name__}: {exc}"
        else:
            x = materialized.plan.retained_symbols[0]
            info["materialized"] = True
            info["numerator_degrees"] = [
                int(sp.degree(sp.expand(num), x)) for num in materialized.numerators
            ]
            info["denominators"] = [str(d) for d in materialized.denominators]
            info["cleared_factors"] = [str(f) for f in materialized.cleared_factors]
            info["regularity_determinant"] = str(materialized.regularity_determinant)
        charts.append(info)
    return {
        "model": getattr(adapter.model, "name", type(adapter.model).__name__),
        "parameter_domains": dict(adapter.parameter_domains),
        "scalar_charts": charts,
        "rejected_partitions": [
            {
                "reason": entry.reason,
                "retained_ids": list(entry.retained_ids),
                "retained_equations": list(entry.retained_equations),
            }
            for entry in search.rejected_partitions
        ],
    }


def _prem_reduce(f: Any, e: Any, x: Any) -> Any:
    """One pseudo-remainder reduction step; preserves common zeros of f, e.

    The result is normalized to its primitive part in ``x``: the removed
    content is a polynomial in the controls alone, whose zeros remain common
    zeros of every reduced equation (so no original solution is lost), while
    dropping the leading-coefficient-power blowup prem accumulates.
    """
    degree_f = sp.degree(f, x)
    degree_e = sp.degree(e, x)
    if degree_f is None or degree_e is None or degree_e < 1 or degree_f < degree_e:
        return f
    reduced = sp.expand(sp.prem(f, e, x))
    if reduced == 0 or (sp.degree(reduced, x) or 0) < 1:
        return reduced
    return sp.expand(sp.Poly(reduced, x).primitive()[1].as_expr())


def _reduced_derivative_system(p: Any, x: Any, order: int) -> list[Any]:
    """Pseudo-remainder reduction of ``{d_x^i p, i < order}`` in ``x``.

    Every common zero of the original derivative system is a zero of the
    reduced system (prem is a polynomial combination), so solving the reduced
    system never loses solutions; candidates are verified exactly against the
    original equations afterwards.  For ``degree(p) == order`` the lowest
    equation is linear in ``x`` and the chain reproduces the proven
    elimination ``x = -b/(q*a)``.
    """
    derivatives = [sp.expand(sp.diff(p, x, i)) for i in range(order)]
    reduced: list[Any] = [derivatives[-1]]
    for i in range(order - 2, -1, -1):
        f = derivatives[i]
        for e in sorted(reduced, key=lambda item: (sp.degree(item, x) or 0)):
            f = _prem_reduce(f, e, x)
        if f != 0:
            reduced.append(f)
    return reduced


def _filter_solution(
    solution: dict[Any, Any],
    *,
    stratum_level: int,
    lead: Any,
    p_level: Any,
    x: Any,
    order: int,
    controls: tuple[Any, ...],
    parameter_domains: dict[str, str],
    chart_factors: tuple[Any, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Classify one exact solution.

    Returns ``(solution_record, None)`` for an accepted solution (classified
    ``regular`` or ``chart_singular``) or ``(None, filtered_record)``.
    """
    exact = {str(symbol): str(value) for symbol, value in solution.items()}
    filtered = {"stratum": stratum_level, "exact": exact}

    floats: dict[str, float] = {}
    reality_methods: set[str] = set()
    for symbol, value in solution.items():
        try:
            real = _real_value(value)
        except ExactCheckError as exc:
            return None, {**filtered, "reason": "exact_check_error", "detail": str(exc)}
        if real is None:
            return None, {**filtered, "reason": "nonreal", "detail": str(symbol)}
        floats[str(symbol)], method = real
        reality_methods.add(method)

    try:
        if _is_exactly_zero(lead.subs(solution)):
            return None, {
                **filtered,
                "reason": "deeper_stratum",
                "detail": "leading coefficient vanishes; handled at a deeper stratum",
            }
        dq = sp.diff(p_level, x, order)
        if _is_exactly_zero(dq.subs(solution)):
            return None, {
                **filtered,
                "reason": "higher_multiplicity",
                "detail": f"d_x^{order} p vanishes; multiplicity exceeds the target",
            }
        vanishing = [
            str(factor)
            for factor in chart_factors
            if _is_exactly_zero(factor.subs(solution))
        ]
    except ExactCheckError as exc:
        return None, {**filtered, "reason": "exact_check_error", "detail": str(exc)}

    for symbol in controls:
        if parameter_domains.get(str(symbol), "real") != "nonnegative":
            continue
        if _is_negative(solution[symbol]):
            return None, {
                **filtered,
                "reason": "domain_violation",
                "detail": f"{symbol} is negative inside a nonnegative domain",
            }

    record = {
        "stratum": stratum_level,
        "classification": (
            CLASSIFICATION_CHART_SINGULAR if vanishing else CLASSIFICATION_REGULAR
        ),
        "exact": exact,
        "values": floats,
        "checks": {
            "reality": "exact" if reality_methods == {"exact"} else "numeric",
            "exact_multiplicity": True,
            "chart_factors_nonzero": not vanishing,
        },
    }
    if vanishing:
        record["vanishing_chart_factors"] = vanishing
    return record, None


def _solve_zero_dimensional(
    equations: list[Any], unknowns: Sequence[Any]
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    """Solve a zero-dimensional polynomial system exactly.

    Returns ``(solutions, None)`` or ``(None, error)``; failure here is a
    solver failure, never conflated with a positive-dimensional system.
    """
    try:
        raw = sp.solve_poly_system(equations, *unknowns)
    except Exception as exc:  # noqa: BLE001 - SymPy raises many exception types
        return None, f"solve_poly_system: {type(exc).__name__}: {exc}"
    if raw is None:
        return None, "solve_poly_system returned None on a zero-dimensional system"
    seen: set[tuple[str, ...]] = set()
    solutions: list[tuple[Any, ...]] = []
    for entry in raw:
        key = tuple(str(value) for value in entry)
        if key not in seen:
            seen.add(key)
            solutions.append(tuple(entry))
    return solutions, None


def _elimination_system(p_level: Any, x: Any, order: int) -> tuple[list[Any], Any]:
    """Proven ``x = -b/(q*a)`` elimination; valid only when ``degree(p) == order``.

    For a degree-``q`` polynomial a ``q``-fold root must be the root of the
    linear ``(q-1)``-st derivative, i.e. ``x = -b/(q*a)`` with ``a``, ``b``
    the two leading coefficients.  Substituting into the first ``q - 1``
    derivatives and clearing the (nonzero on the stratum) denominators gives
    ``q - 1`` polynomial equations in the controls alone.
    """
    coefficients = sp.Poly(p_level, x).all_coeffs()
    a, b = coefficients[0], coefficients[1]
    r0 = -b / (order * a)
    equations = []
    for i in range(order - 1):
        deriv = sp.together(sp.diff(p_level, x, i).subs(x, r0))
        equations.append(sp.expand(sp.fraction(deriv)[0]))
    return equations, r0


def _lacunary_reduction(
    equations: list[Any], unknowns: tuple[Any, ...]
) -> tuple[list[Any], list[Any], dict[Any, tuple[Any, int]]]:
    """Monomial covering ``u = v**k`` for lacunary unknowns.

    When an unknown ``v`` appears in every term of every equation with an
    exponent that is a multiple of ``k > 1`` (currently only ``k == 2`` is
    applied), substitute ``u = v**k``.  The map ``v -> v**k`` is a finite
    surjective morphism, so the zero-/positive-dimension decision on the
    substituted system is exactly the dimension of the original system;
    solution branches are lifted back exactly by :func:`_lift_branches`.
    """
    reduced = list(equations)
    new_unknowns: list[Any] = []
    branches: dict[Any, tuple[Any, int]] = {}
    for variable in unknowns:
        exponents: set[int] = set()
        for equation in reduced:
            exponents.update(monom[0] for monom in sp.Poly(equation, variable).monoms())
        exponents.discard(0)
        k = 0
        for exponent in exponents:
            k = gcd(k, exponent)
        if k == 2 and exponents:
            replacement = sp.Dummy(f"lac_{variable.name}")
            reduced = [
                sp.expand(equation.subs(variable**k, replacement))
                for equation in reduced
            ]
            new_unknowns.append(replacement)
            branches[replacement] = (variable, k)
        else:
            new_unknowns.append(variable)
    return reduced, new_unknowns, branches


def _lift_branches(
    values: dict[Any, Any], branches: dict[Any, tuple[Any, int]]
) -> list[dict[Any, Any]]:
    """Lift solutions of a lacunary-substituted system to the original unknowns."""
    lifted: list[dict[Any, Any]] = [
        {variable: value for variable, value in values.items()}
    ]
    for replacement, (variable, _k) in branches.items():
        w = lifted[0].get(replacement)
        if w is None:
            continue
        if _is_exactly_zero(w):
            roots = [sp.Integer(0)]
        else:
            base = sp.sqrt(w)
            roots = [base, -base]
        extended: list[dict[Any, Any]] = []
        for branch in lifted:
            for root in roots:
                extended.append(
                    {v: val for v, val in branch.items() if v is not replacement}
                    | {variable: root}
                )
        lifted = extended
    return lifted


def _factors_divide_lead(coeff: Any, lead: Any, variables: tuple[Any, ...]) -> bool:
    """Return True when every nonconstant factor of ``coeff`` divides ``lead``.

    Equivalently the zero set of ``coeff`` is contained in the degree-drop
    boundary ``lead == 0``, so dividing by ``coeff`` loses nothing on the
    regular stratum.
    """
    _, factors = sp.factor_list(coeff)
    for factor, _multiplicity in factors:
        if not factor.free_symbols:
            continue
        _, remainder = sp.div(lead, factor, *variables)
        if sp.expand(remainder) != 0:
            return False
    return True


def _linear_elimination(
    equations: list[Any], unknowns: list[Any], lead: Any
) -> tuple[list[Any], list[Any], list[tuple[Any, Any, Any]]]:
    """Eliminate unknowns that appear linearly in some equation.

    For ``A*v + B = 0`` with ``A``, ``B`` free of ``v``, substitute
    ``v = -B/A`` into the remaining equations (clearing denominators).  The
    step is applied only when every factor of ``A`` divides the stratum
    leading coefficient, so solutions with ``A == 0`` lie on the degree-drop
    boundary and are covered by deeper strata.  Returns the reduced
    equations, the remaining unknowns and the ordered substitution chain
    ``(v, A, B)`` used to reconstruct eliminated values.
    """
    remaining = list(equations)
    variables = list(unknowns)
    substitutions: list[tuple[Any, Any, Any]] = []
    changed = True
    while changed:
        changed = False
        for v in list(variables):
            for equation in list(remaining):
                if sp.degree(equation, v) != 1:
                    continue
                coeff = sp.expand(sp.diff(equation, v))
                rest = sp.expand(equation.subs(v, 0))
                if coeff == 0 or not _factors_divide_lead(
                    coeff, lead, tuple(variables)
                ):
                    continue
                value = sp.together(-rest / coeff)
                remaining = [
                    sp.expand(sp.fraction(sp.together(g.subs(v, value)))[0])
                    for g in remaining
                    if g is not equation
                ]
                remaining = [g for g in remaining if g != 0]
                substitutions.append((v, coeff, rest))
                variables.remove(v)
                changed = True
                break
            if changed:
                break
    return remaining, variables, substitutions


#: Maximum degree of a univariate elimination factor processed with exact
#: quotient-field recovery in the resultant path.  Larger factors are screened
#: numerically and recorded under ``skipped_factors`` (completeness "partial").
MAX_EXACT_FACTOR_DEGREE = 40


def _qf_reduce(expr: Any, modulus: Any, gen: Any) -> Any:
    """Reduce a rational expression in ``gen`` into ``QQ[gen]/(modulus)``.

    ``modulus`` must be irreducible over ``QQ`` so the quotient is a field.
    Returns the canonical representative (a polynomial in ``gen`` of degree
    below ``degree(modulus)``).  Raises :class:`ExactCheckError` when a
    denominator is the zero element of the field.
    """
    mp = sp.Poly(modulus, gen, domain=sp.QQ)
    numerator, denominator = sp.fraction(sp.together(expr))
    try:
        rn = sp.rem(sp.Poly(sp.expand(numerator), gen, domain=sp.QQ), mp)
        rd = sp.rem(sp.Poly(sp.expand(denominator), gen, domain=sp.QQ), mp)
        if rd.is_zero:
            raise ExactCheckError(f"denominator vanishes modulo {modulus!r}")
        if rd.degree() == 0:
            return sp.expand(rn.as_expr() / rd.all_coeffs()[0])
        inverse = sp.invert(rd, mp, gen).as_expr()
        return sp.rem(
            sp.Poly(sp.expand(rn.as_expr() * inverse), gen, domain=sp.QQ), mp
        ).as_expr()
    except ExactCheckError:
        raise
    except Exception as exc:  # noqa: BLE001 - any SymPy failure means "undecided"
        raise ExactCheckError(
            f"quotient-field reduction failed modulo {modulus!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _qf_gcd(f: Any, g: Any, var: Any, modulus: Any, gen: Any) -> Any:
    """Monic gcd of ``f``, ``g`` in ``var`` over the field ``QQ[gen]/(modulus)``.

    Euclidean algorithm with monic normalization at every step; all
    coefficient arithmetic is exact formal reduction by :func:`_qf_reduce`.
    Returns 0 when both polynomials vanish modulo ``modulus``.
    """

    def coeffs(poly_expr: Any) -> list[Any]:
        dense = [
            _qf_reduce(c, modulus, gen) for c in sp.Poly(poly_expr, var).all_coeffs()
        ]
        while dense and dense[0] == 0:
            dense.pop(0)
        return dense

    def normalize(dense: list[Any]) -> list[Any]:
        inverse = _qf_reduce(sp.Integer(1) / dense[0], modulus, gen)
        return [_qf_reduce(c * inverse, modulus, gen) for c in dense]

    fcs, gcs = coeffs(f), coeffs(g)
    if fcs:
        fcs = normalize(fcs)
    if gcs:
        gcs = normalize(gcs)
    while gcs:
        if len(fcs) < len(gcs):
            fcs, gcs = gcs, fcs
        remainder = list(fcs)
        degree_g = len(gcs) - 1
        while len(remainder) - 1 >= degree_g and remainder:
            head = remainder[0]
            if head != 0:
                remainder = [
                    _qf_reduce(a - _qf_reduce(head * gc, modulus, gen), modulus, gen)
                    for a, gc in zip(remainder, gcs, strict=False)
                ] + remainder[len(gcs) :]
            else:
                remainder = remainder[1:]
            while remainder and remainder[0] == 0:
                remainder.pop(0)
        fcs, gcs = gcs, (normalize(remainder) if remainder else remainder)
    if not fcs:
        return sp.Integer(0)
    degree = len(fcs) - 1
    return sp.expand(sum(c * var ** (degree - i) for i, c in enumerate(fcs)))


def _numeric_factor_screen(
    r0: Any, r1: Any, h: Any, ca: Any, cb: Any, digits: int = 50
) -> dict[str, Any]:
    """Numerically decide whether ``r0``, ``r1`` share a root above ``h == 0``.

    Evaluates at one root ``a0`` of ``h``; the gcd degree is constant across
    the conjugate roots of an irreducible factor, so one embedding decides
    for all of them.  The verdict is numeric evidence, always recorded in the
    case record — never an exactness criterion.
    """
    try:
        a0 = sp.nroots(sp.Poly(h, ca), n=digits, maxsteps=2000)[0]
        f0 = sp.Poly(sp.N(r0.subs(ca, a0), digits), cb)
        f1_coeffs = sp.Poly(sp.N(r1.subs(ca, a0), digits), cb)
    except Exception as exc:  # noqa: BLE001
        return {"decision": "undecided", "detail": f"{type(exc).__name__}: {exc}"}
    if f0.degree() < 1:
        return {"decision": "undecided", "detail": "r0 loses all degree at the root"}
    try:
        roots = sp.nroots(f0, n=digits - 5, maxsteps=2000)
    except Exception as exc:  # noqa: BLE001
        return {
            "decision": "undecided",
            "detail": f"nroots: {type(exc).__name__}: {exc}",
        }
    terms = [(exp[0], abs(complex(coeff))) for exp, coeff in f1_coeffs.terms()]
    best_rel: float | None = None
    best_b: complex | None = None
    for b_ in roots:
        b_abs = abs(complex(b_))
        scale = sum(c * b_abs**e for e, c in terms)
        residual = abs(complex(sp.N(r1.subs({ca: a0, cb: b_}), digits - 5)))
        relative = residual / max(scale, 1e-300)
        if best_rel is None or relative < best_rel:
            best_rel, best_b = relative, complex(b_)
    if best_rel is None:
        return {"decision": "undecided", "detail": "no roots to test"}
    record = {
        "sample_root": str(a0),
        "sample_partner": str(best_b),
        "min_relative_residual": float(best_rel),
    }
    if best_rel < 1e-20:
        record["decision"] = "common_root_evidence"
    elif best_rel > 1e-6:
        record["decision"] = "no_common_root"
    else:
        record["decision"] = "undecided"
    return record


def _emit_certified_family(
    *,
    modulus: Any | None,
    gen: Any | None,
    value_map: dict[Any, Any],
    original: list[Any],
    lead: Any,
    dq: Any,
    x: Any,
    order: int,
    controls: tuple[Any, ...],
    parameter_domains: dict[str, str],
    chart_factors: tuple[Any, ...],
    stratum_level: int,
    solutions: list[Any],
    filtered: list[Any],
    extra_required: Iterable[Any] = (),
) -> None:
    """Certify one solution family and emit per-root solution records.

    ``value_map`` assigns every unknown an exact value: rationals (or
    radicals) when ``modulus`` is None, otherwise expressions in ``gen`` with
    the irreducible ``modulus``.  Acceptance checks are exact: plain
    substitution for modulus-None families, formal reduction modulo
    ``modulus`` otherwise (sound for every root of an irreducible modulus).
    Reality and parameter-domain checks are per root via exact ``CRootOf``
    properties.  ``extra_required`` holds the stratum degree-drop conditions:
    families not vanishing on them are zeros of the degree-dropped polynomial
    alone, not of this stratum, and are filtered as ``stratum_condition``.
    """
    label = {
        "stratum": stratum_level,
        "family": {str(k): str(v) for k, v in value_map.items()},
    }
    if modulus is not None:
        label["family_modulus"] = str(modulus)

    def decide(expr: Any) -> bool:
        if modulus is None:
            return _is_exactly_zero(expr)
        return _qf_reduce(expr, modulus, gen) == 0

    try:
        if not all(decide(equation.subs(value_map)) for equation in original):
            filtered.append(
                {
                    **label,
                    "reason": "reduction_artifact",
                    "detail": "resultant family failed the exact derivative check",
                }
            )
            return
        failed_drops = [
            str(condition)
            for condition in extra_required
            if not decide(condition.subs(value_map))
        ]
        if failed_drops:
            filtered.append(
                {
                    **label,
                    "reason": "stratum_condition",
                    "detail": (
                        "degree-drop conditions not satisfied: "
                        + ", ".join(failed_drops)
                    ),
                }
            )
            return
        if decide(lead.subs(value_map)):
            filtered.append(
                {
                    **label,
                    "reason": "deeper_stratum",
                    "detail": (
                        "leading coefficient vanishes; handled at a deeper stratum"
                    ),
                }
            )
            return
        if decide(dq.subs(value_map)):
            filtered.append(
                {
                    **label,
                    "reason": "higher_multiplicity",
                    "detail": (
                        f"d_x^{order} p vanishes; multiplicity exceeds the target"
                    ),
                }
            )
            return
        vanishing = [
            str(factor) for factor in chart_factors if decide(factor.subs(value_map))
        ]
    except ExactCheckError as exc:
        filtered.append({**label, "reason": "exact_check_error", "detail": str(exc)})
        return

    roots: list[tuple[Any | None, int | None]]
    if modulus is None:
        roots = [(None, None)]
    else:
        degree = int(sp.degree(modulus, gen))
        roots = [(sp.CRootOf(modulus, gen, i), i) for i in range(degree)]
    for root, index in roots:
        if root is not None and root.is_real is not True:
            filtered.append(
                {
                    **label,
                    "reason": "nonreal",
                    "detail": f"root {index} of the family modulus",
                }
            )
            continue
        values = {
            symbol: (value if root is None else value.subs(gen, root))
            for symbol, value in value_map.items()
        }
        record_label = {
            "stratum": stratum_level,
            "exact": {str(k): str(v) for k, v in values.items()},
        }
        floats: dict[str, float] = {}
        reality_methods: set[str] = set()
        rejected = False
        for symbol, value in values.items():
            try:
                real = _real_value(value)
            except ExactCheckError as exc:
                filtered.append(
                    {**record_label, "reason": "exact_check_error", "detail": str(exc)}
                )
                rejected = True
                break
            if real is None:
                filtered.append(
                    {**record_label, "reason": "nonreal", "detail": str(symbol)}
                )
                rejected = True
                break
            floats[str(symbol)], method = real
            reality_methods.add(method)
        if rejected:
            continue
        domain_hit = False
        for symbol in controls:
            if parameter_domains.get(str(symbol), "real") != "nonnegative":
                continue
            if _is_negative(values[symbol]):
                filtered.append(
                    {
                        **record_label,
                        "reason": "domain_violation",
                        "detail": f"{symbol} is negative inside a nonnegative domain",
                    }
                )
                domain_hit = True
                break
        if domain_hit:
            continue
        record = {
            "stratum": stratum_level,
            "classification": (
                CLASSIFICATION_CHART_SINGULAR if vanishing else CLASSIFICATION_REGULAR
            ),
            "exact": record_label["exact"],
            "values": floats,
            "checks": {
                "reality": "exact" if reality_methods == {"exact"} else "numeric",
                "exact_multiplicity": True,
                "chart_factors_nonzero": not vanishing,
                "zero_certificate": (
                    "substitution" if modulus is None else "quotient_field_mod_minpoly"
                ),
            },
        }
        if vanishing:
            record["vanishing_chart_factors"] = vanishing
        solutions.append(record)


def _resultant_stratum_solve(
    p_level: Any,
    *,
    x: Any,
    order: int,
    controls: tuple[Any, ...],
    lead: Any,
    parameter_domains: dict[str, str],
    chart_factors: tuple[Any, ...],
    stratum_level: int,
    drop_equations: Iterable[Any] = (),
    max_exact_factor_degree: int = MAX_EXACT_FACTOR_DEGREE,
) -> dict[str, Any]:
    """Solve one ``degree(p) > order`` stratum with exactly two controls.

    Eliminates ``x`` by resultants ``resultant_x(p^(q-1), p^(i))`` stripped of
    the degree-drop boundary factors, factors the univariate control
    elimination polynomial over ``QQ`` and recovers the second control and
    ``x`` by quotient-field gcds (:func:`_qf_gcd`).  ``drop_equations`` are
    the stratum degree-drop conditions; families not vanishing on them are
    filtered as ``stratum_condition``.  Returns ``status`` ``"solved"`` with
    certified solutions, or ``"fallback"`` with a reason when the resultant
    system is degenerate (the caller then uses the prem-chain + saturated
    Groebner path).
    """
    ca, cb = controls
    original = [sp.expand(sp.diff(p_level, x, i)) for i in range(order)]
    drops = [sp.expand(condition) for condition in drop_equations]
    dq = sp.diff(p_level, x, order)
    top = original[-1]
    info: dict[str, Any] = {"skipped_factors": [], "ghost_factors": []}
    out: dict[str, Any] = {
        "status": None,
        "solutions": [],
        "filtered": [],
        "info": info,
    }

    boundary_lead = sp.Poly(top, x).all_coeffs()[0]
    _, boundary_factors = sp.factor_list(boundary_lead)
    resultants: list[Any] = []
    for i in range(order - 1):
        ri = sp.resultant(sp.Poly(top, x), sp.Poly(original[i], x), x)
        ri = sp.expand(ri.as_expr() if isinstance(ri, sp.Poly) else ri)
        if ri == 0:
            out["status"] = "fallback"
            out["reason"] = (
                f"resultant(d_x^{order - 1} p, d_x^{i} p) vanishes identically; "
                "the derivative system has a structural common factor"
            )
            return out
        for factor, _multiplicity in boundary_factors:
            if not factor.free_symbols:
                continue
            while True:
                quotient, remainder = sp.div(ri, factor, ca, cb)
                if sp.expand(remainder) != 0:
                    break
                ri = sp.expand(quotient)
        if ri == 0:
            out["status"] = "fallback"
            out["reason"] = "resultant absorbed by boundary factors"
            return out
        resultants.append(sp.Poly(ri, ca, cb).primitive()[1].as_expr())
    info["resultant_degrees"] = [
        int(sp.Poly(ri, ca, cb).total_degree()) for ri in resultants
    ]

    r0, r1 = resultants
    if not r0.free_symbols or not r1.free_symbols:
        # A constant nonzero resultant admits no common x root: empty stratum.
        out["status"] = "solved"
        return out

    # Control families: (modulus, gen, assignments for ca and cb as
    # expressions in gen; modulus None with rational assignments).
    control_families: list[tuple[Any | None, Any | None, dict[Any, Any]]] = []
    if drops:
        # Degree-drop stratum: pin controls rationally through linear drop
        # factors, then solve the remaining univariate control system by an
        # exact gcd over QQ.  (Without the drop conditions the resultant
        # system alone is underdetermined here.)
        substitutions: dict[Any, Any] = {}
        for condition in drops:
            cond = sp.expand(condition.subs(substitutions))
            if cond == 0:
                continue
            pinned: tuple[Any, Any] | None = None
            _, cond_factors = sp.factor_list(cond)
            for c_factor, _mult in cond_factors:
                if not c_factor.free_symbols:
                    continue
                for v in (ca, cb):
                    if v in substitutions or sp.degree(c_factor, v) != 1:
                        continue
                    coeff = sp.expand(sp.diff(c_factor, v))
                    if coeff.free_symbols:
                        continue  # only constant-coefficient pins are sound
                    rest = sp.expand(c_factor.subs(v, 0))
                    pinned = (v, sp.expand(-rest / coeff))
                    break
                if pinned is not None:
                    break
            if pinned is None:
                out["status"] = "fallback"
                out["reason"] = (
                    f"degree-drop condition {cond} has no linear factor with "
                    "constant coefficient"
                )
                return out
            v, sigma = pinned
            for key, earlier in list(substitutions.items()):
                substitutions[key] = sp.expand(earlier.subs(v, sigma))
            substitutions[v] = sigma
        info["drop_substitutions"] = {
            str(key): str(value) for key, value in substitutions.items()
        }
        r0s = sp.expand(r0.subs(substitutions))
        r1s = sp.expand(r1.subs(substitutions))
        remaining = [c for c in (ca, cb) if c not in substitutions]
        if not remaining:
            if r0s == 0 and r1s == 0:
                control_families.append((None, None, dict(substitutions)))
        else:
            s = remaining[0]
            common1d = sp.gcd(r0s, r1s, s)
            common1d = sp.expand(
                common1d.as_expr() if isinstance(common1d, sp.Poly) else common1d
            )
            if common1d == 0:
                out["status"] = "fallback"
                out["reason"] = (
                    "both resultants vanish after the drop substitution; a "
                    "positive-dimensional component needs the saturated decision"
                )
                return out
            if (sp.degree(common1d, s) or 0) >= 1:
                _, k_factors = sp.factor_list(common1d)
                for k_factor, _mult in k_factors:
                    if not k_factor.free_symbols:
                        continue
                    if int(sp.degree(k_factor, s)) == 1:
                        kc = sp.Poly(k_factor, s).all_coeffs()
                        root_value = sp.Rational(-kc[1] / kc[0])
                        control_families.append(
                            (
                                None,
                                None,
                                {
                                    key: sp.expand(value.subs(s, root_value))
                                    for key, value in substitutions.items()
                                }
                                | {s: root_value},
                            )
                        )
                    else:
                        control_families.append((k_factor, s, {**substitutions, s: s}))
    else:
        elimination_poly = sp.resultant(sp.Poly(r0, ca, cb), sp.Poly(r1, ca, cb), cb)
        if not isinstance(elimination_poly, sp.Poly):
            elimination_poly = sp.Poly(sp.expand(elimination_poly), ca)
        if elimination_poly.as_expr() == 0:
            out["status"] = "fallback"
            out["reason"] = (
                "control resultants share a common component; the control "
                "system is not zero-dimensional"
            )
            return out
        info["elimination_degree"] = int(elimination_poly.degree())
        _, u_factors = sp.factor_list(elimination_poly.as_expr())
        u_factors = [
            (sp.Poly(factor, ca).primitive()[1].as_expr(), multiplicity)
            for factor, multiplicity in u_factors
            if factor.free_symbols
        ]
        info["elimination_factors"] = [
            {"degree": int(sp.degree(factor, ca)), "multiplicity": multiplicity}
            for factor, multiplicity in u_factors
        ]

        for h, _multiplicity in u_factors:
            degree_h = int(sp.degree(h, ca))
            factor_label = {"factor": str(h), "degree": degree_h}
            if degree_h > max_exact_factor_degree:
                screen = _numeric_factor_screen(r0, r1, h, ca, cb)
                info["skipped_factors"].append({**factor_label, "screen": screen})
                continue
            try:
                common = _qf_gcd(r0, r1, cb, h, ca)
            except ExactCheckError as exc:
                out["filtered"].append(
                    {
                        "stratum": stratum_level,
                        **factor_label,
                        "reason": "exact_check_error",
                        "detail": str(exc),
                    }
                )
                continue
            degree_g = sp.degree(common, cb) if common != 0 else None
            if common == 0:
                out["status"] = "fallback"
                out["reason"] = (
                    f"both resultants vanish modulo the factor of degree "
                    f"{degree_h}; a positive-dimensional component needs the "
                    "saturated decision"
                )
                return out
            if degree_g is None or degree_g < 1:
                info["ghost_factors"].append(factor_label)
                continue
            if degree_h == 1:
                coeffs = sp.Poly(h, ca).all_coeffs()
                alpha = sp.Rational(-coeffs[1] / coeffs[0])
                _, g_factors = sp.factor_list(sp.expand(common))
                for k_factor, _mult in g_factors:
                    if not k_factor.free_symbols:
                        continue
                    if int(sp.degree(k_factor, cb)) == 1:
                        kc = sp.Poly(k_factor, cb).all_coeffs()
                        beta = sp.Rational(-kc[1] / kc[0])
                        control_families.append((None, None, {ca: alpha, cb: beta}))
                    else:
                        control_families.append((k_factor, cb, {ca: alpha, cb: cb}))
            elif degree_g == 1:
                beta = sp.expand(-sp.Poly(common, cb).all_coeffs()[1])
                control_families.append((h, ca, {ca: ca, cb: beta}))
            else:
                out["filtered"].append(
                    {
                        "stratum": stratum_level,
                        **factor_label,
                        "reason": "recovery_unsupported",
                        "detail": (
                            f"gcd over the quotient field has degree {degree_g} "
                            f"in {cb}; nonlinear recovery over a nonrational "
                            "modulus is not implemented"
                        ),
                    }
                )
                continue

    for modulus, gen, assignment in control_families:
        substituted = [sp.expand(equation.subs(assignment)) for equation in original]
        try:
            if modulus is None:
                dx = substituted[0]
                for equation in substituted[1:]:
                    dx = sp.gcd(dx, equation, x)
                dx = sp.expand(dx.as_expr() if isinstance(dx, sp.Poly) else dx)
            else:
                dx = substituted[0]
                for equation in substituted[1:]:
                    dx = _qf_gcd(dx, equation, x, modulus, gen)
                    if dx == 0:
                        break
        except ExactCheckError as exc:
            out["filtered"].append(
                {
                    "stratum": stratum_level,
                    "reason": "exact_check_error",
                    "detail": str(exc),
                }
            )
            continue
        if dx == 0:
            out["filtered"].append(
                {
                    "stratum": stratum_level,
                    "reason": "deeper_stratum",
                    "detail": "every derivative vanishes modulo the factor",
                    "family": {str(k): str(v) for k, v in assignment.items()},
                }
            )
            continue
        degree_dx = sp.degree(dx, x)
        if degree_dx is None or degree_dx < 1:
            out["filtered"].append(
                {
                    "stratum": stratum_level,
                    "reason": "reduction_artifact",
                    "detail": "resultant family shares no common x root",
                    "family": {str(k): str(v) for k, v in assignment.items()},
                }
            )
            continue
        x_families: list[tuple[Any | None, Any | None, dict[Any, Any]]] = []
        if degree_dx == 1:
            if modulus is None:
                xc = sp.Poly(dx, x).all_coeffs()
                x_value = sp.Rational(-xc[1] / xc[0])
            else:
                # dx is monic over the quotient field: x + c0.
                x_value = sp.expand(-sp.Poly(dx, x).all_coeffs()[1])
            x_families.append((modulus, gen, {**assignment, x: x_value}))
        elif modulus is None:
            _, dx_factors = sp.factor_list(dx)
            for x_factor, _mult in dx_factors:
                if not x_factor.free_symbols:
                    continue
                if int(sp.degree(x_factor, x)) == 1:
                    xc = sp.Poly(x_factor, x).all_coeffs()
                    x_families.append(
                        (
                            None,
                            None,
                            {**assignment, x: sp.Rational(-xc[1] / xc[0])},
                        )
                    )
                else:
                    x_families.append((x_factor, x, {**assignment, x: x}))
        else:
            out["filtered"].append(
                {
                    "stratum": stratum_level,
                    "reason": "recovery_unsupported",
                    "detail": (
                        f"common x gcd has degree {degree_dx} over a "
                        "nonrational modulus; not implemented"
                    ),
                    "family": {str(k): str(v) for k, v in assignment.items()},
                }
            )
            continue
        for family_modulus, family_gen, values in x_families:
            _emit_certified_family(
                modulus=family_modulus,
                gen=family_gen,
                value_map=values,
                original=original,
                lead=lead,
                dq=dq,
                x=x,
                order=order,
                controls=controls,
                parameter_domains=parameter_domains,
                chart_factors=chart_factors,
                stratum_level=stratum_level,
                solutions=out["solutions"],
                filtered=out["filtered"],
                extra_required=drops,
            )

    out["status"] = "solved"
    return out


def _solve_stratum(
    p_level: Any,
    *,
    x: Any,
    order: int,
    controls: tuple[Any, ...],
    lead: Any,
    drop_conditions: list[Any],
    stratum_level: int,
    parameter_domains: dict[str, str],
    chart_factors: tuple[Any, ...],
) -> dict[str, Any]:
    """Solve one stratum of the multiplicity problem."""
    started = time.perf_counter()
    unknowns = (x, *controls)
    original = [sp.expand(sp.diff(p_level, x, i)) for i in range(order)]
    drop_equations = [sp.expand(condition) for condition in drop_conditions]
    elimination = sp.degree(p_level, x) == order
    r0: Any = None
    if elimination:
        # degree(p) == q: the proven x = -b/(q*a) elimination reduces the
        # problem to q-1 polynomial equations in the controls alone.
        eliminated, r0 = _elimination_system(p_level, x, order)
        system = [e for e in (*drop_equations, *eliminated) if e != 0]
        solve_unknowns = controls
        method = "elimination"  # x = -b/(q*a), proven for degree == order
    else:
        # degree(p) > q: prem chains shrink the x-degrees; every candidate is
        # verified exactly against the original derivative equations below.
        reduced = _reduced_derivative_system(p_level, x, order)
        system = [e for e in (*drop_equations, *reduced) if e != 0]
        solve_unknowns = unknowns
        method = "prem_reduced"
    stratum: dict[str, Any] = {
        "level": stratum_level,
        "effective_degree": int(sp.degree(p_level, x)),
        "leading_coefficient": str(lead),
        "drop_conditions": [str(c) for c in drop_conditions],
        "method": method,
        "equations": [str(e) for e in original],
        "status": None,
        "solutions": [],
        "filtered": [],
    }

    if not elimination and len(controls) == 2:
        # degree(p) > q with two controls: the prem-reduced 3-variable
        # Groebner step is prohibitively slow on real charts (pair-hopping
        # exceeds 600 s), so eliminate x by resultants and solve the square
        # control system by univariate factorization and quotient-field gcds.
        # Degree-drop conditions of deeper strata are certified per family.
        try:
            resultant = _resultant_stratum_solve(
                p_level,
                x=x,
                order=order,
                controls=controls,
                lead=lead,
                parameter_domains=parameter_domains,
                chart_factors=chart_factors,
                stratum_level=stratum_level,
                drop_equations=drop_equations,
            )
        except Exception as exc:  # noqa: BLE001 - fall back, never lose the stratum
            resultant = {
                "status": "fallback",
                "reason": f"{type(exc).__name__}: {exc}",
                "info": {},
            }
        stratum["resultant"] = resultant.get("info", {})
        if resultant["status"] == "solved":
            stratum["method"] = "resultant_elimination"
            stratum["status"] = STATUS_SOLVED
            stratum["zero_dimensional"] = True
            stratum["solutions"] = resultant["solutions"]
            stratum["filtered"] = resultant["filtered"]
            stratum["completeness"] = (
                "partial" if resultant["info"].get("skipped_factors") else "complete"
            )
            stratum["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            return stratum
        stratum["resultant_fallback"] = resultant["reason"]

    work_system, work_unknowns, branches = _lacunary_reduction(system, solve_unknowns)
    work_system, work_unknowns, linear_subs = _linear_elimination(
        work_system, work_unknowns, lead
    )
    if branches:
        stratum["lacunary"] = {str(u): [str(v), k] for u, (v, k) in branches.items()}
    if linear_subs:
        stratum["linear_eliminations"] = [str(v) for v, _, _ in linear_subs]
    stratum["reduced_equations"] = [str(e) for e in work_system]

    def finish(
        entries: list[tuple[Any, ...]],
        entry_unknowns: Sequence[Any],
        entry_branches: dict[Any, tuple[Any, int]],
        entry_linear_subs: list[tuple[Any, Any, Any]],
    ) -> None:
        for entry in entries:
            known = dict(zip(entry_unknowns, entry[: len(entry_unknowns)], strict=True))
            eliminated_ok = True
            for variable, coeff, rest in reversed(entry_linear_subs):
                try:
                    coeff_value = coeff.subs(known)
                    if _is_exactly_zero(coeff_value):
                        # Zeros of the elimination coefficient lie on the
                        # degree-drop boundary: deeper strata cover them.
                        eliminated_ok = False
                        stratum["filtered"].append(
                            {
                                "stratum": stratum_level,
                                "exact": {str(k): str(v) for k, v in known.items()},
                                "reason": "deeper_stratum",
                                "detail": "linear elimination coefficient vanishes",
                            }
                        )
                        break
                    known[variable] = sp.simplify(-rest.subs(known) / coeff_value)
                except ExactCheckError as exc:
                    eliminated_ok = False
                    stratum["filtered"].append(
                        {
                            "stratum": stratum_level,
                            "exact": {str(k): str(v) for k, v in known.items()},
                            "reason": "exact_check_error",
                            "detail": str(exc),
                        }
                    )
                    break
            if not eliminated_ok:
                continue
            for lifted in _lift_branches(known, entry_branches):
                # Drop auxiliary unknowns (e.g. the saturation variable).
                lifted = {
                    variable: value
                    for variable, value in lifted.items()
                    if variable in solve_unknowns
                }
                if elimination:
                    # x is recovered rationally; the stratum condition
                    # lead != 0 is checked before forming r0 (a = 0 belongs
                    # to a deeper stratum).
                    try:
                        if _is_exactly_zero(lead.subs(lifted)):
                            stratum["filtered"].append(
                                {
                                    "stratum": stratum_level,
                                    "exact": {
                                        str(k): str(v) for k, v in lifted.items()
                                    },
                                    "reason": "deeper_stratum",
                                    "detail": "leading coefficient vanishes",
                                }
                            )
                            continue
                        x_value = sp.simplify(r0.subs(lifted))
                    except ExactCheckError as exc:
                        stratum["filtered"].append(
                            {
                                "stratum": stratum_level,
                                "exact": {str(k): str(v) for k, v in lifted.items()},
                                "reason": "exact_check_error",
                                "detail": str(exc),
                            }
                        )
                        continue
                    candidates.append({x: x_value, **lifted})
                else:
                    candidates.append(lifted)

    candidates: list[dict[Any, Any]] = []
    if not work_system and not work_unknowns:
        # Every equation eliminated rationally with no unknown left: a single
        # trivial zero-dimensional solution.
        stratum["status"] = STATUS_SOLVED
        stratum["zero_dimensional"] = True
        finish([()], work_unknowns, branches, linear_subs)
    elif not work_system:
        stratum["status"] = STATUS_POSITIVE_DIMENSIONAL
        stratum["zero_dimensional"] = False
    else:
        try:
            basis = sp.groebner(work_system, *work_unknowns)
            inconsistent = basis.contains(sp.Integer(1))
            zero_dimensional = bool(basis.is_zero_dimensional)
        except Exception as exc:  # noqa: BLE001
            stratum["status"] = STATUS_SOLVER_ERROR
            stratum["error"] = f"groebner: {type(exc).__name__}: {exc}"
        else:
            stratum["zero_dimensional"] = zero_dimensional or inconsistent
            if inconsistent:
                # Inconsistent system: a complete empty answer for this stratum.
                stratum["status"] = STATUS_SOLVED
            elif zero_dimensional:
                raw, error = _solve_zero_dimensional(work_system, work_unknowns)
                if error is not None:
                    stratum["status"] = STATUS_SOLVER_ERROR
                    stratum["error"] = error
                else:
                    stratum["status"] = STATUS_SOLVED
                    assert raw is not None  # error is None => solutions present
                    finish(raw, work_unknowns, branches, linear_subs)
            else:
                # Re-decide the dimension on the original system saturated
                # with the stratum leading coefficient (Rabinowitsch) before
                # declaring positive_dimensional: prem reduction may add
                # extraneous components, and the elimination is an
                # isomorphism only where the leading coefficient is nonzero.
                z = sp.Symbol(f"_sat_{stratum_level}")
                if elimination:
                    saturated = [*system, z * lead - 1]
                else:
                    saturated = [*drop_equations, *original, z * lead - 1]
                saturated_unknowns = (*solve_unknowns, z)
                try:
                    saturated_basis = sp.groebner(saturated, *saturated_unknowns)
                    saturated_inconsistent = saturated_basis.contains(sp.Integer(1))
                    saturated_zero_dim = bool(saturated_basis.is_zero_dimensional)
                except Exception as exc:  # noqa: BLE001
                    stratum["status"] = STATUS_SOLVER_ERROR
                    stratum["error"] = (
                        f"groebner(saturated): {type(exc).__name__}: {exc}"
                    )
                else:
                    stratum["zero_dimensional"] = saturated_zero_dim
                    stratum["dimension_decided_by"] = "saturated_system"
                    if not saturated_zero_dim:
                        stratum["status"] = STATUS_POSITIVE_DIMENSIONAL
                    elif saturated_inconsistent:
                        stratum["status"] = STATUS_SOLVED
                    else:
                        raw, error = _solve_zero_dimensional(
                            saturated, saturated_unknowns
                        )
                        if error is not None:
                            stratum["status"] = STATUS_SOLVER_ERROR
                            stratum["error"] = error
                        else:
                            stratum["status"] = STATUS_SOLVED
                            assert raw is not None  # error is None => present
                            finish(raw, saturated_unknowns, {}, [])

    if stratum["status"] == STATUS_SOLVED:
        for solution in candidates:
            # Admit only exact zeros of the original derivative equations:
            # prem reduction may add extraneous candidates where intermediate
            # leading coefficients vanish.
            try:
                exact_zero = all(
                    _is_exactly_zero(equation.subs(solution))
                    for equation in (*drop_equations, *original)
                )
            except ExactCheckError as exc:
                exact_zero = False
                stratum["filtered"].append(
                    {
                        "stratum": stratum_level,
                        "exact": {str(k): str(v) for k, v in solution.items()},
                        "reason": "exact_check_error",
                        "detail": str(exc),
                    }
                )
            if not exact_zero:
                if not any(
                    entry["exact"] == {str(k): str(v) for k, v in solution.items()}
                    for entry in stratum["filtered"]
                ):
                    stratum["filtered"].append(
                        {
                            "stratum": stratum_level,
                            "exact": {str(k): str(v) for k, v in solution.items()},
                            "reason": "reduction_artifact",
                            "detail": "zero of the reduced system only",
                        }
                    )
                continue
            record, filtered = _filter_solution(
                solution,
                stratum_level=stratum_level,
                lead=lead,
                p_level=p_level,
                x=x,
                order=order,
                controls=controls,
                parameter_domains=parameter_domains,
                chart_factors=chart_factors,
            )
            if record is not None:
                stratum["solutions"].append(record)
            else:
                stratum["filtered"].append(filtered)

    stratum["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return stratum


def solve_scalar_multiplicity(
    p: Any,
    x: Any,
    order: int,
    controls: Sequence[Any],
    *,
    parameter_domains: dict[str, str] | None = None,
    denominators: Iterable[Any] = (),
    cleared_factors: Iterable[Any] = (),
    nominal_degree: int | None = None,
) -> dict[str, Any]:
    """Solve the exact multiplicity-``order`` problem for one scalar polynomial.

    ``p`` is a SymPy expression, polynomial in ``x`` with coefficients
    depending only on ``controls`` (all other parameters already substituted
    by exact rationals).  ``nominal_degree`` may declare the chart-level
    degree of ``p`` before substitution; when it exceeds the actual degree
    the case is reported on the degree-drop stratum instead of being
    silently excluded.
    """
    started = time.perf_counter()
    controls = tuple(controls)
    domains = dict(parameter_domains or {})
    chart_factors = tuple(denominators) + tuple(cleared_factors)
    result: dict[str, Any] = {
        "status": None,
        "unknowns": [str(x), *(str(c) for c in controls)],
        "order": order,
        "strata": [],
    }
    expanded = sp.expand(p)
    poly = sp.Poly(expanded, x)
    degree = int(poly.degree())
    nominal = max(degree, nominal_degree if nominal_degree is not None else degree)
    result["nominal_degree"] = nominal
    result["degree"] = degree
    if nominal < order:
        result["status"] = STATUS_UNSUPPORTED
        result["detail"] = f"polynomial degree {nominal} below target order {order}"
        return result

    # Coefficient list of length nominal+1, zero-padded on top so that a
    # leading coefficient killed by substitution shows up as deeper strata.
    actual_coeffs = poly.all_coeffs()
    coefficients = [sp.Integer(0)] * (nominal - degree) + list(actual_coeffs)

    for level in range(0, nominal - order + 1):
        lead = coefficients[level]
        if _is_exactly_zero(lead):
            # Empty stratum: the leading coefficient vanishes identically, so
            # the effective degree is lower for every control value.
            result["strata"].append(
                {
                    "level": level,
                    "effective_degree": nominal - level,
                    "leading_coefficient": str(lead),
                    "drop_conditions": [str(c) for c in coefficients[:level]],
                    "status": "empty_stratum",
                    "solutions": [],
                    "filtered": [],
                }
            )
            continue
        p_level = sum(
            coefficients[i] * x ** (nominal - i) for i in range(level, nominal + 1)
        )
        drop_conditions = [c for c in coefficients[:level] if not _is_exactly_zero(c)]
        stratum = _solve_stratum(
            sp.expand(p_level),
            x=x,
            order=order,
            controls=controls,
            lead=lead,
            drop_conditions=drop_conditions,
            stratum_level=level,
            parameter_domains=domains,
            chart_factors=chart_factors,
        )
        result["strata"].append(stratum)

    processed = [s for s in result["strata"] if s["status"] != "empty_stratum"]
    if not processed:
        result["status"] = STATUS_UNSUPPORTED
        result["detail"] = "every stratum has effective degree below the target order"
    else:
        primary = processed[0]
        solutions = primary.get("solutions", [])
        if primary["status"] == STATUS_SOLVER_ERROR:
            result["status"] = STATUS_SOLVER_ERROR
        elif primary["status"] == STATUS_POSITIVE_DIMENSIONAL:
            result["status"] = STATUS_POSITIVE_DIMENSIONAL
        elif primary["level"] > 0:
            result["status"] = STATUS_DEGREE_DROP
        elif solutions and all(
            s["classification"] == CLASSIFICATION_CHART_SINGULAR for s in solutions
        ):
            result["status"] = STATUS_CHART_SINGULAR
        else:
            result["status"] = STATUS_SOLVED
    counts: dict[str, Any] = {"regular": 0, "chart_singular": 0, "filtered": {}}
    for stratum in result["strata"]:
        for solution in stratum.get("solutions", []):
            key = (
                "regular"
                if solution["classification"] == CLASSIFICATION_REGULAR
                else "chart_singular"
            )
            counts[key] += 1
        for entry in stratum.get("filtered", []):
            reason = entry.get("reason", "unknown")
            counts["filtered"][reason] = counts["filtered"].get(reason, 0) + 1
    result["counts"] = counts
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _case_symbols(adapter: Any, names: Iterable[str]) -> dict[str, Any]:
    symbols = {str(symbol): symbol for symbol in adapter.parameter_symbols}
    missing = [name for name in names if name not in symbols]
    if missing:
        raise ValueError(f"unknown model parameters: {missing}")
    return {name: symbols[name] for name in names}


def _reconstruct_and_check(
    adapter: Any,
    materialized: Any,
    solution: dict[Any, Any],
    fixed_substitutions: dict[Any, Any],
    fixed_values: dict[str, float],
) -> dict[str, Any]:
    """Reconstruct the full state and run the numerical physicality checks."""
    full = {**fixed_substitutions, **solution}
    state: list[float] = []
    imag_relative_max = 0.0
    for expr in materialized.reconstruct_full_state():
        numeric = complex(sp.N(expr.subs(full), 40))
        scale = max(1.0, abs(numeric.real))
        imag_relative_max = max(imag_relative_max, abs(numeric.imag) / scale)
        state.append(numeric.real)
    vector = np.asarray(state, dtype=float)
    params: dict[str, Any] = dict(fixed_values)
    for symbol, value in solution.items():
        name = str(symbol)
        if name in adapter.parameter_names:
            params[name] = float(sp.N(value, 40))
    matrix = adapter.state_matrix(vector, params)
    hermitian = (matrix + matrix.conjugate().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hermitian)
    residual = float(np.linalg.norm(adapter.rhs(vector, params)))
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(eigenvalues))))
    return {
        "state": [float(v) for v in vector],
        "state_imag_relative_max": float(imag_relative_max),
        "min_state_eigenvalue": float(eigenvalues.min()),
        "physical": bool(
            eigenvalues.min() >= -tolerance and imag_relative_max <= 1e-12
        ),
        # Full CAM residual in float64: a recorded diagnostic, never an
        # exactness criterion.
        "full_residual": residual,
    }


def run_case(case: MultiplicityCase, *, adapter: Any = None) -> dict[str, Any]:
    """Execute one multiplicity case in-process and return its JSON record."""
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "record_type": "case",
        "case_key": case.key(),
        "case": case.to_dict(),
        "algebraic_status": None,
    }
    try:
        if adapter is None:
            from qphase_cam.core.fpgen import FPGenDynamicsAdapter

            adapter = FPGenDynamicsAdapter.from_model(load_model(case.model))
        names = set(adapter.parameter_names)
        missing = names - set(case.controls) - set(case.fixed_parameters)
        extra = (set(case.controls) | set(case.fixed_parameters)) - names
        if missing or extra:
            record["algebraic_status"] = STATUS_UNSUPPORTED
            record["detail"] = (
                f"case parameters do not partition the model parameters: "
                f"missing={sorted(missing)} unknown={sorted(extra)}"
            )
            return record
        record["parameter_domains"] = {
            name: adapter.parameter_domains[name] for name in case.controls
        }

        search = adapter.search_linear_reductions(
            retained_dimension=1,
            retained_ids=None,
            equation_partitions="all",
            return_limit=None,
            partition_limit=None,
            materialization_limit=None,
        )
        by_id = {candidate.chart_id: candidate for candidate in search.candidates}
        candidate = by_id.get(case.chart_id)
        record["coverage"] = {
            "scalar_chart_ids": sorted(by_id),
            "rejected_partitions": [
                {"reason": entry.reason, "retained_ids": list(entry.retained_ids)}
                for entry in search.rejected_partitions
            ],
        }
        if candidate is None:
            record["algebraic_status"] = STATUS_UNSUPPORTED
            record["detail"] = f"chart_id {case.chart_id!r} not found"
            return record
        try:
            materialized = adapter.materialized_linear_reduction(candidate=candidate)
        except Exception as exc:  # noqa: BLE001
            record["algebraic_status"] = STATUS_UNSUPPORTED
            record["detail"] = f"materialization_failed: {type(exc).__name__}: {exc}"
            return record
        record["chart"] = {
            "chart_id": candidate.chart_id,
            "retained_ids": list(candidate.retained_ids),
            "retained_equations": list(candidate.retained_equations),
            "eliminated_ids": list(candidate.eliminated_ids),
            "reduced_degree": candidate.reduced_degree,
            "denominators": [str(d) for d in materialized.denominators],
            "cleared_factors": [str(f) for f in materialized.cleared_factors],
            "regularity_determinant": str(materialized.regularity_determinant),
        }

        x = materialized.plan.retained_symbols[0]
        fixed_symbols = _case_symbols(adapter, case.fixed_parameters)
        control_symbols = _case_symbols(adapter, case.controls)
        fixed_substitutions = {
            fixed_symbols[name]: sp.Rational(value)
            for name, value in case.fixed_parameters.items()
        }
        fixed_values = {
            name: float(sp.Rational(value))
            for name, value in case.fixed_parameters.items()
        }
        numerator = sp.expand(materialized.numerators[0])
        substituted = sp.expand(numerator.subs(fixed_substitutions))
        # Chart regularity factors must be specialized to the fixed parameter
        # values as well: left symbolic, a factor that vanishes at the fixed
        # values would be misread as nonzero (and quotient-field certificates
        # require univariate expressions in the controls).
        chart_denominators = tuple(
            sp.expand(factor.subs(fixed_substitutions))
            for factor in materialized.denominators
        )
        chart_cleared = tuple(
            sp.expand(factor.subs(fixed_substitutions))
            for factor in materialized.cleared_factors
        )

        result = solve_scalar_multiplicity(
            substituted,
            x,
            case.order,
            tuple(control_symbols[name] for name in case.controls),
            parameter_domains={
                name: adapter.parameter_domains.get(name, "real")
                for name in case.controls
            },
            denominators=chart_denominators,
            cleared_factors=chart_cleared,
            nominal_degree=int(candidate.reduced_degree),
        )
        record.update(
            {
                "algebraic_status": result["status"],
                "nominal_degree": result["nominal_degree"],
                "degree": result["degree"],
                "strata": result["strata"],
                "counts": result["counts"],
            }
        )
        if "detail" in result:
            record["detail"] = result["detail"]

        # Reconstruct full states only for regular solutions, after the exact
        # filters; cross-chart dedup keys on these reconstructed states.
        name_map = {str(s): s for s in (*adapter.parameter_symbols, x)}
        for stratum in record["strata"]:
            for solution in stratum.get("solutions", []):
                if solution["classification"] != CLASSIFICATION_REGULAR:
                    continue
                symbol_solution = {
                    name_map[name]: sp.sympify(value)
                    for name, value in solution["exact"].items()
                }
                solution.update(
                    _reconstruct_and_check(
                        adapter,
                        materialized,
                        symbol_solution,
                        fixed_substitutions,
                        fixed_values,
                    )
                )
    except Exception as exc:  # noqa: BLE001 - top-level guard: never lose the case
        record["algebraic_status"] = STATUS_SOLVER_ERROR
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def _failure_record(case: MultiplicityCase, status: str, detail: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "record_type": "case",
        "case_key": case.key(),
        "case": case.to_dict(),
        "algebraic_status": status,
        "detail": detail,
    }


def completed_case_keys(jsonl_path: str | Path) -> set[str]:
    """Case keys already recorded in a JSONL file (interrupt recovery)."""
    path = Path(jsonl_path)
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") == "case" and record.get("case_key"):
                keys.add(record["case_key"])
    return keys


def run_cases(
    cases: Iterable[MultiplicityCase],
    out_path: str | Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    command: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run cases in isolated subprocesses, appending JSONL after each case.

    Cases whose key already exists in ``out_path`` are skipped (resume after
    interruption).  ``command`` optionally overrides the subprocess argv; the
    literal ``{case_json}`` is replaced by the serialized case.  A subprocess
    that exceeds ``timeout_seconds`` is recorded as ``timeout``; a non-zero
    exit is recorded as ``solver_error``.
    """
    out = Path(out_path)
    done = completed_case_keys(out)
    records: list[dict[str, Any]] = []
    for case in cases:
        key = case.key()
        if key in done:
            continue
        case_json = json.dumps(case.to_dict())
        if command is None:
            argv = [
                sys.executable,
                "-m",
                "qphase_cam.solver.analytic_multiplicity",
                "run-case",
                "--case-json",
                case_json,
            ]
        else:
            argv = [part.replace("{case_json}", case_json) for part in command]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            record = _failure_record(
                case, STATUS_TIMEOUT, f"exceeded wall-clock timeout {timeout_seconds}s"
            )
        else:
            lines = proc.stdout.strip().splitlines()
            if proc.returncode == 0 and lines:
                try:
                    record = json.loads(lines[-1])
                except json.JSONDecodeError:
                    record = _failure_record(
                        case, STATUS_SOLVER_ERROR, "subprocess emitted invalid JSON"
                    )
            else:
                record = _failure_record(
                    case,
                    STATUS_SOLVER_ERROR,
                    f"subprocess exit {proc.returncode}: {proc.stderr[-400:]}",
                )
        record["case_key"] = key
        record.setdefault("schema", SCHEMA)
        record.setdefault("record_type", "case")
        record.setdefault("case", case.to_dict())
        record["runner_elapsed_seconds"] = round(time.perf_counter() - started, 3)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        done.add(key)
        records.append(record)
    return records


def deduplicate_solutions(
    records: Iterable[dict[str, Any]], *, digits: int = 9
) -> list[dict[str, Any]]:
    """Group regular solutions across cases by reconstructed full state.

    Dedup happens only after full-state reconstruction and keeps every
    provenance entry (case key, chart, stratum) of each occurrence.
    """
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("record_type") != "case":
            continue
        case = record.get("case", {})
        for stratum in record.get("strata", []):
            for solution in stratum.get("solutions", []):
                if solution.get("classification") != CLASSIFICATION_REGULAR:
                    continue
                state = solution.get("state")
                if state is None:
                    continue
                key = (
                    case.get("model"),
                    tuple(round(float(v), digits) for v in state),
                    tuple(
                        sorted(
                            (name, round(float(value), digits))
                            for name, value in solution.get("values", {}).items()
                        )
                    ),
                )
                group = groups.setdefault(
                    key,
                    {
                        "state": state,
                        "values": solution.get("values"),
                        "provenance": [],
                    },
                )
                group["provenance"].append(
                    {
                        "case_key": record.get("case_key"),
                        "chart_id": case.get("chart_id"),
                        "stratum": stratum.get("level"),
                    }
                )
    return list(groups.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qphase_cam.solver.analytic_multiplicity")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_case_parser = subcommands.add_parser(
        "run-case", help="run a single case and print its JSON record"
    )
    run_case_parser.add_argument("--case-json", required=True)
    run_parser = subcommands.add_parser(
        "run", help="run a list of cases with subprocess isolation and JSONL resume"
    )
    run_parser.add_argument("--cases", required=True, help="JSON file with case dicts")
    run_parser.add_argument("--out", required=True, help="JSONL output path")
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-case wall-clock timeout in seconds",
    )
    enumerate_parser = subcommands.add_parser(
        "enumerate", help="print the scalar chart coverage of a model"
    )
    enumerate_parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    if args.command == "run-case":
        record = run_case(MultiplicityCase.from_dict(json.loads(args.case_json)))
        print(json.dumps(record))
        return 0
    if args.command == "run":
        with open(args.cases, encoding="utf-8") as handle:
            cases = [MultiplicityCase.from_dict(item) for item in json.load(handle)]
        records = run_cases(cases, args.out, timeout_seconds=args.timeout)
        print(json.dumps({"ran": len(records)}, indent=1))
        return 0
    if args.command == "enumerate":
        from qphase_cam.core.fpgen import FPGenDynamicsAdapter

        adapter = FPGenDynamicsAdapter.from_model(load_model(args.model))
        print(json.dumps(enumerate_scalar_charts(adapter), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
