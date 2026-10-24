"""qphase_cam: solver-owned identity derivation
---------------------------------------------------------
The solver owns three identities, all deterministic, all content-driven, none of
them a slot:

``search identity``
    What design was searched: the model (name plus the scalar parameters that the
    search does *not* sweep), the target and its order, the control bounds and
    sampling, the perturbation spec and the strategy mode.
``case identity``
    One search context: the search identity plus the control values of this
    solve call. Two grid points of the same campaign therefore differ while
    sharing one search identity.
``candidate identity``
    One physical candidate: the model, target/order, controls and the candidate
    state, preferring the verified decimal values when the solver produced them.
    It deliberately contains **no case or search scope**, so the same physical
    candidate found by a different search keeps the same identity.
``accepted-start identity``
    One *accepted start*: the case scope, the discovery method and the start's
    own content. Two starts that converge to the same candidate therefore have
    two identities, and both are listed on that candidate.

Every identity is ``cam:<kind>:<hex>`` where ``<hex>`` is the first 16 hex digits
of a SHA-256 over canonical JSON. No path, no ``hash()``, no ``id()``, no array
slot and no wall clock enters the digest, so two identical solves agree and two
different pieces of science do not collide.

Precision boundary: when a candidate carries verified decimal values the identity
is derived from those digits, so it is stable for a solver that reproduces the
same verified digits. It is *not* a claim that the identity survives a different
verification precision; the accompanying ``identity_basis`` says which basis was
used, and a candidate without decimals falls back to the raw
``state_vector`` payload, which is explicitly a numerical representation rather
than a precision-independent one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from qphase.core.utils import canonical_json

__all__ = [
    "CANDIDATE_IDENTITY_BASIS",
    "IDENTITY_SCHEME",
    "case_identity",
    "candidate_content",
    "candidate_identity",
    "digest_payload",
    "search_identity",
    "search_scope",
    "start_identity",
]

#: Scheme marker mixed into every identity payload, so a future scheme change
#: cannot silently reuse the old digests.
IDENTITY_SCHEME = "qphase_cam.identity/1"

#: Number of hex digits kept from the SHA-256 digest.
DIGEST_LENGTH = 16

#: Identity bases a candidate may use.
CANDIDATE_IDENTITY_BASIS = {
    "verified_full_state_decimal_values": (
        "identity is derived from the verified full-state decimal digits"
    ),
    "state_vector_and_controls": (
        "identity is derived from the raw state vector and controls; a numerical "
        "representation, not a precision-independent one"
    ),
}


def digest_payload(payload: Mapping[str, Any]) -> str:
    """Return the deterministic digest of a canonical JSON payload."""
    text = canonical_json(_jsonable(payload))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:DIGEST_LENGTH]


def _jsonable(value: Any) -> Any:
    """Return a JSON form of a value without letting a nan/inf break hashing."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, complex):
        return {
            "real": value.real if np.isfinite(value.real) else repr(value.real),
            "imag": value.imag if np.isfinite(value.imag) else repr(value.imag),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return repr(value)


def _scalar_parameters(model: Any) -> dict[str, Any]:
    """Return the scalar model parameters as an identity payload."""
    params = getattr(model, "params", None) or {}
    payload: dict[str, Any] = {}
    for name, value in params.items():
        array = np.asarray(value)
        if array.ndim != 0:
            # An array-valued parameter is not search context; it is payload, and
            # hashing it here would make every identity depend on its layout.
            continue
        payload[str(name)] = _jsonable(array.item())
    return payload


def search_scope(
    model: Any,
    *,
    target: str,
    order: int,
    controls: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """Return the search-design payload of one campaign.

    The swept parameters (the control names) are **not** part of the search
    scope: they are what the case identity adds, so every point of one campaign
    shares the search identity.
    """
    params = _scalar_parameters(model)
    for name in controls:
        params.pop(str(name), None)
    return {
        "scheme": IDENTITY_SCHEME,
        "model": str(getattr(model, "name", type(model).__name__)),
        "params": params,
        "target": str(target),
        "order": int(order),
        "controls": {
            str(name): {
                "bounds": [float(spec.min), float(spec.max)],
                "sampling": str(getattr(spec, "sampling", "")),
            }
            for name, spec in controls.items()
        },
        "perturbation": {
            str(key): _jsonable(value) for key, value in perturbation.items()
        },
        "strategy": str(strategy),
    }


def search_identity(scope: Mapping[str, Any]) -> str:
    """Return the identity of one search design."""
    return f"cam:search:{digest_payload(scope)}"


def case_identity(search: str, case_params: Mapping[str, float]) -> str:
    """Return the identity of one search case (a search plus its control values)."""
    payload = {
        "scheme": IDENTITY_SCHEME,
        "search": str(search),
        "case_params": {
            str(name): _jsonable(float(value)) for name, value in case_params.items()
        },
    }
    return f"cam:case:{digest_payload(payload)}"


def candidate_content(
    *,
    model_name: str,
    target: str,
    order: int,
    state_vector: Any,
    controls: Mapping[str, float],
    verified_full_state_decimal_values: Any = (),
) -> tuple[dict[str, Any], str]:
    """Return the candidate content payload and the identity basis it uses."""
    decimals = tuple(str(value) for value in (verified_full_state_decimal_values or ()))
    if decimals:
        basis = "verified_full_state_decimal_values"
        state: dict[str, Any] = {"decimals": list(decimals)}
    else:
        basis = "state_vector_and_controls"
        vector = np.asarray(state_vector, dtype=float).reshape(-1)
        state = {"state_vector": [float(value) for value in vector]}
    return (
        {
            "scheme": IDENTITY_SCHEME,
            "model": str(model_name),
            "target": str(target),
            "order": int(order),
            "state": state,
            "controls": {str(name): float(value) for name, value in controls.items()},
        },
        basis,
    )


def candidate_identity(content: Mapping[str, Any]) -> str:
    """Return the identity of one physical candidate.

    The payload carries no case or search scope on purpose: the same physical
    candidate found by another search keeps this identity, which is what makes it
    reusable across searches.
    """
    return f"cam:candidate:{digest_payload(content)}"


def start_identity(*, case: str, method: str, content: Mapping[str, Any]) -> str:
    """Return the identity of one accepted start inside one case."""
    payload = {
        "scheme": IDENTITY_SCHEME,
        "case": str(case),
        "method": str(method),
        "candidate": content,
    }
    return f"cam:start:{digest_payload(payload)}"
