"""Reading a typed state product as the rows an annotation is written over.

A state product stores its solutions in fixed-capacity slots over a scan axis. An
annotation is written per *solution*, so this module turns the product into the
filled rows — state matrix, residual, success flag, parameters of the scan point
and the solution identity — and back into typed field columns. Nothing here
reaches into a solver result: the only input is the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.data import Dataset

from qphase_cam.outcomes import CAMOutcomeStatus

__all__ = [
    "StateRow",
    "annotation_status",
    "field_column",
    "state_rows",
]


@dataclass(frozen=True)
class StateRow:
    """One filled solution slot of a state product."""

    scan_index: int
    solution: int
    index: tuple[int, ...]
    state: np.ndarray
    residual: float
    success: bool
    parameters: dict[str, float]
    identity: str


def state_rows(product: Dataset) -> list[StateRow]:
    """Return every filled solution slot of a state/state_scan product."""
    states = np.asarray(product.handle("states").materialize())
    valid = np.asarray(product.handle("valid_mask").materialize(), dtype=bool)
    residuals = np.asarray(product.handle("residuals").materialize(), dtype=float)
    success = np.asarray(product.handle("success").materialize(), dtype=bool)
    identities = np.asarray(product.handle("solution_identity").materialize())
    grid_shape = tuple(int(value) for value in product.attributes.get("grid_shape", ()))
    parameters = {
        variable.name.removeprefix("parameter."): np.asarray(
            product.handle(variable.name).materialize()
        )
        for variable in product.variables
        if variable.name.startswith("parameter.")
    }
    rows: list[StateRow] = []
    for scan, solution in zip(*np.nonzero(valid), strict=True):
        index = (
            tuple(int(value) for value in np.unravel_index(int(scan), grid_shape))
            if grid_shape
            else ()
        )
        rows.append(
            StateRow(
                scan_index=int(scan),
                solution=int(solution),
                index=index,
                state=np.asarray(states[int(scan), int(solution)]),
                residual=float(residuals[int(scan), int(solution)]),
                success=bool(success[int(scan), int(solution)]),
                parameters={
                    name: float(values[int(scan)])
                    for name, values in parameters.items()
                },
                identity=str(identities[int(scan), int(solution)]),
            )
        )
    return rows


def field_column(
    rows: list[StateRow],
    *,
    tail: tuple[int, ...] = (),
    dtype: Any = float,
    fill: Any = None,
) -> np.ndarray:
    """Return one annotation field column over ``rows`` with trailing axes."""
    width = np.dtype(dtype)
    if fill is None:
        fill = np.nan if width.kind == "f" else 0
        if width.kind == "c":
            fill = np.nan + 1j * np.nan
        if width.kind in "US":
            fill = ""
        if width.kind == "b":
            fill = False
    return np.full((len(rows), *tail), fill, dtype=dtype)


def annotation_status(
    rows: list[StateRow],
    *,
    healthy: list[bool] | None = None,
    unhealthy: CAMOutcomeStatus = CAMOutcomeStatus.NONPHYSICAL,
) -> list[str]:
    """Return one outcome status per annotated row.

    A row whose solve did not succeed is ``not_converged``; a row that succeeded
    but fails the annotation's own check carries ``unhealthy``. Nothing collapses
    into a boolean.
    """
    statuses: list[str] = []
    for position, row in enumerate(rows):
        if not row.success:
            statuses.append(CAMOutcomeStatus.NOT_CONVERGED.value)
        elif healthy is not None and not healthy[position]:
            statuses.append(unhealthy.value)
        else:
            statuses.append(CAMOutcomeStatus.VALID.value)
    return statuses
