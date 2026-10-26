"""qphase_cam: observables and their derivative contract
---------------------------------------------------------
An observable is the readout a stochastic or response assessment projects onto:
it answers "what do we actually measure, in which charge sector, and how does it
move when the state moves?". This is a **plain public contract**, not a plugin
class — an observable is a small value object a caller constructs, so registering
one would add a registry slot nothing selects from.

Three pieces:

* :class:`ObservableIdentity` states what an observable *is* — its controlled
  kind, its charge sector, its name and the formula that produced it as readable
  text. Nothing infers a sector from a name.
* :class:`CAMObservable` is the structural protocol: ``identity``, ``value`` and
  ``gradient``. :meth:`CAMObservable.gradient` returns a
  :class:`DerivativeRecord`, so a gradient can never be used without the step,
  the coordinate scale and the repeat check that produced it.
* Three implementations ship here: :class:`RayleighObservable`
  (``Re Tr(H R)/Tr(R)`` from the model Hamiltonian and the adapter's state matrix, no
  mode or model hard-coded), :class:`DiagonalRayleighObservable`
  (``Re (H R)[m,m]/R[m,m]``, the frequency of one mode's amplitude) and
  :class:`FunctionalObservable` (a caller-supplied value and optional analytic
  gradient), which is enough for any other readout without a new plugin registration.

The trace readout and the diagonal readout each own their denominator rule and never
share one: a state matrix whose *trace* sits at rounding scale cannot normalise
``Re Tr(H R)/Tr(R)`` at all, so :class:`RayleighObservable` refuses it by raising, while
:class:`DiagonalRayleighObservable` reports ``NaN`` when a mode amplitude's diagonal
element is below its own ``denominator_floor`` — a node of one mode is a real state, not
an unusable one. Neither rule is applied to the other readout.

The default gradient is a central difference with the step tied to each
coordinate's own scale, repeated at half the step so the record can say how
stable the derivative is. A non-finite or unresolved derivative is reported
through the record's fields, never clamped into a plausible-looking number.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CAMObservable",
    "ChargeSector",
    "DerivativeRecord",
    "DiagonalRayleighObservable",
    "FunctionalObservable",
    "ObservableIdentity",
    "ObservableKind",
    "RayleighObservable",
    "coordinate_scale",
    "observable_values",
    "finite_difference_gradient",
]


class ObservableKind(StrEnum):
    """What an observable is attached to.

    ``state``
        A property of the density matrix itself (photons, occupation, parity).
    ``carrier``
        A carrier-frequency readout derived from the Hamiltonian.
    ``response``
        A response/spectral quantity rather than a state expectation.
    """

    STATE = "state"
    CARRIER = "carrier"
    RESPONSE = "response"


class ChargeSector(StrEnum):
    """The charge sector an observable is projective onto."""

    CHARGE_ZERO = "charge_zero"
    CHARGE_ONE = "charge_one"


class ObservableIdentity(BaseModel):
    """The controlled identity of one observable.

    ``measurement_mapping`` is the formula in readable text ("Re Tr(H R) / Tr(R)")
    so a stored annotation can be re-derived without guessing which readout
    produced it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ObservableKind
    sector: ChargeSector
    name: str = Field(min_length=1)
    measurement_mapping: str = Field(min_length=1)


class DerivativeRecord(BaseModel):
    """A gradient together with how it was obtained and how well it converged.

    ``step`` and ``scale`` are per coordinate. ``repeat_relative_error`` is the
    relative change between the gradient at ``step`` and at ``step / 2``: a large
    value means the derivative is not resolved and the caller must not use it as
    if it were. ``condition`` is ``max|g| / min|g|`` over the non-zero entries and
    is ``None`` when the gradient has a zero entry (the ratio is undefined, not
    infinite).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["central_difference", "analytic"]
    order: int = Field(ge=0)
    gradient: tuple[float, ...]
    step: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    condition: float | None = None
    repeat_relative_error: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_finite(self) -> bool:
        """Whether every gradient entry is finite."""
        return all(np.isfinite(value) for value in self.gradient)


@runtime_checkable
class CAMObservable(Protocol):
    """The readout contract an assessment projects onto.

    ``value`` returns one real scalar at one state; ``gradient`` returns it
    together with the derivative provenance. A discrete or unbranched observable
    still has to state a gradient record, even if it refuses (analytic with a
    stated reason) — the assessment never invents one.

    ``values`` is the optional batch form: ``states`` is a ``(n, coordinates)``
    stack and the result is one value per row. An observable that cannot evaluate a
    stack simply does not provide it, and :func:`observable_values` then samples the
    single-point ``value`` row by row — the same arithmetic, without the batching.
    A batch implementation is the *same readout* evaluated for every row at once: it
    exists to avoid re-entering the adapter and the model once per point, not to
    define a second readout, and the two forms agree to rounding rather than to the
    last bit because the summation order differs.
    """

    identity: ObservableIdentity

    def value(self, adapter: Any, state: Any, params: Mapping[str, Any]) -> float: ...

    def gradient(
        self, adapter: Any, state: Any, params: Mapping[str, Any]
    ) -> DerivativeRecord: ...


def observable_values(
    observable: Any, adapter: Any, states: Any, params: Mapping[str, Any]
) -> np.ndarray:
    """Return one readout value per row of a state stack.

    A readout that implements the batch contract answers directly; one that only
    implements ``value`` is sampled row by row, which is the same arithmetic on the
    same states. The result is validated against the stack length so a readout cannot
    silently return a different number of values.
    """
    stack = np.asarray(states, dtype=float)
    if stack.ndim != 2:
        raise ValueError(
            f"a state stack must be (n, coordinates); got shape {stack.shape}"
        )
    batched = getattr(observable, "values", None)
    if callable(batched):
        values = np.asarray(batched(adapter, stack, params), dtype=float).reshape(-1)
        if values.size != stack.shape[0]:
            raise ValueError(
                f"the readout returned {values.size} values for {stack.shape[0]} states"
            )
        return values
    return np.asarray(
        [float(observable.value(adapter, vector, params)) for vector in stack],
        dtype=float,
    )


def _state_matrices(
    adapter: Any, stack: np.ndarray, params: Mapping[str, Any]
) -> np.ndarray:
    """Return one state matrix per row of a state stack.

    An adapter that declares ``supports_batched_states`` is asked once; one that only
    implements the single-state contract is sampled row by row, which is the same
    arithmetic. A batch answer that is not one square matrix per row raises: that is a
    broken contract, not an unusable state.
    """
    if getattr(adapter, "supports_batched_states", False):
        matrices = np.asarray(adapter.state_matrix(stack, params), dtype=complex)
    else:
        matrices = np.stack(
            [
                np.asarray(adapter.state_matrix(row, params), dtype=complex)
                for row in stack
            ]
        )
    if matrices.shape[:-2] != (stack.shape[0],) or (
        matrices.shape[-2] != matrices.shape[-1]
    ):
        raise ValueError(
            f"the adapter returned state matrices of shape {matrices.shape} for "
            f"{stack.shape[0]} states"
        )
    return matrices


def coordinate_scale(state: Any) -> np.ndarray:
    """Return the per-coordinate scale a finite difference steps against.

    The scale is ``max(|x_i|, 1)``: a coordinate far from zero is stepped in
    proportion to itself, and a coordinate near zero is stepped in absolute
    units, so neither case collapses the step to a rounding error.
    """
    values = np.abs(np.asarray(state, dtype=float).reshape(-1))
    return np.maximum(values, 1.0)


def finite_difference_gradient(
    value: Callable[[np.ndarray], float],
    state: Any,
    *,
    scale: Any | None = None,
    relative_step: float = 1.0e-5,
) -> DerivativeRecord:
    """Return the central-difference gradient of one scalar value.

    Every coordinate is stepped by ``relative_step * scale_i`` and the whole
    gradient is recomputed at half that step; the two results are compared so the
    record reports whether the derivative is resolved instead of presenting a
    single unqualified number.

    Raises
    ------
    ValueError
        If the state is not a finite one-dimensional vector or the relative step
        is not finite and positive.

    """
    point = np.asarray(state, dtype=float).reshape(-1)
    if point.size == 0:
        raise ValueError("a finite difference needs at least one coordinate")
    if not np.all(np.isfinite(point)):
        raise ValueError("a finite difference needs a finite state")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError(
            f"relative_step must be finite and positive; got {relative_step}"
        )
    resolved_scale = (
        coordinate_scale(point) if scale is None else np.asarray(scale, dtype=float)
    )
    if resolved_scale.shape != point.shape:
        raise ValueError(
            f"scale has shape {resolved_scale.shape} for a state of shape {point.shape}"
        )
    if not np.all(np.isfinite(resolved_scale)) or np.any(resolved_scale <= 0.0):
        raise ValueError("scale must be finite and strictly positive")

    def gradient_at(step: np.ndarray) -> np.ndarray:
        output = np.empty(point.size)
        for index in range(point.size):
            offset = np.zeros(point.size)
            offset[index] = step[index]
            upper = float(value(point + offset))
            lower = float(value(point - offset))
            output[index] = (upper - lower) / (2.0 * step[index])
        return output

    step = relative_step * resolved_scale
    gradient = gradient_at(step)
    refined = gradient_at(step / 2.0)
    denominator = float(np.linalg.norm(gradient))
    repeat_error = (
        float(np.linalg.norm(gradient - refined) / denominator)
        if denominator > 0.0 and np.all(np.isfinite(gradient))
        else None
    )
    magnitude = np.abs(gradient)
    nonzero = magnitude[magnitude > 0.0]
    # A non-finite gradient has no max/min ratio: computing one would divide infinity
    # by infinity. The record already says the gradient is not finite.
    condition = (
        float(np.max(nonzero) / np.min(nonzero))
        if (
            nonzero.size == magnitude.size
            and nonzero.size
            and np.all(np.isfinite(gradient))
        )
        else None
    )
    return DerivativeRecord(
        method="central_difference",
        order=2,
        gradient=tuple(float(item) for item in gradient),
        step=tuple(float(item) for item in step),
        scale=tuple(float(item) for item in resolved_scale),
        condition=condition,
        repeat_relative_error=repeat_error,
        metadata={
            "relative_step": float(relative_step),
            "repeat_check": "recomputed at half the step",
            "scale_rule": "max(|x_i|, 1)",
        },
    )


@dataclass(frozen=True)
class RayleighObservable:
    """``Re Tr(H R) / Tr(R)`` readout for the carrier frequency.

    The Hamiltonian comes from the model (``cam_hamiltonian``) and the state
    matrix from the adapter, so the observable is model- and mode-agnostic: any
    CAM model that provides a Hamiltonian and any adapter that regroups the state
    vector work. The charge sector is zero because the readout is a frequency.
    """

    model: Any
    name: str = "rayleigh_frequency"
    relative_step: float = 1.0e-5
    identity: ObservableIdentity = field(init=False)

    def __post_init__(self) -> None:
        """Fill the identity and reject a model without a Hamiltonian."""
        if self.model is None:
            raise ValueError("a Rayleigh observable needs a model with a Hamiltonian")
        if not callable(getattr(self.model, "cam_hamiltonian", None)):
            raise TypeError(
                "a Rayleigh observable needs a model that implements "
                "cam_hamiltonian(state, params)"
            )
        object.__setattr__(
            self,
            "identity",
            ObservableIdentity(
                kind=ObservableKind.CARRIER,
                sector=ChargeSector.CHARGE_ZERO,
                name=str(self.name),
                measurement_mapping="Re Tr(H R) / Tr(R)",
            ),
        )

    def value(self, adapter: Any, state: Any, params: Mapping[str, Any]) -> float:
        """Return the Rayleigh frequency at one state."""
        parameter_map = dict(params or {})
        matrix = np.asarray(adapter.state_matrix(state, parameter_map), dtype=complex)
        denominator = np.trace(matrix)
        # A trace at rounding-error scale cannot normalise the readout; refusing
        # is honest, and 1e-300 is the smallest normal double rather than a
        # tuned threshold.
        if abs(denominator) < 1.0e-300:
            raise ValueError(
                "the state matrix has a vanishing trace, so the Rayleigh readout "
                "is undefined at this state"
            )
        hamiltonian = np.asarray(
            self.model.cam_hamiltonian(matrix, parameter_map), dtype=complex
        )
        if hamiltonian.shape != matrix.shape:
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} does not match the state "
                f"matrix shape {matrix.shape}"
            )
        return float(np.real(np.trace(hamiltonian @ matrix) / denominator))

    def values(
        self, adapter: Any, states: Any, params: Mapping[str, Any]
    ) -> np.ndarray:
        """Return the Rayleigh frequency of every row of a state stack.

        One Hamiltonian call for the whole stack when the model can answer a batch,
        one row at a time when it cannot: the product, the trace and the ratio are the
        point readout's, evaluated for every state at once.
        """
        parameter_map = dict(params or {})
        stack = np.asarray(states, dtype=float)
        if stack.ndim != 2:
            raise ValueError(
                f"a state stack must be (n, coordinates); got shape {stack.shape}"
            )
        matrices = _state_matrices(adapter, stack, parameter_map)
        if getattr(self.model, "supports_batched_matrices", False):
            hamiltonian = np.asarray(
                self.model.cam_hamiltonian(matrices, parameter_map), dtype=complex
            )
        else:
            hamiltonian = np.stack(
                [
                    np.asarray(
                        self.model.cam_hamiltonian(
                            np.array(matrix, dtype=complex, copy=True), parameter_map
                        ),
                        dtype=complex,
                    )
                    for matrix in matrices
                ]
            )
        if hamiltonian.shape != matrices.shape:
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} does not match the state "
                f"matrix stack shape {matrices.shape}"
            )
        denominator = np.trace(matrices, axis1=-2, axis2=-1)
        numerator = np.trace(hamiltonian @ matrices, axis1=-2, axis2=-1)
        # A trace at rounding-error scale cannot normalise the readout; the same
        # refusal as the point readout, so one undefined row makes the batch undefined.
        if np.any(np.abs(denominator) < 1.0e-300):
            raise ValueError(
                "the state matrix has a vanishing trace, so the Rayleigh readout "
                "is undefined at this state"
            )
        return np.asarray(np.real(numerator / denominator), dtype=float)

    def gradient(
        self, adapter: Any, state: Any, params: Mapping[str, Any]
    ) -> DerivativeRecord:
        """Return the central-difference gradient of the Rayleigh frequency."""
        return finite_difference_gradient(
            lambda vector: self.value(adapter, vector, params),
            state,
            relative_step=self.relative_step,
        )


@dataclass(frozen=True)
class DiagonalRayleighObservable:
    """``Re (H R)[m, m] / R[m, m]``: the frequency of one mode's amplitude.

    The Hamiltonian comes from the model (``cam_hamiltonian``) and the state matrix
    from the adapter, so the readout is model-agnostic; the only mode-specific input is
    the index ``mode``. The charge sector is zero because the readout is a frequency.

    A mode amplitude can vanish — that is a node of the mode, not a broken state — so a
    diagonal element smaller than ``denominator_floor`` is reported as ``NaN``. With
    ``denominator_floor = 0`` the guard is skipped and the plain division keeps its own
    semantics. Trace readouts use :class:`RayleighObservable`, whose denominator rule
    (refusing a trace at rounding scale) is its own and is *not* shared with this class.
    Errors from the adapter, the Hamiltonian or the model are never turned into ``NaN``.
    """

    model: Any
    mode: int
    name: str | None = None
    relative_step: float = 1.0e-5
    denominator_floor: float = 1.0e-14
    identity: ObservableIdentity = field(init=False)

    def __post_init__(self) -> None:
        """Fill the identity and reject an unusable model, mode or floor."""
        if self.model is None:
            raise ValueError(
                "a diagonal Rayleigh observable needs a model with a Hamiltonian"
            )
        if not callable(getattr(self.model, "cam_hamiltonian", None)):
            raise TypeError(
                "a diagonal Rayleigh observable needs a model that implements "
                "cam_hamiltonian(state, params)"
            )
        mode = int(self.mode)
        if mode < 0:
            raise ValueError(f"the mode index must be non-negative; got {mode}")
        floor = float(self.denominator_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError(
                "denominator_floor must be finite and non-negative; got "
                f"{self.denominator_floor!r}"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "denominator_floor", floor)
        object.__setattr__(
            self,
            "identity",
            ObservableIdentity(
                kind=ObservableKind.CARRIER,
                sector=ChargeSector.CHARGE_ZERO,
                name=(
                    str(self.name)
                    if self.name is not None
                    else f"mode_frequency_{mode}"
                ),
                measurement_mapping=f"Re (H R)[{mode},{mode}] / R[{mode},{mode}]",
            ),
        )

    def _hamiltonian(self, matrix: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the model Hamiltonian of one matrix or one matrix stack."""
        hamiltonian = np.asarray(
            self.model.cam_hamiltonian(matrix, params), dtype=complex
        )
        if hamiltonian.shape != matrix.shape:
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} does not match the state "
                f"matrix shape {matrix.shape}"
            )
        return hamiltonian

    def value(self, adapter: Any, state: Any, params: Mapping[str, Any]) -> float:
        """Return the frequency of one mode's amplitude at one state."""
        parameter_map = dict(params or {})
        matrix = np.asarray(adapter.state_matrix(state, parameter_map), dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"a state matrix must be square; got shape {matrix.shape}")
        if self.mode >= matrix.shape[0]:
            raise ValueError(
                f"the mode index {self.mode} is out of range for a "
                f"{matrix.shape[0]}x{matrix.shape[1]} state matrix"
            )
        hamiltonian = self._hamiltonian(matrix, parameter_map)
        product = hamiltonian @ matrix
        denominator = matrix[self.mode, self.mode]
        if self.denominator_floor > 0.0 and abs(denominator) < self.denominator_floor:
            return float("nan")
        return float(np.real(product[self.mode, self.mode] / denominator))

    def values(
        self, adapter: Any, states: Any, params: Mapping[str, Any]
    ) -> np.ndarray:
        """Return the frequency of one mode's amplitude for every row of a stack.

        One state-matrix call and one Hamiltonian call for the whole stack when the
        adapter and the model can answer a batch, one row at a time when they cannot:
        the quotient and the denominator floor are the point readout's, evaluated for
        every state at once.
        """
        parameter_map = dict(params or {})
        stack = np.asarray(states, dtype=float)
        if stack.ndim != 2:
            raise ValueError(
                f"a state stack must be (n, coordinates); got shape {stack.shape}"
            )
        matrices = _state_matrices(adapter, stack, parameter_map)
        if getattr(self.model, "supports_batched_matrices", False):
            hamiltonian = self._hamiltonian(matrices, parameter_map)
        else:
            hamiltonian = np.stack(
                [
                    self._hamiltonian(
                        np.array(matrix, dtype=complex, copy=True), parameter_map
                    )
                    for matrix in matrices
                ]
            )
        if self.mode >= matrices.shape[-1]:
            raise ValueError(
                f"the mode index {self.mode} is out of range for a "
                f"{matrices.shape[-1]}x{matrices.shape[-2]} state matrix stack"
            )
        product = hamiltonian @ matrices
        denominator = matrices[:, self.mode, self.mode]
        numerator = product[:, self.mode, self.mode]
        with np.errstate(divide="ignore", invalid="ignore"):
            output = np.real(numerator / denominator)
        if self.denominator_floor > 0.0:
            output = np.where(
                np.abs(denominator) < self.denominator_floor, np.nan, output
            )
        return np.asarray(output, dtype=float)

    def gradient(
        self, adapter: Any, state: Any, params: Mapping[str, Any]
    ) -> DerivativeRecord:
        """Return the central-difference gradient of the mode frequency."""
        return finite_difference_gradient(
            lambda vector: self.value(adapter, vector, params),
            state,
            relative_step=self.relative_step,
        )


@dataclass(frozen=True)
class FunctionalObservable:
    """A caller-supplied readout, so other measurements need no new plugin.

    ``gradient_value`` may be omitted; the central difference is then used and
    the record says so. When it is given the record is marked analytic, with no
    step, because no finite difference was evaluated.
    """

    identity: ObservableIdentity
    value_fn: Callable[[Any, Any, Mapping[str, Any]], float]
    gradient_fn: Callable[[Any, Any, Mapping[str, Any]], Any] | None = None
    relative_step: float = 1.0e-5
    values_fn: Callable[[Any, Any, Mapping[str, Any]], Any] | None = None

    def value(self, adapter: Any, state: Any, params: Mapping[str, Any]) -> float:
        """Return the caller's readout at one state."""
        return float(self.value_fn(adapter, state, params))

    def values(
        self, adapter: Any, states: Any, params: Mapping[str, Any]
    ) -> np.ndarray:
        """Return the caller's readout for every row of a state stack.

        Without ``values_fn`` the caller supplied only a point readout, so the stack
        is sampled row by row: the same function, the same arithmetic, no batching.
        """
        if self.values_fn is None:
            return np.asarray(
                [float(self.value_fn(adapter, state, params)) for state in states],
                dtype=float,
            )
        return np.asarray(self.values_fn(adapter, states, params), dtype=float).reshape(
            -1
        )

    def gradient(
        self, adapter: Any, state: Any, params: Mapping[str, Any]
    ) -> DerivativeRecord:
        """Return the caller's gradient, or its central difference."""
        if self.gradient_fn is not None:
            values = np.asarray(
                self.gradient_fn(adapter, state, params), dtype=float
            ).reshape(-1)
            magnitude = np.abs(values)
            nonzero = magnitude[magnitude > 0.0]
            # As above: a non-finite gradient has no condition ratio.
            condition = (
                float(np.max(nonzero) / np.min(nonzero))
                if (
                    nonzero.size == magnitude.size
                    and nonzero.size
                    and np.all(np.isfinite(values))
                )
                else None
            )
            return DerivativeRecord(
                method="analytic",
                order=1,
                gradient=tuple(float(item) for item in values),
                condition=condition,
                metadata={"source": "caller_supplied"},
            )
        return finite_difference_gradient(
            lambda vector: self.value(adapter, vector, params),
            state,
            relative_step=self.relative_step,
        )
