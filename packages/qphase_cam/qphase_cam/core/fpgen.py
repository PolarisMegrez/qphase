"""Validated bridge from fpgen symbolic dynamics to CAM numerical code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from qphase_cam.errors import BifurcationCapabilityError, FPGenCompatibilityError

SUPPORTED_FPGEN_SERIES = (0, 4)
SUPPORTED_MODEL_SCHEMAS = frozenset({"1.1"})
SUPPORTED_STATE_LAYOUT = "hermitian-declared-index-v1"


def _version_series(value: str) -> tuple[int, int]:
    try:
        major, minor, *_ = value.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise FPGenCompatibilityError(
            f"cannot parse fpgen version {value!r}"
        ) from exc


def validate_fpgen_runtime() -> None:
    """Reject fpgen releases outside the reviewed public API contract."""
    import fpgen

    if _version_series(fpgen.__version__) != SUPPORTED_FPGEN_SERIES:
        raise FPGenCompatibilityError(
            "qphase_cam requires the reviewed fpgen 0.4.x API; "
            f"found {fpgen.__version__}"
        )
    if fpgen.MODEL_SCHEMA_VERSION not in SUPPORTED_MODEL_SCHEMAS:
        raise FPGenCompatibilityError(
            f"unsupported fpgen model schema {fpgen.MODEL_SCHEMA_VERSION!r}"
        )
    if fpgen.STATE_LAYOUT_VERSION != SUPPORTED_STATE_LAYOUT:
        raise FPGenCompatibilityError(
            f"unsupported fpgen state layout {fpgen.STATE_LAYOUT_VERSION!r}"
        )


def canonical_state_ids(n_modes: int) -> tuple[str, ...]:
    """Return fpgen IDs in QPhase's canonical Hermitian coordinate order."""
    diagonal = tuple(f"r_diag_{i}" for i in range(n_modes))
    pairs = tuple(
        (i, j) for i in range(n_modes) for j in range(i + 1, n_modes)
    )
    real = tuple(f"r_re_{i}_{j}" for i, j in pairs)
    imag = tuple(f"r_im_{i}_{j}" for i, j in pairs)
    return diagonal + real + imag


@dataclass(frozen=True)
class FPGenDynamicsAdapter:
    """A checked fpgen dynamics object bound to one CAM model instance."""

    model: Any
    dynamics: Any

    @classmethod
    def from_model(cls, model: Any) -> FPGenDynamicsAdapter:
        validate_fpgen_runtime()
        from fpgen import CovarianceDynamics

        provider = getattr(model, "cam_fpgen_dynamics", None)
        if not callable(provider):
            raise BifurcationCapabilityError(
                f"model {getattr(model, 'name', type(model).__name__)!r} does "
                "not provide cam_fpgen_dynamics()"
            )
        dynamics = provider()
        if not isinstance(dynamics, CovarianceDynamics):
            raise BifurcationCapabilityError(
                "cam_fpgen_dynamics() must return fpgen.CovarianceDynamics"
            )
        adapter = cls(model=model, dynamics=dynamics)
        adapter._validate_contract()
        return adapter

    @cached_property
    def spec(self) -> Any:
        return self.dynamics.to_model_spec(name=str(self.model.name))

    @property
    def fingerprint(self) -> str:
        return str(self.spec.fingerprint)

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.dynamics.state_spec)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.dynamics.parameter_spec)

    def parameter_vector(self, params: dict[str, Any] | None = None) -> np.ndarray:
        source = self.model.params if params is None else params
        missing = [name for name in self.parameter_names if name not in source]
        if missing:
            raise BifurcationCapabilityError(
                f"missing fpgen parameters for {self.model.name}: {missing}"
            )
        return np.asarray([source[name] for name in self.parameter_names], dtype=float)

    @cached_property
    def numpy_functions(self) -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        exec(self.spec.numpy_source(), namespace)
        return namespace

    def rhs(self, vector: Any, params: dict[str, Any] | None = None) -> np.ndarray:
        return np.asarray(
            self.numpy_functions["rhs"](vector, self.parameter_vector(params))
        )

    def jacobian(
        self, vector: Any, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        return np.asarray(
            self.numpy_functions["jacobian"](
                vector, self.parameter_vector(params)
            )
        )

    def parameter_jacobian(
        self, vector: Any, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        return np.asarray(
            self.numpy_functions["parameter_jacobian"](
                vector, self.parameter_vector(params)
            )
        )

    def directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> np.ndarray:
        return self.derivative_evaluator.directional(
            order, vector, params, *directions
        )

    def mpmath_rhs(self, vector: Any, params: dict[str, Any]) -> Any:
        return self.derivative_evaluator.mpmath_rhs(vector, params)

    def mpmath_jacobian(self, vector: Any, params: dict[str, Any]) -> Any:
        return self.derivative_evaluator.mpmath_jacobian(vector, params)

    def mpmath_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> Any:
        return self.derivative_evaluator.mpmath_directional(
            order, vector, params, *directions
        )

    @cached_property
    def derivative_evaluator(self) -> FPGenDerivativeEvaluator:
        return FPGenDerivativeEvaluator(self)

    def provenance(self) -> dict[str, Any]:
        return {
            "fpgen_version": self._fpgen_version(),
            "model_schema": self.spec.schema_version,
            "state_layout": self.spec.provenance.state_layout_version,
            "fingerprint": self.fingerprint,
            "derivation": self.spec.provenance.manifest(),
        }

    def _validate_contract(self) -> None:
        expected = canonical_state_ids(int(self.model.n_modes))
        actual = tuple(item.id for item in self.dynamics.state_spec)
        if actual != expected:
            raise FPGenCompatibilityError(
                f"model {self.model.name!r} state layout mismatch: "
                f"expected {expected}, found {actual}"
            )
        if len(actual) != int(self.model.n_modes) ** 2:
            raise FPGenCompatibilityError(
                f"model {self.model.name!r} has an invalid CAM state size"
            )
        parameter_names = tuple(
            item.name for item in self.dynamics.parameter_spec
        )
        if len(set(parameter_names)) != len(parameter_names):
            raise FPGenCompatibilityError("fpgen parameter names are not unique")
        model_names = set(self.model.params)
        if set(parameter_names) != model_names:
            raise FPGenCompatibilityError(
                f"model {self.model.name!r} parameter mismatch: fpgen has "
                f"{sorted(parameter_names)}, model has {sorted(model_names)}"
            )
        if self.dynamics.state_layout_version != SUPPORTED_STATE_LAYOUT:
            raise FPGenCompatibilityError(
                f"model {self.model.name!r} uses unsupported state layout "
                f"{self.dynamics.state_layout_version!r}"
            )

    @staticmethod
    def _fpgen_version() -> str:
        import fpgen

        return str(fpgen.__version__)


class FPGenDerivativeEvaluator:
    """Compile exact fpgen directional contractions on demand."""

    def __init__(self, adapter: FPGenDynamicsAdapter) -> None:
        self.adapter = adapter
        self._functions: dict[tuple[int, str], Any] = {}

    def directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> np.ndarray:
        if len(directions) != order:
            raise ValueError("direction count must equal derivative order")
        key = (order, "numpy")
        function = self._functions.get(key)
        if function is None:
            function = self._compile(order, "numpy")
            self._functions[key] = function
        state = np.asarray(vector, dtype=float).reshape(-1)
        parameter_values = self.adapter.parameter_vector(params)
        direction_values = [
            item
            for direction in directions
            for item in np.asarray(direction, dtype=float).reshape(-1)
        ]
        return np.asarray(
            function(*state, *parameter_values, *direction_values),
            dtype=float,
        ).reshape(-1)

    def mpmath_rhs(self, vector: Any, params: dict[str, Any]) -> Any:
        return self._mp_rhs(
            *vector, *self._mp_parameter_values(params)
        )

    def mpmath_jacobian(self, vector: Any, params: dict[str, Any]) -> Any:
        return self._mp_jacobian(
            *vector, *self._mp_parameter_values(params)
        )

    def mpmath_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> Any:
        if len(directions) != order:
            raise ValueError("direction count must equal derivative order")
        key = (order, "mpmath")
        function = self._functions.get(key)
        if function is None:
            function = self._compile(order, "mpmath")
            self._functions[key] = function
        return function(
            *vector,
            *self._mp_parameter_values(params),
            *(item for direction in directions for item in direction),
        )

    @cached_property
    def _mp_rhs(self) -> Any:
        import sympy as sp

        dynamics = self.adapter.dynamics
        arguments = (
            *dynamics.coordinates,
            *(item.symbol for item in dynamics.parameter_spec),
        )
        return sp.lambdify(arguments, dynamics.rhs, modules="mpmath")

    @cached_property
    def _mp_jacobian(self) -> Any:
        import sympy as sp

        dynamics = self.adapter.dynamics
        arguments = (
            *dynamics.coordinates,
            *(item.symbol for item in dynamics.parameter_spec),
        )
        return sp.lambdify(arguments, dynamics.jacobian(), modules="mpmath")

    def _mp_parameter_values(self, params: dict[str, Any]) -> tuple[Any, ...]:
        import mpmath as mp

        return tuple(
            mp.mpf(str(params[name])) for name in self.adapter.parameter_names
        )

    def _compile(self, order: int, modules: str) -> Any:
        import sympy as sp

        dynamics = self.adapter.dynamics
        n_state = len(dynamics.coordinates)
        direction_symbols = tuple(
            sp.Matrix(
                sp.symbols(f"_d{slot}_0:{n_state}", real=True)
            )
            for slot in range(order)
        )
        expression = dynamics.directional(
            order, state_directions=direction_symbols
        )
        arguments = (
            *dynamics.coordinates,
            *(item.symbol for item in dynamics.parameter_spec),
            *(item for direction in direction_symbols for item in direction),
        )
        return sp.lambdify(arguments, expression, modules=modules)
