"""Validated bridge from fpgen symbolic dynamics to CAM numerical code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

import numpy as np

from qphase_cam.errors import BifurcationCapabilityError, FPGenCompatibilityError

SUPPORTED_FPGEN_SERIES = (0, 5)
SUPPORTED_MODEL_SCHEMAS = frozenset({"2.0"})
SUPPORTED_MOMENT_API = "1.0"
SUPPORTED_REDUCTION_API = "1.0"
SUPPORTED_STATE_LAYOUTS = frozenset(
    {
        "hermitian-declared-index-v1",
        "hermitian-normal-anomalous-declared-index-v2",
    }
)


class FPGenReductionSearchProtocol(Protocol):
    """Narrow reduction-search surface consumed by qphase_cam."""

    candidates: tuple[Any, ...]

    def manifest(self) -> dict[str, Any]: ...


class FPGenLinearReductionProtocol(Protocol):
    """Narrow linear-reduction plan consumed by qphase_cam."""

    retained_symbols: tuple[Any, ...]

    def materialize(self, *, method: str = "fraction_free") -> Any: ...


class FPGenCovarianceDynamicsProtocol(Protocol):
    """Reviewed fpgen public surface hidden behind the CAM adapter."""

    coordinates: tuple[Any, ...]
    parameter_spec: tuple[Any, ...]
    hamiltonian: Any
    rhs: Any

    def to_model_spec(self, *, name: str) -> Any: ...

    def jacobian(self) -> Any: ...

    def directional(
        self,
        order: int,
        *,
        state_directions: tuple[Any, ...] = (),
        parameter_directions: tuple[Any, ...] = (),
    ) -> Any: ...

    def search_linear_reductions(
        self, **kwargs: Any
    ) -> FPGenReductionSearchProtocol: ...

    def linear_reduce(
        self, *, candidate: Any = None, order_parameters: Any = None
    ) -> FPGenLinearReductionProtocol: ...


def _version_series(value: str) -> tuple[int, int]:
    try:
        major, minor, *_ = value.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise FPGenCompatibilityError(f"cannot parse fpgen version {value!r}") from exc


def validate_fpgen_runtime() -> None:
    """Reject fpgen releases outside the reviewed public API contract."""
    import fpgen

    if _version_series(fpgen.__version__) != SUPPORTED_FPGEN_SERIES:
        raise FPGenCompatibilityError(
            "qphase_cam requires the reviewed fpgen 0.5.x API; "
            f"found {fpgen.__version__}"
        )
    if fpgen.MODEL_SCHEMA_VERSION not in SUPPORTED_MODEL_SCHEMAS:
        raise FPGenCompatibilityError(
            f"unsupported fpgen model schema {fpgen.MODEL_SCHEMA_VERSION!r}"
        )
    if fpgen.MOMENT_DYNAMICS_API_VERSION != SUPPORTED_MOMENT_API:
        raise FPGenCompatibilityError(
            f"unsupported fpgen moment API {fpgen.MOMENT_DYNAMICS_API_VERSION!r}"
        )
    if fpgen.REDUCTION_API_VERSION != SUPPORTED_REDUCTION_API:
        raise FPGenCompatibilityError(
            f"unsupported fpgen reduction API {fpgen.REDUCTION_API_VERSION!r}"
        )


def canonical_state_ids(n_modes: int) -> tuple[str, ...]:
    """Return fpgen IDs in QPhase's canonical Hermitian coordinate order."""
    diagonal = tuple(f"r_diag_{i}" for i in range(n_modes))
    pairs = tuple((i, j) for i in range(n_modes) for j in range(i + 1, n_modes))
    real = tuple(f"r_re_{i}_{j}" for i, j in pairs)
    imag = tuple(f"r_im_{i}_{j}" for i, j in pairs)
    return diagonal + real + imag


@dataclass(frozen=True)
class FPGenDynamicsAdapter:
    """A checked fpgen dynamics object bound to one CAM model instance."""

    model: Any
    _dynamics: FPGenCovarianceDynamicsProtocol

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
        adapter = cls(model=model, _dynamics=dynamics)
        adapter._validate_contract()
        return adapter

    @cached_property
    def spec(self) -> Any:
        return self._dynamics.to_model_spec(name=str(self.model.name))

    @property
    def state_symbols(self) -> tuple[Any, ...]:
        return tuple(self._dynamics.coordinates)

    @property
    def parameter_symbols(self) -> tuple[Any, ...]:
        return tuple(item.symbol for item in self._dynamics.parameter_spec)

    @property
    def symbolic_hamiltonian(self) -> Any:
        return self._dynamics.hamiltonian

    @property
    def symbolic_rhs(self) -> Any:
        return self._dynamics.rhs

    def symbolic_jacobian(self) -> Any:
        return self._dynamics.jacobian()

    def symbolic_directional(
        self,
        order: int,
        *,
        state_directions: tuple[Any, ...] = (),
        parameter_directions: tuple[Any, ...] = (),
    ) -> Any:
        return self._dynamics.directional(
            order,
            state_directions=state_directions,
            parameter_directions=parameter_directions,
        )

    def search_linear_reductions(self, **kwargs: Any) -> FPGenReductionSearchProtocol:
        return self._dynamics.search_linear_reductions(**kwargs)

    def linear_reduction(self, *, candidate: Any) -> FPGenLinearReductionProtocol:
        return self._dynamics.linear_reduce(candidate=candidate)

    def materialized_linear_reduction(self, *, candidate: Any) -> Any:
        return self.linear_reduction(candidate=candidate).materialize()

    @property
    def fingerprint(self) -> str:
        return str(self.spec.fingerprint)

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.spec.state)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.spec.parameters)

    @property
    def parameter_domains(self) -> dict[str, str]:
        return {item.name: str(item.domain) for item in self.spec.parameters}

    @property
    def state_size(self) -> int:
        return self.spec.state_size

    @property
    def state_layout(self) -> str:
        return str(self.spec.provenance.state_layout_version)

    @property
    def moment_layout(self) -> str:
        return str(self.spec.provenance.moment_layout)

    @property
    def state_matrix_shape(self) -> tuple[int, int]:
        shape = self.spec.state_matrix_shape
        if shape is None:
            raise FPGenCompatibilityError("fpgen model has no state matrix")
        return shape

    @property
    def diagonal_state_indices(self) -> tuple[int, ...]:
        return tuple(int(item.index) for item in self.spec.state if item.kind == "diag")

    def parameter_vector(self, params: dict[str, Any] | None = None) -> np.ndarray:
        source = self.model.params if params is None else params
        missing = [name for name in self.parameter_names if name not in source]
        if missing:
            raise BifurcationCapabilityError(
                f"missing fpgen parameters for {self.model.name}: {missing}"
            )
        return np.asarray([source[name] for name in self.parameter_names], dtype=float)

    @cached_property
    def compiled(self) -> Any:
        return self.spec.compile_numpy()

    def rhs(self, vector: Any, params: dict[str, Any] | None = None) -> np.ndarray:
        return np.asarray(self.compiled.rhs(vector, self.parameter_vector(params)))

    def jacobian(self, vector: Any, params: dict[str, Any] | None = None) -> np.ndarray:
        return np.asarray(self.compiled.jacobian(vector, self.parameter_vector(params)))

    def state_matrix(
        self, vector: Any, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        return np.asarray(
            self.compiled.state_matrix(vector, self.parameter_vector(params)),
            dtype=complex,
        )

    def physical_eigenvalues(
        self, vector: Any, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        if self.moment_layout not in {"normal", "augmented"}:
            raise BifurcationCapabilityError(
                f"moment layout {self.moment_layout!r} has no Hermitian PSD domain"
            )
        matrix = self.state_matrix(vector, params)
        return np.linalg.eigvalsh((matrix + matrix.conjugate().T) / 2.0)

    def parameter_jacobian(
        self, vector: Any, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        return np.asarray(
            self.compiled.parameter_jacobian(vector, self.parameter_vector(params))
        )

    def directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> np.ndarray:
        return self.derivative_evaluator.directional(order, vector, params, *directions)

    def mixed_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *,
        state_directions: tuple[Any, ...] = (),
        parameter_directions: tuple[Any, ...] = (),
    ) -> np.ndarray:
        return self.derivative_evaluator.mixed_directional(
            order,
            vector,
            params,
            state_directions=state_directions,
            parameter_directions=parameter_directions,
        )

    def parameter_direction(self, name: str, scale: float = 1.0) -> np.ndarray:
        try:
            index = self.parameter_names.index(name)
        except ValueError as exc:
            raise BifurcationCapabilityError(
                f"unknown fpgen parameter {name!r}"
            ) from exc
        direction = np.zeros(len(self.parameter_names), dtype=float)
        direction[index] = float(scale)
        return direction

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

    def mpmath_mixed_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *,
        state_directions: tuple[Any, ...] = (),
        parameter_directions: tuple[Any, ...] = (),
    ) -> Any:
        return self.derivative_evaluator.mpmath_mixed_directional(
            order,
            vector,
            params,
            state_directions=state_directions,
            parameter_directions=parameter_directions,
        )

    @cached_property
    def derivative_evaluator(self) -> FPGenDerivativeEvaluator:
        return FPGenDerivativeEvaluator(self)

    def provenance(self) -> dict[str, Any]:
        return {
            "fpgen_version": self._fpgen_version(),
            "model_schema": self.spec.schema_version,
            "moment_api": SUPPORTED_MOMENT_API,
            "reduction_api": SUPPORTED_REDUCTION_API,
            "state_layout": self.spec.provenance.state_layout_version,
            "matrix_semantics": self.spec.matrix_semantics,
            "physical_domain_hint": self.spec.physical_domain_hint,
            "fingerprint": self.fingerprint,
            "derivation": self.spec.provenance.manifest(),
        }

    def closure_provenance(self) -> dict[str, Any]:
        """Return a compact exactness summary for downstream diagnostics."""
        derivation = dict(self.spec.provenance.manifest())
        closure = derivation.get("moment_closure")
        closure_exact = closure in (None, "none", "exact")
        fpe_exact = bool(derivation.get("fpe_is_exact", False))
        warnings = []
        if not fpe_exact:
            warnings.append("phase_space_fpe_truncated")
        if not closure_exact:
            warnings.append("moment_hierarchy_factorized")
        return {
            "representation": derivation.get("representation", "unknown"),
            "original_kramers_moyal_order": derivation.get(
                "original_kramers_moyal_order"
            ),
            "fpe_truncation_order": derivation.get("fpe_truncation_order"),
            "discarded_term_count": int(derivation.get("discarded_term_count", 0)),
            "fpe_is_exact": fpe_exact,
            "moment_closure": closure,
            "moment_closure_is_exact": closure_exact,
            "deterministic_cam_is_exact": fpe_exact and closure_exact,
            "warnings": tuple(warnings),
        }

    def _validate_contract(self) -> None:
        actual = tuple(item.id for item in self.spec.state)
        if not self.spec.supports("state_matrix"):
            raise FPGenCompatibilityError("fpgen model has no state-matrix capability")
        if self.state_layout not in SUPPORTED_STATE_LAYOUTS:
            raise FPGenCompatibilityError(
                f"unsupported fpgen state layout {self.state_layout!r}"
            )
        if tuple(item.index for item in self.spec.state) != tuple(range(len(actual))):
            raise FPGenCompatibilityError(
                f"model {self.model.name!r} has non-contiguous state indices"
            )
        if self.moment_layout == "normal":
            expected = canonical_state_ids(int(self.model.n_modes))
            if actual != expected:
                raise FPGenCompatibilityError(
                    f"model {self.model.name!r} state layout mismatch: "
                    f"expected {expected}, found {actual}"
                )
        parameter_names = tuple(item.name for item in self.spec.parameters)
        if len(set(parameter_names)) != len(parameter_names):
            raise FPGenCompatibilityError("fpgen parameter names are not unique")
        model_names = set(self.model.params)
        if set(parameter_names) != model_names:
            raise FPGenCompatibilityError(
                f"model {self.model.name!r} parameter mismatch: fpgen has "
                f"{sorted(parameter_names)}, model has {sorted(model_names)}"
            )

    @staticmethod
    def _fpgen_version() -> str:
        import fpgen

        return str(fpgen.__version__)


class FPGenDerivativeEvaluator:
    """Compile exact fpgen directional contractions on demand."""

    def __init__(self, adapter: FPGenDynamicsAdapter) -> None:
        self.adapter = adapter
        self._functions: dict[tuple[int, int, str], Any] = {}

    def directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> np.ndarray:
        return self.mixed_directional(
            order,
            vector,
            params,
            state_directions=tuple(directions),
        )

    def mixed_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *,
        state_directions: tuple[Any, ...] = (),
        parameter_directions: tuple[Any, ...] = (),
    ) -> np.ndarray:
        if len(state_directions) + len(parameter_directions) != order:
            raise ValueError("direction count must equal derivative order")
        key = (len(state_directions), len(parameter_directions), "numpy")
        function = self._functions.get(key)
        if function is None:
            function = self._compile(
                len(state_directions), len(parameter_directions), "numpy"
            )
            self._functions[key] = function
        state = np.asarray(vector, dtype=float).reshape(-1)
        parameter_values = self.adapter.parameter_vector(params)
        state_direction_values = [
            item
            for direction in state_directions
            for item in np.asarray(direction, dtype=float).reshape(-1)
        ]
        parameter_direction_values = [
            item
            for direction in parameter_directions
            for item in np.asarray(direction, dtype=float).reshape(-1)
        ]
        return np.asarray(
            function(
                *state,
                *parameter_values,
                *state_direction_values,
                *parameter_direction_values,
            ),
            dtype=float,
        ).reshape(-1)

    def mpmath_rhs(self, vector: Any, params: dict[str, Any]) -> Any:
        return self._mp_rhs(*vector, *self._mp_parameter_values(params))

    def mpmath_jacobian(self, vector: Any, params: dict[str, Any]) -> Any:
        return self._mp_jacobian(*vector, *self._mp_parameter_values(params))

    def mpmath_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> Any:
        return self.mpmath_mixed_directional(
            order,
            vector,
            params,
            state_directions=tuple(directions),
        )

    def mpmath_mixed_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *,
        state_directions: tuple[Any, ...] = (),
        parameter_directions: tuple[Any, ...] = (),
    ) -> Any:
        if len(state_directions) + len(parameter_directions) != order:
            raise ValueError("direction count must equal derivative order")
        key = (len(state_directions), len(parameter_directions), "mpmath")
        function = self._functions.get(key)
        if function is None:
            function = self._compile(
                len(state_directions), len(parameter_directions), "mpmath"
            )
            self._functions[key] = function
        return function(
            *vector,
            *self._mp_parameter_values(params),
            *(item for direction in state_directions for item in direction),
            *(item for direction in parameter_directions for item in direction),
        )

    @cached_property
    def _mp_rhs(self) -> Any:
        import sympy as sp

        arguments = (*self.adapter.state_symbols, *self.adapter.parameter_symbols)
        return sp.lambdify(arguments, self.adapter.symbolic_rhs, modules="mpmath")

    @cached_property
    def _mp_jacobian(self) -> Any:
        import sympy as sp

        arguments = (*self.adapter.state_symbols, *self.adapter.parameter_symbols)
        return sp.lambdify(
            arguments, self.adapter.symbolic_jacobian(), modules="mpmath"
        )

    def _mp_parameter_values(self, params: dict[str, Any]) -> tuple[Any, ...]:
        import mpmath as mp

        return tuple(mp.mpf(str(params[name])) for name in self.adapter.parameter_names)

    def _compile(self, state_order: int, parameter_order: int, modules: str) -> Any:
        import sympy as sp

        n_state = len(self.adapter.state_symbols)
        direction_symbols = tuple(
            sp.Matrix(sp.symbols(f"_d{slot}_0:{n_state}", real=True))
            for slot in range(state_order)
        )
        parameter_direction_symbols = tuple(
            sp.Matrix(
                sp.symbols(
                    f"_p{slot}_0:{len(self.adapter.parameter_symbols)}", real=True
                )
            )
            for slot in range(parameter_order)
        )
        expression = self.adapter.symbolic_directional(
            state_order + parameter_order,
            state_directions=direction_symbols,
            parameter_directions=parameter_direction_symbols,
        )
        arguments = (
            *self.adapter.state_symbols,
            *self.adapter.parameter_symbols,
            *(item for direction in direction_symbols for item in direction),
            *(item for direction in parameter_direction_symbols for item in direction),
        )
        return sp.lambdify(arguments, expression, modules=modules)
