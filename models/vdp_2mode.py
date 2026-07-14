"""Van der Pol Oscillator (Itô SDE) Plugin
-----------------------------------------------------
Defines the VDP model directly as a DiffusiveSDEModel.
Implemented as a QPhase Plugin.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel

from .kernels.base import ModelKernelRegistry
from .kernels.cayley_maruyama import VDP2ModeCayleyCuPyKernel
from .kernels.euler_maruyama import VDP2ModeEulerCuPyKernel


class VDPLevel3Config(BaseModel):
    """Configuration for VDP Level 3 Model.

    Fields accept a scalar or a one-dimensional array so that parameter scans
    can be fused into a single batched simulation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    omega_a: Any = Field(
        description="Frequency of mode a (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )
    omega_b: Any = Field(
        description="Frequency of mode b (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )
    gamma_a: Any = Field(
        description="Damping rate of mode a (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )
    gamma_b: Any = Field(
        description="Damping rate of mode b (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )
    Gamma: Any = Field(
        description="Nonlinear gain coefficient (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )
    g: Any = Field(
        description="Coupling strength between modes (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )
    D: Any = Field(
        default=1.0,
        description="Diffusion coefficient (scalar or 1-D array)",
        json_schema_extra={"scanable": True},
    )


class VDPLevel3Model:
    """Van der Pol Oscillator - Level 3 (SDE Model).

    This class implements the Plugin protocol and the SDEModel protocol.
    """

    name: ClassVar[str] = "vdp_2mode"
    description: ClassVar[str] = "Van der Pol Oscillator (SDE Model)"
    config_schema: ClassVar[type[VDPLevel3Config]] = VDPLevel3Config

    def __init__(self, config: VDPLevel3Config | None = None, **kwargs: Any) -> None:
        if config is None:
            config = VDPLevel3Config(**kwargs)
        self.config = config
        self._params = config.model_dump()
        self._kernel_registry = ModelKernelRegistry()
        self._kernel_registry.register(VDP2ModeEulerCuPyKernel())
        self._kernel_registry.register(VDP2ModeCayleyCuPyKernel())

    @property
    def n_modes(self) -> int:
        return 2

    @property
    def noise_basis(self) -> str:
        return "complex"

    @property
    def noise_dim(self) -> int:
        return 4

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def drift(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Drift Vector."""
        xp = get_xp(y)
        alpha = y[:, 0]
        beta = y[:, 1]

        # Convert array-like parameters to the active backend. This is required
        # for batched parameter scans where the scheduler broadcasts scan values
        # as NumPy arrays and the active backend may be CuPy.
        def _param(name: str) -> Any:
            val = p[name]
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                return xp.asarray(val)
            return val

        omega_a = _param("omega_a")
        omega_b = _param("omega_b")
        gamma_a = _param("gamma_a")
        gamma_b = _param("gamma_b")
        Gamma = _param("Gamma")
        g = _param("g")

        dalpha = (
            (-1j * omega_a) + (gamma_a / 2.0) + Gamma * (1.0 - xp.abs(alpha) ** 2)
        ) * alpha - 1j * g * beta
        dbeta = ((-1j * omega_b) - (gamma_b / 2.0)) * beta - 1j * g * alpha

        out = xp.empty_like(y)
        out[:, 0] = dalpha
        out[:, 1] = dbeta
        return out

    def drift_matrix(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Return the state-dependent matrix ``A`` satisfying ``drift=A@y``."""
        del t
        xp = get_xp(y)
        alpha = y[:, 0]

        def _param(name: str) -> Any:
            val = p[name]
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                return xp.asarray(val)
            return val

        omega_a = _param("omega_a")
        omega_b = _param("omega_b")
        gamma_a = _param("gamma_a")
        gamma_b = _param("gamma_b")
        Gamma = _param("Gamma")
        g = _param("g")

        matrix = xp.zeros((y.shape[0], 2, 2), dtype=y.dtype)
        matrix[:, 0, 0] = (
            gamma_a / 2.0 + Gamma * (1.0 - xp.abs(alpha) ** 2) - 1j * omega_a
        )
        matrix[:, 0, 1] = -1j * g
        matrix[:, 1, 0] = -1j * g
        matrix[:, 1, 1] = -gamma_b / 2.0 - 1j * omega_b
        return matrix

    def diffusion(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Matrix."""
        xp = get_xp(y)
        alpha = y[:, 0]

        def _param(name: str) -> Any:
            val = p[name]
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                return xp.asarray(val)
            return val

        gamma_a = _param("gamma_a")
        gamma_b = _param("gamma_b")
        Gamma = _param("Gamma")
        D = _param("D")

        D_alpha = D * (gamma_a / 2.0 + Gamma * (2.0 * xp.abs(alpha) ** 2 - 1.0))
        D_beta = D * (gamma_b / 2.0)

        n = y.shape[0]
        # Use y.real.dtype to ensure we match the precision of the state
        # (e.g. float32 for complex64, float64 for complex128)
        rdtype = y.real.dtype

        if not hasattr(D_alpha, "shape") or D_alpha.shape != (n,):
            D_alpha = xp.full((n,), float(D_alpha), dtype=rdtype)
        if not hasattr(D_beta, "shape") or D_beta.shape != (n,):
            D_beta = xp.full((n,), float(D_beta), dtype=rdtype)

        # Ensure D_alpha/D_beta are cast to the correct real dtype
        # (In case they were promoted to float64 by scalar ops)
        if hasattr(xp, "asarray"):
            D_alpha = xp.asarray(D_alpha, dtype=rdtype)
            D_beta = xp.asarray(D_beta, dtype=rdtype)

        D_alpha = xp.clip(D_alpha, 0.0, None)
        D_beta = xp.clip(D_beta, 0.0, None)

        Lc = xp.zeros((n, 2, 2), dtype=y.dtype)
        Lc[:, 0, 0] = xp.sqrt(D_alpha)
        Lc[:, 1, 1] = xp.sqrt(D_beta)

        return Lc

    def has_kernelized_terms(self, backend: Any) -> bool:
        """Kernel path is available for the CuPy backend."""
        return self._kernel_registry.supports(
            "euler_maruyama", backend, operation="terms"
        )

    def kernelized_terms(
        self, y: Any, t: float, p: dict[str, Any], backend: Any
    ) -> tuple[Any, Any]:
        """Fused drift+diffusion kernel for CuPy."""
        terms = self._kernel_registry.resolve(
            "euler_maruyama", backend, operation="terms"
        )
        return terms(y, p, backend)

    def supports_fused_step(self, scheme: str, backend: Any) -> bool:
        """Return whether a model-local fused step exists for the scheme."""
        return self._kernel_registry.supports(scheme, backend, operation="step")

    def fused_step(
        self,
        scheme: str,
        y: Any,
        t: float,
        dt: float,
        p: dict[str, Any],
        noise: Any,
        backend: Any,
    ) -> Any:
        """Dispatch a complete integrator step to a model-local kernel."""
        step = self._kernel_registry.resolve(scheme, backend, operation="step")
        return step(y, t, dt, p, noise, backend)

    def to_diffusive_sde_model(self) -> FunctionalSDEModel:
        """Convert to the standard FunctionalSDEModel dataclass."""
        return FunctionalSDEModel(
            name=self.name,
            n_modes=self.n_modes,
            noise_basis=self.noise_basis,
            noise_dim=self.noise_dim,
            params=self.params,
            drift=self.drift,
            diffusion=self.diffusion,
            drift_matrix=self.drift_matrix,
        )
