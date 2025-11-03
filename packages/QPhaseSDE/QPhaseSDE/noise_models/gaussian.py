"""
QPhaseSDE: Gaussian Noise Models
--------------------------------
Backend-agnostic Gaussian white-noise generators supporting independent and
correlated channels with per-step increments for SDE integrators.

Behavior
--------
- Use backend RNG and array ops; correlated noise is produced via a cached
    Cholesky factorization of the covariance when provided by the spec.

Notes
-----
- Dimension, basis, and covariance semantics are defined by NoiseSpec and
    documented at class/method level.
"""

from typing import Any, Sequence
from ..core.protocols import BackendBase as Backend
from ..core.protocols import NoiseSpec
from ..core.errors import QPSConfigError, QPSModelError
from ..core.registry import register as _register_decorator

__all__ = [
	"GaussianNoiseModel",
]

@_register_decorator("noise_model", "gaussian")
class GaussianNoiseModel:
    """Backend-agnostic Gaussian white-noise model supporting independent and correlated noise.

    This class implements multi-channel Gaussian noise generation for SDEs,
    supporting both independent and correlated noise via a covariance matrix.
    It is backend-agnostic and uses the provided backend for RNG, stacking,
    and contractions. For correlated noise, the Cholesky factor of the covariance
    is cached for efficient sampling.

    Parameters
    ----------
    spec : NoiseSpec
        Noise specification dataclass (kind, dim, covariance, etc.).
    backend : BackendBase
        Backend instance providing array operations and random number generation.

    Attributes
    ----------
    spec : NoiseSpec
        Noise specification used for this model.
    backend : BackendBase
        Backend used for all array and RNG operations.
    _chol : Any
        Cholesky factor of the covariance matrix (for correlated noise).
    _correlate : Callable or None
        Fast-path correlation function (backend-specific).

    Methods
    -------
    sample(rng, n_traj, dt)
        Draw one step of Wiener increments with variance dt using backend RNG.

    Examples
    --------
    >>> from QPhaseSDE.noise_models.gaussian import GaussianNoiseModel
    >>> model = GaussianNoiseModel(spec, backend)
    >>> dW = model.sample(rng, n_traj=128, dt=0.01)
    """
    def __init__(self, spec: NoiseSpec, backend: Backend):
        """Initialize the Gaussian noise model with spec and backend."""
        self.spec = spec
        self.backend = backend
        self._chol = None
        self._correlate = None  # set to a function (z, chol_T) -> correlated z
        if spec.kind == "correlated":
            if spec.covariance is None:
                raise QPSConfigError("[599] Covariance must be provided for correlated noise")
            # Ensure covariance is on the backend
            C = backend.asarray(spec.covariance, dtype=float)
            L = backend.cholesky(C)
            self._chol = L
            try:
                # Cache L^T to avoid per-step transpose
                self._chol_T = L.T  # type: ignore[attr-defined]
            except Exception:
                self._chol_T = None
            # Choose fast-path correlate function based on backend capabilities
            try:
                be_name = str(backend.backend_name()).lower()
            except Exception:
                be_name = ""
            if be_name == "torch":
                try:
                    import torch as _th  # type: ignore
                    self._correlate = lambda z, chol_T: z @ chol_T  # (T,M)@(M,K)->(T,K)
                except Exception:
                    self._correlate = None
            else:
                self._correlate = None

    def sample(self, rng: Any, n_traj: int, dt: float):
        """Draw one step of Wiener increments with variance ``dt`` using backend RNG.

        For independent noise, samples are drawn directly. For correlated noise,
        samples are transformed using the Cholesky factor of the covariance matrix.
        Supports per-trajectory RNGs for independent streams.

        Parameters
        ----------
        rng : Any
            Backend RNG instance or a sequence of per-trajectory RNGs.
        n_traj : int
            Number of trajectories to sample.
        dt : float
            Time step size; variance of increments is ``dt``.

        Returns
        -------
        Any
            Array of shape ``(n_traj, dim)`` with sampled increments.

        Raises
        ------
        QPSModelError
            - [599] Covariance must be provided for correlated noise.
            - [600] Failed to assemble per-trajectory noise rows.

        Examples
        --------
        >>> dW = model.sample(rng, n_traj=128, dt=0.01)
        """
        M = int(self.spec.dim)
        # If rng is a sequence, sample per-trajectory for independent streams
        if isinstance(rng, (list, tuple)):
            rows = []
            for r in rng:  # type: ignore[assignment]
                rows.append(self.backend.randn(r, (M,), dtype=float))
            # Prefer backend.stack if available
            try:
                z = self.backend.stack(tuple(rows), axis=0)  # type: ignore[attr-defined]
            except Exception:
                # Try domain libraries by dtype
                try:
                    import torch as _th  # type: ignore
                    if rows and isinstance(rows[0], _th.Tensor):
                        z = _th.stack(rows, dim=0)
                        z = z.to(dtype=_th.float64)
                    else:
                        raise RuntimeError
                except Exception:
                    try:
                        import cupy as _cp  # type: ignore
                        if rows and isinstance(rows[0], _cp.ndarray):  # type: ignore[attr-defined]
                            z = _cp.stack(rows, axis=0)
                            z = z.astype(_cp.float64)
                        else:
                            raise RuntimeError
                    except Exception:
                        try:
                            import numpy as _np
                            z = _np.stack(rows, axis=0).astype(_np.float64, copy=False)
                        except Exception as e:
                            # Last resort: backend.asarray
                            try:
                                z = self.backend.asarray(rows, dtype=float)
                            except Exception:
                                raise QPSModelError(f"[600] Failed to assemble per-trajectory noise rows: {e}")
        else:
            z = self.backend.randn(rng, (n_traj, M), dtype=float)
        if self.spec.kind == "independent":
            return (dt ** 0.5) * z
        assert self._chol is not None
        chol_T = getattr(self, '_chol_T', None)
        if chol_T is None:
            chol_T = self._chol.T  # type: ignore[attr-defined]
        if self._correlate is not None:
            try:
                zk = self._correlate(z, chol_T)
                return (dt ** 0.5) * zk
            except Exception:
                pass
        return (dt ** 0.5) * self.backend.einsum('tm,mk->tk', z, chol_T)
