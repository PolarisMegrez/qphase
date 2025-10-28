from __future__ import annotations

"""Gaussian noise models implemented against the Backend protocol."""

from typing import Any, Sequence

from ..core.protocols import BackendBase as Backend
from ..core.protocols import NoiseSpec
from ..core.errors import NoiseModelError
from ..core.registry import register as _register_decorator


@_register_decorator("noise_model", "gaussian")
class GaussianNoiseModel:
    """Backend-agnostic Gaussian white-noise model (independent/correlated)."""
    def __init__(self, spec: NoiseSpec, backend: Backend):
        self.spec = spec
        self.backend = backend
        self._chol = None
        if spec.kind == "correlated":
            if spec.covariance is None:
                raise NoiseModelError("[400] Covariance must be provided for correlated noise")
            # Ensure covariance is on the backend
            C = backend.asarray(spec.covariance, dtype=float)
            self._chol = backend.cholesky(C)

    def sample(self, rng: Any, n_traj: int, dt: float):
        """Draw one step of Wiener increments with variance dt using backend RNG.

        rng can be a single RNG or a sequence of per-trajectory RNGs.
        """
        M = int(self.spec.dim)
        # If rng is a sequence, sample per-trajectory for independent streams
        if isinstance(rng, (list, tuple)):
            rows = []
            for r in rng:  # type: ignore[assignment]
                rows.append(self.backend.randn(r, (M,), dtype=float))
            z = self.backend.asarray(rows, dtype=float)  # (n_traj, M)
        else:
            z = self.backend.randn(rng, (n_traj, M), dtype=float)
        if self.spec.kind == "independent":
            return (dt ** 0.5) * z
        assert self._chol is not None
        return (dt ** 0.5) * self.backend.einsum('tm,mk->tk', z, self._chol.T)
