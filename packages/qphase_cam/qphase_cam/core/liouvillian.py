"""CAM Liouvillian and residual evaluation."""

from __future__ import annotations

from typing import Any

from qphase.backend.xputil import get_xp

from .coordinates import matrix_to_vector


def liouvillian(hamiltonian: Any, state: Any, diffusion: Any) -> Any:
    """Evaluate ``-i H R + i R H^dagger + D`` for batches or scalars."""
    xp = get_xp(state)
    adjoint = xp.swapaxes(xp.conj(hamiltonian), -1, -2)
    return -1j * (hamiltonian @ state) + 1j * (state @ adjoint) + diffusion


def model_liouvillian(model: Any, state: Any, params: dict[str, Any]) -> Any:
    """Evaluate a CAM-capable model's Liouvillian."""
    return liouvillian(
        model.cam_hamiltonian(state, params),
        state,
        model.cam_diffusion(state, params),
    )


def residual_vector(model: Any, state: Any, params: dict[str, Any]) -> Any:
    """Return the canonical real residual vector."""
    return matrix_to_vector(model_liouvillian(model, state, params))
