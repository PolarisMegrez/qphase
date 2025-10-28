from __future__ import annotations

"""
Vectorized Euler–Maruyama step for complex-valued multi-mode SDE systems.

Contract and shapes
-------------------
- State y: ndarray complex128 with shape (n_traj, n_modes)
- Model provides:
	- drift(y, t, params) -> (n_traj, n_modes) complex
	- diffusion(y, t, params) -> (n_traj, n_modes, M_b) complex, where
		M_b is the number of noise channels in the model's declared basis.
	- noise_basis: 'real' or 'complex'. If 'complex', we expand to real basis
		with 2*M_b real channels using expand_complex_noise().
- Noise increment dW: ndarray float64 with shape (n_traj, M), where M is the
	number of real noise channels actually sampled by the engine.

Update rule (Itô):
	y_{t+dt} = y_t + a(y_t, t) * dt + L(y_t, t) @ dW
where '@' denotes contraction over the noise-channel dimension.

This solver is backend-agnostic for v0.1.1 and uses NumPy einsum for clarity
and vectorization. Future backends (e.g., numba, cupy) can re-implement the
same interface to swap at runtime.
"""

from typing import Dict, Any

from .protocols import Integrator
from ..core.protocols import BackendBase as Backend
from ..core.registry import register as _register_decorator, registry as _registry


def _expand_complex_noise_backend(Lc: Any, backend: Backend) -> Any:
	"""Expand complex-basis diffusion matrix using backend ops only.

	Input Lc: (B?, n_modes, M_c) complex -> Output: (B?, n_modes, 2*M_c) complex
	"""
	a = backend.real(Lc)
	b = backend.imag(Lc)
	s = (2.0) ** 0.5
	Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
	Lr_imag = backend.concatenate((b / s, a / s), axis=-1)
	return Lr_real + 1j * Lr_imag


@_register_decorator("integrator", "euler_maruyama")
class EulerMaruyama(Integrator):
	"""Backend-aware Euler–Maruyama Integrator.

	Uses the provided backend for contractions so it can work across backends.
	"""

	def step(self, y: Any, t: float, dt: float, model: Any, dW: Any, backend: Backend) -> Any:
		a = model.drift(y, t, model.params)  # (n_traj, n_modes)
		L = model.diffusion(y, t, model.params)  # (n_traj, n_modes, M_b)
		if getattr(model, 'noise_basis', 'real') == 'complex':
			L = _expand_complex_noise_backend(L, backend)
		# Contract noise channels: (tnm, tm) -> (tn)
		return a * dt + backend.einsum('tnm,tm->tn', L, dW)


# Register commonly used aliases mapping to Euler–Maruyama
try:
    _registry.register("integrator", "euler", EulerMaruyama)
    _registry.register("integrator", "em", EulerMaruyama)
    # Temporary alias: map 'milstein' to EM in v0.1.1
    _registry.register("integrator", "milstein", EulerMaruyama)
except Exception:
    # Ignore duplicate-registration errors during reloads/tests
    pass

