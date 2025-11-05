"""
QPhaseSDE: Milstein Integrator
------------------------------
Strong order-1.0 Milstein scheme for Itô SDEs under the commutative-noise
assumption (diagonal or commuting diffusion fields). Falls back to
Euler–Maruyama when a diffusion Jacobian is unavailable or the model uses a
complex noise basis that is not expanded with a matching Jacobian.

Behavior
--------
- Update rule (component-wise) under commutative noise:
	y_{t+dt} = y_t + a(y_t,t)·dt + sum_k L_k(y_t,t)·dW_k
			   + 1/2 · sum_k [ (∑_j L_{j,k} ∂L_{:,k}/∂y_j ) · (dW_k^2 - dt) ]
  where L_{:,k} is the k-th diffusion column, and ∂L_{:,k}/∂y_j denotes the
  Jacobian slice of L along the state dimension.

Notes
-----
- Multi-dimensional, non-commutative noise generally requires Lévy area terms;
  those are not included here. For typical diagonal/commutative cases this
  implementation provides the classic Milstein correction.
- If ``model.noise_basis == 'complex'``, this implementation currently falls
  back to Euler–Maruyama unless the provided Jacobian matches a real-expanded
  diffusion; see implementation notes.
"""

from typing import Any, Optional, Callable
from .protocols import Integrator
from ..core.protocols import BackendBase as Backend
from ..core.registry import register as _register_decorator

__all__ = [
	"Milstein",
]

def _expand_complex_noise_backend(Lc: Any, backend: Backend) -> Any:
	"""Expand complex-basis diffusion into an equivalent real basis.

	Transforms L_c ∈ C^{..., n_modes, M_c} into L_r ∈ C^{..., n_modes, 2·M_c}
	using only backend operations, preserving contraction with real noise.
	"""
	a = backend.real(Lc)
	b = backend.imag(Lc)
	s = (2.0) ** 0.5
	Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
	Lr_imag = backend.concatenate((b / s, a / s), axis=-1)
	return Lr_real + 1j * Lr_imag


@_register_decorator("integrator", "milstein")
class Milstein(Integrator):
	"""Milstein integrator (commutative-noise variant).

	Requires a diffusion Jacobian ``model.diffusion_jacobian`` with shape
	compatible with ``diffusion``. If unavailable or incompatible, falls back
	to Euler–Maruyama behavior (no correction term).
	"""

	def __init__(self) -> None:
		self._contract_fn: Optional[Callable[[Backend, Any, Any], Any]] = None

	def _contract(self, backend: Backend, L: Any, dW: Any) -> Any:
		# Initialize fast-path at first use
		if self._contract_fn is None:
			try:
				be_name = str(backend.backend_name()).lower()
			except Exception:
				be_name = ""
			if be_name == 'torch':
				try:
					import torch as _th  # type: ignore
					def _contract(_backend: Backend, _L: Any, _dW: Any):
						return _th.bmm(_L, _dW.unsqueeze(-1)).squeeze(-1)
					self._contract_fn = _contract
				except Exception:
					self._contract_fn = None
			if self._contract_fn is None:
				self._contract_fn = lambda _backend, _L, _dW: _backend.einsum('tnm,tm->tn', _L, _dW)
		return self._contract_fn(backend, L, dW)

	def step(self, y: Any, t: float, dt: float, model: Any, dW: Any, backend: Backend) -> Any:
		a = model.drift(y, t, model.params)             # (T, N)
		L = model.diffusion(y, t, model.params)         # (T, N, M_b)

		# Expand complex-basis diffusion if declared; Jacobian handling for complex
		# basis is not implemented here, so we will skip Milstein correction when
		# noise basis is complex.
		noise_basis = getattr(model, 'noise_basis', 'real')
		if str(noise_basis) == 'complex':
			L_eff = _expand_complex_noise_backend(L, backend)
			# EM increment only (no Milstein correction under complex basis)
			return a * dt + self._contract(backend, L_eff, dW)

		# EM part (real basis)
		dy = a * dt + self._contract(backend, L, dW)

		# Milstein correction requires diffusion Jacobian.
		jac = getattr(model, 'diffusion_jacobian', None)
		if jac is None:
			return dy

		try:
			J = jac(y, t, model.params)  # expected shape (T, N, M_b, N): ∂L_{i,k}/∂y_j
			# xi_k = dW_k^2 - dt, shape (T, M_b)
			xi = dW * dW - dt
			# G_{i,k} = sum_j L_{j,k} * J_{i,k,j} -> einsum('tjk,tikj->tik')
			G = backend.einsum('tjk,tikj->tik', L, J)
			corr = 0.5 * backend.einsum('tik,tk->ti', G, xi)
			return dy + corr
		except Exception:
			# Shape/capability mismatch — fall back to EM increment
			return dy

