"""
QPhaseSDE: Euler-Maruyama Integrator
------------------------------------
Reference Itô SDE solver with backend-optimized contractions, integrated with
the central registry for discovery and composition.

Behavior
--------
- Backend-agnostic step rule ``dy = a(y,t)·dt + L(y,t) @ dW``; contraction over
  noise channels is specialized per backend when possible. Complex noise bases
  are expanded to an equivalent real basis prior to contraction.

Notes
-----
- The short-term alias 'milstein' may map to Euler–Maruyama in this release.
"""

from typing import Dict, Any, Callable, Optional
from .protocols import Integrator
from ..core.protocols import BackendBase as Backend
from ..core.registry import register as _register_decorator, registry as _registry

__all__ = [
	"EulerMaruyama",
]

def _expand_complex_noise_backend(Lc: Any, backend: Backend) -> Any:
	"""Expand complex-basis diffusion matrix to an equivalent real basis.

	Transforms L_c ∈ C^{..., n_modes, M_c} into L_r ∈ C^{..., n_modes, 2·M_c}
	using only backend operations, preserving contraction with real noise.
	"""
	a = backend.real(Lc)
	b = backend.imag(Lc)
	s = (2.0) ** 0.5
	Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
	Lr_imag = backend.concatenate((b / s, a / s), axis=-1)
	return Lr_real + 1j * Lr_imag

@_register_decorator("integrator", "euler_maruyama")
class EulerMaruyama(Integrator):
	"""Euler–Maruyama integrator for SDEs with backend-optimized contractions.

	This solver implements the classic Euler–Maruyama time stepping under the
	Itô interpretation. It is backend-agnostic: contractions over noise channels
	are delegated to the active backend (NumPy/Numba/CuPy/Torch). When the model
	declares a complex noise basis, the diffusion is expanded to an equivalent
	real basis internally to match the engine's real-valued noise increments.

	Attributes
	----------
	_contract_fn : Optional[Callable[[Backend, Any, Any], Any]]
		An internal fast-path contraction function specialized on first use based
		on the backend. For Torch, a batched-matmul is used; otherwise falls back
		to ``backend.einsum('tnm,tm->tn', L, dW)``.

	Methods
	-------
	step(y, t, dt, model, dW, backend)
		Advance the state by one step according to
		``y_{t+dt} = y_t + a(y_t,t)·dt + L(y_t,t) @ dW``.

	Examples
	--------
	>>> # Typical usage via the registry (names may be aliased to 'euler', 'em')
	>>> from QPhaseSDE.core.registry import registry
	>>> em_cls = registry.create("integrator", "euler_maruyama")  # doctest: +SKIP
	>>> integrator = em_cls()  # doctest: +SKIP
	>>> # integrator.step(y, t, dt, model, dW, backend)  # doctest: +SKIP

	References
	----------
	- Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic
	  Differential Equations. Springer. (Euler–Maruyama scheme)
	  doi:10.1007/978-3-662-12616-5
	- Higham, D. J. (2001). An Algorithmic Introduction to Numerical Simulation
	  of Stochastic Differential Equations. SIAM Review, 43(3), 525–546.
	  doi:10.1137/S0036144500378302
	- Gardiner, C. W. (2009). Stochastic Methods: A Handbook for the Natural and
	  Social Sciences (4th ed.). Springer.
	"""
	def __init__(self) -> None:
		"""Initialize the integrator and internal contraction cache.

		Lazily specializes a fast-path contraction function on first use based on
		the active backend.
		"""
		# Lazily initialized fast-path functions based on backend
		self._contract_fn: Optional[Callable[[Any, Any, Any], Any]] = None  # (backend, L, dW) -> (tn)

	def step(self, y: Any, t: float, dt: float, model: Any, dW: Any, backend: Backend) -> Any:
		"""Compute one-step increment ``dy`` using the Euler–Maruyama scheme.

		The update follows ``dy = a(y,t)·dt + L(y,t) @ dW``, where ``a`` is the
		drift and ``L`` the diffusion matrix. If the model declares a complex noise
		basis (``noise_basis == 'complex'``), the diffusion is expanded to a real
		basis before contracting with the real-valued increment ``dW``.

		Parameters
		----------
		y : Any
			State array with shape ``(n_traj, n_modes)`` (complex).
		t : float
			Current simulation time.
		dt : float
			Time step size (positive).
		model : Any
			Object providing ``drift(y, t, params)`` and ``diffusion(y, t, params)``
			evaluated on ``y``; may define ``noise_basis`` in {'real','complex'}.
		dW : Any
			Noise increment array with shape ``(n_traj, M)`` (real), sampled by the
			engine according to the model's noise specification.
		backend : Backend
			Active backend implementing array operations and contractions.

		Returns
		-------
		Any
			Increment ``dy`` with the same shape as ``y`` (complex).

		Examples
		--------
		>>> # dy = em.step(y, t, dt, model, dW, backend)  # doctest: +SKIP
		"""
		a = model.drift(y, t, model.params)  # (n_traj, n_modes)
		L = model.diffusion(y, t, model.params)  # (n_traj, n_modes, M_b)
		if getattr(model, 'noise_basis', 'real') == 'complex':
			L = _expand_complex_noise_backend(L, backend)
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
						# (_T, N, M) bmm (_T, M, 1) -> (_T, N)
						return _th.bmm(_L, _dW.unsqueeze(-1)).squeeze(-1)
					self._contract_fn = _contract
				except Exception:
					self._contract_fn = None
			# default fallback
			if self._contract_fn is None:
				self._contract_fn = lambda _backend, _L, _dW: _backend.einsum('tnm,tm->tn', _L, _dW)
		# Contract noise channels: (tnm, tm) -> (tn)
		return a * dt + self._contract_fn(backend, L, dW)

# Register commonly used aliases mapping to Euler–Maruyama
try:
	_registry.register("integrator", "euler", EulerMaruyama)
	_registry.register("integrator", "em", EulerMaruyama)
except Exception:
	# Ignore duplicate-registration errors during reloads/tests
	pass