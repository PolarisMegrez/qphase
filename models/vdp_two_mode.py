from __future__ import annotations

import numpy as np
from typing import Dict


def build_sde(params: Dict):
	"""
	Two-mode van der Pol coupled cavities (Itô SDE) with complex noise basis.

	dα/dt = [-i ω_a + γ_a/2 + Γ(1 - |α|^2)] α - i g β + ξ_α
	dβ/dt = [-i ω_b - γ_b/2] β - i g α + ξ_β

	Diffusion (complex basis, diagonal):
	  D_α^c = D [ γ_a/2 + Γ(2|α|^2 - 1) ]
	  D_β^c = D γ_b/2
	We set L_c = diag( sqrt(max(D_α^c, 0)), sqrt(max(D_β^c, 0)) ).
	"""
	# Required params: omega_a, omega_b, gamma_a, gamma_b, Gamma, g, D
	def drift(y: np.ndarray, t: float, p: Dict) -> np.ndarray:
		# y: (n_traj, 2) complex, y[:,0]=α, y[:,1]=β
		alpha = y[:, 0]
		beta = y[:, 1]
		omega_a = p["omega_a"]
		omega_b = p["omega_b"]
		gamma_a = p["gamma_a"]
		gamma_b = p["gamma_b"]
		Gamma = p["Gamma"]
		g = p["g"]

		dalpha = ((-1j * omega_a) + (gamma_a / 2.0) + Gamma * (1.0 - np.abs(alpha) ** 2)) * alpha - 1j * g * beta
		dbeta = ((-1j * omega_b) - (gamma_b / 2.0)) * beta - 1j * g * alpha
		out = np.empty_like(y)
		out[:, 0] = dalpha
		out[:, 1] = dbeta
		return out

	def diffusion(y: np.ndarray, t: float, p: Dict) -> np.ndarray:
		alpha = y[:, 0]
		gamma_a = p["gamma_a"]
		gamma_b = p["gamma_b"]
		Gamma = p["Gamma"]
		D = p["D"]

		D_alpha = D * (gamma_a / 2.0 + Gamma * (2.0 * np.abs(alpha) ** 2 - 1.0))
		D_beta = D * (gamma_b / 2.0)
		# Clip to non-negative for numerical stability
		D_alpha = np.clip(D_alpha, 0.0, None)
		D_beta = np.clip(D_beta, 0.0, None)
		Lc = np.zeros((y.shape[0], 2, 2), dtype=np.complex128)
		Lc[:, 0, 0] = np.sqrt(D_alpha)
		Lc[:, 1, 1] = np.sqrt(D_beta)
		return Lc

	class _Model:
		def __init__(self, name, n_modes, noise_basis, noise_dim, params, drift, diffusion, diffusion_jacobian=None):
			self.name = name
			self.n_modes = n_modes
			self.noise_basis = noise_basis
			self.noise_dim = noise_dim
			self.params = params
			self.drift = drift
			self.diffusion = diffusion
			self.diffusion_jacobian = diffusion_jacobian

	return _Model(
		name="vdp_two_mode",
		n_modes=2,
		noise_basis="complex",
		noise_dim=2,
		params=params,
		drift=drift,
		diffusion=diffusion,
		diffusion_jacobian=None,
	)

