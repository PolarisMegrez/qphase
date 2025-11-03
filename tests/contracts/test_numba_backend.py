from __future__ import annotations

import numpy as np
import pytest

from QPhaseSDE.backends.factory import get_backend
from QPhaseSDE.core.engine import run as engine_run
from QPhaseSDE.core.protocols import NoiseSpec, SDEModel


def build_toy_model(n_modes: int = 2) -> SDEModel:
    def _drift(y: np.ndarray, t: float, params):
        # linear stable drift to keep values bounded
        lam = -0.1
        return lam * y
    def _diffusion(y: np.ndarray, t: float, params):
        # diagonal diffusion in complex basis for simplicity
        Lc = np.zeros((y.shape[0], y.shape[1], 1), dtype=np.complex128)
        Lc[:, :, 0] = 0.5 + 0.0j
        return Lc
    return SDEModel(
        name="toy",
        n_modes=n_modes,
        noise_basis="complex",
        noise_dim=1,
        params={},
        drift=_drift,
        diffusion=_diffusion,
        diffusion_jacobian=None,
    )


def test_numba_backend_engine_runs():
    try:
        be = get_backend('numba')
    except Exception:
        pytest.skip('Numba backend not available')
    model = build_toy_model(2)
    y0 = np.array([1.0+0j, -1.0+0j], dtype=np.complex128)
    ts = engine_run(
        model=model,
        ic=y0,
        time={"t0": 0.0, "dt": 1e-3, "steps": 100},
        n_traj=8,
        solver='euler',
        backend=be,
        noise_spec=NoiseSpec(kind='independent', dim=2),
        master_seed=123,
        return_stride=5,
    )
    assert getattr(ts, 'data', None) is not None
    assert ts.data.shape == (8, (100//5)+1, model.n_modes)
    assert ts.dt == pytest.approx(5e-3)
