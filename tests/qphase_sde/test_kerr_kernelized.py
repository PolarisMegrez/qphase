"""GPU consistency tests for Kerr model kernels."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


@pytest.fixture(params=["kerr_2mode", "kerr_3mode"])
def model(request):
    if request.param == "kerr_2mode":
        from models.kerr_2mode import Kerr2ModeModel

        return Kerr2ModeModel(
            omega_a=0.5, omega_b=0.3, chi=0.01, gamma_a=1.0,
            gamma_b=0.5, g=0.1,
        )
    from models.kerr_3mode import Kerr3ModeModel

    return Kerr3ModeModel(
        omega_a=0.5, omega_b=0.3, omega_c=0.2, chi=0.01,
        gamma_a=1.0, gamma_b=0.5, gamma_c=1.0, g_ab=0.1, g_ac=0.05,
    )


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_kernelized_terms_match_python(model):
    import cupy as cp
    from qphase.backend.cupy_backend import CuPyBackend

    n = 64
    rng = np.random.default_rng(42)
    y_np = (
        rng.standard_normal((n, model.n_modes))
        + 1j * rng.standard_normal((n, model.n_modes))
    ).astype(np.complex64)
    y = cp.asarray(y_np)

    drift, diffusion = model.kernelized_terms(y, 0.0, model.params, CuPyBackend())
    cp.testing.assert_allclose(
        drift, cp.asarray(model.drift(y_np, 0.0, model.params)), rtol=1e-4, atol=1e-5
    )
    cp.testing.assert_allclose(
        diffusion,
        cp.asarray(model.diffusion(y_np, 0.0, model.params)),
        rtol=1e-4,
        atol=1e-5,
    )
