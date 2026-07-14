"""Tests for the Kerr 3-mode kernelized terms path."""

from __future__ import annotations

import numpy as np
import pytest


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


@pytest.fixture
def model():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.kerr_3mode import Kerr3ModeConfig, Kerr3ModeModel

    return Kerr3ModeModel(Kerr3ModeConfig(
        omega_a=0.5,
        omega_b=0.3,
        omega_c=0.2,
        chi=0.01,
        kappa_a=1.0,
        kappa_b=0.5,
        kappa_c=1.0,
        g_ab=0.1,
        g_ac=0.05,
    ))


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_kernelized_terms_match_python(model):
    """CuPy kernelized drift/diffusion matches the Python implementation."""
    import cupy as cp

    from qphase.backend.cupy_backend import CuPyBackend

    backend = CuPyBackend()
    n = 64
    rng = np.random.default_rng(42)
    y_np = (rng.standard_normal((n, 3)) + 1j * rng.standard_normal((n, 3))).astype(
        np.complex64
    )
    y = cp.asarray(y_np)

    a_py = cp.asarray(model.drift(y_np, 0.0, model.params))
    L_py = cp.asarray(model.diffusion(y_np, 0.0, model.params))

    a_k, L_k = model.kernelized_terms(y, 0.0, model.params, backend)

    cp.testing.assert_allclose(a_k, a_py, rtol=1e-4, atol=1e-5)
    cp.testing.assert_allclose(L_k, L_py, rtol=1e-4, atol=1e-5)
