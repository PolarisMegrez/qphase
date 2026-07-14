"""CuPy RawKernel for the Kerr 3-mode model.

Computes fused drift and diffusion for all trajectories in one launch. The
per-trajectory diffusion includes a small 2x2 Cholesky decomposition for mode a
and isotropic noise for modes b and c.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qphase.backend.base import BackendBase

from qphase_sde.kernels import compile_cached_kernel

_KERR_SOURCE = r"""
extern "C" __global__
void __kerr_3mode_terms_func__(
    const $CT$* __restrict__ y,
    const double* __restrict__ omega_a,
    const double* __restrict__ omega_b,
    const double* __restrict__ omega_c,
    const double* __restrict__ chi,
    const double* __restrict__ kappa_a,
    const double* __restrict__ kappa_b,
    const double* __restrict__ kappa_c,
    const double* __restrict__ g_ab,
    const double* __restrict__ g_ac,
    int n,
    $CT$* __restrict__ drift,
    $CT$* __restrict__ diffusion
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    $CT$ alpha = y[i * 3 + 0];
    $CT$ beta  = y[i * 3 + 1];
    $CT$ gamma = y[i * 3 + 2];

    double oa = omega_a[i];
    double ob = omega_b[i];
    double oc = omega_c[i];
    double c  = chi[i];
    double ka = kappa_a[i];
    double kb = kappa_b[i];
    double kc = kappa_c[i];
    double gab = g_ab[i];
    double gac = g_ac[i];

    double ar = alpha.x, ai = alpha.y;
    double br = beta.x,  bi = beta.y;
    double gr = gamma.x, gi = gamma.y;

    // Drift for mode a
    double n_alpha2 = ar * ar + ai * ai;
    double term_kerr = 2.0 * c * n_alpha2;
    double dar = -0.5 * ka * ar + oa * ai + term_kerr * ai + gab * bi + gac * gi;
    double dai = -0.5 * ka * ai - oa * ar - term_kerr * ar - gab * br - gac * gr;

    // Drift for mode b (gain)
    double dbr = 0.5 * kb * br + ob * bi + gab * ai;
    double dbi = 0.5 * kb * bi - ob * br - gab * ar;

    // Drift for mode c (loss)
    double dgr = -0.5 * kc * gr + oc * gi + gac * ai;
    double dgi = -0.5 * kc * gi - oc * gr - gac * ar;

    drift[i * 3 + 0].x = dar; drift[i * 3 + 0].y = dai;
    drift[i * 3 + 1].x = dbr; drift[i * 3 + 1].y = dbi;
    drift[i * 3 + 2].x = dgr; drift[i * 3 + 2].y = dgi;

    // Diffusion for mode a: 2x2 Cholesky of covariance
    // M = -2i * chi * alpha^2
    double ar2 = ar * ar - ai * ai;
    double ai2 = 2.0 * ar * ai;
    // M = c * (ai2 - i*ar2) because -2i*(ar+i ai)^2 = -2i*(ar2 - ai2 + 2i ar ai)
    //                              = 4 ar ai - 2i (ar2 - ai2)
    double Mr =  4.0 * c * ar * ai;
    double Mi = -2.0 * c * (ar2);

    double Dval = ka;
    double absM = sqrt(Mr * Mr + Mi * Mi);
    if (Dval > 1e-9 && absM > Dval) {
        double scale = 0.999 * Dval / (absM + 1e-16);
        Mr *= scale;
        Mi *= scale;
    }

    double Sig_xx = 0.5 * (Dval + Mr);
    double Sig_yy = 0.5 * (Dval - Mr);
    double Sig_xy = 0.5 * Mi;
    if (Sig_xx < 0.0) Sig_xx = 0.0;

    double L11 = sqrt(Sig_xx);
    double L11_safe = L11 + 1e-16;
    double L21 = Sig_xy / L11_safe;
    if (L11 < 1e-9) L21 = 0.0;
    double term_sq = Sig_yy - L21 * L21;
    if (term_sq < 0.0) term_sq = 0.0;
    double L22 = sqrt(term_sq);

    $T$ nb = sqrt(0.5 * kb);
    $T$ nc = sqrt(0.5 * kc);

    // Mode a -> noise channels 0,1 (row offset 0)
    diffusion[i * 18 + 0].x = ($T$)L11; diffusion[i * 18 + 0].y = ($T$)L21;
    diffusion[i * 18 + 1].x = ($T$)0.0; diffusion[i * 18 + 1].y = ($T$)L22;
    // Mode b -> noise channels 2,3 (row offset 6, column offset 2)
    diffusion[i * 18 + 8].x = nb; diffusion[i * 18 + 8].y = ($T$)0.0;
    diffusion[i * 18 + 9].x = ($T$)0.0; diffusion[i * 18 + 9].y = nb;
    // Mode c -> noise channels 4,5 (row offset 12, column offset 4)
    diffusion[i * 18 + 16].x = nc; diffusion[i * 18 + 16].y = ($T$)0.0;
    diffusion[i * 18 + 17].x = ($T$)0.0; diffusion[i * 18 + 17].y = nc;
}
"""


def _broadcast_param(be: Any, p: Any, n: int) -> Any:
    """Broadcast a scalar or array parameter to a CuPy array of shape (n,)."""
    import cupy as cp

    if hasattr(p, "__len__") and not isinstance(p, (str, bytes)):
        arr = cp.asarray(p)
        if arr.shape == (n,):
            return arr
        if arr.size == 1:
            return cp.full((n,), float(arr.item()), dtype=cp.float64)
    return cp.full((n,), float(p), dtype=cp.float64)


def kernelized_terms(
    y: Any,
    params: dict[str, Any],
    backend: BackendBase,
) -> tuple[Any, Any]:
    """Return (drift, diffusion) for Kerr 3-mode using a CuPy kernel."""
    import cupy as cp

    n = int(y.shape[0])
    rdtype = y.real.dtype
    if rdtype == np.float32:
        source = _KERR_SOURCE.replace("$T$", "float").replace("$CT$", "float2")
        ctype = "complex<float>"
    else:
        source = _KERR_SOURCE.replace("$T$", "double").replace("$CT$", "double2")
        ctype = "complex<double>"

    kernel = compile_cached_kernel("kerr_3mode_terms", ctype, source)

    omega_a = _broadcast_param(backend, params["omega_a"], n)
    omega_b = _broadcast_param(backend, params["omega_b"], n)
    omega_c = _broadcast_param(backend, params["omega_c"], n)
    chi = _broadcast_param(backend, params["chi"], n)
    kappa_a = _broadcast_param(backend, params["kappa_a"], n)
    kappa_b = _broadcast_param(backend, params["kappa_b"], n)
    kappa_c = _broadcast_param(backend, params["kappa_c"], n)
    g_ab = _broadcast_param(backend, params["g_ab"], n)
    g_ac = _broadcast_param(backend, params["g_ac"], n)

    drift = cp.empty((n, 3), dtype=y.dtype)
    diffusion = cp.zeros((n, 3, 6), dtype=y.dtype)

    threads = 256
    blocks = (n + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            y,
            omega_a,
            omega_b,
            omega_c,
            chi,
            kappa_a,
            kappa_b,
            kappa_c,
            g_ab,
            g_ac,
            n,
            drift,
            diffusion,
        ),
    )
    return drift, diffusion
