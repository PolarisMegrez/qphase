"""
QPhaseSDE: Power Spectral Density
---------------------------------
Compute power spectral density (PSD) from multi-trajectory time series for one
or more modes using FFT-based periodograms.

Behavior
--------
- Support two input interpretations: complex-valued directly (``kind='complex'``)
    or magnitude-based (``kind='modular'``).
- Provide common PSD conventions: unitary/symmetric (angular frequency ω) and
    pragmatic (frequency f). Exact scaling, return shapes, and error semantics are
    specified by the function docstrings.

Notes
-----
- These utilities are backend-agnostic with NumPy implementations and are used
    by visualizers as well as analysis pipelines.
"""

from typing import Dict, List, Tuple
import numpy as np
from ..core.errors import QPSConfigError

__all__ = [
    "compute_psd_single",
    "compute_psd_for_modes",
]

def compute_psd_single(x: np.ndarray, dt: float, *, kind: str = "complex", convention: str = "symmetric") -> Tuple[np.ndarray, np.ndarray]:
    """Compute two-sided power spectral density (PSD) for a single mode.

    Computes the mean PSD across trajectories using either a unitary two-sided
    FFT with an angular-frequency axis (rad/s) or a pragmatic periodogram with
    a frequency axis (Hz). The input can be treated as complex-valued directly
    (``kind='complex'``) or via its magnitude (``kind='modular'``).

    Parameters
    ----------
    x : numpy.ndarray
        Complex-like time series array of shape ``(n_traj, n_time)``.
    dt : float
        Sampling interval in seconds.
    kind : {'complex', 'modular'}, optional
        Interpretation of the input. ``'complex'`` uses ``x`` directly;
        ``'modular'`` uses ``abs(x)``. Default is ``'complex'``.
    convention : {'symmetric', 'unitary', 'pragmatic'}, optional
        PSD scaling and frequency axis convention. ``'symmetric'`` and
        ``'unitary'`` select a unitary FFT (``norm='ortho'``) with angular
        frequency axis ω and scaling ``S(ω_k) = (dt / 2π) · |X_unitary[k]|^2``.
        ``'pragmatic'`` selects a standard FFT (``norm=None``) with frequency
        axis f and scaling ``S(f_k) = (dt / N) · |X[k]|^2``. Default is
        ``'symmetric'``.

    Returns
    -------
    axis : numpy.ndarray
        One-dimensional array of shape ``(n_time,)``. Angular frequency ω for
        unitary/symmetric conventions [rad/s], or frequency f for pragmatic
        convention [Hz].
    P : numpy.ndarray
        Mean PSD over trajectories with shape ``(n_time,)`` and the chosen
        normalization.

    Raises
    ------
    QPSConfigError
        Raised for invalid configuration parameters:

        - [521] ``kind`` is not one of {'complex', 'modular'}
        - [522] ``convention`` is not one of {'symmetric', 'unitary', 'pragmatic'}

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((8, 256)) + 1j * rng.standard_normal((8, 256))
    >>> axis, P = compute_psd_single(x, dt=1e-2, kind='complex', convention='unitary')
    >>> axis.shape, P.shape
    ((256,), (256,))
    """
    if kind not in ("complex", "modular"):
        raise QPSConfigError("[521] kind must be 'complex' or 'modular'")
    if convention not in ("symmetric", "unitary", "pragmatic"):
        raise QPSConfigError("[522] convention must be 'symmetric'|'unitary'|'pragmatic'")

    x_proc = np.abs(x) if kind == "modular" else x
    n_traj, n_time = x_proc.shape

    if convention in ("symmetric", "unitary"):
        # Unitary FFT (energy-preserving in l2). For spectral density per rad/s,
        # multiply by dt/(2π) so that sum_k S(ω_k) Δω ≈ average power, Δω = 2π/(N·dt).
        X = np.fft.fft(x_proc, axis=1, norm="ortho")
        P = (dt / (2.0 * np.pi)) * np.mean(np.abs(X) ** 2, axis=0)
        f = np.fft.fftfreq(n_time, d=dt)
        axis = 2.0 * np.pi * f
    else:
        # Standard FFT; periodogram scaling per Hz: S(f_k) = (dt/N) |X[k]|^2
        X = np.fft.fft(x_proc, axis=1, norm=None)
        P = (dt / float(n_time)) * np.mean(np.abs(X) ** 2, axis=0)
        axis = np.fft.fftfreq(n_time, d=dt)

    return axis, P

def compute_psd_for_modes(data: np.ndarray, dt: float, modes: List[int], *, kind: str, convention: str) -> Dict:
    """Compute PSD for multiple modes and stack results into a dict payload.

    Computes the PSD per requested mode using :func:`compute_psd_single` and
    stacks the results column-wise so that ``psd`` has shape ``(n_time, len(modes))``.

    Parameters
    ----------
    data : numpy.ndarray
        Complex-like time series array of shape ``(n_traj, n_time, n_modes)``.
    dt : float
        Sampling interval in seconds.
    modes : list of int
        Non-empty list of mode indices to analyze.
    kind : {'complex', 'modular'}
        See :func:`compute_psd_single`.
    convention : {'symmetric', 'unitary', 'pragmatic'}
        See :func:`compute_psd_single`.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``'axis'``: numpy.ndarray of shape ``(n_time,)`` (ω or f depending on convention)
        - ``'psd'``: numpy.ndarray of shape ``(n_time, len(modes))``
        - ``'modes'``: list of int (echo of input indices)
        - ``'kind'``: str (echo of input)
        - ``'convention'``: str (echo of input)

    Raises
    ------
    QPSConfigError
        Raised for invalid configuration parameters:

        - [521] ``kind`` is not one of {'complex', 'modular'}
        - [522] ``convention`` is not one of {'symmetric', 'unitary', 'pragmatic'}
        - [523] ``modes`` is empty

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> data = rng.standard_normal((4, 128, 3)) + 1j * rng.standard_normal((4, 128, 3))
    >>> out = compute_psd_for_modes(data, dt=1e-2, modes=[0, 2], kind='complex', convention='pragmatic')
    >>> sorted(out.keys())
    ['axis', 'convention', 'kind', 'modes', 'psd']
    >>> out['psd'].shape
    (128, 2)
    """
    if not modes:
        raise QPSConfigError("[523] modes must be non-empty")
    # Compute first to get axis
    axis0, P0 = compute_psd_single(data[:, :, modes[0]], dt, kind=kind, convention=convention)
    P_list = [P0]
    for m in modes[1:]:
        _, Pm = compute_psd_single(data[:, :, m], dt, kind=kind, convention=convention)
        P_list.append(Pm)
    P_mat = np.vstack(P_list).T  # shape (n_freq, n_modes)
    return {
        "axis": axis0,
        "psd": P_mat,
        "modes": modes,
        "kind": kind,
        "convention": convention,
    }