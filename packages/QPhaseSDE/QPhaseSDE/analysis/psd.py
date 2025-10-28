from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def compute_psd_single(x: np.ndarray, dt: float, *, kind: str = "complex", convention: str = "symmetric") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-sided PSD for a batch of trajectories and a single mode.

    x: shape (n_traj, n_time) complex array (time series)
    kind: 'complex' (use x) | 'modular' (use |x|)
    convention:
      - 'symmetric'/'unitary': unitary FFT (norm='ortho') with angular frequency axis ω
      - 'pragmatic': standard FFT (norm=None) with frequency axis f and 1/N amplitude scaling

    Returns (axis, P) where axis is ω or f, and P is mean PSD over trajectories with shape (n_time,).
    """
    if kind not in ("complex", "modular"):
        raise ValueError("kind must be 'complex' or 'modular'")
    if convention not in ("symmetric", "unitary", "pragmatic"):
        raise ValueError("convention must be 'symmetric'|'unitary'|'pragmatic'")

    x_proc = np.abs(x) if kind == "modular" else x
    n_traj, n_time = x_proc.shape

    if convention in ("symmetric", "unitary"):
        X = np.fft.fft(x_proc, axis=1, norm="ortho")
        P = np.mean(np.abs(X) ** 2, axis=0)
        f = np.fft.fftfreq(n_time, d=dt)
        axis = 2.0 * np.pi * f
    else:
        X = np.fft.fft(x_proc, axis=1, norm=None)
        P = np.mean(np.abs(X) ** 2, axis=0) / float(n_time)
        axis = np.fft.fftfreq(n_time, d=dt)

    return axis, P


essential_fields = ("axis", "psd", "modes", "kind", "convention")


def compute_psd_for_modes(data: np.ndarray, dt: float, modes: List[int], *, kind: str, convention: str) -> Dict:
    """
    Compute PSD for multiple modes and return a dict with axis and stacked PSD.

    data: (n_traj, n_time, n_modes) complex
    modes: list of mode indices

    Returns dict: { axis: np.ndarray, psd: np.ndarray (n_time, len(modes)), modes: list }
    """
    if not modes:
        raise ValueError("modes must be non-empty")
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
