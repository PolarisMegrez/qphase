"""qphase_sde: Real-valued sampled-signal spectral math (Phase 6 frozen).

Frozen numerical definitions of the ``sampled_scalar`` PSD measurement path.
The spectral estimator child plugins keep producing full two-sided DFT power
spectra; this module contributes the strict, backend-duck-typed steps that
turn a real-valued sampled signal into a one-sided density estimate:

- :func:`detrend_series` — the strict ``none``/``constant`` detrend modes.
  The default is ``none``; a real input never silently changes it.
- :func:`fold_one_sided_powers` — one-sided folding of two-sided DFT powers
  in unshifted bin order: the DC bin and, for even transform lengths, the
  Nyquist bin are counted once; every interior bin is doubled, merging the
  Hermitian positive/negative frequency pair. Folding applies per trajectory,
  before any cross-realization mean/std/SEM reduction, so the uncertainty
  keeps its realization basis. Total integrated power is preserved exactly
  for the Hermitian-symmetric spectra of real signals.
- :func:`one_sided_frequency_axis` — the non-negative coordinate of a folded
  spectrum. One-sidedness is expressed by this coordinate and the
  ``one_sided`` sidedness attribute, never by the orientation name.
- :func:`sampled_scalar_psd_units` — canonical units of a sampled-scalar PSD
  variable: the input variable units squared per frequency-axis unit.

No ``rfft`` fast path is taken here: the first implementation computes the
full FFT and folds strictly; the performance variant is deferred (Phase 9).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = [
    "DetrendMode",
    "PsdConvention",
    "detrend_series",
    "fold_one_sided_powers",
    "one_sided_bin_count",
    "one_sided_frequency_axis",
    "sampled_scalar_psd_units",
]

#: Strict detrend modes of the sampled-scalar spectral path.
DetrendMode = Literal["none", "constant"]

#: Frozen PSD conventions shared with the spectral estimators: ``symmetric``
#: and ``unitary`` report angular frequency (axis scaled by ``2*pi``),
#: ``pragmatic`` reports cyclic frequency; all three are densities whose
#: frequency-axis units identifier is ``inverse_time``.
PsdConvention = Literal["symmetric", "unitary", "pragmatic"]

_CONVENTIONS = ("symmetric", "unitary", "pragmatic")


def detrend_series(y: Any, mode: DetrendMode) -> Any:
    """Apply the strict detrend ``mode`` along the trailing (time) axis.

    ``none`` returns the input unchanged; ``constant`` subtracts the
    per-record mean. Any other mode is rejected — detrending is an explicit,
    provenance-recorded choice, never inferred from the input dtype. The
    input stays on its backend (NumPy/CuPy duck typing, no host copy).
    """
    if mode == "none":
        return y
    if mode == "constant":
        return y - y.mean(axis=-1, keepdims=True)
    raise ValueError(f"unsupported detrend mode: {mode!r}")


def one_sided_bin_count(n_fft: int) -> int:
    """Return the non-negative bin count of an ``n_fft``-point DFT."""
    if n_fft < 1:
        raise ValueError(f"transform length must be positive, got {n_fft}")
    return n_fft // 2 + 1


def fold_one_sided_powers(powers: Any) -> Any:
    """Fold two-sided DFT powers to one-sided, per record.

    ``powers`` holds two-sided power values in unshifted DFT bin order along
    the trailing axis (bin ``k`` at index ``k``), with any leading batch
    dimensions (e.g. trajectories). The result keeps bins
    ``0 .. n_fft//2``: the DC bin and, for even ``n_fft``, the Nyquist bin
    ``n_fft//2`` are counted once; interior bins are doubled, merging each
    Hermitian positive/negative frequency pair of a real-valued signal.

    The operation is a multiplicative reweighting, so it commutes with the
    estimator's per-bin density scaling and applies to per-trajectory powers
    before cross-realization statistics. The input stays on its backend
    (NumPy/CuPy duck typing, no host copy).
    """
    n_fft = int(powers.shape[-1])
    n_one = one_sided_bin_count(n_fft)
    folded = powers[..., :n_one].copy()
    # Interior bins: everything except DC, and except Nyquist for even n_fft.
    interior_stop = n_one - 1 if n_fft % 2 == 0 else n_one
    folded[..., 1:interior_stop] *= 2
    return folded


def one_sided_frequency_axis(
    n_fft: int, dt: float, *, convention: PsdConvention
) -> np.ndarray:
    """Non-negative frequency coordinate of a folded ``n_fft``-bin spectrum.

    Bins sit at ``k / (n_fft * dt)`` for ``k = 0 .. n_fft//2``; angular
    conventions (``symmetric``/``unitary``) scale the axis by ``2*pi``. The
    axis is strictly non-negative and increasing — the one-sidedness of the
    product is expressed by this coordinate together with the ``one_sided``
    sidedness attribute.
    """
    if dt <= 0.0:
        raise ValueError(f"sampling interval must be positive, got {dt}")
    if convention not in _CONVENTIONS:
        raise ValueError(f"unsupported PSD convention: {convention!r}")
    axis = np.arange(one_sided_bin_count(n_fft), dtype=np.float64) / (n_fft * dt)
    if convention in ("symmetric", "unitary"):
        axis *= 2.0 * np.pi
    return axis


def sampled_scalar_psd_units(
    variable_units: str, *, convention: PsdConvention
) -> str:
    """Canonical units of a sampled-scalar PSD variable.

    The PSD is a density whose integral over frequency is the variance of the
    sampled signal, so its units are the input variable units squared per
    frequency-axis unit. All frozen conventions share the ``inverse_time``
    frequency-axis units identifier (angular vs. cyclic differs only by the
    numeric ``2*pi`` axis scaling), hence the canonical string is
    ``"<units>^2/inverse_time"`` — or ``"1/inverse_time"`` for a
    dimensionless input. No general unit algebra is introduced.
    """
    if convention not in _CONVENTIONS:
        raise ValueError(f"unsupported PSD convention: {convention!r}")
    numerator = f"{variable_units}^2" if variable_units else "1"
    return f"{numerator}/inverse_time"
