from typing import Any, ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.optimize import curve_fit

from .base import BasePeakFinderConfig, PeakFinder, PeakInfo


class RationalPeakFinderConfig(BasePeakFinderConfig):
    """Configuration for Rational Function peak finder."""

    method: Literal["rational"] = "rational"
    num_order: int = Field(2, description="Numerator maximum order")
    den_order: int = Field(4, description="Denominator maximum order")
    parity: Literal["none", "even", "odd"] = Field(
        "even", description="Parity constraint"
    )
    initial_guess: list[float] | None = None
    feature: Literal["analytical_peak", "denominator_zero", "denominator_min"] = Field(
        "analytical_peak",
        description=(
            "Primary feature: analytical peak, denominator zero, or "
            "denominator minimum."
        ),
    )


class RationalPeakFinder(PeakFinder):
    """Fit a rational function to the PSD and find analytical peaks."""

    name: ClassVar[str] = "rational"
    description: ClassVar[str] = "Rational function fitting peak finder"
    config_schema: ClassVar[type[RationalPeakFinderConfig]] = RationalPeakFinderConfig

    def __init__(
        self, config: RationalPeakFinderConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(config, **kwargs)

    def _rational_func(self, w, *params):
        config = cast(RationalPeakFinderConfig, self.config)
        num_order = config.num_order
        den_order = config.den_order
        parity = config.parity

        # Parameter Unpacking
        # Identify which powers are present
        num_powers = self._get_powers(num_order, parity, is_denominator=False)
        den_powers = self._get_powers(den_order, parity, is_denominator=True)

        n_num = len(num_powers)

        p_num = params[:n_num]
        p_den = params[n_num:]

        num_val = np.zeros_like(w, dtype=float)
        for c, p in zip(p_num, num_powers, strict=False):
            num_val += c * (w**p)

        den_val = np.zeros_like(w, dtype=float)
        for c, p in zip(p_den, den_powers, strict=False):
            den_val += c * (w**p)

        # Avoid division by zero
        return num_val / (den_val + 1e-20)

    def _get_powers(self, order, parity, is_denominator=False):
        powers = []
        for i in range(order + 1):
            if parity == "none":
                powers.append(i)
            elif parity == "even":
                # Denominator: Even only
                if is_denominator:
                    if i % 2 == 0:
                        powers.append(i)
                else:
                    # Numerator: Even only (usually PSD is even)
                    if i % 2 == 0:
                        powers.append(i)
            elif parity == "odd":
                # Denominator: Even only (magnitude squared of transfer function)
                if is_denominator:
                    if i % 2 == 0:
                        powers.append(i)
                else:
                    # Numerator: Odd only
                    if i % 2 != 0:
                        powers.append(i)
        return powers

    def _construct_polys(self, params):
        config = cast(RationalPeakFinderConfig, self.config)
        num_powers = self._get_powers(config.num_order, config.parity, False)
        den_powers = self._get_powers(config.den_order, config.parity, True)

        n_num = len(num_powers)
        p_num = params[:n_num]
        p_den = params[n_num:]

        # Helper to create dense coeff array [c_n, c_n-1, ..., c_0]
        def to_dense(coeffs, powers):
            max_p = max(powers) if powers else 0
            dense = np.zeros(max_p + 1)
            for c, p in zip(coeffs, powers, strict=False):
                dense[p] = c
            return dense[::-1]  # numpy poly1d expects high-to-low order

        N = np.poly1d(to_dense(p_num, num_powers))
        D = np.poly1d(to_dense(p_den, den_powers))

        return N, D

    def find_peaks(self, freqs: np.ndarray, psd: np.ndarray) -> PeakInfo:
        # Step 1: Fit
        config = cast(RationalPeakFinderConfig, self.config)
        num_powers = self._get_powers(config.num_order, config.parity, False)
        den_powers = self._get_powers(config.den_order, config.parity, True)
        n_params = len(num_powers) + len(den_powers)

        p0: NDArray[np.float64] = np.ones(n_params, dtype=np.float64)
        p0[0] = np.mean(psd)
        if config.initial_guess and len(config.initial_guess) == n_params:
            p0 = np.asarray(config.initial_guess, dtype=np.float64)

        try:
            popt, _ = curve_fit(self._rational_func, freqs, psd, p0=p0, maxfev=10000)

            # Step 2: Analytical Analysis
            N, D = self._construct_polys(popt)

            # Derivative Numerator G(w) = N'D - ND'
            # R' = G / D^2. R'=0 <=> G=0.
            N_deriv = np.polyder(N)
            D_deriv = np.polyder(D)
            G = N_deriv * D - N * D_deriv

            # Find roots of G (Critical points)
            crit_points = G.roots

            # Filter real roots within freq range
            valid_peaks = []
            f_min, f_max = freqs.min(), freqs.max()

            for root in crit_points:
                if np.isreal(root):
                    r_real = np.real(root)
                    if f_min <= r_real <= f_max:
                        # Eval second derivative or just value to check if Max
                        val = self._eval_rational(r_real, popt)
                        # Crude check: value > neighbors?
                        # Or check sign of G around root.
                        # Simple check: psd should be positive.
                        if val > 0:
                            # Actually need to check if it's a Peak or Valid
                            # Rational function can have local minima or inflection.
                            # We can check R''(w) or check if value is "high enough".
                            # Let's assume positive peaks.
                            valid_peaks.append((r_real, val))

            # Sort by value desc
            valid_peaks.sort(key=lambda x: x[1], reverse=True)

            analytical_freqs = [p[0] for p in valid_peaks]
            analytical_vals = [p[1] for p in valid_peaks]

            # Reconstruct fitted curve for plotting
            y_fit = self._rational_func(freqs, *popt)

            # 1. Denom Roots (Poles/Zeros)
            poles = [r for r in D.roots]
            denom_zeros = [complex(r) for r in poles]

            # 2. Denom Minima (Roots of D')
            denom_min_roots = D_deriv.roots
            denom_minima = [
                np.real(r)
                for r in denom_min_roots
                if np.isreal(r) and f_min <= np.real(r) <= f_max
            ]

            props = {
                "params": popt.tolist(),
                "fitted_curve": y_fit,
                "analytical_peaks": analytical_freqs,
                "denominator_zeros": denom_zeros,
                "denominator_minima": denom_minima,
            }

            # Select output feature based on config
            target_freqs = []
            target_vals = []
            feat = config.feature

            if feat == "analytical_peak":
                target_freqs = analytical_freqs
                target_vals = analytical_vals

            elif feat == "denominator_zero":
                # Return Real part as Freq, Imag part as Value
                for p in denom_zeros:
                    r_real = p.real
                    if f_min <= r_real <= f_max:
                        target_freqs.append(r_real)
                        target_vals.append(p.imag)
                # Sort by freq (Real part)
                if target_freqs:
                    pairs = sorted(
                        zip(target_freqs, target_vals, strict=False), key=lambda x: x[0]
                    )
                    target_freqs = [pair[0] for pair in pairs]
                    target_vals = [pair[1] for pair in pairs]

            elif feat == "denominator_min":
                # Return Real part as Freq, Value of D as Value
                for m in denom_minima:
                    target_freqs.append(float(m))
                    val_D = float(np.polyval(D, m))
                    target_vals.append(val_D)
                # Sort by freq
                if target_freqs:
                    pairs = sorted(
                        zip(target_freqs, target_vals, strict=False), key=lambda x: x[0]
                    )
                    target_freqs = [pair[0] for pair in pairs]
                    target_vals = [pair[1] for pair in pairs]

            # Map to nearest grid indices
            indices = []
            for tf in target_freqs:
                idx = np.argmin(np.abs(freqs - tf))
                indices.append(int(idx))

            return PeakInfo(
                indices=indices,
                frequencies=target_freqs,
                values=target_vals,
                properties=props,
            )

        except Exception:
            # Fit failed
            return PeakInfo(indices=[], frequencies=[], values=[])

    def _eval_rational(self, w, params):
        # Quick eval helper
        # Could reuse _rational_func but it unpacks params every time
        # Reuse construct
        N, D = self._construct_polys(params)
        return np.polyval(N, w) / (np.polyval(D, w) + 1e-20)
