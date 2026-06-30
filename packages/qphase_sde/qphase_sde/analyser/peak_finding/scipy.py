from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field
from scipy.signal import find_peaks, savgol_filter

from .base import BasePeakFinderConfig, PeakFinder, PeakInfo


class ScipyPeakFinderConfig(BasePeakFinderConfig):
    """Configuration for Scipy-based peak finder."""

    method: Literal["scipy"] = "scipy"
    min_height: float | None = Field(None, description="Minimum peak height")
    prominence: float | None = Field(None, description="Peak prominence")
    distance: int | None = Field(None, description="Minimum horizontal distance")
    smooth_window: int | None = Field(
        5, description="Window length for smoothing before peak finding"
    )
    noise_threshold: float | None = Field(
        3.0, description="Threshold relative to noise floor"
    )
    max_peaks: int | None = Field(None, description="Maximum number of peaks to return")


class ScipyPeakFinder(PeakFinder):
    """Standard peak finding using scipy.signal.find_peaks with smoothing."""

    name: ClassVar[str] = "scipy"
    description: ClassVar[str] = "Standard peak finding using scipy.signal.find_peaks"
    config_schema: ClassVar[type[ScipyPeakFinderConfig]] = ScipyPeakFinderConfig

    def __init__(
        self, config: ScipyPeakFinderConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(config, **kwargs)

    def find_peaks(self, freqs: np.ndarray, psd: np.ndarray) -> PeakInfo:
        config = cast(ScipyPeakFinderConfig, self.config)

        # Robust Smoothing Strategy
        # 1. Work in Log domain (dB)
        # 2. Apply Savitzky-Golay filter
        # 3. Convert back to linear

        p_log = np.log10(psd + 1e-20)

        w_len = config.smooth_window or 5
        if w_len % 2 == 0:
            w_len += 1

        try:
            # Polyorder 2 preserves peak shapes better than higher orders
            p_log_smooth = savgol_filter(p_log, w_len, 2)
            p_smooth = 10**p_log_smooth
        except Exception:
            p_smooth = psd.copy()
            p_log_smooth = p_log

        fp_kwargs = {}

        # Height Threshold logic
        calc_min_h = None
        noise_thresh = config.noise_threshold

        if noise_thresh is not None:
            noise_floor = np.median(p_smooth)
            calc_min_h = noise_floor * noise_thresh

        final_min_h = calc_min_h
        if config.min_height is not None:
            if final_min_h is not None:
                final_min_h = max(final_min_h, config.min_height)
            else:
                final_min_h = config.min_height

        if final_min_h is not None:
            fp_kwargs["height"] = final_min_h

        # Prominence
        if config.prominence is not None:
            fp_kwargs["prominence"] = config.prominence
        elif calc_min_h is not None:
            fp_kwargs["prominence"] = calc_min_h * 0.5

        if config.distance is not None:
            fp_kwargs["distance"] = config.distance
        else:
            fp_kwargs["distance"] = w_len

        # Execute
        peaks, props = find_peaks(p_smooth, **fp_kwargs)

        # Quadratic Refinement
        refined_freqs = []
        refined_vals = []

        for pk in peaks:
            f_pk = freqs[pk]
            v_pk = p_smooth[pk]

            # 3-point Gaussian/Parabolic fit on Log
            if 0 < pk < len(p_log_smooth) - 1:
                y1 = p_log_smooth[pk - 1]
                y2 = p_log_smooth[pk]
                y3 = p_log_smooth[pk + 1]

                denom = 2 * (y1 - 2 * y2 + y3)
                if denom != 0:
                    delta = (y1 - y3) / denom
                    if -0.5 <= delta <= 0.5:
                        df = freqs[1] - freqs[0]
                        f_pk = freqs[pk] + delta * df
                        # v_pk update skipped for simplicity, using smoothed value

            refined_freqs.append(f_pk)
            refined_vals.append(v_pk)

        r_freqs = np.array(refined_freqs)
        r_vals = np.array(refined_vals)

        # Max Peaks Filter
        if config.max_peaks is not None and len(peaks) > config.max_peaks:
            top_indices = np.argsort(r_vals)[-config.max_peaks :]
            top_indices = np.sort(top_indices)

            peaks = peaks[top_indices]
            r_freqs = r_freqs[top_indices]
            r_vals = r_vals[top_indices]

            for k, v in props.items():
                props[k] = v[top_indices]

        return PeakInfo(
            indices=peaks.tolist(),
            frequencies=r_freqs.tolist(),
            values=r_vals.tolist(),
            properties=props,
        )
