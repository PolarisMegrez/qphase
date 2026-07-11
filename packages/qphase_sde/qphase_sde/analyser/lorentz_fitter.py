"""qphase_sde: Lorentzian Fitter (cross-job analyzer)
---------------------------------------------------------
Analyzer plugin that operates on aggregated SDE results (or a run directory)
to perform cross-job Lorentzian fitting, PSD merging, and distribution
collection. It replaces the legacy root-level ``qphase_sde.postprocess`` module.

Public API
----------
``LorentzFitter`` : Cross-job PSD analyzer / Lorentzian fitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from qphase.backend.base import BackendBase
from qphase.core.aggregation import (
    AggregateResult,
    DirectoryInputResult,
    iter_directory_results,
    write_columns_csv,
    write_npz_bundle,
    write_pkl_bundle,
    write_table_csv,
)
from qphase.core.errors import QPhaseError
from qphase.core.protocols import PluginConfigBase, ResultProtocol
from scipy.optimize import curve_fit

from ..result import SDEResult
from .base import Analyzer
from .result import AnalysisResult

__all__ = ["LorentzFitter", "LorentzFitterConfig"]


class LorentzFitterConfig(PluginConfigBase):
    """Configuration for the Lorentzian cross-job analyzer."""

    scan_param: str = Field(..., description="Parameter to use as the scan axis")
    psd_key: str = Field("psd", description="Analysis key containing the PSD payload")
    mode: int = Field(0, description="PSD mode/column index to fit")
    fit_window: float | None = Field(
        None, description="Half-width around the strongest peak for fitting"
    )
    freq_min: float | None = Field(None, description="Minimum frequency for fitting")
    freq_max: float | None = Field(None, description="Maximum frequency for fitting")
    min_r2: float | None = Field(
        None, description="Minimum R^2; below marks status as low_quality"
    )
    min_peak_height: float | None = Field(
        None, description="Minimum peak intensity threshold"
    )
    max_linewidth: float | None = Field(
        None, description="Maximum allowed linewidth threshold"
    )
    clip_by_std: bool = Field(
        False,
        description=(
            "Clip the fitting window to mean +/- clip_sigma * std "
            "using the squared PSD as a weight distribution"
        ),
    )
    clip_sigma: float = Field(
        10.0,
        description=(
            "Number of standard deviations to keep when clip_by_std is enabled "
            "(default 10.0 corresponds to ~5 raw-FWHM for a squared Lorentzian, "
            "since std = linewidth/2)"
        ),
    )
    export_dist: bool = Field(
        False, description="Also collect and export dist/pdist payloads"
    )
    export: list[str] = Field(
        default_factory=lambda: ["fit_results.csv", "psd_merged.csv"],
        description="List of output files to write",
    )
    output_dir: str | None = Field(
        None, description="Output directory; usually injected by the engine"
    )
    pattern: str = Field("*.npz", description="Glob pattern for result files")


@dataclass(frozen=True)
class LorentzFitResult:
    """Lorentzian fit result for a single PSD trace."""

    center: float = float("nan")
    linewidth: float = float("nan")
    base: float = float("nan")
    peak_intensity: float = float("nan")
    R2: float = float("nan")
    status: str = "failed"
    error: str = ""
    warning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "linewidth": self.linewidth,
            "base": self.base,
            "amplitude": self.peak_intensity - self.base,
            "peak_intensity": self.peak_intensity,
            "R2": self.R2,
            "status": self.status,
            "error": self.error,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class LoadedResult:
    """Normalized view of one saved SDE job result."""

    path: Path
    job_name: str
    result: SDEResult

    @property
    def meta(self) -> dict[str, Any]:
        return self.result.meta

    @property
    def analysis(self) -> dict[str, Any]:
        return self.result.analysis


def _lorentzian_with_baseline(
    x: np.ndarray, center: float, gamma: float, amplitude: float, base: float
) -> np.ndarray:
    """Return a Lorentzian peak on a constant baseline."""
    return amplitude * (gamma**2 / ((x - center) ** 2 + gamma**2)) + base


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _squared_weighted_moments(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return the mean and std of x weighted by (y - min(y))**2.

    Squaring the PSD emphasizes the peak region and yields a finite second
    moment for Lorentzian-like data. Returns (mean, std).
    """
    weights = (y - np.min(y)) ** 2
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or x.size < 4:
        return float(np.mean(x)), float(np.std(x))

    mean = float(np.sum(x * weights) / weight_sum)
    variance = float(np.sum(weights * (x - mean) ** 2) / weight_sum)
    std = float(np.sqrt(max(variance, 0.0)))
    return mean, std


def _apply_std_clip(
    x: np.ndarray, y: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a clipped (x, y) window around the squared-PSD-weighted mean.

    Weights are derived from ``(y - y.min())**2`` to reduce baseline influence.
    If the clip would leave fewer than 4 samples, the original arrays are
    returned unchanged.
    """
    mean, std = _squared_weighted_moments(x, y)
    if std <= 0.0:
        return x, y

    mask = np.abs(x - mean) <= sigma * std
    if int(np.count_nonzero(mask)) < 4:
        return x, y
    return x[mask], y[mask]


def fit_lorentzian(
    axis: np.ndarray,
    psd: np.ndarray,
    *,
    fit_window: float | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    min_r2: float | None = None,
    min_peak_height: float | None = None,
    max_linewidth: float | None = None,
    clip_by_std: bool = False,
    clip_sigma: float = 10.0,
) -> LorentzFitResult:
    """Fit one PSD trace with a Lorentzian plus baseline.

    Optional ``clip_by_std`` first tightens the frequency window around the
    squared-PSD-weighted center, which helps ignore distant long-tail bumps and
    speeds up fitting on very wide frequency grids.

    After a successful fit, the squared-PSD-weighted standard deviation of the
    full input is compared with the Lorentzian expectation ``std = linewidth/2``.
    If the two differ by more than a factor of two, a warning is recorded in the
    ``warning`` field, indicating the data may not be single-peaked Lorentzian.

    Optional quality thresholds mark the result as ``low_quality`` (rather than
    ``failed``) when the fit succeeds numerically but does not meet the requested
    criteria.
    """
    try:
        x = np.asarray(axis, dtype=float).reshape(-1)
        y = np.asarray(psd, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 4:
            raise ValueError("axis and psd must have the same length >= 4")

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 4:
            raise ValueError("not enough finite samples for Lorentzian fit")

        if freq_min is not None or freq_max is not None:
            freq_mask = np.ones_like(x, dtype=bool)
            if freq_min is not None:
                freq_mask &= x >= freq_min
            if freq_max is not None:
                freq_mask &= x <= freq_max
            x = x[freq_mask]
            y = y[freq_mask]
            if x.size < 4:
                raise ValueError(
                    "not enough samples in requested frequency range for fit"
                )

        # Characterize the full input before any windowing. This is used both for
        # optional tail clipping and for post-fit single-peak validation.
        _, full_std = _squared_weighted_moments(x, y)

        peak_idx = int(np.argmax(y))
        if fit_window is not None and fit_window > 0:
            center_guess = x[peak_idx]
            window_mask = np.abs(x - center_guess) <= fit_window
            if int(np.count_nonzero(window_mask)) >= 4:
                x = x[window_mask]
                y = y[window_mask]
                peak_idx = int(np.argmax(y))

        if clip_by_std:
            x, y = _apply_std_clip(x, y, clip_sigma)
            if x.size < 4:
                raise ValueError("not enough samples after std-based tail clipping")
            peak_idx = int(np.argmax(y))

        step = float(np.median(np.abs(np.diff(np.sort(x))))) if x.size > 1 else 1.0
        span = float(np.max(x) - np.min(x))
        gamma_init = max(step * 5.0, span / max(x.size, 1), 1e-12)
        amplitude_init = max(float(np.max(y) - np.min(y)), 1e-12)
        base_init = float(np.min(y))

        popt, _ = curve_fit(
            _lorentzian_with_baseline,
            x,
            y,
            p0=[float(x[peak_idx]), gamma_init, amplitude_init, base_init],
            bounds=(
                [float(np.min(x)), 1e-12, 0.0, -np.inf],
                [float(np.max(x)), max(span, 1e-12), np.inf, np.inf],
            ),
            maxfev=10000,
        )

        center, gamma, amplitude, base = (float(value) for value in popt)
        fitted = _lorentzian_with_baseline(x, *popt)
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        linewidth = 2.0 * gamma
        peak_intensity = amplitude + base

        reasons: list[str] = []
        if min_r2 is not None and r2 < min_r2:
            reasons.append(f"R2={r2:.4f} < {min_r2}")
        if min_peak_height is not None and peak_intensity < min_peak_height:
            reasons.append(f"peak_intensity={peak_intensity:.4e} < {min_peak_height}")
        if max_linewidth is not None and linewidth > max_linewidth:
            reasons.append(f"linewidth={linewidth:.4e} > {max_linewidth}")

        if reasons:
            status = "low_quality"
            error = "; ".join(reasons)
        else:
            status = "ok"
            error = ""

        # Validate single-peak / Lorentzian-like character using the squared-PSD
        # weighted standard deviation. For a normalized squared Lorentzian the
        # standard deviation equals gamma = linewidth/2.
        warning = ""
        if full_std > 0.0 and linewidth > 0.0:
            expected_std = linewidth / 2.0
            ratio = full_std / expected_std
            if ratio < 0.5 or ratio > 2.0:
                warning = (
                    f"squared-weighted std {full_std:.3e} deviates from "
                    f"Lorentzian expectation {expected_std:.3e} by factor "
                    f"{ratio:.2f}; data may not be single-peaked Lorentzian"
                )

        return LorentzFitResult(
            center=center,
            linewidth=linewidth,
            base=base,
            peak_intensity=peak_intensity,
            R2=r2,
            status=status,
            error=error,
            warning=warning,
        )
    except Exception as exc:
        return LorentzFitResult(error=str(exc))


def extract_psd_trace(
    loaded: LoadedResult,
    *,
    psd_key: str = "psd",
    mode: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract one mode trace from an analyser PSD payload."""
    if psd_key not in loaded.analysis:
        raise QPhaseError(f"{loaded.path} has no analysis[{psd_key!r}] payload")

    psd_payload = loaded.analysis[psd_key]
    axis = np.asarray(psd_payload["axis"], dtype=float).reshape(-1)
    psd_matrix = np.asarray(psd_payload["psd"], dtype=float)
    modes = list(psd_payload.get("modes", []))

    if psd_matrix.ndim == 1:
        trace = psd_matrix
    else:
        if mode in modes:
            mode_index = modes.index(mode)
        elif 0 <= mode < psd_matrix.shape[1]:
            mode_index = mode
        else:
            raise QPhaseError(
                f"Mode {mode} not found in {loaded.path}; available modes: {modes}"
            )
        trace = psd_matrix[:, mode_index]

    trace = np.asarray(trace, dtype=float).reshape(-1)
    if axis.size != trace.size:
        raise QPhaseError(
            f"PSD axis length {axis.size} does not match trace length "
            f"{trace.size} in {loaded.path}"
        )
    return axis, trace, psd_payload


def _collect_distribution(
    loaded: LoadedResult,
    key: str,
    mode: int,
    scan_param: str,
    scan_value: Any,
    rows: list[dict[str, Any]],
) -> None:
    payload = loaded.analysis.get(key)
    if not isinstance(payload, dict):
        return
    distributions = payload.get("distributions")
    if not isinstance(distributions, dict) or mode not in distributions:
        return
    row = dict(distributions[mode])
    row[scan_param] = scan_value
    row["job_name"] = loaded.job_name
    rows.append(row)


def _load_input(data: Any, pattern: str) -> list[LoadedResult]:
    """Normalize analyzer input into a list of ``LoadedResult``."""
    if isinstance(data, DirectoryInputResult):
        data = data.path

    if isinstance(data, (str, Path)):
        paths = iter_directory_results(data, pattern)
        return [
            LoadedResult(
                path=path,
                job_name=path.parent.name or path.stem,
                result=SDEResult.load(path),
            )
            for path in paths
        ]

    if isinstance(data, AggregateResult):
        data = data.results

    if isinstance(data, dict):
        loaded: list[LoadedResult] = []
        for name, result in data.items():
            if isinstance(result, SDEResult):
                loaded.append(
                    LoadedResult(
                        path=Path("."),
                        job_name=name,
                        result=result,
                    )
                )
            else:
                # Try to unwrap a GenericResult that wraps an SDEResult path/dict
                res_data = getattr(result, "data", None)
                if isinstance(res_data, SDEResult):
                    loaded.append(
                        LoadedResult(
                            path=Path("."),
                            job_name=name,
                            result=res_data,
                        )
                    )
        if not loaded:
            raise QPhaseError(
                "LorentzFitter received a dict but no SDEResult values were found"
            )
        return loaded

    if isinstance(data, SDEResult):
        return [LoadedResult(path=Path("."), job_name="single", result=data)]

    raise QPhaseError(
        f"LorentzFitter received unsupported input type: {type(data).__name__}"
    )


class LorentzFitter(Analyzer):
    """Cross-job analyzer that fits Lorentzians to aggregated PSD data."""

    name: ClassVar[str] = "lorentz_fitter"
    description: ClassVar[str] = (
        "Cross-job Lorentzian fitter for aggregated SDE PSD results"
    )
    config_schema: ClassVar[type[LorentzFitterConfig]] = LorentzFitterConfig

    def __init__(self, config: LorentzFitterConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        """Fit Lorentzians to aggregated PSD data and export merged outputs.

        Parameters
        ----------
        data : Any
            ``AggregateResult``, directory path, or ``DirectoryInputResult``.
        backend : BackendBase
            Backend instance (unused; kept for protocol compatibility).

        Returns
        -------
        AnalysisResult
            Result containing fit rows and export metadata.

        """
        del backend  # not used; we operate on already-computed PSD arrays
        config = self.config
        if not isinstance(config, LorentzFitterConfig):
            raise RuntimeError("LorentzFitter config not initialized")

        loaded_results = _load_input(data, config.pattern)
        if not loaded_results:
            raise QPhaseError("LorentzFitter received no input results")

        fit_rows: list[dict[str, Any]] = []
        psd_columns: dict[str, np.ndarray] = {}
        dist_rows: list[dict[str, Any]] = []
        pdist_rows: list[dict[str, Any]] = []
        reference_axis: np.ndarray | None = None

        for loaded in loaded_results:
            params = loaded.meta.get("params", {})
            if config.scan_param not in params:
                raise QPhaseError(
                    f"{loaded.path} meta.params is missing {config.scan_param!r}"
                )
            scan_value = params[config.scan_param]

            axis, trace, _ = extract_psd_trace(
                loaded, psd_key=config.psd_key, mode=config.mode
            )
            if reference_axis is None:
                reference_axis = axis
            elif reference_axis.shape != axis.shape or not np.allclose(
                reference_axis, axis
            ):
                raise QPhaseError(f"PSD frequency axis differs in {loaded.path}")

            fit_result = fit_lorentzian(
                axis,
                trace,
                fit_window=config.fit_window,
                freq_min=config.freq_min,
                freq_max=config.freq_max,
                min_r2=config.min_r2,
                min_peak_height=config.min_peak_height,
                max_linewidth=config.max_linewidth,
                clip_by_std=config.clip_by_std,
                clip_sigma=config.clip_sigma,
            )
            row = {
                "job_name": loaded.job_name,
                config.scan_param: scan_value,
                **fit_result.as_dict(),
            }
            fit_rows.append(row)
            psd_columns[str(scan_value)] = trace

            if config.export_dist:
                _collect_distribution(
                    loaded,
                    "dist",
                    config.mode,
                    config.scan_param,
                    scan_value,
                    dist_rows,
                )
                _collect_distribution(
                    loaded,
                    "pdist",
                    config.mode,
                    config.scan_param,
                    scan_value,
                    pdist_rows,
                )

        fit_rows.sort(key=lambda row: _sort_key(row[config.scan_param]))
        psd_columns = dict(
            sorted(psd_columns.items(), key=lambda item: _sort_key(item[0]))
        )

        output_dir = self._resolve_output_dir(config)
        written: dict[str, Path] = {}
        if output_dir is not None and config.export:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            if "fit_results.csv" in config.export:
                fieldnames = [
                    "job_name",
                    config.scan_param,
                    "center",
                    "linewidth",
                    "base",
                    "amplitude",
                    "peak_intensity",
                    "R2",
                    "status",
                    "error",
                    "warning",
                ]
                written["fit_results"] = write_table_csv(
                    fit_rows, out / "fit_results.csv", fieldnames=fieldnames
                )

            if "psd_merged.csv" in config.export and reference_axis is not None:
                written["psd_merged"] = write_columns_csv(
                    reference_axis, psd_columns, out / "psd_merged.csv"
                )

            if config.export_dist and dist_rows:
                written["dist_merged"] = write_npz_bundle(
                    out / "dist_merged.npz",
                    dist_list=np.array(dist_rows, dtype=object),
                    scan_params=np.array(
                        [row[config.scan_param] for row in dist_rows], dtype=object
                    ),
                )

            if config.export_dist and pdist_rows:
                written["pdist_merged"] = write_pkl_bundle(
                    out / "pdist_merged.pkl", pdist_rows
                )

        return AnalysisResult(
            data_dict={
                "fit_rows": fit_rows,
                "psd_columns": psd_columns,
                "axis": reference_axis,
                "written": {k: str(v) for k, v in written.items()},
            },
            meta={
                "scan_param": config.scan_param,
                "mode": config.mode,
                "count": len(loaded_results),
            },
        )

    def _resolve_output_dir(self, config: LorentzFitterConfig) -> Path | None:
        """Return the output directory, preferring the engine-injected value."""
        injected = getattr(self, "output_dir", None)
        if injected is not None:
            return Path(injected)
        if config.output_dir is not None:
            return Path(config.output_dir)
        return None
