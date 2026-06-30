"""Result postprocessing helpers for SDE runs."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.core.errors import QPhaseError
from scipy.optimize import curve_fit

from .result import SDEResult


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "linewidth": self.linewidth,
            "base": self.base,
            "peak_intensity": self.peak_intensity,
            "R2": self.R2,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class PostprocessBundle:
    """Data collected while postprocessing a run directory."""

    fit_rows: list[dict[str, Any]]
    psd_columns: dict[str, np.ndarray]
    axis: np.ndarray | None
    dist_rows: list[dict[str, Any]] = field(default_factory=list)
    pdist_rows: list[dict[str, Any]] = field(default_factory=list)


def lorentzian_with_baseline(
    x: np.ndarray, center: float, gamma: float, amplitude: float, base: float
) -> np.ndarray:
    """Return a Lorentzian peak on a constant baseline."""
    return amplitude * (gamma**2 / ((x - center) ** 2 + gamma**2)) + base


def load_result(path: str | Path) -> LoadedResult:
    """Load one SDE ``.npz`` result file."""
    result_path = Path(path)
    result = SDEResult.load(result_path)
    return LoadedResult(
        path=result_path,
        job_name=result_path.parent.name or result_path.stem,
        result=result,
    )


def iter_result_files(run_dir: str | Path, pattern: str = "*.npz") -> list[Path]:
    """Find saved job result files under a run directory."""
    root = Path(run_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        raise QPhaseError(f"Run directory does not exist: {root}")

    files = sorted(path for path in root.glob(f"*/{pattern}") if path.is_file())
    if not files:
        files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise QPhaseError(f"No result files matching {pattern!r} found under {root}")
    return files


def load_run_results(run_dir: str | Path, pattern: str = "*.npz") -> list[LoadedResult]:
    """Load all result files from a run directory or a single result file."""
    return [load_result(path) for path in iter_result_files(run_dir, pattern)]


def fit_lorentzian(
    axis: np.ndarray,
    psd: np.ndarray,
    *,
    fit_window: float | None = None,
) -> LorentzFitResult:
    """Fit one PSD trace with a Lorentzian plus baseline."""
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

        peak_idx = int(np.argmax(y))
        if fit_window is not None and fit_window > 0:
            center_guess = x[peak_idx]
            window_mask = np.abs(x - center_guess) <= fit_window
            if int(np.count_nonzero(window_mask)) >= 4:
                x = x[window_mask]
                y = y[window_mask]
                peak_idx = int(np.argmax(y))

        step = float(np.median(np.abs(np.diff(np.sort(x))))) if x.size > 1 else 1.0
        span = float(np.max(x) - np.min(x))
        gamma_init = max(step * 5.0, span / max(x.size, 1), 1e-12)
        amplitude_init = max(float(np.max(y) - np.min(y)), 1e-12)
        base_init = float(np.min(y))

        popt, _ = curve_fit(
            lorentzian_with_baseline,
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
        fitted = lorentzian_with_baseline(x, *popt)
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return LorentzFitResult(
            center=center,
            linewidth=2.0 * gamma,
            base=base,
            peak_intensity=amplitude + base,
            R2=r2,
            status="ok",
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
            f"PSD axis length {axis.size} does not match trace length"
            f" {trace.size} in {loaded.path}"
        )
    return axis, trace, psd_payload


def postprocess_run(
    run_dir: str | Path,
    *,
    scan_param: str,
    psd_key: str = "psd",
    mode: int = 0,
    fit_window: float | None = None,
    export_dist: bool = False,
    pattern: str = "*.npz",
) -> PostprocessBundle:
    """Load a run directory, fit PSD traces, and collect exportable data."""
    fit_rows: list[dict[str, Any]] = []
    psd_columns: dict[str, np.ndarray] = {}
    dist_rows: list[dict[str, Any]] = []
    pdist_rows: list[dict[str, Any]] = []
    reference_axis: np.ndarray | None = None

    for loaded in load_run_results(run_dir, pattern):
        params = loaded.meta.get("params", {})
        if scan_param not in params:
            raise QPhaseError(f"{loaded.path} meta.params is missing {scan_param!r}")
        scan_value = params[scan_param]

        axis, trace, _ = extract_psd_trace(loaded, psd_key=psd_key, mode=mode)
        if reference_axis is None:
            reference_axis = axis
        elif reference_axis.shape != axis.shape or not np.allclose(
            reference_axis, axis
        ):
            raise QPhaseError(f"PSD frequency axis differs in {loaded.path}")

        fit_result = fit_lorentzian(axis, trace, fit_window=fit_window)
        row = {
            "job_name": loaded.job_name,
            scan_param: scan_value,
            **fit_result.as_dict(),
        }
        fit_rows.append(row)
        psd_columns[str(scan_value)] = trace

        if export_dist:
            _collect_distribution(
                loaded, "dist", mode, scan_param, scan_value, dist_rows
            )
            _collect_distribution(
                loaded, "pdist", mode, scan_param, scan_value, pdist_rows
            )

    fit_rows.sort(key=lambda row: _sort_key(row[scan_param]))
    psd_columns = dict(sorted(psd_columns.items(), key=lambda item: _sort_key(item[0])))
    return PostprocessBundle(
        fit_rows, psd_columns, reference_axis, dist_rows, pdist_rows
    )


def export_postprocess_bundle(
    bundle: PostprocessBundle,
    output_dir: str | Path,
    *,
    scan_param: str,
    overwrite: bool = False,
    export_dist: bool = False,
) -> dict[str, Path]:
    """Write postprocess outputs to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    fit_path = out / "fit_results.csv"
    psd_path = out / "psd_merged.csv"
    _ensure_writable(fit_path, overwrite)
    _ensure_writable(psd_path, overwrite)

    _write_fit_csv(bundle.fit_rows, fit_path, scan_param)
    _write_psd_csv(bundle.axis, bundle.psd_columns, psd_path)
    written["fit_results"] = fit_path
    written["psd_merged"] = psd_path

    if export_dist and bundle.dist_rows:
        dist_path = out / "dist_merged.npz"
        _ensure_writable(dist_path, overwrite)
        np.savez_compressed(
            dist_path,
            dist_list=np.array(bundle.dist_rows, dtype=object),
            scan_params=np.array(
                [row[scan_param] for row in bundle.dist_rows], dtype=object
            ),
        )
        written["dist_merged"] = dist_path

    if export_dist and bundle.pdist_rows:
        pdist_path = out / "pdist_merged.pkl"
        _ensure_writable(pdist_path, overwrite)
        with pdist_path.open("wb") as handle:
            pickle.dump(bundle.pdist_rows, handle)
        written["pdist_merged"] = pdist_path

    return written


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


def _write_fit_csv(rows: list[dict[str, Any]], path: Path, scan_param: str) -> None:
    fieldnames = [
        "job_name",
        scan_param,
        "center",
        "linewidth",
        "base",
        "peak_intensity",
        "R2",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_psd_csv(
    axis: np.ndarray | None, columns: dict[str, np.ndarray], path: Path
) -> None:
    if axis is None:
        raise QPhaseError("No PSD data available to export")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        labels = list(columns.keys())
        writer.writerow(["frequency", *labels])
        for row_index, freq in enumerate(axis):
            writer.writerow([freq, *(columns[label][row_index] for label in labels)])


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise QPhaseError(f"Output file already exists: {path}; pass --overwrite")


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))
