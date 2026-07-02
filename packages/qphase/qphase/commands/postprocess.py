"""Postprocess saved QPhase SDE results."""

from __future__ import annotations

from pathlib import Path

import typer

from qphase.core.errors import QPhaseError

# Module-level singletons for Typer defaults to satisfy B008.
_RUN_DIR_ARG = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=True,
    readable=True,
    help="Run directory or single SDE .npz result file",
)

_SCAN_PARAM_OPT = typer.Option(
    ..., "--scan-param", "-s", help="Parameter name under meta.params"
)

_MODE_OPT = typer.Option(0, "--mode", "-m", help="PSD mode to fit/export")

_PSD_KEY_OPT = typer.Option("psd", help="Analysis key containing PSD payload")

_OUTPUT_DIR_OPT = typer.Option(
    None,
    "--output-dir",
    "-o",
    file_okay=False,
    dir_okay=True,
    help="Directory for fit_results.csv and psd_merged.csv",
)

_FIT_WINDOW_OPT = typer.Option(
    None,
    help="Optional half-width around the strongest peak used for fitting",
)

_FREQ_MIN_OPT = typer.Option(
    None, "--freq-min", help="Minimum frequency to include in the Lorentz fit"
)

_FREQ_MAX_OPT = typer.Option(
    None, "--freq-max", help="Maximum frequency to include in the Lorentz fit"
)

_MIN_R2_OPT = typer.Option(
    None, "--min-r2", help="Minimum R^2; lower values are marked low_quality"
)

_MIN_PEAK_HEIGHT_OPT = typer.Option(
    None,
    "--min-peak-height",
    help="Minimum peak intensity; lower values are marked low_quality",
)

_MAX_LINEWIDTH_OPT = typer.Option(
    None,
    "--max-linewidth",
    help="Maximum linewidth; larger values are marked low_quality",
)

_EXPORT_DIST_OPT = typer.Option(
    False, help="Also export experimental dist_merged.npz/pdist_merged.pkl"
)

_OVERWRITE_OPT = typer.Option(False, help="Overwrite existing output files")

_DRY_RUN_OPT = typer.Option(
    False, "--dry-run", help="List files that would be processed without writing"
)


def postprocess_command(
    run_dir: Path = _RUN_DIR_ARG,
    scan_param: str = _SCAN_PARAM_OPT,
    mode: int = _MODE_OPT,
    psd_key: str = _PSD_KEY_OPT,
    output_dir: Path | None = _OUTPUT_DIR_OPT,
    fit_window: float | None = _FIT_WINDOW_OPT,
    freq_min: float | None = _FREQ_MIN_OPT,
    freq_max: float | None = _FREQ_MAX_OPT,
    min_r2: float | None = _MIN_R2_OPT,
    min_peak_height: float | None = _MIN_PEAK_HEIGHT_OPT,
    max_linewidth: float | None = _MAX_LINEWIDTH_OPT,
    export_dist: bool = _EXPORT_DIST_OPT,
    overwrite: bool = _OVERWRITE_OPT,
    dry_run: bool = _DRY_RUN_OPT,
):
    """Fit Lorentzian PSD peaks and export merged result data."""
    try:
        from qphase_sde.postprocess import (
            export_postprocess_bundle,
            iter_result_files,
            load_result,
            postprocess_run,
        )
    except ImportError as exc:
        typer.echo(f"qphase-sde is required for postprocess: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out_dir = output_dir or (run_dir.parent if run_dir.is_file() else run_dir)

    if dry_run:
        try:
            files = iter_result_files(run_dir)
        except QPhaseError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        typer.echo(f"Would process {len(files)} result file(s):")
        for path in files:
            loaded = load_result(path)
            scan_value = loaded.meta.get("params", {}).get(scan_param, "N/A")
            typer.echo(
                f"  {path.name}: {scan_param}={scan_value}, mode={mode}, "
                f"psd_key={psd_key}"
            )
        typer.echo(f"Outputs would be written to: {out_dir}")
        return

    try:
        bundle = postprocess_run(
            run_dir,
            scan_param=scan_param,
            psd_key=psd_key,
            mode=mode,
            fit_window=fit_window,
            freq_min=freq_min,
            freq_max=freq_max,
            min_r2=min_r2,
            min_peak_height=min_peak_height,
            max_linewidth=max_linewidth,
            export_dist=export_dist,
        )
        written = export_postprocess_bundle(
            bundle,
            out_dir,
            scan_param=scan_param,
            overwrite=overwrite,
            export_dist=export_dist,
        )
    except QPhaseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Processed {len(bundle.fit_rows)} result file(s).")
    for label, path in written.items():
        typer.echo(f"  {label}: {path}")
