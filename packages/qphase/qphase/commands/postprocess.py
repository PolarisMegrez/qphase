"""Postprocess saved QPhase SDE results."""

from __future__ import annotations

from pathlib import Path

import typer

from qphase.core.errors import QPhaseError


def postprocess_command(
    run_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Run directory or single SDE .npz result file",
    ),
    scan_param: str = typer.Option(
        ..., "--scan-param", "-s", help="Parameter name under meta.params"
    ),
    mode: int = typer.Option(0, "--mode", "-m", help="PSD mode to fit/export"),
    psd_key: str = typer.Option("psd", help="Analysis key containing PSD payload"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for fit_results.csv and psd_merged.csv",
    ),
    fit_window: float | None = typer.Option(
        None,
        help="Optional half-width around the strongest peak used for fitting",
    ),
    export_dist: bool = typer.Option(
        False, help="Also export experimental dist_merged.npz/pdist_merged.pkl"
    ),
    overwrite: bool = typer.Option(False, help="Overwrite existing output files"),
):
    """Fit Lorentzian PSD peaks and export merged result data."""
    try:
        from qphase_sde.postprocess import export_postprocess_bundle, postprocess_run

        out_dir = output_dir or (run_dir.parent if run_dir.is_file() else run_dir)
        bundle = postprocess_run(
            run_dir,
            scan_param=scan_param,
            psd_key=psd_key,
            mode=mode,
            fit_window=fit_window,
            export_dist=export_dist,
        )
        written = export_postprocess_bundle(
            bundle,
            out_dir,
            scan_param=scan_param,
            overwrite=overwrite,
            export_dist=export_dist,
        )
    except ImportError as exc:
        typer.echo(f"qphase-sde is required for postprocess: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except QPhaseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Processed {len(bundle.fit_rows)} result file(s).")
    for label, path in written.items():
        typer.echo(f"  {label}: {path}")
