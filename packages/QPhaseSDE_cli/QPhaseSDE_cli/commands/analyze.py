from __future__ import annotations

"""Analyze commands that operate on existing run outputs without recompute.

Phase subcommand renders phase-portrait figures using saved specs or overrides.
"""

from pathlib import Path
import json
import yaml
import importlib
from typing import Dict, Any, List
import numpy as np
import typer

app = typer.Typer()


def _resolve_run_dir(from_run: str | None, runs_root: str) -> Path:
	"""Resolve a run directory from an absolute path or from a run-id under root."""
	if from_run is None:
		raise typer.BadParameter("--from-run required")
	p = Path(from_run)
	if p.exists():
		return p
	# else treat as run-id under runs_root
	return Path(runs_root) / from_run


@app.command()
def phase(
	from_run: str = typer.Option(..., help="Run directory path or run-id under --runs-root"),
	runs_root: str = typer.Option("runs", help="Root directory for runs if --from-run is an id"),
	# Optional: use stored specs or override via JSON
	use_snapshot: bool = typer.Option(True, help="Use viz specs saved in snapshot if available"),
	specs_json: str = typer.Option("", help="Optional JSON list of viz specs to override, each with kind/modes/t_range"),
):
	"""Render phase portraits from an existing run's time series.
	If use_snapshot is True and a triad snapshot or viz info exists, those specs are used.
	Otherwise, you can pass specs_json to specify figures.
	"""
	run_dir = _resolve_run_dir(from_run, runs_root)
	ts_dir = run_dir / "time_series"
	if not ts_dir.exists():
		raise typer.BadParameter(f"No time_series folder found at {ts_dir}")
	ts_files = sorted(ts_dir.glob("timeseries*.npz"))
	if not ts_files:
		raise typer.BadParameter(f"No time series files found under {ts_dir}")

	specs = []
	psd_specs = []
	style_cfg = None
	style_psd_cfg = None
	replot_mode = 'append'
	snap_cfg_yaml = run_dir / "config_snapshot" / "config.yaml"
	snap_viz_yaml = run_dir / "config_snapshot" / "visualization.yaml"
	# Back-compat JSON fallbacks
	snap_cfg_json = run_dir / "config_snapshot" / "config.json"
	snap_viz_json = run_dir / "config_snapshot" / "visualization.json"
	snap_triad_yaml = run_dir / "config_snapshot" / "triad.yaml"

	if use_snapshot:
		# prefer visualization.yaml (decoupled from config.yaml)
		if snap_viz_yaml.exists():
			try:
				meta = yaml.safe_load(snap_viz_yaml.read_text(encoding='utf-8')) or {}
				viz_meta = meta.get('visualization') or {}
				phase_meta = viz_meta.get('phase_portrait') if isinstance(viz_meta, dict) else None
				if isinstance(phase_meta, list):
					specs = phase_meta
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
				style_all = meta.get('profile_visualization', {})
				if isinstance(style_all, dict):
					style_cfg = style_all.get('phase_portrait') or style_all.get('phase')
					style_psd_cfg = style_all.get('psd')
					# replot_mode may be inside the phase_portrait dict or top-level
					if isinstance(style_cfg, dict) and isinstance(style_cfg.get('replot_mode'), str):
						replot_mode = style_cfg.get('replot_mode')  # type: ignore
					elif isinstance(style_all.get('replot_mode'), str):
						replot_mode = style_all.get('replot_mode')  # type: ignore
			except Exception:
				pass
		# fallback to visualization.json if yaml not present
		if not (specs or psd_specs) and snap_viz_json.exists():
			try:
				meta = json.loads(snap_viz_json.read_text(encoding='utf-8'))
				viz_meta = meta.get('visualization') or {}
				phase_meta = viz_meta.get('phase_portrait') if isinstance(viz_meta, dict) else None
				if isinstance(phase_meta, list):
					specs = phase_meta
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
				style_all = meta.get('profile_visualization', {})
				if isinstance(style_all, dict):
					style_cfg = style_all.get('phase_portrait') or style_all.get('phase')
					style_psd_cfg = style_all.get('psd')
					# replot_mode may be inside the phase_portrait dict or top-level
					if isinstance(style_cfg, dict) and isinstance(style_cfg.get('replot_mode'), str):
						replot_mode = style_cfg.get('replot_mode')  # type: ignore
					elif isinstance(style_all.get('replot_mode'), str):
						replot_mode = style_all.get('replot_mode')  # type: ignore
			except Exception:
				pass
		# prefer YAML triad if present
		if not (specs or psd_specs) and snap_triad_yaml.exists():
			try:
				from QPhaseSDE_cli.config.loader import load_triad_config
				triad = load_triad_config(snap_triad_yaml)
				rv = getattr(triad.run, 'visualization', None) or getattr(triad.run, 'viz', None)
				viz_phase = getattr(rv, 'phase_portrait', None) or getattr(rv, 'phase', None)
				if viz_phase:
					specs = [s.model_dump() for s in viz_phase]
				viz_psd = getattr(rv, 'psd', None)
				if viz_psd:
					psd_specs = [s.model_dump() for s in viz_psd]
				prof_viz = getattr(triad.profile, 'visualization', None) or getattr(triad.profile, 'viz', None)
				if isinstance(prof_viz, dict):
					style_cfg = prof_viz.get('phase_portrait') or prof_viz.get('phase')
					style_psd_cfg = prof_viz.get('psd')
					pm = None
					if isinstance(style_cfg, dict):
						pm = style_cfg.get('replot_mode')
					if pm is None and isinstance(prof_viz, dict):
						pm = prof_viz.get('replot_mode')
					if isinstance(pm, str):
						replot_mode = pm
			except Exception:
				pass
		# else fallback to config.yaml/json viz (legacy embedding)
		if not (specs or psd_specs) and snap_cfg_yaml.exists():
			try:
				meta = yaml.safe_load(snap_cfg_yaml.read_text(encoding='utf-8')) or {}
				viz_meta = meta.get('visualization') or meta.get('viz') or {}
				phase_meta = viz_meta.get('phase_portrait') if isinstance(viz_meta, dict) else None
				if isinstance(phase_meta, list):
					specs = phase_meta
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
				style_all = meta.get('profile_visualization', {})
				if isinstance(style_all, dict):
					style_cfg = style_all.get('phase_portrait') or style_all.get('phase')
					style_psd_cfg = style_all.get('psd')
			except Exception:
				pass
		if not (specs or psd_specs) and snap_cfg_json.exists():
			try:
				meta = json.loads(snap_cfg_json.read_text(encoding='utf-8'))
				viz_meta = meta.get('visualization') or meta.get('viz') or {}
				phase_meta = viz_meta.get('phase_portrait') if isinstance(viz_meta, dict) else None
				if isinstance(phase_meta, list):
					specs = phase_meta
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
				style_all = meta.get('profile_visualization', {})
				if isinstance(style_all, dict):
					style_cfg = style_all.get('phase_portrait') or style_all.get('phase')
					style_psd_cfg = style_all.get('psd')
			except Exception:
				pass

	if specs_json:
		try:
			specs = json.loads(specs_json)
		except Exception as e:
			raise typer.BadParameter(f"Invalid specs_json: {e}")

	if not (specs or psd_specs):
		raise typer.BadParameter("No viz specs found. Use use_snapshot=True with saved viz, or pass --specs-json.")

	service_mod = importlib.import_module('QPhaseSDE.visualizers.service')
	render_from_spec = getattr(service_mod, 'render_from_spec')

	saved_all = []
	for f in ts_files:
		ic_tag = f.stem.replace('timeseries', '').strip('_') or 'ic00'
		with np.load(f) as npz:
			data = npz["data"]
			t0 = float(npz["t0"]) if "t0" in npz else 0.0
			dt = float(npz["dt"]) if "dt" in npz else 1.0
		out_dir = run_dir / 'figures' / ic_tag
		# ensure output directory exists
		out_dir.mkdir(parents=True, exist_ok=True)
		# handle clean mode by wiping directory first
		if replot_mode == 'clean' and out_dir.exists():
			for pf in out_dir.glob('*.png'):
				try:
					pf.unlink()
				except Exception:
					pass
		saved = []
		if specs:
			for spec in specs:
				k = spec.get('kind') if isinstance(spec, dict) else None
				per_kind_style = style_cfg.get(str(k)) if isinstance(style_cfg, dict) and k is not None else None
				meta = render_from_spec(spec, data, t0=t0, dt=dt, outdir=out_dir, style_overrides=per_kind_style, save=True)
				p = meta.get('path') if isinstance(meta, dict) else None
				if p is not None:
					saved.append(p)
		if psd_specs:
			for spec in psd_specs:
				if isinstance(spec, dict):
					spec.setdefault('kind', 'psd')
				meta = render_from_spec(spec, data, t0=t0, dt=dt, outdir=out_dir, style_overrides=style_psd_cfg, save=True)
				p = meta.get('path') if isinstance(meta, dict) else None
				if p is not None:
					saved.append(p)
		saved_all.extend(saved)

	typer.echo(json.dumps({
		"run_dir": str(run_dir),
		"saved": [str(p) for p in saved_all],
	}))


@app.command()
def psd(
	from_run: str = typer.Option(..., help="Run directory path or run-id under --runs-root"),
	runs_root: str = typer.Option("runs", help="Root directory for runs if --from-run is an id"),
	use_snapshot: bool = typer.Option(True, help="Use PSD specs saved in snapshot if available"),
	specs_json: str = typer.Option("", help="Optional JSON list of PSD specs to override, each with kind/modes/xlim/t_range"),
):
	"""Render PSD figures from an existing run's time series.
	If use_snapshot is True and a triad snapshot or viz info exists, those PSD specs are used.
	Otherwise, you can pass specs_json to specify figures.
	"""
	run_dir = _resolve_run_dir(from_run, runs_root)
	ts_dir = run_dir / "time_series"
	if not ts_dir.exists():
		raise typer.BadParameter(f"No time_series folder found at {ts_dir}")
	ts_files = sorted(ts_dir.glob("timeseries*.npz"))
	if not ts_files:
		raise typer.BadParameter(f"No time series files found under {ts_dir}")

	psd_specs: List[Dict[str, Any]] = []
	style_psd_cfg = None
	# snapshot sources
	snap_viz_yaml = run_dir / "config_snapshot" / "visualization.yaml"
	snap_viz_json = run_dir / "config_snapshot" / "visualization.json"
	snap_triad_yaml = run_dir / "config_snapshot" / "triad.yaml"
	snap_cfg_yaml = run_dir / "config_snapshot" / "config.yaml"
	snap_cfg_json = run_dir / "config_snapshot" / "config.json"

	if use_snapshot:
		if snap_viz_yaml.exists():
			try:
				meta = yaml.safe_load(snap_viz_yaml.read_text(encoding='utf-8')) or {}
				viz_meta = meta.get('visualization') or {}
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
				style_all = meta.get('profile_visualization', {})
				if isinstance(style_all, dict):
					style_psd_cfg = style_all.get('psd')
			except Exception:
				pass
		if not psd_specs and snap_viz_json.exists():
			try:
				meta = json.loads(snap_viz_json.read_text(encoding='utf-8'))
				viz_meta = meta.get('visualization') or {}
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
				style_all = meta.get('profile_visualization', {})
				if isinstance(style_all, dict):
					style_psd_cfg = style_all.get('psd')
			except Exception:
				pass
		if not psd_specs and snap_triad_yaml.exists():
			try:
				from QPhaseSDE_cli.config.loader import load_triad_config
				triad = load_triad_config(snap_triad_yaml)
				rv = getattr(triad.run, 'visualization', None) or getattr(triad.run, 'viz', None)
				viz_psd = getattr(rv, 'psd', None)
				if viz_psd:
					psd_specs = [s.model_dump() for s in viz_psd]
				prof_viz = getattr(triad.profile, 'visualization', None) or getattr(triad.profile, 'viz', None)
				if isinstance(prof_viz, dict):
					style_psd_cfg = prof_viz.get('psd')
			except Exception:
				pass
		if not psd_specs and snap_cfg_yaml.exists():
			try:
				meta = yaml.safe_load(snap_cfg_yaml.read_text(encoding='utf-8')) or {}
				viz_meta = meta.get('visualization') or meta.get('viz') or {}
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
			except Exception:
				pass
		if not psd_specs and snap_cfg_json.exists():
			try:
				meta = json.loads(snap_cfg_json.read_text(encoding='utf-8'))
				viz_meta = meta.get('visualization') or meta.get('viz') or {}
				psd_meta = viz_meta.get('psd') if isinstance(viz_meta, dict) else None
				if isinstance(psd_meta, list):
					psd_specs = psd_meta
			except Exception:
				pass

	if specs_json:
		try:
			psd_specs = json.loads(specs_json)
		except Exception as e:
			raise typer.BadParameter(f"Invalid specs_json: {e}")

	if not psd_specs:
		raise typer.BadParameter("No PSD specs found. Use use_snapshot=True with saved viz, or pass --specs-json.")

	service_mod = importlib.import_module('QPhaseSDE.visualizers.service')
	render_from_spec = getattr(service_mod, 'render_from_spec')

	saved_all = []
	for f in ts_files:
		ic_tag = f.stem.replace('timeseries', '').strip('_') or 'ic00'
		with np.load(f) as npz:
			data = npz["data"]
			t0 = float(npz["t0"]) if "t0" in npz else 0.0
			dt = float(npz["dt"]) if "dt" in npz else 1.0
		out_dir = run_dir / 'figures' / ic_tag
		out_dir.mkdir(parents=True, exist_ok=True)
		saved = []
		for spec in psd_specs:
			if isinstance(spec, dict):
				spec.setdefault('kind', 'psd')
			meta = render_from_spec(spec, data, t0=t0, dt=dt, outdir=out_dir, style_overrides=style_psd_cfg, save=True)
			p = meta.get('path') if isinstance(meta, dict) else None
			if p is not None:
				saved.append(p)
		saved_all.extend(saved)

	typer.echo(json.dumps({
		"run_dir": str(run_dir),
		"saved": [str(p) for p in saved_all],
	}))

