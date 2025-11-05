from __future__ import annotations

"""Run commands for launching simulations via CLI.

Uses a jobs-based YAML configuration (profile/run/jobs). Legacy triad files
are migrated automatically to a single-job configuration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import importlib
import typer
from QPhaseSDE.core.errors import (
	QPSError,
	QPSConfigError,
	get_logger,
	configure_logging,
)
from QPhaseSDE.core.config import (
	EngineConfig,
	VisualizerConfig,
	ModelArgs,
	VizJobArgs,
	ConfigPipeline,
	get_default,
)
from QPhaseSDE.core.scheduler import run as scheduler_run


app = typer.Typer()


def _coerce_option_default(value, fallback=None):
	"""Return a plain Python value when this function is called directly (not via Typer).
	Typer passes OptionInfo objects as defaults when called programmatically; here we
	unwrap to their `.default` or use an explicit fallback.
	"""
	# Avoid importing typer models; duck-type on presence of 'default'
	default_attr = getattr(value, "default", None)
	if default_attr is not None:
		return default_attr
	return value if value is not None else fallback


def _parse_kv(pairs: List[str]) -> Dict:
	"""Parse key=value pairs from CLI into a dict with basic type casting."""
	out: Dict = {}
	for s in pairs:
		if "=" not in s:
			continue
		k, v = s.split("=", 1)
		k = k.strip()
		v = v.strip()
		# try to cast
		for caster in (int, float, complex):
			try:
				v_cast = caster(v)
				v = v_cast
				break
			except Exception:
				continue
		out[k] = v
	return out


## No longer loading models directly in CLI; scheduler handles model import/build.


@app.command()
def sde(
	config: Optional[str] = typer.Option(None, help="YAML config with sections: profile/run/jobs (legacy triad migrates automatically)"),
	verbose: bool = typer.Option(False, help="Enable verbose logging"),
	log_file: Optional[str] = typer.Option(None, help="Write logs to file path"),
	log_json: bool = typer.Option(False, help="Log in JSON format"),
	suppress_warnings: bool = typer.Option(False, help="Suppress warnings output"),
	max_storage_gb: Optional[float] = typer.Option(None, help="Override default 1 GiB time-series storage guard (in GB)"),
):
	"""Run SDE simulation(s) and persist results.

	When --config is provided, parameters are read from a triad YAML; otherwise
	legacy flags are used. A run directory is created with a snapshot and NPZ
	results, and optional figures are rendered if configured.
	"""
	# Configure logging early (coerce Typer OptionInfo when called programmatically)
	verbose_val = bool(_coerce_option_default(verbose, fallback=False))
	log_file_val = _coerce_option_default(log_file)
	log_json_val = bool(_coerce_option_default(log_json, fallback=False))
	suppress_val = bool(_coerce_option_default(suppress_warnings, fallback=False))
	configure_logging(verbose=verbose_val, log_file=log_file_val, as_json=log_json_val, suppress_warnings=suppress_val)
	log = get_logger()

	# Resolve system defaults accessor
	cfgsys = importlib.import_module('QPhaseSDE.core.config')
	_get_system = getattr(cfgsys, 'get_system')

	try:
		if config is None:
			raise QPSConfigError("[500] --config is required and the legacy flags path has been removed.")
		# Ensure user workspace paths are importable for model modules like 'models.*'
		cfg_path = Path(config).resolve()
		# Try both the config directory and its parent (common repo root)
		for cand in (cfg_path.parent, cfg_path.parent.parent):
			if cand.exists():
				pstr = str(cand)
				if pstr not in sys.path:
					sys.path.insert(0, pstr)
		# Load jobs-based config (with legacy triad migration)
		from QPhaseSDE_cli.config.loader import load_root_config
		root = load_root_config(config)
		# Build structured pipeline parts
		backend = root.profile.backend or get_default('engine.default_backend', 'numpy')
		solver = root.profile.solver or get_default('engine.default_solver', 'euler')
		rng_stream = root.run.trajectories.rng_stream or get_default('cli.run.trajectories.rng_stream', 'per_trajectory')
		eng_cfg = EngineConfig(backend=backend, solver=solver, progress={}, rng_stream=rng_stream)
		viz_cfg = VisualizerConfig(styles=dict(getattr(root.profile, 'visualizer', {}) or {}))

		# Common runtime time/trajectories
		time_spec = {"t0": float(root.run.time.t0), "dt": float(root.run.time.dt), "steps": int(root.run.time.steps)}
		traj_spec = {"n_traj": int(root.run.trajectories.n_traj)}

		model_args: List[ModelArgs] = []
		viz_args: List[VizJobArgs] = []
		for j in root.jobs:
			ma = ModelArgs(
				name=j.name,
				module=j.module,
				function=j.function,
				params=dict(j.params),
				ic=[list(vec) for vec in j.ic],
				noise=j.noise.model_dump(),  # type: ignore[attr-defined]
				time=dict(time_spec),
				trajectories=dict(traj_spec),
				solver=None,
				backend=None,
			)
			model_args.append(ma)
			if j.visualizer:
				viz_args.append(VizJobArgs(name=j.name, specs=[s.model_dump() for s in j.visualizer]))  # type: ignore[attr-defined]
			else:
				viz_args.append(VizJobArgs(name=j.name, specs=[]))

		# Misc save policy
		se = root.profile.save.save_every if root.profile.save.save_every is not None else get_default('cli.profile.save.save_every', 1)
		misc = {
			"save": {
				"root": root.profile.save.root,
				"save_every": int(se),
				"save_timeseries": bool(root.profile.save.save_timeseries),
				"save_psd_complex": bool(root.profile.save.save_psd_complex),
				"save_psd_modular": bool(root.profile.save.save_psd_modular),
			},
		}

		pipeline = ConfigPipeline.from_parts(
			engine_config=eng_cfg,
			visualizer_config=viz_cfg,
			model_args_list=model_args,
			viz_job_args_list=viz_args,
			misc=misc,
		)

		# Define progress and run_dir callbacks
		def _on_progress(job_label: str, ic_index: int, ic_total: int, step_done: int, steps_total: int, eta_seconds: float):
			pct = 100.0 * float(step_done) / float(steps_total)
			# eta can be NaN early on
			eta_ok = (eta_seconds == eta_seconds and eta_seconds >= 0.0)
			mm = int(eta_seconds // 60) if eta_ok else 0
			ss = int(eta_seconds % 60) if eta_ok else 0
			eta_str = f"{mm:02d}:{ss:02d}" if eta_ok else "--:--"
			typer.echo(f"[{job_label}] [IC {ic_index+1}/{ic_total}] {pct:5.1f}% eta {eta_str}", nl=False)
			print("\r", end="")

		def _on_run_dir(p: Path):
			# Print final run dir for this job
			typer.echo(str(p))

		_ = scheduler_run(pipeline, on_progress=_on_progress, on_run_dir=_on_run_dir)

	except QPSError as e:
		log.error(str(e))
		raise typer.Exit(code=1)

	# (Per-IC plotting handled above for each job.)

