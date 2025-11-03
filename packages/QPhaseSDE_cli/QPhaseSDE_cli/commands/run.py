from __future__ import annotations

"""Run commands for launching simulations via CLI or programmatic calls.

Supports either a triad YAML configuration or legacy flag-based parameters.
"""

import importlib
import importlib.util
import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import typer
from QPhaseSDE.core.errors import (
	QPSError,
	QPSConfigError,
	get_logger,
	configure_logging,
)
from QPhaseSDE.core.xputil import to_numpy as _to_numpy_array


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


## Removed local _to_numpy_array; now using QPhaseSDE.core.xputil.to_numpy


def _load_model_from_path(path: str):
	p = Path(path)
	"""Load a Python module from a file path and verify it exposes build_sde()."""
	spec = importlib.util.spec_from_file_location(p.stem, p)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)  # type: ignore
	if not hasattr(mod, "build_sde"):
			raise QPSConfigError("[501] Model file must define build_sde(params) -> SDEModel")
	return mod  # type: ignore


@app.command()
def sde(
	config: Optional[str] = typer.Option(None, help="YAML config with sections: model/profile/run"),
	model_path: Optional[str] = typer.Option(None, help="Path to model .py exposing build_sde(params); ignored if --config provided"),
	backend: str = typer.Option("numpy", help="Backend: numpy|numba|torch|cupy; overridden by profile.backend if --config provided"),
	solver: str = typer.Option("euler", help="Solver: euler|milstein; overridden by profile.solver if --config provided"),
	dt: float = typer.Option(1e-3, help="Time step; ignored if --config provided"),
	steps: int = typer.Option(1000, help="Number of steps; ignored if --config provided"),
	n_traj: int = typer.Option(16, help="Number of trajectories; ignored if --config provided"),
	seed: Optional[int] = typer.Option(None, help="Random seed; ignored if --config provided"),
	save_every: int = typer.Option(1, help="Save every N steps; overridden by profile.save.save_every if --config provided"),
	run_root: str = typer.Option("runs", help="Runs root directory; overridden by profile.save.root if --config provided"),
	params: List[str] = typer.Option([], "--set", help="Model parameter overrides as key=value; ignored if --config provided"),
	ic: Optional[str] = typer.Option(None, help="Initial condition JSON list of complex; ignored if --config provided"),
	noise_kind: str = typer.Option("independent", help="Noise: independent|correlated; ignored if --config provided"),
	cov_json: Optional[str] = typer.Option(None, help="Covariance JSON when correlated; ignored if --config provided"),
	rng_stream: Optional[str] = typer.Option(None, help="RNG stream: per_trajectory|batched; overridden by run.trajectories.rng_stream if --config provided"),
	verbose: bool = typer.Option(False, help="Enable verbose logging"),
	log_file: Optional[str] = typer.Option(None, help="Write logs to file path"),
	log_json: bool = typer.Option(False, help="Log in JSON format"),
	suppress_warnings: bool = typer.Option(False, help="Suppress warnings output"),
	max_storage_gb: Optional[float] = typer.Option(None, help="Override default 1 GiB time-series storage guard (in GB); only applies when using --config"),
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

	# Resolve engine/noise/io modules dynamically
	engine_mod = importlib.import_module('QPhaseSDE.core.engine')
	noise_mod = importlib.import_module('QPhaseSDE.core.protocols')
	backend_factory = importlib.import_module('QPhaseSDE.backends.factory')
	io_results = importlib.import_module('QPhaseSDE.io.results')
	io_snapshot = importlib.import_module('QPhaseSDE.io.snapshot')
	NoiseSpec = getattr(noise_mod, 'NoiseSpec')

	try:
		if config is not None:
			# Use YAML triad config
			from QPhaseSDE_cli.config.loader import load_triad_config
			triad = load_triad_config(config)

			# Build model from module/function
			try:
				mod = importlib.import_module(triad.model.module)
			except ModuleNotFoundError:
				# allow direct file path in module field as fallback, or resolve relative to config
				module_spec = triad.model.module
				cand = Path(module_spec)
				if cand.suffix == ".py" and cand.exists():
					mod = _load_model_from_path(str(cand))
				else:
					cfg_path = Path(config).resolve()
					repo_root = cfg_path.parent.parent
					rel_path = Path(module_spec.replace('.', os.sep) + '.py')
					candidate = repo_root / rel_path
					if candidate.exists():
						mod = _load_model_from_path(str(candidate))
					else:
						raise QPSConfigError(f"[504] Model module not found: {module_spec}")
			build_fn = getattr(mod, triad.model.function)
			model = build_fn(triad.model.params)
			# Prepare IC sets
			ic_sets_str = triad.model.ic
			ic_sets: List[List[complex]] = []
			for vec in ic_sets_str:
				vec_c = [complex(s) for s in vec]
				if len(vec_c) != model.n_modes:
					raise QPSConfigError(f"[502] IC length {len(vec_c)} does not match model.n_modes={model.n_modes}")
				ic_sets.append(vec_c)

			real_dim = 2 * model.noise_dim if model.noise_basis == "complex" else model.noise_dim
			if triad.model.noise.kind == "independent":
				noise_spec = NoiseSpec(kind="independent", dim=real_dim)
			else:
				C = np.array(triad.model.noise.covariance, dtype=float)
				noise_spec = NoiseSpec(kind="correlated", dim=real_dim, covariance=C)

			backend = triad.profile.backend
			solver = triad.profile.solver
			run_root = triad.profile.save.root
			if triad.profile.save.save_every is not None:
				save_every = triad.profile.save.save_every

			dt = triad.run.time.dt
			steps = triad.run.time.steps
			n_traj = triad.run.trajectories.n_traj
			seed_file = getattr(triad.run.trajectories, 'seed_file', None)
			master_seed_cfg = getattr(triad.run.trajectories, 'master_seed', None)
			rng_stream_cfg = getattr(triad.run.trajectories, 'rng_stream', 'per_trajectory')
			model_path = None
			noise_kind = triad.model.noise.kind
		else:
			# Legacy CLI path
			log.warning("[991] Legacy flag-based CLI path is deprecated and will be removed in a future release; please use triad YAML via --config.")
			if model_path is None:
				raise QPSConfigError("[500] When --config is not provided, --model-path is required.")
			model_path = str(_coerce_option_default(model_path))
			backend = _coerce_option_default(backend, fallback="numpy")
			solver = _coerce_option_default(solver, fallback="euler")
			dt = float(_coerce_option_default(dt, fallback=1e-3))
			steps = int(_coerce_option_default(steps, fallback=1000))
			n_traj = int(_coerce_option_default(n_traj, fallback=16))
			seed = _coerce_option_default(seed)
			seed_file = None
			master_seed_cfg = None  # legacy flags don't expose these; will auto-generate in RNG strategy if needed
			rng_stream_cfg = str(_coerce_option_default(rng_stream, fallback="per_trajectory")) if rng_stream is not None else "per_trajectory"
			save_every = int(_coerce_option_default(save_every, fallback=1))
			run_root = _coerce_option_default(run_root, fallback="runs")
			params = list(params or [])

			mod = _load_model_from_path(model_path)
			p_over = _parse_kv(params)
			model = mod.build_sde(p_over)
			if ic is not None:
				lst = json.loads(ic)
				y0 = np.array([complex(x) for x in lst], dtype=np.complex128)
			else:
				y0 = np.zeros(model.n_modes, dtype=np.complex128)
			real_dim = 2 * model.noise_dim if model.noise_basis == "complex" else model.noise_dim
			if noise_kind == "independent":
				noise_spec = NoiseSpec(kind="independent", dim=real_dim)
			else:
				if cov_json is None:
					raise QPSConfigError("[503] Covariance JSON required for correlated noise")
				C = np.array(json.loads(cov_json), dtype=float)
				noise_spec = NoiseSpec(kind="correlated", dim=real_dim, covariance=C)

		# Guard: estimate time-series storage if saving enabled
		try:
			from math import ceil
			save_cfg = triad.profile.save
			if bool(getattr(save_cfg, 'save_timeseries', False)):
				sv_every = int(save_every) if save_every is not None else 1
				saved_steps = int(ceil(steps / max(1, sv_every)))
				ic_num = len(ic_sets)
				n_modes = int(getattr(model, 'n_modes', 0))
				bytes_per_complex = 16  # complex128
				est_bytes = ic_num * n_traj * saved_steps * n_modes * bytes_per_complex
				threshold = int((max_storage_gb if max_storage_gb is not None else 1.0) * (1024**3))
				if est_bytes > threshold:
					raise QPSConfigError(f"[520] Estimated time-series size {est_bytes/1024**3:.2f} GiB exceeds limit ({threshold/1024**3:.2f} GiB). Disable profile.save.save_timeseries or increase --max-storage-gb.")
		except QPSError:
			raise
		except Exception:
			# Non-fatal estimation failure
			pass

		# Prepare run id and dirs
		run_root = _coerce_option_default(run_root, fallback="runs")
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
		run_id = f"{ts}_{uuid.uuid4().hex[:8]}"
		run_dir = Path(run_root) / run_id
		run_dir.mkdir(parents=True, exist_ok=True)

		# Prepare engine and backend
		time_spec = {"t0": 0.0, "dt": dt, "steps": steps}
		engine_run = getattr(engine_mod, 'run')
		get_backend = getattr(backend_factory, 'get_backend')
		backend_instance = get_backend(backend)

		# Snapshot
		run_meta = {
			"source": "config" if config is not None else "cli",
			"model_path": str(model_path) if model_path else None,
			"model_name": model.name,
			"params": model.params,
			"backend": backend,
			"solver": solver,
			"time": time_spec,
			"n_traj": n_traj,
			"save_every": save_every,
			"rng_stream": rng_stream_cfg,
			"noise": {"kind": noise_kind, "dim": real_dim},
		}
		write_run_snapshot = getattr(io_snapshot, 'write_run_snapshot')
		save_time_series = getattr(io_results, 'save_time_series')
		save_manifest = getattr(io_results, 'save_manifest')
		if config is not None and Path(config).exists():
			viz_payload = None
			prof_viz_payload = None
			if getattr(triad, 'run', None) is not None and (getattr(triad.run, 'visualization', None) is not None or getattr(triad.run, 'viz', None) is not None):
				try:
					viz_obj = triad.run.visualization if getattr(triad.run, 'visualization', None) is not None else triad.run.viz
					viz_payload = viz_obj.model_dump()  # type: ignore[attr-defined]
				except Exception:
					pass
			if getattr(triad, 'profile', None) is not None and (getattr(triad.profile, 'visualization', None) is not None or getattr(triad.profile, 'viz', None) is not None):
				try:
					prof_viz = triad.profile.visualization if getattr(triad.profile, 'visualization', None) is not None else triad.profile.viz
					prof_viz_payload = prof_viz
				except Exception:
					pass
			write_run_snapshot(run_dir, config=run_meta, model_path=model_path, visualization=viz_payload, profile_visualization=prof_viz_payload)
			try:
				from shutil import copy2
				snap_dir = Path(run_dir) / 'config_snapshot'
				copy2(config, snap_dir / 'triad.yaml')
			except Exception:
				pass
		else:
			write_run_snapshot(run_dir, config=run_meta, model_path=model_path)

		# Determine RNG strategy before simulations
		rng_info = None
		per_traj_seeds: Optional[List[int]] = None
		seed_file_used = None
		if seed_file is not None:
			# Priority: seed file overrides master seed if both provided
			seed_file_used = str(seed_file)
			try:
				with open(seed_file, 'r', encoding='utf-8') as f:
					per_traj_seeds = [int(line.strip()) for line in f if str(line).strip()]
			except Exception as e:
				raise QPSConfigError(f"[506] Failed to read seed_file: {seed_file}: {e}")
			if len(per_traj_seeds) != n_traj:
				raise QPSConfigError(f"[507] seed_file contains {len(per_traj_seeds)} seeds, expected n_traj={n_traj}")
			# Log warning if master_seed also provided
			if master_seed_cfg is not None:
				log.warning("[900] seed_file provided; master_seed is ignored")
			rng_info = {
				"backend": getattr(backend_instance, 'name', 'backend'),
				"algorithm": "PCG64",
				"master_seed": None,
				"source": "seed_file",
			}
		else:
			# derive per-trajectory seeds from master_seed (config or auto)
			import time as _t
			master_seed_val = int(master_seed_cfg) if master_seed_cfg is not None else int(_t.time_ns() & 0x7FFFFFFFFFFFFFFF)
			# We store explicit per-trajectory seeds using NumPy SeedSequence to generate stable int seeds
			try:
				import numpy as _np
				ss = _np.random.SeedSequence(master_seed_val)
				children = ss.spawn(n_traj)
				# For portability, derive a 64-bit integer from each child state
				per_traj_seeds = [int(child.generate_state(1, dtype=_np.uint64)[0]) for child in children]
			except Exception:
				# Fallback: simple offset seeds
				per_traj_seeds = [int(master_seed_val + i) for i in range(n_traj)]
			rng_info = {
				"backend": getattr(backend_instance, 'name', 'backend'),
				"algorithm": "PCG64",
				"master_seed": master_seed_val,
				"source": "master_seed",
			}


		# Persist RNG metadata and registry snapshot
		try:
			from QPhaseSDE.core.registry import registry as _registry
			reg_meta = _registry.list()
		except Exception:
			reg_meta = None
		write_run_snapshot(run_dir, config=run_meta, model_path=model_path, rng_info=rng_info, per_traj_seeds=per_traj_seeds, seed_file_used=seed_file_used, registry_meta=reg_meta)

		# Execute simulations
		if config is not None:
			viz_specs = None
			viz_psd_specs = None
			style_cfg = None
			style_psd_cfg = None
			replot_mode = 'append'
			if getattr(triad, 'run', None) is not None and getattr(triad.run, 'visualization', None) is not None:
				viz_specs = getattr(triad.run.visualization, 'phase_portrait', None)
				viz_psd_specs = getattr(triad.run.visualization, 'psd', None)
			if getattr(triad, 'profile', None) is not None:
				prof_viz = triad.profile.visualization if getattr(triad.profile, 'visualization', None) is not None else triad.profile.viz
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

			# Use new visualizer service API (dispatcher removed)
			service_mod = importlib.import_module('QPhaseSDE.visualizers.service')
			render_from_spec = getattr(service_mod, 'render_from_spec')
			save_time_series = getattr(io_results, 'save_time_series')

			for ic_idx, ic_vec in enumerate(ic_sets):
				y0 = np.asarray(ic_vec, dtype=np.complex128)
				def _progress_cb(step_done: int, steps_total: int, eta_seconds: float, ic_i: int, ic_n: int):
					pct = 100.0 * float(step_done) / float(steps_total)
					eta_str = "--:--" if not (eta_seconds == eta_seconds and eta_seconds >= 0.0) else f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
					typer.echo(f"[IC {ic_i+1}/{ic_n}] {pct:5.1f}% eta {eta_str}", nl=False)
					print("\r", end="")
				ts_obj = engine_run(
					model=model,
					ic=y0,
					time=time_spec,
					n_traj=n_traj,
					solver=solver,
					backend=backend_instance,
					noise_spec=noise_spec,
					seed=None,
					master_seed=rng_info.get("master_seed") if rng_info else None,
					per_traj_seeds=per_traj_seeds,
					save_every=save_every,
					return_stride=1,
					rng_stream=rng_stream_cfg,
					progress_cb=_progress_cb,
					progress_interval_seconds=1.0,
					ic_index=ic_idx,
					ic_total=len(ic_sets),
					warmup_min_steps=max(10, int(0.01 * steps)),
					warmup_min_seconds=1.0,
				)
				print()
				fname = f"timeseries_ic{ic_idx:02d}.npz"
				# Save timeseries if enabled
				if bool(getattr(triad.profile.save, 'save_timeseries', False)):
					# Decimate only for saving using save_every, keeping analysis at full resolution
					from QPhaseSDE.states.numpy_state import TrajectorySet as _NpTS
					se = max(1, int(save_every) if save_every is not None else 1)
					data_s = ts_obj.data[:, ::se, :]
					data_s_np = _to_numpy_array(data_s)
					ts_s = _NpTS(data=data_s_np, t0=ts_obj.t0, dt=ts_obj.dt * se, meta=getattr(ts_obj, 'meta', {}))
					save_time_series(ts_s, run_dir, filename=fname)

				# Compute/save PSD data if requested in profile.save
				psd_conf = (triad.profile.visualization or {}).get('psd', {}) if isinstance(getattr(triad.profile, 'visualization', None), dict) else {}
				convention = psd_conf.get('convention', 'symmetric') if isinstance(psd_conf, dict) else 'symmetric'
				modes_psd: List[int] = []
				if viz_psd_specs:
					for s in viz_psd_specs:
						modes_psd.extend([int(m) for m in getattr(s, 'modes', [])])
					modes_psd = sorted(set(modes_psd))
				if not modes_psd:
					modes_psd = list(range(model.n_modes))
				from QPhaseSDE.analysis.psd import compute_psd_for_modes as _compute_psd_batch
				save_psd_npz = getattr(io_results, 'save_psd')
				ic_tag = f"ic{ic_idx:02d}"
				_ts_data_np = _to_numpy_array(ts_obj.data)
				if bool(getattr(triad.profile.save, 'save_psd_complex', False)):
					res = _compute_psd_batch(_ts_data_np, ts_obj.dt, modes_psd, kind='complex', convention=convention)
					save_psd_npz(run_dir, ic_tag, kind='complex', convention=convention, axis=res['axis'], psd=res['psd'], modes=res['modes'])
				if bool(getattr(triad.profile.save, 'save_psd_modular', False)):
					res = _compute_psd_batch(_ts_data_np, ts_obj.dt, modes_psd, kind='modular', convention=convention)
					save_psd_npz(run_dir, ic_tag, kind='modular', convention=convention, axis=res['axis'], psd=res['psd'], modes=res['modes'])
				if viz_specs or viz_psd_specs:
					out_dir = Path(run_dir) / 'figures' / f'ic{ic_idx:02d}'
					# ensure output directory exists
					out_dir.mkdir(parents=True, exist_ok=True)
					if replot_mode == 'clean' and out_dir.exists():
						for f in out_dir.glob('*.png'):
							try:
								f.unlink()
							except Exception:
								pass
					if viz_specs:
						specs_serialized = [s.model_dump() for s in viz_specs]
						# Render phase-portrait specs via service
						_render_data_np = _ts_data_np
						for spec in specs_serialized:
							k = spec.get('kind') if isinstance(spec, dict) else None
							per_kind_style = style_cfg.get(str(k)) if isinstance(style_cfg, dict) and k is not None else None
							_ = render_from_spec(spec, _render_data_np, t0=ts_obj.t0, dt=ts_obj.dt, outdir=out_dir, style_overrides=per_kind_style, save=True)
					if viz_psd_specs:
						psd_specs_serialized = [s.model_dump() for s in viz_psd_specs]
						for spec in psd_specs_serialized:
							# Ensure the PSD spec is marked with kind='psd' for service dispatch
							spec.setdefault('kind', 'psd')
							per_kind_style = style_psd_cfg if isinstance(style_psd_cfg, dict) else None
							_ = render_from_spec(spec, _ts_data_np, t0=ts_obj.t0, dt=ts_obj.dt, outdir=out_dir, style_overrides=per_kind_style, save=True)
		else:
			# Legacy single run
			ts_obj = engine_run(
				model=model,
				ic=y0,
				time=time_spec,
				n_traj=n_traj,
				solver=solver,
				backend=backend_instance,
				noise_spec=noise_spec,
				seed=None,
				master_seed=rng_info.get("master_seed") if rng_info else None,
				per_traj_seeds=per_traj_seeds,
				save_every=save_every,
				return_stride=1,
				rng_stream=rng_stream_cfg,
			)
			# Save default time series filename with decimation for saving
			from QPhaseSDE.states.numpy_state import TrajectorySet as _NpTS
			se = max(1, int(save_every) if save_every is not None else 1)
			data_s = ts_obj.data[:, ::se, :]
			data_s_np = _to_numpy_array(data_s)
			ts_s = _NpTS(data=data_s_np, t0=ts_obj.t0, dt=ts_obj.dt * se, meta=getattr(ts_obj, 'meta', {}))
			save_time_series(ts_s, run_dir)

		# Save manifest and echo run dir
		manifest = {"run_id": run_id, "package": "QPhaseSDE", "version": "0.1.2"}
		save_manifest(run_dir, manifest)
		typer.echo(str(run_dir))
	except QPSError as e:
		log.error(str(e))
		raise typer.Exit(code=1)

	# (Per-IC plotting handled above in the config path.)

