"""
QPhaseSDE: Core Scheduler
-------------------------
Serial orchestration of engine jobs, analysis, visualization, and IO based on
the structured configuration pipeline (ConfigPipeline).

Responsibilities
----------------
- Build EngineJob and VizJob lists from a ConfigPipeline
- For each job and IC, call engine.run and persist results according to policy
- Compute requested analyses (e.g., PSD) using the central registry (analysis:*)
- Render figures via the visualizer service with profile style overrides
- Save a snapshot and a minimal manifest per job

Notes
-----
- This implementation is intentionally conservative and serial; future work may
  add parallel scheduling and richer policy hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib
import importlib.util
import uuid

import numpy as np
import numpy as _np
from typing import cast as _cast

from .config import (
    ConfigPipeline,
    EngineJob,
)
from .errors import QPSConfigError, get_logger
from .engine import run as engine_run
from .protocols import NoiseSpec
from ..io.results import save_time_series, save_manifest, save_npz
from ..io.snapshot import write_run_snapshot
from ..analysis import __name__ as _analysis_pkg  # ensure package import side-effects
from ..visualizers.service import render_from_spec
from ..core.xputil import to_numpy as _to_numpy_array
from ..states.numpy_state import TrajectorySet as _NpTS

log = get_logger()


def _load_module(module_spec: str) -> Any:
    """Load a Python module by dotted name or from a file path."""
    try:
        return importlib.import_module(module_spec)
    except ModuleNotFoundError:
        p = Path(module_spec)
        if p.suffix == ".py" and p.exists():
            spec = importlib.util.spec_from_file_location(p.stem, p)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        raise


@dataclass
class RunResult:
    run_dir: Path
    job_index: int
    job_name: str


def run(
    pipeline: ConfigPipeline,
    *,
    on_progress: Optional[
        Any
    ] = None,  # callable(job_label: str, ic_index: int, ic_total: int, step_done: int, steps_total: int, eta_seconds: float)
    on_run_dir: Optional[Any] = None,  # callable(run_dir: Path)
) -> List[RunResult]:
    """Run the configured jobs serially and return a summary per job.

    This expects ``pipeline`` to be constructed via ``ConfigPipeline.from_parts``
    or equivalent with scalar-params ModelArgs (no DSL, no arrays).
    """
    parts = pipeline.export_parts()
    engine_jobs = pipeline.build_engine_jobs()
    viz_jobs = pipeline.build_viz_jobs()
    misc = parts.get("misc", {}) or {}
    engine_config = parts.get("engine_config")
    visualizer_config = parts.get("visualizer_config")

    results: List[RunResult] = []

    # Common knobs (optional; callers may embed them in EngineConfig)
    # Defaults are pulled from engine jobs/common config as needed.

    for j_idx, ej in enumerate(engine_jobs):
        # Resolve model builder
        try:
            mod = _load_module(ej.module)
        except ModuleNotFoundError:
            raise QPSConfigError(f"[504] Model module not found: {ej.module}")
        if not hasattr(mod, ej.function):
            raise QPSConfigError(f"[505] Function '{ej.function}' not found in module '{ej.module}'")
        build_fn = getattr(mod, ej.function)
        model = build_fn(ej.params)

        # Validate IC vectors
        ic_sets: List[List[complex]] = []
        for vec in (ej.ic or []):
            vec_c = [complex(s) for s in vec]
            if len(vec_c) != getattr(model, 'n_modes', None):
                raise QPSConfigError(f"[502] IC length {len(vec_c)} does not match model.n_modes={getattr(model, 'n_modes', None)}")
            ic_sets.append(vec_c)
        if not ic_sets:
            # Permit empty IC list to indicate a single zero vector
            ic_sets = [[0j] * int(getattr(model, 'n_modes', 0))]

        # Noise spec
        real_dim = 2 * int(model.noise_dim) if str(model.noise_basis) == "complex" else int(model.noise_dim)
        if (ej.noise or {}).get('kind', 'independent') == 'independent':
            noise_spec = NoiseSpec(kind='independent', dim=real_dim)
        else:
            C = np.array((ej.noise or {}).get('covariance', []), dtype=float)
            noise_spec = NoiseSpec(kind='correlated', dim=real_dim, covariance=C)

        # Time/trajectory/solver/backend
        time_spec = {
            't0': float((ej.time or {}).get('t0', 0.0)),
            'dt': float((ej.time or {}).get('dt', 1e-3)),
            'steps': int((ej.time or {}).get('steps', 1000)),
        }
        n_traj = int((ej.trajectories or {}).get('n_traj', 1))
        solver = ej.solver or 'euler'
        backend = ej.backend or 'numpy'
        save_every = int((misc.get('save', {}) or {}).get('save_every', 1))
        rng_stream = (ej.trajectories or {}).get('rng_stream', 'per_trajectory')

        # Per-job run directory
        run_root = Path((misc.get('save', {}) or {}).get('root', 'runs'))
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_id = f"{ts}_{uuid.uuid4().hex[:8]}"
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot structured config: separate EngineConfig vs EngineJob (ModelArgs-like)
        eng_cfg_payload = {
            "backend": backend,
            "solver": solver,
            "progress": dict(getattr(engine_config, 'progress', {}) or {}),
            "rng_stream": rng_stream,
        }
        eng_job_payload = {
            "name": ej.name,
            "module": ej.module,
            "function": ej.function,
            "params": ej.params,
            "ic": ej.ic,
            "noise": ej.noise,
            "time": time_spec,
            "trajectories": ej.trajectories,
            "model_name": getattr(model, 'name', None),
        }
        snapshot_cfg = {
            "engine_config": eng_cfg_payload,
            "engine_job": eng_job_payload,
            "job_index": j_idx,
            "job_name": ej.name or f"job_{j_idx:02d}",
        }
        write_run_snapshot(run_dir, config=snapshot_cfg, model_path=None, visualizer=None, profile_visualizer=dict(getattr(visualizer_config, 'styles', {}) or {}))

        # Execute per-IC
        for ic_idx, ic_vec in enumerate(ic_sets):
            y0 = np.asarray(ic_vec, dtype=np.complex128)
            # Fast, quiet progress callback (scheduler keeps logs slim)
            def _progress_cb(step_done: int, steps_total: int, eta_seconds: float, ic_i: int, ic_n: int):
                if (step_done % max(1, int(0.1 * steps_total))) == 0:
                    try:
                        pct = 100.0 * float(step_done) / float(steps_total)
                        log.info(f"[{run_id}] [IC {ic_i+1}/{ic_n}] {pct:5.1f}%")
                        if callable(on_progress):
                            on_progress(ej.name or f"job_{j_idx:02d}", ic_i, ic_n, step_done, steps_total, eta_seconds)
                    except Exception:
                        pass
            ts_obj = engine_run(
                model=model,
                ic=y0,
                time=time_spec,
                n_traj=n_traj,
                solver=solver,
                backend=backend,
                noise_spec=noise_spec,
                seed=None,
                master_seed=None,
                per_traj_seeds=None,
                save_every=save_every,
                return_stride=1,
                rng_stream=rng_stream,
                progress_cb=_progress_cb,
                progress_interval_seconds=1.0,
                ic_index=ic_idx,
                ic_total=len(ic_sets),
                warmup_min_steps=max(10, int(0.01 * int(time_spec['steps']))),
                warmup_min_seconds=1.0,
            )

            # Save downsampled time series if enabled
            if bool((misc.get('save', {}) or {}).get('save_timeseries', False)):
                se = max(1, int(save_every))
                data_s = ts_obj.data[:, ::se, :]
                data_s_np = _cast(_np.ndarray, _to_numpy_array(data_s))
                ts_s = _NpTS(data=data_s_np, t0=ts_obj.t0, dt=ts_obj.dt * se, meta=getattr(ts_obj, 'meta', {}))
                save_time_series(ts_s, run_dir, filename=f"timeseries_ic{ic_idx:02d}.npz")

            # Analyses/PSD
            try:
                from ..core.registry import registry as _registry
                analysis_psd = _registry.create("analysis:psd")
            except Exception:
                analysis_psd = None

            if analysis_psd is not None:
                # Determine which modes to analyze: collect from viz specs if any
                modes_psd: List[int] = []
                vj = viz_jobs[j_idx] if j_idx < len(viz_jobs) else None
                if vj is not None:
                    for spec in vj.specs or []:
                        k = spec.get('kind') if isinstance(spec, dict) else None
                        if k in ("complex", "modular") and isinstance(spec.get('modes', None), list):
                            modes_psd.extend([int(m) for m in spec['modes']])
                if not modes_psd:
                    modes_psd = list(range(int(getattr(model, 'n_modes', 0))))
                # Convention from visualizer_config styles if present
                v_styles = dict(getattr(visualizer_config, 'styles', {}) or {})
                psd_conf = v_styles.get('psd', {}) if isinstance(v_styles, dict) else {}
                convention = psd_conf.get('convention', 'symmetric') if isinstance(psd_conf, dict) else 'symmetric'
                _ts_np = _to_numpy_array(ts_obj.data)
                # Artifact naming: use existing 'analysis' namespace to fetch a PSD naming callable
                try:
                    from ..core.registry import registry as _registry
                    psd_namer = _registry.create("analysis:psd_namer")
                except Exception:
                    psd_namer = None
                if bool((misc.get('save', {}) or {}).get('save_psd_complex', False)):
                    res = analysis_psd(_ts_np, ts_obj.dt, modes_psd, kind='complex', convention=convention)
                    name = str(psd_namer(kind='complex', convention=convention)) if callable(psd_namer) else f"psd_complex_{convention}"
                    payload = {"axis": res['axis'], "psd": res['psd'], "modes": res['modes'], "kind": 'complex', "convention": convention}
                    _ = save_npz(run_dir, f"ic{ic_idx:02d}", name, payload)
                if bool((misc.get('save', {}) or {}).get('save_psd_modular', False)):
                    res = analysis_psd(_ts_np, ts_obj.dt, modes_psd, kind='modular', convention=convention)
                    name = str(psd_namer(kind='modular', convention=convention)) if callable(psd_namer) else f"psd_modular_{convention}"
                    payload = {"axis": res['axis'], "psd": res['psd'], "modes": res['modes'], "kind": 'modular', "convention": convention}
                    _ = save_npz(run_dir, f"ic{ic_idx:02d}", name, payload)

            # Render figures per spec
            vj = viz_jobs[j_idx] if j_idx < len(viz_jobs) else None
            if vj is not None and vj.specs:
                outdir = run_dir / 'figures' / f'ic{ic_idx:02d}'
                outdir.mkdir(parents=True, exist_ok=True)
                all_styles = vj.styles or {}
                for spec in vj.specs:
                    k = spec.get('kind') if isinstance(spec, dict) else None
                    per_kind_style = None
                    if isinstance(all_styles, dict) and k is not None:
                        if str(k) in ('re_im','Re_Im','abs_abs','Abs_Abs'):
                            per_kind_style = (all_styles.get('phase_portrait') or all_styles.get('phase') or {})
                            # Normalize legacy kind case
                            if str(k) in ('Re_Im','re_im'):
                                spec['kind'] = 're_im'
                            elif str(k) in ('Abs_Abs','abs_abs'):
                                spec['kind'] = 'abs_abs'
                        elif str(k) in ('complex','modular'):
                            per_kind_style = all_styles.get('psd') if isinstance(all_styles, dict) else None
                    _ = render_from_spec(spec, _to_numpy_array(ts_obj.data), t0=ts_obj.t0, dt=ts_obj.dt, outdir=outdir, style_overrides=per_kind_style, save=True)

        # Manifest
        manifest = {"run_id": run_id, "package": "QPhaseSDE", "job_index": j_idx, "job_name": ej.name or f"job_{j_idx:02d}"}
        save_manifest(run_dir, manifest)
        results.append(RunResult(run_dir=run_dir, job_index=j_idx, job_name=ej.name or f"job_{j_idx:02d}"))
        if callable(on_run_dir):
            try:
                on_run_dir(run_dir)
            except Exception:
                pass

    return results

# Backward-compatible alias
launch = run
