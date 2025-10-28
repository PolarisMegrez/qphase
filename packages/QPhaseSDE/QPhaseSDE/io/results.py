from __future__ import annotations

"""IO helpers to persist and restore time series and run metadata."""

import json
from pathlib import Path
import numpy as np
from ..core.protocols import TrajectorySetBase as TrajectorySetLike
from ..states.numpy_state import TrajectorySet as NpTrajectorySet
from ..core.errors import SDEIOError


def save_time_series(ts: TrajectorySetLike, run_dir: str | Path, filename: str = "timeseries.npz") -> None:
	"""Save trajectory set data to a compressed NPZ in run_dir/time_series."""
	run_dir = Path(run_dir)
	ts_dir = run_dir / "time_series"
	ts_dir.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(ts_dir / filename, data=ts.data, t0=ts.t0, dt=ts.dt)


def save_manifest(run_dir: str | Path, manifest: dict) -> None:
	"""Write a manifest.json with basic run information into run_dir."""
	run_dir = Path(run_dir)
	(run_dir / "logs").mkdir(parents=True, exist_ok=True)
	with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2, default=str)


def save_psd(run_dir: str | Path, ic_tag: str, *, kind: str, convention: str, axis: np.ndarray, psd: np.ndarray, modes: list[int]) -> Path:
	"""Save PSD arrays into run_dir/psd/{ic_tag}/ as compressed NPZ.

	axis shape: (n_freq,), psd shape: (n_freq, n_modes)
	Returns saved file path.
	"""
	run_dir = Path(run_dir)
	psd_dir = run_dir / "psd" / ic_tag
	psd_dir.mkdir(parents=True, exist_ok=True)
	fname = f"psd_{kind}_{convention}.npz"
	fpath = psd_dir / fname
	np.savez_compressed(fpath, axis=axis, psd=psd, modes=np.array(modes, dtype=int), kind=kind, convention=convention)
	return fpath


def load_time_series(run_dir: str | Path, filename: str = "timeseries.npz") -> TrajectorySetLike:
	"""Load a previously saved trajectory set from run_dir/time_series."""
	run_dir = Path(run_dir)
	ts_path = run_dir / "time_series" / filename
	if not ts_path.exists():
		raise SDEIOError(f"[601] Time series file not found: {ts_path}")
	with np.load(ts_path) as npz:
		data = npz["data"]
		t0 = float(npz["t0"]) if "t0" in npz else 0.0
		dt = float(npz["dt"]) if "dt" in npz else 1.0
	return NpTrajectorySet(data=data, t0=t0, dt=dt, meta={})
