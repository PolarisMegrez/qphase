"""
QPhaseSDE: Results IO
---------------------
Lightweight helpers to persist and load simulation artifacts: multi-trajectory
time series, power spectral density (PSD) arrays, and a minimal run manifest,
using a predictable directory layout under each run.

Behavior
--------
- Provides stable subfolders for results under a run directory and delegates
  exact filenames/keys and formats to individual function docstrings.

Notes
-----
- Uses NumPy NPZ for portability and compression; see function docstrings for
  exact storage schema and error codes.
"""

import json
from pathlib import Path
import numpy as np
from ..core.protocols import TrajectorySetBase as TrajectorySetLike
from ..states.numpy_state import TrajectorySet as NpTrajectorySet
from ..core.errors import QPSIOError

__all__ = [
	"save_time_series",
	"save_manifest",
	"save_npz",
	"save_psd",
	"load_time_series",
]

def save_time_series(ts: TrajectorySetLike, run_dir: str | Path, filename: str = "timeseries.npz") -> None:
	"""Save a trajectory set to ``run_dir/time_series/<filename>`` as NPZ.

	Parameters
	----------
	ts : TrajectorySetLike
		Trajectory set with fields ``data`` (array of shape ``(n_traj, n_keep, n_modes)``),
		``t0`` (float) and ``dt`` (float).
	run_dir : str or pathlib.Path
		Run directory under which the ``time_series`` subfolder will be created.
	filename : str, default "timeseries.npz"
		Target NPZ filename written inside the ``time_series`` folder.

	Raises
	------
	QPSIOError
		- [103] Failed to save the time series NPZ due to an IO error.

	Examples
	--------
	>>> # save_time_series(ts, run_dir)  # doctest: +SKIP
	"""
	try:
		run_dir = Path(run_dir)
		ts_dir = run_dir / "time_series"
		ts_dir.mkdir(parents=True, exist_ok=True)
		np.savez_compressed(ts_dir / filename, data=ts.data, t0=ts.t0, dt=ts.dt)
	except Exception as e:
		raise QPSIOError(f"[103] Failed to save time series: {e}")

def save_manifest(run_dir: str | Path, manifest: dict) -> None:
	"""Write ``manifest.json`` with run information into ``run_dir``.

	Parameters
	----------
	run_dir : str or pathlib.Path
		Run directory where ``manifest.json`` will be written.
	manifest : dict
		JSON-serializable mapping describing the run (times, config pointers, etc.).

	Raises
	------
	QPSIOError
		- [103] Failed to write the manifest due to an IO or serialization error.

	Examples
	--------
	>>> # save_manifest(run_dir, {"seed": 123})  # doctest: +SKIP
	"""
	try:
		run_dir = Path(run_dir)
		(run_dir / "logs").mkdir(parents=True, exist_ok=True)
		with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
			json.dump(manifest, f, indent=2, default=str)
	except Exception as e:
		raise QPSIOError(f"[103] Failed to write manifest: {e}")

def save_npz(run_dir: str | Path, ic_tag: str, name: str, payload: dict) -> Path:
	"""Save an analysis artifact payload into ``run_dir/<prefix>/<ic_tag>/<name>.npz``.

	Parameters
	----------
	run_dir : str or pathlib.Path
		Run directory where the artifact subfolder will be created.
	ic_tag : str
		Identifier for the initial condition bucket; used as a subfolder name.
	name : str
		Base artifact name (without extension). The first token before '_' is used
		as the top-level subfolder under the run directory (e.g., 'psd_...').
	payload : dict
		A JSON/NumPy-serializable mapping to persist into the NPZ file. Arrays
		will be saved directly; lists will be converted as-is.

	Returns
	-------
	pathlib.Path
		Path to the saved NPZ file.

	Raises
	------
	QPSIOError
		- [103] Failed to save the NPZ due to an IO error.

	Examples
	--------
	>>> # save_npz(run_dir, "ic00", "psd_complex_symmetric", {"axis": axis, "psd": psd})  # doctest: +SKIP
	"""
	try:
		run_dir = Path(run_dir)
		prefix = str(name).split("_", 1)[0] if "_" in str(name) else str(name)
		art_dir = run_dir / prefix / ic_tag
		art_dir.mkdir(parents=True, exist_ok=True)
		fpath = art_dir / f"{name}.npz"
		# Flatten dict to savez kwargs
		kwargs = {k: (np.asarray(v) if hasattr(v, "__array__") else v) for k, v in (payload or {}).items()}
		np.savez_compressed(fpath, **kwargs)
		return fpath
	except Exception as e:
		raise QPSIOError(f"[103] Failed to save artifact '{name}': {e}")

def save_psd(run_dir: str | Path, ic_tag: str, *, kind: str, convention: str, axis: np.ndarray, psd: np.ndarray, modes: list[int]) -> Path:
	"""Compatibility wrapper: delegate PSD saving to ``save_npz``.

	Uses artifact name ``psd_{kind}_{convention}`` and stores the standard keys
	(axis, psd, modes, kind, convention).
	"""
	name = f"psd_{kind}_{convention}"
	payload = {
		"axis": axis,
		"psd": psd,
		"modes": np.asarray(modes, dtype=int),
		"kind": kind,
		"convention": convention,
	}
	return save_npz(run_dir, ic_tag, name, payload)

def load_time_series(run_dir: str | Path, filename: str = "timeseries.npz") -> TrajectorySetLike:
	"""Load a trajectory set saved by ``save_time_series``.

	Parameters
	----------
	run_dir : str or pathlib.Path
		Run directory where the ``time_series`` subfolder resides.
	filename : str, default "timeseries.npz"
		NPZ filename to load from the ``time_series`` folder.

	Returns
	-------
	TrajectorySetLike
		A NumPy-based trajectory set reconstructed from the NPZ contents.

	Raises
	------
	QPSIOError
		- [101] Time series file not found.
		- [102] Failed to parse or load the NPZ file.

	Examples
	--------
	>>> # ts = load_time_series(run_dir)  # doctest: +SKIP
	"""
	run_dir = Path(run_dir)
	ts_path = run_dir / "time_series" / filename
	if not ts_path.exists():
		raise QPSIOError(f"[101] Time series file not found: {ts_path}")
	try:
		with np.load(ts_path) as npz:
			data = npz["data"]
			t0 = float(npz["t0"]) if "t0" in npz else 0.0
			dt = float(npz["dt"]) if "dt" in npz else 1.0
		return NpTrajectorySet(data=data, t0=t0, dt=dt, meta={})
	except Exception as e:
		raise QPSIOError(f"[102] Failed to load time series: {e}")
