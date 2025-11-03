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

def save_psd(run_dir: str | Path, ic_tag: str, *, kind: str, convention: str, axis: np.ndarray, psd: np.ndarray, modes: list[int]) -> Path:
	"""Save PSD arrays into ``run_dir/psd/<ic_tag>/psd_<kind>_<convention>.npz``.

	Parameters
	----------
	run_dir : str or pathlib.Path
		Run directory where the ``psd/<ic_tag>`` folder will be created.
	ic_tag : str
		Identifier for the initial condition bucket; used as a subfolder name.
	kind : str
		PSD kind (e.g., "complex" or "modular").
	convention : str
		PSD normalization convention (e.g., "symmetric" | "unitary" | "pragmatic").
	axis : numpy.ndarray
		Frequency axis with shape ``(n_freq,)``.
	psd : numpy.ndarray
		PSD values with shape ``(n_freq, n_modes)``.
	modes : list[int]
		Indices of modes included in the PSD.

	Returns
	-------
	pathlib.Path
		Path to the saved NPZ file.

	Raises
	------
	QPSIOError
		- [103] Failed to save the PSD NPZ due to an IO error.

	Examples
	--------
	>>> # save_psd(run_dir, "ic0", kind="complex", convention="symmetric", axis=axis, psd=psd, modes=[0,1])  # doctest: +SKIP
	"""
	try:
		run_dir = Path(run_dir)
		psd_dir = run_dir / "psd" / ic_tag
		psd_dir.mkdir(parents=True, exist_ok=True)
		fname = f"psd_{kind}_{convention}.npz"
		fpath = psd_dir / fname
		np.savez_compressed(
			fpath,
			axis=axis,
			psd=psd,
			modes=np.array(modes, dtype=int),
			kind=kind,
			convention=convention,
		)
		return fpath
	except Exception as e:
		raise QPSIOError(f"[103] Failed to save PSD: {e}")

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
