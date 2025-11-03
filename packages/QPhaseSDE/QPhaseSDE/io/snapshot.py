"""
QPhaseSDE: Run Snapshot
-----------------------
Create reproducible run snapshots (config, visualization, RNG metadata, optional
model copy) in a stable directory layout under each run.

Behavior
--------
- Persist configuration and optional artifacts under ``config_snapshot/``,
  deferring exact filenames and payload shapes to function docstrings.

Notes
-----
- Prefers ruamel.yaml; falls back to PyYAML when available. Raises if neither is installed.
- Certain legacy visualization keys may be stripped for forward compatibility.
"""

__all__ = [
	"write_run_snapshot",
]

import shutil
from pathlib import Path
from typing import Dict, Optional, List
from ..core.errors import QPSIOError

# Prefer ruamel.yaml for YAML writing; fall back to PyYAML if available
_USE_RUAMEL = False
_USE_PYYAML = False
try:
	from ruamel.yaml import YAML as _RUYAML  # type: ignore
	_USE_RUAMEL = True
except Exception:
	try:
		import yaml as _PYYAML  # type: ignore
		_USE_PYYAML = True
	except Exception:
		_USE_PYYAML = False

def _dump_yaml(path: Path, data: Dict) -> None:
	"""Write a dict to YAML using ruamel.yaml or PyYAML (internal helper)."""
	if not (_USE_RUAMEL or _USE_PYYAML):
		raise QPSIOError("[002] YAML library not found. Please install 'ruamel.yaml' or 'PyYAML'.")
	if _USE_RUAMEL:
		yaml_obj = _RUYAML()
		yaml_obj.default_flow_style = False
		with open(path, "w", encoding="utf-8") as f:
			yaml_obj.dump(data, f)
	elif _USE_PYYAML:
		with open(path, "w", encoding="utf-8") as f:
			_PYYAML.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def write_run_snapshot(run_dir: str | Path,
					   config: Dict,
					   model_path: Optional[str | Path] = None,
					   seed: Optional[int] = None,
					   visualization: Optional[Dict] = None,
					   profile_visualization: Optional[Dict] = None,
					   rng_info: Optional[Dict] = None,
					   per_traj_seeds: Optional[List[int]] = None,
					   seed_file_used: Optional[str] = None,
					   registry_meta: Optional[Dict] = None) -> None:
	"""Write reproducible run snapshot files under ``run_dir/config_snapshot``.

	Persists config, visualization, RNG metadata, per-trajectory seeds, model copy,
	and registry provenance as YAML or text files in a stable directory layout.

	Parameters
	----------
	run_dir : str or pathlib.Path
		Run directory under which the ``config_snapshot`` folder will be created.
	config : dict
		Core run configuration (visualization keys are stripped).
	model_path : str or pathlib.Path, optional
		Path to the model file to copy for provenance.
	seed : int, optional
		Random seed (legacy; now stored in ``rng_info``).
	visualization : dict, optional
		Visualization configuration payload.
	profile_visualization : dict, optional
		Profile visualization configuration payload.
	rng_info : dict, optional
		RNG metadata to write to ``seed.yaml``.
	per_traj_seeds : list[int], optional
		List of per-trajectory seeds to write to ``seeds.txt``.
	seed_file_used : str, optional
		Path to the seed file used (for provenance).
	registry_meta : dict, optional
		Registry metadata to write to ``registry.yaml``.

	Raises
	------
	QPSIOError
		- [002] YAML library not found (ruamel.yaml or PyYAML required).

	Examples
	--------
	>>> write_run_snapshot(run_dir, config, model_path="model.py", rng_info={"seed": 42})  # doctest: +SKIP
	"""
	run_dir = Path(run_dir)
	snap = run_dir / "config_snapshot"
	logs = run_dir / "logs"
	snap.mkdir(parents=True, exist_ok=True)
	logs.mkdir(parents=True, exist_ok=True)
	# Remove any legacy visualization keys from config.json to keep it focused
	cfg_clean = dict(config)
	for k in ("visualization", "profile_visualization", "viz"):
		if k in cfg_clean:
			cfg_clean.pop(k, None)
	_dump_yaml(snap / "config.yaml", cfg_clean)
	# Write separate visualization.yaml if provided
	if visualization is not None or profile_visualization is not None:
		# drop legacy 'phase' key if present under visualization
		viz_clean = None
		if isinstance(visualization, dict):
			viz_clean = dict(visualization)
			if 'phase' in viz_clean:
				viz_clean.pop('phase', None)
		else:
			viz_clean = visualization
		viz_payload = {
			"visualization": viz_clean,
			"profile_visualization": profile_visualization,
		}
		_dump_yaml(snap / "visualization.yaml", viz_payload)
	# seed.txt removed in favor of seed.yaml + seeds.txt only
	# RNG metadata snapshot
	if rng_info is not None:
		_dump_yaml(snap / "seed.yaml", rng_info)
	if per_traj_seeds is not None:
		with open(snap / "seeds.txt", "w", encoding="utf-8") as f:
			for s in per_traj_seeds:
				f.write(f"{s}\n")
	if model_path:
		mp = Path(model_path)
		if mp.exists():
			shutil.copy2(mp, snap / mp.name)

	# Registry metadata snapshot (optional)
	if registry_meta is not None:
		_dump_yaml(snap / "registry.yaml", registry_meta)
