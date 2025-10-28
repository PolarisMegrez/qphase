from __future__ import annotations

"""Snapshot utilities to capture run configuration and seed for reproducibility."""

import shutil
from pathlib import Path
from typing import Dict, Optional, List
from ..core.errors import SDEIOError

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
	if not (_USE_RUAMEL or _USE_PYYAML):
		raise SDEIOError("[600] YAML library not found. Please install 'ruamel.yaml' or 'PyYAML'.")
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
	"""Persist snapshot files in run_dir/config_snapshot.

	Writes:
	- config.json: core run configuration (excluding visualization fields)
	- visualization.json: contains { "visualization": ..., "profile_visualization": ... }
	- seed.txt: random seed (if provided)
	- a copy of the model file (if provided)
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
