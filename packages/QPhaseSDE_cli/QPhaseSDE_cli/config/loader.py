from __future__ import annotations

"""Configuration loader for triad YAML (model/profile/run).

Prefers ruamel.yaml for round-trip fidelity; falls back to PyYAML if available.
"""

from pathlib import Path
from typing import Tuple, Callable, Any

# Configure YAML loader backend
_YAML_BACKEND: str
_yaml_loader_factory: Callable[[], Any]
pyyaml = None  # type: ignore

try:
    # Preferred: ruamel for comment-preserving round-trips
    from ruamel.yaml import YAML as _RuamelYAML  # type: ignore

    def _make_ruamel():
        return _RuamelYAML(typ="rt")

    _yaml_loader_factory = _make_ruamel
    _YAML_BACKEND = 'ruamel'
except Exception:
    # Fallback: PyYAML if available (no comment preservation)
    try:
        import yaml as _pyyaml  # type: ignore

        pyyaml = _pyyaml  # type: ignore
        _yaml_loader_factory = lambda: None  # not used for PyYAML path
        _YAML_BACKEND = 'pyyaml'
    except Exception:
        _yaml_loader_factory = lambda: None
        _YAML_BACKEND = 'none'

from .schemas import TriadConfig
from QPhaseSDE.core.errors import ConfigError, SDEIOError


def load_triad_config(path: str | Path) -> TriadConfig:
    """Load and validate a triad configuration from a YAML file path."""
    p = Path(path)
    if _YAML_BACKEND == 'ruamel':
        yaml_loader = _yaml_loader_factory()
        with p.open("r", encoding="utf-8") as f:
            data = yaml_loader.load(f)
    elif _YAML_BACKEND == 'pyyaml' and pyyaml is not None:
        with p.open("r", encoding="utf-8") as f:
            data = pyyaml.safe_load(f)  # type: ignore
    else:
        raise SDEIOError("[600] No YAML loader available. Install 'ruamel.yaml' or 'PyYAML'.")

    if not isinstance(data, dict):
        raise ConfigError("[505] Config must be a mapping with sections: model, profile, run")
    return TriadConfig.model_validate(data)
