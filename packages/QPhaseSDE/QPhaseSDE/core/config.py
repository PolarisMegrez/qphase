"""
QPhaseSDE: Core Config Pipeline
-------------------------------
Centralized configuration pipeline that manages:
- System parameters (expert-only) from core/system.yaml (+ optional env override)
- User defaults from core/defaults.yaml (+ optional site/env override)
- Completion of user-provided config using defaults
- Optional updates to a base config (fallback feature)
- Distribution helpers that transform config into module-specific kwargs
- Snapshot helpers and auxiliary validation/import utilities (lightweight)

Public API
----------
- get_system_params() -> dict
- get_system(path, default=None) -> any
- get_defaults() -> dict
- get_default(path, default=None) -> any
- complete_user_config(user: dict | str | Path | None) -> dict
- build_pipeline(user: dict | str | Path | None) -> ConfigPipeline
- class ConfigPipeline: encapsulates effective config and distribution helpers
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Mapping, List
from pathlib import Path
import os
import re
from dataclasses import dataclass, field

from QPhaseSDE.io.snapshot import write_run_snapshot

try:
    import importlib.resources as ilr  # Python >=3.9
except Exception:  # pragma: no cover
    import importlib_resources as ilr  # type: ignore

from .errors import get_logger

__all__ = [
	"get_system",
	"get_defaults",
	"get_default",
    "complete_user_config",
    "build_pipeline",
    "expand_sweeps",
    # Structured API exports
    "ModelArgs",
    "VizJobArgs",
    "EngineConfig",
    "VisualizerConfig",
    "EngineJob",
    "VizJob",
    "make_model_args",
    "make_viz_job_args",
    "make_engine_config",
    "make_visualizer_config",
    "make_engine_job",
    "make_viz_job",
    "expand_model_args",
    "validate_scalar_params",
	"ConfigPipeline",
]

_logger = get_logger()

_HAS_RUAMEL = False
_HAS_PYYAML = False
try:
    from ruamel.yaml import YAML as _RUYAML  # type: ignore
    _HAS_RUAMEL = True
except Exception:
    try:
        import yaml as _PYYAML  # type: ignore
        _HAS_PYYAML = True
    except Exception:
        _HAS_PYYAML = False


def _read_yaml_file(path: Path) -> Dict[str, Any]:
    if _HAS_RUAMEL:
        y = _RUYAML(typ="safe")
        with open(path, "r", encoding="utf-8") as f:
            return dict(y.load(f) or {})
    if _HAS_PYYAML:
        with open(path, "r", encoding="utf-8") as f:
            return dict(_PYYAML.safe_load(f) or {})  # type: ignore[name-defined]
    raise RuntimeError("No YAML parser installed; install 'ruamel.yaml' or 'PyYAML'.")


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _read_yaml_any(path: Path) -> Dict[str, Any]:
    """Read YAML from an arbitrary path using available YAML backend."""
    return _read_yaml_file(path)


def complete_user_config(user: Optional[Mapping[str, Any] | str | Path]) -> Dict[str, Any]:
    """Return a completed user config where unspecified keys are filled from defaults.

    - Accepts a dict or YAML file path; if None, returns defaults.
    - Unknown keys are preserved to allow free-form sections (e.g., jobs[*].params).
    """
    defaults = get_defaults()
    if user is None:
        return dict(defaults)
    if isinstance(user, (str, Path)):
        data = _read_yaml_any(Path(user))
    else:
        data = dict(user)
    return _deep_merge(defaults, data)


# CLI now owns DSL parsing. Keep core limited to arrays/scalars only.


def _expand_params_variants(params: Mapping[str, Any], *, combinator: str = "cartesian") -> list[dict[str, Any]]:
    """Internal helper: expand a mapping of arrays/scalars to concrete dicts."""
    keys: list[str] = []
    values_list: list[list[Any]] = []
    for k, v in dict(params or {}).items():
        keys.append(k)
        if isinstance(v, (list, tuple)):
            values_list.append(list(v))
        else:
            values_list.append([v])

    if not keys:
        return [dict()]

    out: list[dict[str, Any]] = []
    if combinator == "zipped":
        lengths = [len(vs) for vs in values_list]
        L = max(lengths)
        if any((l not in (1, L)) for l in lengths):
            raise ValueError("[540] zipped sweep requires all lists to have the same length or be scalars")
        def _get(vs, i):
            return vs[0] if len(vs) == 1 else vs[i]
        for i in range(L):
            out.append({k: _get(vs, i) for k, vs in zip(keys, values_list)})
        return out

    from itertools import product
    for combo in product(*values_list):
        out.append({k: val for k, val in zip(keys, combo)})
    return out

def expand_sweeps(model: ModelArgs, *, combinator: str = "cartesian") -> List[ModelArgs]:
    """Expand a ModelArgs into one or more scalar-params ModelArgs.

    - Only arrays/scalars are supported in ``model.params``; DSL is not allowed.
    - Returns a list of ModelArgs (length 1 when there is no sweep).
    """
    variants = _expand_params_variants(model.params, combinator=combinator)
    out: List[ModelArgs] = []
    for p in variants:
        out.append(ModelArgs(
            name=model.name,
            module=model.module,
            function=model.function,
            params=p,
            ic=list(model.ic),
            noise=dict(model.noise),
            time=dict(model.time),
            trajectories=dict(model.trajectories),
            solver=model.solver,
            backend=model.backend,
        ))
    return out

# --------------------------- Structured API (mid-/top-layer) ---------------------------

@dataclass
class ModelArgs:
    """Per-job engine arguments (aka EngineJobArgs).

    params should generally be scalars. Arrays are allowed at this level only
    for pre-expansion convenience; building an Engine job or Pipeline from a
    ModelArgs with array params must raise. DSL forms (string or dict lin/log/values)
    are not allowed in ModelArgs.params and should be normalized by the CLI first.
    """
    name: Optional[str] = None
    module: str = ""
    function: str = "build_sde"
    params: Dict[str, Any] = field(default_factory=dict)
    ic: List[Any] = field(default_factory=list)
    noise: Dict[str, Any] = field(default_factory=dict)
    time: Dict[str, Any] = field(default_factory=dict)
    trajectories: Dict[str, Any] = field(default_factory=dict)
    solver: Optional[str] = None
    backend: Optional[str] = None

@dataclass
class VizJobArgs:
    """Per-job visualizer spec bundle (raw specs; service validates)."""
    name: Optional[str] = None
    specs: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EngineConfig:
    """Common engine-level configuration (applies to all jobs)."""
    backend: Optional[str] = None
    solver: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    rng_stream: Optional[str] = None

@dataclass
class VisualizerConfig:
    """Common visualizer-level configuration (styles, defaults)."""
    styles: Dict[str, Any] = field(default_factory=dict)

_DSL_KEYS = {"lin", "log", "values", "val", "linspace", "logspace"}

def _contains_dsl_value(v: Any) -> bool:
    if isinstance(v, str):
        s = v.strip()
        return bool(re.match(r"^(lin|linspace|log|logspace)\s*\(", s))
    if isinstance(v, dict):
        return any(k in v for k in _DSL_KEYS)
    return False

def _has_array_params(params: Mapping[str, Any]) -> bool:
    for val in params.values():
        if isinstance(val, (list, tuple)):
            return True
    return False

def make_model_args(payload: Mapping[str, Any]) -> ModelArgs:
    """Construct ModelArgs from a dict; forbid DSL forms in params.

    Arrays in params are allowed here (pre-expansion), but no DSL (string or dict).
    """
    d = dict(payload or {})
    params = dict(d.get("params", {}))
    for k, v in params.items():
        if _contains_dsl_value(v):
            raise ValueError(f"[541] DSL not allowed in ModelArgs.params ('{k}'); normalize in CLI before building ModelArgs")
    return ModelArgs(
        name=d.get("name"),
        module=str(d.get("module", "")),
        function=str(d.get("function", "build_sde")),
        params=params,
        ic=list(d.get("ic", [])),
        noise=dict(d.get("noise", {})),
        time=dict(d.get("time", {})),
        trajectories=dict(d.get("trajectories", {})),
        solver=d.get("solver"),
        backend=d.get("backend"),
    )

def make_viz_job_args(payload: Mapping[str, Any]) -> VizJobArgs:
    d = dict(payload or {})
    specs = d.get("specs")
    if specs is None:
        specs = d.get("visualizer", [])
    return VizJobArgs(name=d.get("name"), specs=list(specs or []))

def make_engine_config(payload: Mapping[str, Any]) -> EngineConfig:
    d = dict(payload or {})
    return EngineConfig(
        backend=d.get("backend"),
        solver=d.get("solver"),
        progress=dict(d.get("progress", {})),
        rng_stream=d.get("rng_stream"),
    )

def make_visualizer_config(payload: Mapping[str, Any]) -> VisualizerConfig:
    d = dict(payload or {})
    styles = d.get("styles", d.get("visualizer", {}))
    return VisualizerConfig(styles=dict(styles or {}))

def validate_scalar_params(model: ModelArgs) -> None:
    """Raise if any param is an array/tuple. Use before building jobs."""
    if _has_array_params(model.params):
        raise ValueError("[542] ModelArgs.params contains arrays; expand sweeps before building jobs")

def expand_model_args(model: ModelArgs, *, combinator: str = "cartesian") -> List[ModelArgs]:
    """Alias for expand_sweeps(model, ...)."""
    return expand_sweeps(model, combinator=combinator)

# --------- Mid-layer: EngineJob/VizJob objects and builders ---------

@dataclass
class EngineJob:
    name: Optional[str]
    module: str
    function: str
    params: Dict[str, Any]
    ic: List[Any]
    noise: Dict[str, Any]
    time: Dict[str, Any]
    trajectories: Dict[str, Any]
    solver: Optional[str]
    backend: Optional[str]
    progress: Dict[str, Any] = field(default_factory=dict)
    rng_stream: Optional[str] = None


@dataclass
class VizJob:
    name: Optional[str]
    specs: List[Dict[str, Any]]
    styles: Dict[str, Any] = field(default_factory=dict)


def make_engine_job(model: ModelArgs, common: EngineConfig) -> EngineJob:
    """Build a ready-to-run EngineJob from ModelArgs and EngineConfig.

    Raises if ModelArgs.params contains arrays to enforce pre-expansion.
    """
    validate_scalar_params(model)
    progress = dict(common.progress or {})
    return EngineJob(
        name=model.name,
        module=model.module,
        function=model.function,
        params=dict(model.params),
        ic=list(model.ic),
        noise=dict(model.noise),
        time=dict(model.time),
        trajectories=dict(model.trajectories),
        solver=(model.solver or common.solver),
        backend=(model.backend or common.backend),
        progress=progress,
        rng_stream=common.rng_stream,
    )


def make_viz_job(viz: VizJobArgs, common: VisualizerConfig) -> VizJob:
    return VizJob(
        name=viz.name,
        specs=list(viz.specs or []),
        styles=dict(common.styles or {}),
    )

class ConfigPipeline:
    def __init__(self, user_cfg: Optional[Mapping[str, Any] | str | Path] = None,
                 *,
                 engine_config: Optional[EngineConfig] = None,
                 visualizer_config: Optional[VisualizerConfig] = None,
                 model_args_list: Optional[List[ModelArgs]] = None,
                 viz_job_args_list: Optional[List[VizJobArgs]] = None,
                 misc: Optional[Dict[str, Any]] = None) -> None:
        self._system = get_system_params()
        self._user_effective = complete_user_config(user_cfg)
        # Optional structured parts (when built via from_parts)
        self._engine_config = engine_config
        self._visualizer_config = visualizer_config
        self._model_args_list = list(model_args_list or [])
        self._viz_job_args_list = list(viz_job_args_list or [])
        self._misc = dict(misc or {})

    @property
    def system(self) -> Dict[str, Any]:
        return dict(self._system)

    @property
    def user_effective(self) -> Dict[str, Any]:
        return dict(self._user_effective)

    def for_engine(self) -> Dict[str, Any]:
        """Assemble defaulted kwargs for engine.run.
        - backend: if absent, use defaults engine.default_backend
        - progress defaults are already in user_effective
        Other policy knobs (rng strategy) remain system-level and should be
        applied by callers if they translate to engine args.
        """
        cfg = self._user_effective
        out: Dict[str, Any] = {}
        # Backend
        be = cfg.get("engine", {}).get("default_backend")
        if be is None:
            be = get_default("engine.default_backend", "numpy")
        out["backend"] = be
        # Progress knobs for convenience
        prog = cfg.get("engine", {}).get("progress", {})
        out["progress_interval_seconds"] = prog.get("interval_seconds", 1.0)
        out["warmup_min_steps"] = prog.get("warmup_min_steps", 0)
        out["warmup_min_seconds"] = prog.get("warmup_min_seconds", 0.0)
        return out

    def snapshot(self, run_dir: str | Path, **extras: Any) -> None:
        """Persist a run snapshot of user-effective config (no system params)."""
        write_run_snapshot(run_dir, config=self._user_effective, **extras)

    # ------------- structured API adapters -------------
    @classmethod
    def from_parts(cls,
                   engine_config: EngineConfig,
                   visualizer_config: VisualizerConfig,
                   model_args_list: List[ModelArgs],
                   viz_job_args_list: List[VizJobArgs],
                   *,
                   misc: Optional[Dict[str, Any]] = None) -> "ConfigPipeline":
        """Build a pipeline directly from structured parts (no YAML)."""
        # Enforce that ModelArgs are scalar-only at pipeline build time
        for m in model_args_list:
            if _has_array_params(m.params):
                raise ValueError("[542] ModelArgs.params contains arrays; expand sweeps before building ConfigPipeline")
        return cls(user_cfg=None,
                   engine_config=engine_config,
                   visualizer_config=visualizer_config,
                   model_args_list=model_args_list,
                   viz_job_args_list=viz_job_args_list,
                   misc=misc or {})

    def export_parts(self) -> Dict[str, Any]:
        """Export structured parts if present; returns a mapping of parts."""
        return {
            "engine_config": self._engine_config,
            "visualizer_config": self._visualizer_config,
            "model_args_list": list(self._model_args_list),
            "viz_job_args_list": list(self._viz_job_args_list),
            "misc": dict(self._misc),
        }

    # Convenience builders from stored parts (if provided)
    def build_engine_jobs(self) -> List[EngineJob]:
        if self._engine_config is None:
            return []
        jobs: List[EngineJob] = []
        for m in self._model_args_list:
            jobs.append(make_engine_job(m, self._engine_config))
        return jobs

    def build_viz_jobs(self) -> List[VizJob]:
        if self._visualizer_config is None:
            return []
        jobs: List[VizJob] = []
        for v in self._viz_job_args_list:
            jobs.append(make_viz_job(v, self._visualizer_config))
        return jobs


def build_pipeline(user: Optional[Mapping[str, Any] | str | Path] = None) -> ConfigPipeline:
    """Convenience constructor for ConfigPipeline from a user config payload or path."""
    return ConfigPipeline(user)

# -------------- System parameters --------------
_CACHE_SYS: Optional[Dict[str, Any]] = None


def _system_override_path() -> Optional[Path]:
    # Only allow explicit env override to discourage accidental edits by novices.
    p = os.getenv("QPHASESDE_SYSTEM_PARAMS")
    if p:
        pp = Path(p)
        return pp if pp.exists() else None
    return None


def get_system_params() -> Dict[str, Any]:
    global _CACHE_SYS
    if _CACHE_SYS is None:
        try:
            with ilr.files(__package__).joinpath("system.yaml").open("r", encoding="utf-8") as f:
                base_text = f.read()
        except Exception as e:
            _logger.warning(f"[905] Failed to read packaged system params: {e}; using empty.")
            base = {}
        else:
            try:
                if _HAS_RUAMEL:
                    y = _RUYAML(typ="safe")
                    base = dict(y.load(base_text) or {})
                elif _HAS_PYYAML:
                    base = dict(_PYYAML.safe_load(base_text) or {})  # type: ignore[name-defined]
                else:
                    base = {}
            except Exception as e:
                _logger.warning(f"[906] Failed to parse system params: {e}; using empty.")
                base = {}
        ov_path = _system_override_path()
        if ov_path is not None:
            try:
                override = _read_yaml_file(ov_path)
                _CACHE_SYS = _deep_merge(base, override)
            except Exception as e:
                _logger.warning(f"[907] Failed to read system override {ov_path}: {e}")
                _CACHE_SYS = base
        else:
            _CACHE_SYS = base
    return dict(_CACHE_SYS)


def get_system(path: str, default: Any = None) -> Any:
    cur: Any = get_system_params()
    for p in [s for s in str(path).split(".") if s]:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


# -------------- Defaults (normal) --------------
_CACHE_DEFAULTS: Optional[Dict[str, Any]] = None


def _defaults_override_path() -> Optional[Path]:
    # Permit either env override or site location.
    env = os.getenv("QPHASESDE_DEFAULTS_FILE")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # Site path: Windows AppData or POSIX ~/.config
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            p = Path(appdata) / "QPhaseSDE" / "defaults.yaml"
            return p if p.exists() else None
    else:
        p = Path(os.path.expanduser("~")) / ".config" / "QPhaseSDE" / "defaults.yaml"
        return p if p.exists() else None
    return None


def get_defaults() -> Dict[str, Any]:
    global _CACHE_DEFAULTS
    if _CACHE_DEFAULTS is None:
        try:
            with ilr.files(__package__).joinpath("defaults.yaml").open("r", encoding="utf-8") as f:
                base_text = f.read()
        except Exception as e:
            _logger.warning(f"[915] Failed to read packaged defaults: {e}; using empty.")
            base = {}
        else:
            try:
                if _HAS_RUAMEL:
                    y = _RUYAML(typ="safe")
                    base = dict(y.load(base_text) or {})
                elif _HAS_PYYAML:
                    base = dict(_PYYAML.safe_load(base_text) or {})  # type: ignore[name-defined]
                else:
                    base = {}
            except Exception as e:
                _logger.warning(f"[916] Failed to parse defaults: {e}; using empty.")
                base = {}
        ov = _defaults_override_path()
        if ov is not None:
            try:
                override = _read_yaml_file(ov)
                _CACHE_DEFAULTS = _deep_merge(base, override)
            except Exception as e:
                _logger.warning(f"[917] Failed to read defaults override {ov}: {e}")
                _CACHE_DEFAULTS = base
        else:
            _CACHE_DEFAULTS = base
    return dict(_CACHE_DEFAULTS)


def get_default(path: str, default: Any = None) -> Any:
    cur: Any = get_defaults()
    for p in [s for s in str(path).split(".") if s]:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur
