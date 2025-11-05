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

from .schemas import TriadConfig, RootConfig, JobConfig, JobSpec
from QPhaseSDE.core.config import get_default
from QPhaseSDE.core.errors import QPSConfigError, QPSIOError


def load_triad_config(path: str | Path) -> TriadConfig:
    """Load and validate a legacy triad configuration from a YAML file path."""
    p = Path(path)
    if _YAML_BACKEND == 'ruamel':
        yaml_loader = _yaml_loader_factory()
        with p.open("r", encoding="utf-8") as f:
            data = yaml_loader.load(f)
    elif _YAML_BACKEND == 'pyyaml' and pyyaml is not None:
        with p.open("r", encoding="utf-8") as f:
            data = pyyaml.safe_load(f)  # type: ignore
    else:
        raise QPSIOError("[001] No YAML loader available. Install 'ruamel.yaml' or 'PyYAML'.")

    if not isinstance(data, dict):
        raise QPSConfigError("[505] Config must be a mapping with sections: model, profile, run")
    return TriadConfig.model_validate(data)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def _linspace(a: float, b: float, n: int):
    if n <= 1:
        return [a]
    step = (b - a) / float(n - 1)
    return [a + i * step for i in range(n)]


def _logspace10(a_exp: float, b_exp: float, n: int):
    if n <= 1:
        return [10.0 ** a_exp]
    step = (b_exp - a_exp) / float(n - 1)
    return [10.0 ** (a_exp + i * step) for i in range(n)]


def _values_from_dsl(spec):
    # Direct array
    if isinstance(spec, (list, tuple)):
        return list(spec)
    # String DSL
    if isinstance(spec, str):
        s = spec.strip()
        import re as _re
        m = _re.match(r"^(lin|linspace|log|logspace)\s*\(\s*([^)]*)\s*\)\s*$", s)
        if m:
            kind = m.group(1)
            parts = [p.strip() for p in m.group(2).split(',') if p.strip()]
            if len(parts) >= 3:
                try:
                    a = _as_float(parts[0]); b = _as_float(parts[1]); n = int(float(parts[2]))
                except Exception:
                    return [spec]
                if kind in ("lin", "linspace"):
                    return _linspace(a, b, max(1, int(n)))
                return _logspace10(a, b, max(1, int(n)))
        return [spec]
    # Dict DSL + aliases
    if isinstance(spec, dict):
        d = dict(spec)
        if "val" in d and "values" not in d:
            d["values"] = d.pop("val")
        if "linspace" in d and "lin" not in d:
            d["lin"] = d.pop("linspace")
        if "logspace" in d and "log" not in d:
            d["log"] = d.pop("logspace")
        if "values" in d and isinstance(d["values"], (list, tuple)):
            return list(d["values"])  # explicit list
        if "lin" in d:
            payload = d["lin"]
            if isinstance(payload, dict):
                s = _as_float(payload.get("start")); t = _as_float(payload.get("stop")); n = int(payload.get("num", 1))
            elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                s = _as_float(payload[0]); t = _as_float(payload[1]); n = int(payload[2])
            else:
                return [spec]
            return _linspace(s, t, max(1, int(n)))
        if "log" in d:
            payload = d["log"]
            if isinstance(payload, dict):
                s = _as_float(payload.get("start")); t = _as_float(payload.get("stop")); n = int(payload.get("num", 1))
            elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                s = _as_float(payload[0]); t = _as_float(payload[1]); n = int(payload[2])
            else:
                return [spec]
            return _logspace10(s, t, max(1, int(n)))
        return [spec]
    # Scalar
    return [spec]


def _normalize_params_to_arrays(params: dict) -> dict:
    out = {}
    for k, v in dict(params or {}).items():
        out[k] = _values_from_dsl(v)
    return out


def _expand_params_variants(params: dict, *, combinator: str = "cartesian") -> list[dict]:
    """Expand a mapping of arrays/scalars to a list of concrete param dicts.

    This keeps CLI ownership of DSL/YAML semantics; core only accepts arrays/scalars.
    """
    keys = []
    values_list = []
    for k, v in dict(params or {}).items():
        keys.append(k)
        values_list.append(list(v) if isinstance(v, (list, tuple)) else [v])
    if not keys:
        return [dict()]
    out: list[dict] = []
    if str(combinator) == "zipped":
        lengths = [len(vs) for vs in values_list]
        L = max(lengths)
        if any((l not in (1, L)) for l in lengths):
            raise QPSConfigError("[540] zipped sweep requires equal lengths or scalars for all params")
        def _get(vs, i):
            return vs[0] if len(vs) == 1 else vs[i]
        for i in range(L):
            out.append({k: _get(vs, i) for k, vs in zip(keys, values_list)})
        return out
    # Cartesian
    from itertools import product
    for combo in product(*values_list):
        out.append({k: val for k, val in zip(keys, combo)})
    return out


def load_root_config(path: str | Path) -> RootConfig:
    """Load the new jobs-based configuration. If legacy triad keys are detected,
    migrate to RootConfig with a single job and move run.visualizer into jobs[0].visualizer.
    """
    p = Path(path)
    if _YAML_BACKEND == 'ruamel':
        yaml_loader = _yaml_loader_factory()
        with p.open("r", encoding="utf-8") as f:
            data = yaml_loader.load(f)
    elif _YAML_BACKEND == 'pyyaml' and pyyaml is not None:
        with p.open("r", encoding="utf-8") as f:
            data = pyyaml.safe_load(f)  # type: ignore
    else:
        raise QPSIOError("[001] No YAML loader available. Install 'ruamel.yaml' or 'PyYAML'.")

    if not isinstance(data, dict):
        raise QPSConfigError("[505] Config must be a mapping with sections: profile, run, jobs (or legacy model/profile/run)")

    # Legacy migration path
    if 'model' in data and 'profile' in data and 'run' in data:
        triad = TriadConfig.model_validate(data)
        # Build a single job from legacy 'model', and lift run.visualizer into job.visualizer
        job_dict = {
            'module': triad.model.module,
            'function': triad.model.function,
            'params': triad.model.params,
            'ic': triad.model.ic,
            'noise': triad.model.noise.model_dump(),  # type: ignore[attr-defined]
        }
        # Convert legacy run visualizer into list of JobSpec
        vis_list: list[dict] = []
        rv = getattr(triad.run, 'visualizer', None)
        if rv is not None:
            # phase portraits
            pp = getattr(rv, 'phase_portrait', None)
            if pp:
                for s in pp:
                    vis_list.append({'kind': getattr(s, 'kind'), 'modes': getattr(s, 'modes'), 't_range': getattr(s, 't_range', None)})
            # psd
            psd = getattr(rv, 'psd', None)
            if psd:
                for s in psd:
                    d = {'kind': getattr(s, 'kind'), 'modes': getattr(s, 'modes'), 't_range': getattr(s, 't_range', None)}
                    # xlim is style now; retain here if provided for back-compat (service ignores if style overrides)
                    xlim = getattr(s, 'xlim', None)
                    if xlim is not None:
                        d['xlim'] = xlim  # type: ignore
                    vis_list.append(d)
        if vis_list:
            job_dict['visualizer'] = vis_list
        root = {
            'profile': triad.profile.model_dump(),  # type: ignore[attr-defined]
            'run': triad.run.model_dump(),          # type: ignore[attr-defined]
            'jobs': [job_dict],
        }
        # Expand sweeps into distinct jobs (cartesian default)
        jobs_in = root.get('jobs', [])
        jobs_out: list[dict] = []
        for j in jobs_in:
            params = j.get('params', {})
            combinator = j.get('combinator') or get_default('cli.job.combinator', 'cartesian')
            params_arr = _normalize_params_to_arrays(params)
            variants = _expand_params_variants(params_arr, combinator=str(combinator))
            for vi, pv in enumerate(variants):
                jj = dict(j)
                jj['params'] = pv
                if jj.get('name'):
                    jj['name'] = f"{jj['name']}#{vi+1}" if len(variants) > 1 else jj['name']
                jobs_out.append(jj)
        root['jobs'] = jobs_out
        return RootConfig.model_validate(root)

    # New format: also normalize per-job visualizer dicts into a flat list expected by schema
    try:
        jobs = data.get('jobs') if isinstance(data, dict) else None
        if isinstance(jobs, list):
            for j in jobs:
                if not isinstance(j, dict):
                    continue
                viz = j.get('visualizer')
                if isinstance(viz, dict):
                    flat: list[dict] = []
                    # phase portraits: accept either 'phase_portrait' or legacy 'phase'
                    for key in ('phase_portrait', 'phase'):
                        pp = viz.get(key)
                        if isinstance(pp, list):
                            for s in pp:
                                if isinstance(s, dict) and 'kind' in s and 'modes' in s:
                                    d = {'kind': s.get('kind'), 'modes': s.get('modes')}
                                    if 't_range' in s:
                                        d['t_range'] = s.get('t_range')
                                    flat.append(d)
                    # psd specs
                    psd = viz.get('psd')
                    if isinstance(psd, list):
                        for s in psd:
                            if isinstance(s, dict) and 'kind' in s and 'modes' in s:
                                d = {'kind': s.get('kind'), 'modes': s.get('modes')}
                                if 't_range' in s:
                                    d['t_range'] = s.get('t_range')
                                # drop 'xlim' here; PSD style now lives in profile.visualizer.psd
                                flat.append(d)
                    j['visualizer'] = flat if flat else None
    except Exception:
        # Non-fatal; let schema validation surface precise errors
        pass

    # Expand sweeps into distinct jobs (cartesian default or per-job combinator)
    if isinstance(data, dict):
        jobs_in = data.get('jobs', []) if isinstance(data.get('jobs', None), list) else []
        jobs_out: list[dict] = []
        for j in jobs_in:
            if not isinstance(j, dict):
                continue
            params = j.get('params', {})
            combinator = j.get('combinator') or get_default('cli.job.combinator', 'cartesian')
            params_arr = _normalize_params_to_arrays(params)
            variants = _expand_params_variants(params_arr, combinator=str(combinator))
            for vi, pv in enumerate(variants):
                jj = dict(j)
                jj['params'] = pv
                if jj.get('name'):
                    jj['name'] = f"{jj['name']}#{vi+1}" if len(variants) > 1 else jj['name']
                jobs_out.append(jj)
        data = dict(data)
        data['jobs'] = jobs_out
    return RootConfig.model_validate(data)
