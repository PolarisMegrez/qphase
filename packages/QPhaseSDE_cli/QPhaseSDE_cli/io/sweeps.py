"""
QPhaseSDE_cli.io.sweeps
-----------------------
User-facing sweep helpers for CLI layer. Supports per-job parameter sweeps with
arrays or scalars and two combinators ('cartesian' and 'zipped'). DSL parsing
is handled elsewhere (config.loader) before calling these helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from QPhaseSDE.core.config import get_default

__all__ = [
    "expand_job_params",
]


def _expand_params_variants(params: Mapping[str, Any], *, combinator: str = "cartesian") -> List[Dict[str, Any]]:
    keys: List[str] = []
    values_list: List[List[Any]] = []
    for k, v in dict(params or {}).items():
        keys.append(k)
        values_list.append(list(v) if isinstance(v, (list, tuple)) else [v])
    if not keys:
        return [dict()]
    out: List[Dict[str, Any]] = []
    if str(combinator) == 'zipped':
        lengths = [len(vs) for vs in values_list]
        L = max(lengths)
        if any((l not in (1, L)) for l in lengths):
            raise ValueError("[540] zipped sweep requires equal lengths or scalars for all params")
        def _get(vs, i):
            return vs[0] if len(vs) == 1 else vs[i]
        for i in range(L):
            out.append({k: _get(vs, i) for k, vs in zip(keys, values_list)})
        return out
    from itertools import product
    for combo in product(*values_list):
        out.append({k: val for k, val in zip(keys, combo)})
    return out


def expand_job_params(params: Mapping[str, Any], *, combinator: str | None = None) -> List[Dict[str, Any]]:
    """Expand a job's params mapping into a list of concrete param dicts.

    - params: mapping of parameter names to values/sweep-DSL specs
    - combinator: 'cartesian'|'zipped' or None; when None, use default
      'cli.job.combinator' from core defaults (falls back to 'cartesian').
    """
    comb = combinator or get_default("cli.job.combinator", "cartesian")
    return _expand_params_variants(params, combinator=str(comb))
