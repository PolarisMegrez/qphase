QPhaseSDE Config Pipeline
=========================

This package splits configuration into three parts:

1) System parameters (core/config_system.py)
   - Expert-level, affect semantics/behavior (e.g., RNG strategy, numeric dtype, algorithm aliases).
   - Source: packaged `core/system.yaml` (optionally overridden by env `QPHASESDE_SYSTEM_PARAMS`).
   - Not snapshotted.

2) User parameters + defaults (io/config_user.py)
   - Normal user-editable parameters merged with packaged `core/defaults.yaml`.
   - Unknown keys rejected to avoid typos.

3) Runtime distribution (core/config_runtime.py)
   - Compose and expose helpers to feed modules (e.g., `for_engine()`),
     and to snapshot user-effective configuration only.

Convenience facade: `core/config.py`
------------------------------------
Exports `get_default(path)`, `get_system(path)`, a back-compat `get(path)` (defaults),
and `ConfigPipeline` for end-to-end usage.

Engine ergonomics
-----------------
`engine.run(..., backend)` now accepts `backend=None` at call-sites; if `None`, the
engine falls back to `get_default("engine.default_backend")`.

Snapshot policy
---------------
Only user-effective configuration is written to `config_snapshot/config.yaml`.
System parameters are intentionally excluded.
