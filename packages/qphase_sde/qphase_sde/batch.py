"""qphase_sde: Batch planner and result splitter for parameter scans.

This module lets the generic scheduler fuse multiple expanded scan jobs into a
single GPU tensor operation. The planner decides which jobs can be merged and
builds a combined configuration; the splitter slices the combined trajectory
back into per-scan-point results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.core.batching import BatchPlan, ResultSplitter
from qphase.core.config import JobConfig
from qphase.core.protocols import ResultProtocol
from qphase.core.registry import registry

from qphase_sde.result import SDEResult
from qphase_sde.state import TrajectorySet


class SDEEngineBatchPlanner:
    """Batch planner for the SDE engine.

    Currently supports fusing scan jobs that:

    * use the same engine, integrator, backend and model;
    * have the same time grid (``t0``, ``t1``, ``dt``);
    * have the same ``n_traj`` and initial conditions;
    * differ only in **model parameters** marked as scanable.

    The merged job runs ``n_scan * n_traj`` trajectories in one launch. The
    scanned parameter values are repeated ``n_traj`` times so that each
    trajectory sees its assigned scan point.
    """

    @classmethod
    def can_batch(cls, jobs: list[JobConfig]) -> bool:
        if len(jobs) <= 1:
            return False

        first = jobs[0]
        first_engine = cls._engine_name(first)
        first_model = cls._model_name(first)
        first_backend = cls._backend_name(first)
        first_integrator = cls._integrator_name(first)

        if not first_engine or not first_model:
            return False

        # Extract engine-level invariants.
        first_eng_cfg = first.engine.get(first_engine, {}) if first.engine else {}
        n_traj = first_eng_cfg.get("n_traj")
        t0 = first_eng_cfg.get("t0")
        t1 = first_eng_cfg.get("t1")
        dt = first_eng_cfg.get("dt")
        ic = first_eng_cfg.get("ic")
        seed = first_eng_cfg.get("seed")
        adaptive = first_eng_cfg.get("adaptive")

        # Only scalar numeric invariants are comparable; None is acceptable.
        for job in jobs[1:]:
            if cls._engine_name(job) != first_engine:
                return False
            if cls._model_name(job) != first_model:
                return False
            if cls._backend_name(job) != first_backend:
                return False
            if cls._integrator_name(job) != first_integrator:
                return False

            eng_cfg = job.engine.get(first_engine, {}) if job.engine else {}
            if eng_cfg.get("n_traj") != n_traj:
                return False
            if eng_cfg.get("t0") != t0:
                return False
            if eng_cfg.get("t1") != t1:
                return False
            if eng_cfg.get("dt") != dt:
                return False
            if eng_cfg.get("ic") != ic:
                return False
            if eng_cfg.get("seed") != seed:
                return False
            if eng_cfg.get("adaptive") != adaptive:
                return False

            # No upstream inputs for the first iteration.
            if job.input:
                return False

        # Verify that jobs differ only in model parameters.
        varying = cls._varying_model_params(jobs)
        if not varying:
            return False

        # Conservative check: the model's config schema must accept list/array
        # values for the varying parameters. Models whose fields are strict
        # scalars (e.g. float) cannot be batched because the merged config would
        # fail validation.
        if not cls._model_accepts_array_params(first_model, list(varying.keys())):
            return False

        return True

    @classmethod
    def plan_batch(cls, jobs: list[JobConfig]) -> BatchPlan:
        if not cls.can_batch(jobs):
            raise ValueError("Provided jobs cannot be batched")

        first = jobs[0]
        varying = cls._varying_model_params(jobs)

        # Build the merged job config.
        batch_job = cls._deep_copy_job(first)

        # Determine scan dimension from the first varying parameter.
        n_scan = len(jobs)

        # Repeat each scanned model parameter n_traj times inside the batch.
        eng_cfg = batch_job.engine[cls._engine_name(batch_job)]
        n_traj = eng_cfg.get("n_traj", 1)

        # Keep scan parameters as host arrays so the merged job config remains
        # JSON-serializable for snapshots. The active backend/model converts them
        # to device arrays at execution time.
        for param_path, values in varying.items():
            # Store as a plain Python list so the merged config remains
            # JSON-serializable for snapshots. Models convert back to backend
            # arrays at execution time.
            cls._set_model_param(
                batch_job, param_path, np.repeat(np.asarray(values), n_traj).tolist()
            )

        # Increase trajectory count to cover all scan points.
        eng_cfg["n_traj"] = n_scan * n_traj

        # Tag the merged config so the engine can optionally optimize.
        # EngineConfig allows extra fields, so place batch metadata in the engine
        # config where the engine can read it directly.
        eng_cfg["_batch_scan_count"] = n_scan
        eng_cfg["_batch_scan_params"] = list(varying.keys())

        # Also keep metadata in job.params for downstream bookkeeping / snapshots.
        if batch_job.params is None:
            batch_job.params = {}
        batch_job.params["_batch_scan_count"] = n_scan
        batch_job.params["_batch_scan_params"] = list(varying.keys())

        return BatchPlan(
            batch_job=batch_job,
            original_names=[j.name for j in jobs],
            result_splitter="sde_scan_splitter",
        )

    @classmethod
    def _model_accepts_array_params(
        cls, model_name: str, param_names: list[str]
    ) -> bool:
        """Return True if the model's config schema accepts lists for *param_names*."""
        try:
            model_cls = registry.get_plugin_class("model", model_name)
        except Exception:
            # If we cannot inspect the model, be conservative and do not batch.
            return False

        config_schema = getattr(model_cls, "config_schema", None)
        if config_schema is None:
            return False

        scalar_types = (int, float, bool, str, complex)

        for name in param_names:
            field = config_schema.model_fields.get(name)
            if field is None:
                continue
            ann = field.annotation
            # Plain scalar annotation -> cannot accept a list.
            if ann in scalar_types:
                return False
            # Reject unions of only scalars (e.g. float | int).
            origin = getattr(ann, "__origin__", None)
            if origin is not None:
                args = getattr(ann, "__args__", ())
                if args and all(a in scalar_types for a in args):
                    return False
            # Anything else (Any, list, Sequence, Union with Any, ...) is allowed.
        return True

    @classmethod
    def _varying_model_params(
        cls, jobs: list[JobConfig]
    ) -> dict[str, list[Any]]:
        """Return {dot.path: [val0, val1, ...]} for model params that vary."""
        first = jobs[0]
        first_model_cfg = cls._model_dict(first)
        if not first_model_cfg:
            return {}

        varying: dict[str, list[Any]] = {}
        for key, val in first_model_cfg.items():
            values = [val]
            same = True
            for job in jobs[1:]:
                job_cfg = cls._model_dict(job)
                if job_cfg is None or key not in job_cfg:
                    same = False
                    break
                values.append(job_cfg[key])
                if job_cfg[key] != val:
                    same = False
            if not same:
                varying[key] = values
        return varying

    @classmethod
    def _set_model_param(cls, job: JobConfig, param_name: str, value: Any) -> None:
        """Set a model parameter in a job config by its short name."""
        model_name = cls._model_name(job)
        if not model_name:
            return
        plugins = job.plugins or {}
        model_cfg = plugins.get("model")
        if model_cfg is None:
            model_cfg = job.model_extra.get("model") if job.model_extra else None
        if model_cfg is None:
            return
        if model_name not in model_cfg:
            return
        model_cfg[model_name][param_name] = value

    @classmethod
    def _model_dict(cls, job: JobConfig) -> dict[str, Any] | None:
        model_name = cls._model_name(job)
        if not model_name:
            return None
        plugins = job.plugins or {}
        model_cfg = plugins.get("model")
        if model_cfg is None:
            model_cfg = job.model_extra.get("model") if job.model_extra else None
        if not isinstance(model_cfg, dict):
            return None
        return model_cfg.get(model_name, {})

    @classmethod
    def _model_name(cls, job: JobConfig) -> str:
        plugins = job.plugins or {}
        model_cfg = plugins.get("model") or (job.model_extra or {}).get("model")
        if isinstance(model_cfg, dict) and model_cfg:
            return next(iter(model_cfg.keys()))
        return ""

    @classmethod
    def _engine_name(cls, job: JobConfig) -> str:
        if isinstance(job.engine, dict) and job.engine:
            return next(iter(job.engine.keys()))
        return ""

    @classmethod
    def _backend_name(cls, job: JobConfig) -> str:
        plugins = job.plugins or {}
        backend_cfg = plugins.get("backend") or (job.model_extra or {}).get("backend")
        if isinstance(backend_cfg, dict) and backend_cfg:
            return next(iter(backend_cfg.keys()))
        return ""

    @classmethod
    def _integrator_name(cls, job: JobConfig) -> str:
        plugins = job.plugins or {}
        int_cfg = plugins.get("integrator") or (job.model_extra or {}).get("integrator")
        if isinstance(int_cfg, dict) and int_cfg:
            return next(iter(int_cfg.keys()))
        return ""

    @classmethod
    def _has_analyser(cls, job: JobConfig) -> bool:
        plugins = job.plugins or {}
        return bool(
            plugins.get("analyser")
            or plugins.get("analyzer")
            or (job.model_extra or {}).get("analyser")
            or (job.model_extra or {}).get("analyzer")
        )

    @classmethod
    def _deep_copy_job(cls, job: JobConfig) -> JobConfig:
        import copy

        return copy.deepcopy(job)


class SDEResultSplitter(ResultSplitter):
    """Split a batched SDE result back into per-scan-point results."""

    def split(
        self,
        batched_result: ResultProtocol,
        original_jobs: list[JobConfig],
    ) -> dict[str, ResultProtocol]:
        if not isinstance(batched_result, SDEResult):
            raise TypeError("SDEResultSplitter expects an SDEResult")

        n_scan = len(original_jobs)
        trajectory = batched_result.trajectory

        def _slice_analysis(analysis: dict[str, Any], idx: int) -> dict[str, Any]:
            """Distribute per-scan analysis results to the i-th original job."""
            out: dict[str, Any] = {}
            for key, val in analysis.items():
                if isinstance(val, list) and len(val) == n_scan:
                    out[key] = val[idx]
                else:
                    out[key] = val
            return out

        if trajectory is None:
            # No trajectory data; still split per-scan analysis results if present.
            out: dict[str, ResultProtocol] = {}
            for idx, job in enumerate(original_jobs):
                meta = dict(batched_result.meta)
                meta["job_name"] = job.name
                meta["params"] = self._job_params(job)
                out[job.name] = SDEResult(
                    trajectory=None,
                    meta=meta,
                    analysis=_slice_analysis(batched_result.analysis, idx),
                )
            return out

        data = trajectory.data
        if data.shape[0] != n_scan * trajectory.n_traj // n_scan:
            # shape[0] is n_scan * n_traj. Compute per-scan n_traj.
            pass
        n_total_traj = data.shape[0]
        if n_total_traj % n_scan != 0:
            raise ValueError(
                f"Cannot split trajectory with {n_total_traj} trajectories "
                f"into {n_scan} scan points"
            )
        n_traj = n_total_traj // n_scan

        out = {}
        for idx, job in enumerate(original_jobs):
            start = idx * n_traj
            end = start + n_traj
            slice_data = data[start:end]
            slice_traj = TrajectorySet(
                data=slice_data,
                t0=trajectory.t0,
                dt=trajectory.dt,
                meta=dict(trajectory.meta),
            )
            meta = dict(batched_result.meta)
            meta["job_name"] = job.name
            meta["params"] = self._job_params(job)
            out[job.name] = SDEResult(
                trajectory=slice_traj,
                meta=meta,
                analysis=_slice_analysis(batched_result.analysis, idx),
            )
        return out

    @staticmethod
    def _job_params(job: JobConfig) -> dict[str, Any]:
        """Extract model parameters from a job config."""
        plugins = job.plugins or {}
        model_cfg = plugins.get("model")
        if model_cfg is None:
            model_cfg = (job.model_extra or {}).get("model") if job.model_extra else None
        if isinstance(model_cfg, dict):
            for name, cfg in model_cfg.items():
                if isinstance(cfg, dict):
                    return dict(cfg)
        return {}
