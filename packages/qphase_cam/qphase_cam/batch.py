"""Scheduler batching for independent CAM parameter-scan points."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from qphase.core.batching import BatchPlan, ResultSplitter
from qphase.core.config import JobConfig
from qphase.core.protocols import ResultProtocol

from qphase_cam.result import CAMResult


class CAMBatchPlanner:
    """Fuse compatible CAM jobs when using ``batched_newton``."""

    @classmethod
    def can_batch(cls, jobs: list[JobConfig]) -> bool:
        if len(jobs) <= 1 or any(job.input for job in jobs):
            return False
        first = jobs[0]
        if cls._selected(first, "cam_solver") not in {
            "batched_newton",
            "multistability",
        }:
            return False
        signatures = [
            (
                cls._selected(job, "model"),
                cls._selected(job, "backend"),
                cls._plugin_config(job, "cam_solver"),
                cls._plugin_config(job, "cam_postprocessor"),
            )
            for job in jobs
        ]
        if any(signature != signatures[0] for signature in signatures[1:]):
            return False
        configs = [cls._model_config(job) for job in jobs]
        if not configs or any(not config for config in configs):
            return False
        return any(
            any(config.get(key) != configs[0].get(key) for config in configs[1:])
            for key in configs[0]
        )

    @classmethod
    def plan_batch(cls, jobs: list[JobConfig]) -> BatchPlan:
        if not cls.can_batch(jobs):
            raise ValueError("CAM jobs are not batch-compatible")
        batch_job = copy.deepcopy(jobs[0])
        target = cls._model_config(batch_job)
        configs = [cls._model_config(job) for job in jobs]
        for key in target:
            values = [config[key] for config in configs]
            varies = any(value != values[0] for value in values)
            target[key] = values if varies else values[0]
        batch_job.params = dict(batch_job.params or {})
        batch_job.params["_batch_scan_count"] = len(jobs)
        return BatchPlan(
            batch_job=batch_job,
            original_names=[job.name for job in jobs],
            result_splitter="cam_scan_splitter",
        )

    @staticmethod
    def _plugin_groups(job: JobConfig) -> dict[str, Any]:
        groups = dict(job.plugins or {})
        groups.update(
            {
                key: value
                for key, value in (job.model_extra or {}).items()
                if isinstance(value, dict) and key not in groups
            }
        )
        return groups

    @classmethod
    def _selected(cls, job: JobConfig, namespace: str) -> str:
        group = cls._plugin_groups(job).get(namespace, {})
        return next(iter(group), "") if isinstance(group, dict) else ""

    @classmethod
    def _plugin_config(cls, job: JobConfig, namespace: str) -> Any:
        return cls._plugin_groups(job).get(namespace, {})

    @classmethod
    def _model_config(cls, job: JobConfig) -> dict[str, Any]:
        group = cls._plugin_groups(job).get("model", {})
        if not isinstance(group, dict) or not group:
            return {}
        config = next(iter(group.values()))
        return config if isinstance(config, dict) else {}


class CAMResultSplitter(ResultSplitter):
    """Split the leading batch dimension into original CAM jobs."""

    def split(
        self,
        batched_result: ResultProtocol,
        original_jobs: list[JobConfig],
    ) -> dict[str, ResultProtocol]:
        if not isinstance(batched_result, CAMResult):
            raise TypeError("CAMResultSplitter expects CAMResult")
        if batched_result.states.shape[0] != len(original_jobs):
            raise ValueError("CAM batch dimension does not match original jobs")
        output: dict[str, ResultProtocol] = {}
        for index, job in enumerate(original_jobs):
            postprocess = {
                name: self._slice(value, index, len(original_jobs))
                for name, value in batched_result.postprocess.items()
            }
            meta = dict(batched_result.meta)
            meta["job_name"] = job.name
            params = CAMBatchPlanner._model_config(job)
            output[job.name] = CAMResult(
                states=batched_result.states[index],
                residuals=batched_result.residuals[index],
                success=batched_result.success[index],
                valid_mask=batched_result.valid_mask[index],
                solution_count=batched_result.solution_count[index],
                params=dict(params),
                axes={
                    name: self._slice(value, index, len(original_jobs))
                    for name, value in batched_result.axes.items()
                },
                postprocess=postprocess,
                meta=meta,
            )
        return output

    @staticmethod
    def _slice(value: Any, index: int, batch_size: int) -> Any:
        if isinstance(value, np.ndarray) and value.shape[:1] == (batch_size,):
            return value[index]
        return value
