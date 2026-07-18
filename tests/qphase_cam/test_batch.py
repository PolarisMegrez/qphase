"""CAM scheduler batch planner and result splitter tests."""

from __future__ import annotations

import numpy as np
from qphase.core.config import JobConfig
from qphase_cam.batch import CAMBatchPlanner, CAMResultSplitter
from qphase_cam.result import CAMResult


def _job(name: str, omega_a: float) -> JobConfig:
    return JobConfig.model_construct(
        name=name,
        engine={"cam": {}},
        plugins={
            "backend": {"numpy": {}},
            "model": {
                "vdp_2mode": {
                    "omega_a": omega_a,
                    "omega_b": 0.0,
                    "gamma_a": 2.0,
                    "gamma_b": 0.5,
                    "Gamma": 0.0001,
                    "g": 0.5,
                }
            },
            "cam_solver": {"batched_newton": {"max_iterations": 10}},
            "cam_postprocessor": {"rayleigh_frequency": {}},
        },
    )


def test_batch_planner_merges_only_model_parameters():
    jobs = [_job("left", 0.0), _job("right", 0.001)]
    assert CAMBatchPlanner.can_batch(jobs)
    plan = CAMBatchPlanner.plan_batch(jobs)
    config = plan.batch_job.plugins["model"]["vdp_2mode"]
    assert config["omega_a"] == [0.0, 0.001]
    assert config["gamma_a"] == 2.0
    assert plan.result_splitter == "cam_scan_splitter"


def test_result_splitter_restores_scalar_params_and_postprocess():
    jobs = [_job("left", 0.0), _job("right", 0.001)]
    result = CAMResult(
        states=np.stack((np.eye(2), 2.0 * np.eye(2)))[:, None],
        residuals=np.zeros((2, 1)),
        success=np.ones((2, 1), dtype=bool),
        valid_mask=np.ones((2, 1), dtype=bool),
        solution_count=np.ones(2, dtype=int),
        params={"omega_a": np.array([0.0, 0.001])},
        postprocess={"rayleigh_frequency": np.array([[0.0], [0.001]])},
        meta={"engine": "cam"},
    )
    split = CAMResultSplitter().split(result, jobs)
    assert set(split) == {"left", "right"}
    assert split["right"].params["omega_a"] == 0.001
    assert split["right"].states.shape == (1, 2, 2)
    assert split["right"].postprocess["rayleigh_frequency"][0] == 0.001
