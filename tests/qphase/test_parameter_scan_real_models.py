"""Parameter-scan expansion tests using real model schemas."""

from __future__ import annotations

import pytest
from qphase.core.config import JobConfig, JobList
from qphase.core.registry import registry
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig


@pytest.fixture(autouse=True)
def register_real_models_and_integrator():
    """Register the current production models and a dummy integrator."""
    from models.kerr_3mode import Kerr3ModeModel
    from models.kerr_3pa import Kerr3PAModel
    from models.vdp_2mode import VDPLevel3Model
    from tests.plugins.dummy_plugin import DummyPlugin

    registry.register(
        namespace="model", name="kerr_3pa", builder=Kerr3PAModel, overwrite=True
    )
    registry.register(
        namespace="model", name="kerr_3mode", builder=Kerr3ModeModel, overwrite=True
    )
    registry.register(
        namespace="model", name="vdp_2mode", builder=VDPLevel3Model, overwrite=True
    )
    registry.register(
        namespace="integrator", name="dummy", builder=DummyPlugin, overwrite=True
    )
    yield


def _make_plugins(model_name: str, model_config: dict) -> dict:
    """Build a plugins dict with dummy backend/integrator and the given model."""
    return {
        "model": {model_name: model_config},
        "backend": {"dummy": {"param": 1.0}},
        "integrator": {"dummy": {"param": 1.0}},
    }


def test_kerr_3pa_epsilon_scan():
    """``kerr_3pa.epsilon`` list must expand into individual jobs."""
    job = JobConfig(
        name="kerr_3pa_scan",
        engine={"dummy": {"param": 1.0}},
        plugins=_make_plugins(
            "kerr_3pa",
            {
                "omega0": 1.0,
                "chi": 0.01,
                "kappa3": 0.001,
                "beta": 1.0,
                "kappa1": 1.0,
                "epsilon": [0.025, 0.05, 0.10],
            },
        ),
    )
    scheduler = Scheduler()
    expanded = scheduler._expand_parameter_scans(JobList(jobs=[job]))

    assert len(expanded) == 3
    assert expanded[0].name == "kerr_3pa_scan_001"
    assert expanded[2].name == "kerr_3pa_scan_003"
    values = [j.plugins["model"]["kerr_3pa"]["epsilon"] for j in expanded]
    assert values == [0.025, 0.05, 0.10]


def test_vdp_2mode_omega_a_scan():
    """``vdp_2mode.omega_a`` list must expand into individual jobs."""
    job = JobConfig(
        name="vdp_scan",
        engine={"dummy": {"param": 1.0}},
        plugins=_make_plugins(
            "vdp_2mode",
            {
                "omega_a": [0.9, 1.0, 1.1],
                "omega_b": 1.0,
                "gamma_a": 0.1,
                "gamma_b": 0.1,
                "Gamma": 1.0,
                "g": 0.5,
                "D": 1.0,
            },
        ),
    )
    scheduler = Scheduler()
    expanded = scheduler._expand_parameter_scans(JobList(jobs=[job]))

    assert len(expanded) == 3
    values = [j.plugins["model"]["vdp_2mode"]["omega_a"] for j in expanded]
    assert values == [0.9, 1.0, 1.1]


def test_kerr_3mode_kappa_a_scan():
    """``kerr_3mode.kappa_a`` list must expand into individual jobs."""
    job = JobConfig(
        name="kerr_3mode_scan",
        engine={"dummy": {"param": 1.0}},
        plugins=_make_plugins(
            "kerr_3mode",
            {
                "omega_a": 1.0,
                "omega_b": 1.0,
                "omega_c": 1.0,
                "kappa_a": [0.05, 0.10],
                "kappa_b": 0.1,
                "kappa_c": 0.1,
                "chi": 0.01,
                "g_ab": 0.1,
                "g_ac": 0.1,
            },
        ),
    )
    scheduler = Scheduler()
    expanded = scheduler._expand_parameter_scans(JobList(jobs=[job]))

    assert len(expanded) == 2
    values = [j.plugins["model"]["kerr_3mode"]["kappa_a"] for j in expanded]
    assert values == [0.05, 0.10]


def test_zipped_scan_with_real_model():
    """Zipped scan aligns lists of the same length across real model parameters."""
    job = JobConfig(
        name="vdp_zipped",
        engine={"dummy": {"param": 1.0}},
        plugins=_make_plugins(
            "vdp_2mode",
            {
                "omega_a": [0.9, 1.0, 1.1],
                "omega_b": [0.8, 0.9, 1.0],
                "gamma_a": 0.1,
                "gamma_b": 0.1,
                "Gamma": 1.0,
                "g": 0.5,
                "D": 1.0,
            },
        ),
    )
    system_config = SystemConfig(parameter_scan={"method": "zipped"})
    scheduler = Scheduler(system_config=system_config)
    expanded = scheduler._expand_parameter_scans(JobList(jobs=[job]))

    assert len(expanded) == 3
    assert expanded[0].plugins["model"]["vdp_2mode"]["omega_a"] == 0.9
    assert expanded[0].plugins["model"]["vdp_2mode"]["omega_b"] == 0.8
    assert expanded[2].plugins["model"]["vdp_2mode"]["omega_a"] == 1.1
    assert expanded[2].plugins["model"]["vdp_2mode"]["omega_b"] == 1.0
