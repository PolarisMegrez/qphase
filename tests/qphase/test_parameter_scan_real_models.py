"""Parameter-scan expansion tests using all local model schemas."""

from __future__ import annotations

import pytest
from qphase.core.config import JobConfig, JobList
from qphase.core.registry import registry
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def register_real_models_and_integrator():
    from models.kerr_2mode import Kerr2ModeModel
    from models.kerr_3mode import Kerr3ModeModel
    from models.vdp_2mode import VDP2ModeModel
    from tests.plugins.dummy_plugin import DummyPlugin

    for name, builder in (
        ("vdp_2mode", VDP2ModeModel),
        ("kerr_2mode", Kerr2ModeModel),
        ("kerr_3mode", Kerr3ModeModel),
    ):
        registry.register(namespace="model", name=name, builder=builder, overwrite=True)
    registry.register(
        namespace="integrator", name="dummy", builder=DummyPlugin, overwrite=True
    )
    yield


def _plugins(model_name: str, model_config: dict) -> dict:
    return {
        "model": {model_name: model_config},
        "backend": {"dummy": {"param": 1.0}},
        "integrator": {"dummy": {"param": 1.0}},
    }


@pytest.mark.parametrize(
    ("name", "config", "parameter", "values"),
    [
        (
            "vdp_2mode",
            dict(omega_a=[0.9, 1.0], omega_b=1.0, gamma_a=0.1,
                 gamma_b=0.1, Gamma=1.0, g=0.5),
            "omega_a",
            [0.9, 1.0],
        ),
        (
            "kerr_2mode",
            dict(omega_a=0.0, omega_b=[-0.001, -0.01, -0.1], chi=0.01,
                 gamma_a=0.5, gamma_b=1.8728, g=0.5),
            "omega_b",
            [-0.001, -0.01, -0.1],
        ),
        (
            "kerr_3mode",
            dict(omega_a=1.0, omega_b=1.0, omega_c=1.0, chi=0.01,
                 gamma_a=[0.05, 0.1], gamma_b=0.1, gamma_c=0.1,
                 g_ab=0.1, g_ac=0.1),
            "gamma_a",
            [0.05, 0.1],
        ),
    ],
)
def test_real_model_parameter_scan(name, config, parameter, values):
    job = JobConfig(
        name=f"{name}_scan",
        engine={"dummy": {"param": 1.0}},
        plugins=_plugins(name, config),
    )
    expanded = Scheduler()._expand_parameter_scans(JobList(jobs=[job]))
    assert len(expanded) == len(values)
    assert [job.plugins["model"][name][parameter] for job in expanded] == values


def test_zipped_scan_with_real_model():
    config = dict(
        omega_a=[0.9, 1.0, 1.1], omega_b=[0.8, 0.9, 1.0],
        gamma_a=0.1, gamma_b=0.1, Gamma=1.0, g=0.5,
    )
    job = JobConfig(
        name="vdp_zipped",
        engine={"dummy": {"param": 1.0}},
        plugins=_plugins("vdp_2mode", config),
    )
    scheduler = Scheduler(
        system_config=SystemConfig(parameter_scan={"method": "zipped"})
    )
    expanded = scheduler._expand_parameter_scans(JobList(jobs=[job]))
    assert len(expanded) == 3
    assert expanded[0].plugins["model"]["vdp_2mode"]["omega_b"] == 0.8
    assert expanded[2].plugins["model"]["vdp_2mode"]["omega_a"] == 1.1
