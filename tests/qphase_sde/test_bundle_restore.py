"""Restore SDE scan bundles from v4 artifacts (clean-process chain)."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.artifacts import ArtifactStore
from qphase.core.dataset import iter_dataset_views
from qphase.core.result_loader import load_result
from qphase.core.scan import ScanSpec
from qphase.data import ArtifactCorruptError, BundleDescriptor, DataKind
from qphase.data.resolver import default_artifact_resolver
from qphase.data.store import ArtifactManifest
from qphase_sde.analyser.allan_variance import AllanVarianceAnalyzer
from qphase_sde.analyser.lorentz_fitter import LorentzFitter
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.contracts.bundle import (
    SDE_BUNDLE_ADAPTER_ID,
    SDE_BUNDLE_TYPE_ID,
)
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama
from qphase_sde.result import SDEDataBundle


class _ScanModel:
    name = "restore_scan"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1

    def __init__(self):
        self._params = {"rate": 1.0}

    @property
    def params(self):
        return self._params

    def drift(self, y, t, params):
        del t
        return -np.asarray(params["rate"])[..., None] * y

    def diffusion(self, y, t, params):
        del t, params
        return np.ones(y.shape + (1,))


def _grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.restore_scan.rate",
                    "values": [1.0, 2.0],
                }
            }
        }
    ).compile()


def _scan_bundle():
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            seed=7,
            ic=[["1.0+0.0j"]],
            keep_traj=True,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": _ScanModel(),
            "analyser": {"psd": PsdAnalyzer(kind="complex", modes=[0])},
        },
    )
    return engine.run(context=SimpleNamespace(parameter_grid=_grid(), progress=None))


def _save_scan_bundle(job_dir: Path) -> SDEDataBundle:
    bundle = _scan_bundle()
    job_dir.mkdir(parents=True)
    store = ArtifactStore(job_dir, SimpleNamespace(shard_target_mib=1))
    store.save_result(bundle, "scan")
    return bundle


def test_scan_bundle_restores_from_v3_artifact(tmp_path):
    job_dir = tmp_path / "scan"
    bundle = _save_scan_bundle(job_dir)

    manifest = ArtifactManifest.read(job_dir)
    assert manifest.bundle.type_id == SDE_BUNDLE_TYPE_ID
    assert manifest.bundle.adapter_id == SDE_BUNDLE_ADAPTER_ID
    assert manifest.bundle.product_roles == {
        "trajectories": "trajectories",
        "primary_spectrum": "psd",
    }
    scan = manifest.bundle.descriptor["scan"]
    assert scan["shape"] == [2]
    assert scan["dimension_order"] == ["rate"]
    assert scan["axes"] == {"rate": [1.0, 2.0]}
    assert scan["n_traj_per_point"] == 8
    assert scan["combine"] == "cartesian"
    provenance = manifest.provenance
    assert provenance["engine"] == "sde"
    assert provenance["job_name"] == "scan"
    # Provenance records the real installed distribution versions.
    versions = provenance["versions"]
    assert versions["qphase"] == importlib.metadata.version("qphase")
    assert versions["qphase_sde"] == importlib.metadata.version("qphase-sde")
    # The engine records the fused per-trajectory parameter broadcast; the
    # manifest provenance keeps it as plain JSON (no keys dropped).
    assert provenance["meta"]["params"] == {"rate": [1.0] * 8 + [2.0] * 8}
    assert "meta_dropped" not in provenance

    # Simulate a clean process: drop every in-process artifact binding.
    default_artifact_resolver().clear()
    restored = load_result("scan", job_dir)

    assert isinstance(restored, SDEDataBundle)
    assert restored.shape == (2,)
    assert restored.axes == {"rate": [1.0, 2.0]}
    assert restored.n_traj_per_point == 8
    assert sorted(restored.products) == ["psd", "trajectories"]
    assert restored.provenance.dt == bundle.provenance.dt
    assert restored.metadata["scan_combine"] == "cartesian"
    assert restored.bundle_descriptor.product_roles == manifest.bundle.product_roles

    point = restored.point_view((1,))
    assert point.metadata["scan_point"] == {"rate": 2.0}
    assert point.metadata["params"] == {"rate": 2.0}
    alpha = point.products["trajectories"].handle("alpha").materialize()
    assert alpha.shape[0] == 8
    assert alpha.dtype.kind == "c"


@pytest.mark.parametrize("extent", [2.5, True, 0, -1])
def test_sde_bundle_rejects_non_positive_integer_scan_shape(extent):
    descriptor = BundleDescriptor(
        type_id=SDE_BUNDLE_TYPE_ID,
        adapter_id=SDE_BUNDLE_ADAPTER_ID,
        descriptor_schema=SDE_BUNDLE_TYPE_ID,
        descriptor={
            "scan": {
                "shape": [extent],
                "dimension_order": ["rate"],
                "axes": {"rate": [1.0, 2.0]},
                "n_traj_per_point": 8,
                "combine": "cartesian",
            }
        },
    )

    from qphase_sde.result import _SDE_BUNDLE_ADAPTER

    with pytest.raises(ArtifactCorruptError, match="positive integer"):
        _SDE_BUNDLE_ADAPTER.validate_descriptor(descriptor)


def test_sde_bundle_rejects_product_scan_size_mismatch(tmp_path):
    job_dir = tmp_path / "scan"
    bundle = _scan_bundle()
    descriptor = bundle.bundle_descriptor.model_copy(deep=True)
    descriptor.descriptor["scan"]["shape"] = [3]
    descriptor.descriptor["scan"]["axes"] = {"rate": [1.0, 2.0, 3.0]}

    from qphase.data import save_products

    with pytest.raises(ArtifactCorruptError, match="scan axis size"):
        save_products(
            job_dir,
            bundle.products,
            provenance=bundle.manifest_provenance,
            bundle=descriptor,
        )


def test_sde_bundle_rejects_invalid_stable_product_role(tmp_path):
    job_dir = tmp_path / "scan"
    bundle = _scan_bundle()
    descriptor = bundle.bundle_descriptor.model_copy(
        update={"product_roles": {"primary_spectrum": "trajectories"}}
    )

    from qphase.data import save_products

    with pytest.raises(ArtifactCorruptError, match="primary_spectrum"):
        save_products(
            job_dir,
            bundle.products,
            provenance=bundle.manifest_provenance,
            bundle=descriptor,
        )

    assert bundle.products["trajectories"].kind is DataKind.TIME_SERIES


def test_restored_scan_bundle_feeds_downstream_analysers(tmp_path):
    job_dir = tmp_path / "scan"
    _save_scan_bundle(job_dir)
    default_artifact_resolver().clear()
    restored = load_result("scan", job_dir)
    backend = NumpyBackend()

    # Dataset mode: the cross-job Lorentz fitter consumes the whole bundle.
    fit = LorentzFitter(scan_param="rate").analyze(restored, backend)
    fit_rows = fit.data_dict["fit_rows"]
    assert [row["rate"] for row in fit_rows] == [1.0, 2.0]

    # Map mode: per-point views feed trajectory-level analysers (Allan).
    views = list(iter_dataset_views(restored))
    assert len(views) == 2
    allan = AllanVarianceAnalyzer(
        modes=[0], points=5, min_windows=2, min_independent_windows=1
    )
    for _label, view in views:
        assert isinstance(view, SDEDataBundle)
        legacy = view.legacy_result()
        payload = allan.analyze(legacy.trajectory, backend).data_dict
        assert "mode_results" in payload
        assert legacy.meta["params"]["rate"] in (1.0, 2.0)


def test_scan_bundle_restores_in_clean_subprocess(tmp_path):
    job_dir = tmp_path / "scan"
    _save_scan_bundle(job_dir)

    script = (
        "import json, sys;"
        "from pathlib import Path;"
        "import qphase_sde.result;"  # registers the sde/1 bundle adapter
        "from qphase.core.result_loader import load_result;"
        "restored = load_result('scan', Path(sys.argv[1]));"
        "assert type(restored).__name__ == 'SDEDataBundle';"
        "point = restored.point_view((1,));"
        "psd = restored.products['psd'];"
        "coords = {item.name: list(item.dims) for item in psd.schema.coordinates};"
        "print(json.dumps({"
        "'shape': list(restored.shape),"
        "'axes': restored.axes,"
        "'n_traj_per_point': restored.n_traj_per_point,"
        "'scan_point': point.metadata['scan_point'],"
        "'params': point.metadata['params'],"
        "'coordinates': coords,"
        "'rate': psd.coordinate('rate').tolist(),"
        "}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(job_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload == {
        "shape": [2],
        "axes": {"rate": [1.0, 2.0]},
        "n_traj_per_point": 8,
        "scan_point": {"rate": 2.0},
        "params": {"rate": 2.0},
        "coordinates": {
            "frequency": ["frequency"],
            "mode": ["channel"],
            "rate": ["scan"],
        },
        "rate": [1.0, 2.0],
    }


def test_single_point_bundle_restores_from_v3_artifact(tmp_path):
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.05,
            dt=0.01,
            n_traj=4,
            seed=3,
            ic=[["1.0+0.0j"]],
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": _ScanModel(),
        },
    )
    bundle = engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))
    job_dir = tmp_path / "single"
    job_dir.mkdir()
    ArtifactStore(job_dir, SimpleNamespace(shard_target_mib=1)).save_result(
        bundle, "single"
    )

    default_artifact_resolver().clear()
    restored = load_result("single", job_dir)

    assert isinstance(restored, SDEDataBundle)
    assert restored.shape == ()
    assert restored.axes == {}
    assert restored.point_view(()) is restored
    assert sorted(restored.products) == ["trajectories"]


def test_describe_products_summarizes_sde_scan_artifact(tmp_path):
    from qphase.core.system_config import SystemConfig
    from qphase.service import SchedulerService

    job_dir = tmp_path / "scan"
    _save_scan_bundle(job_dir)

    catalog = SchedulerService(SystemConfig()).describe_products(
        "scan", session_dir=tmp_path
    )

    bundle = catalog.bundle
    assert bundle is not None
    assert bundle.type_id == SDE_BUNDLE_TYPE_ID
    assert bundle.adapter_id == SDE_BUNDLE_ADAPTER_ID
    assert bundle.scan_shape == [2]
    assert bundle.scan_combine == "cartesian"
    assert bundle.scan_axes == {"rate": [1.0, 2.0]}
    assert bundle.n_traj_per_point == 8

    products = {product.name: product for product in catalog.products}
    assert set(products) == {"trajectories", "psd"}
    psd = products["psd"]
    assert psd.kind == "spectral"
    assert psd.materializable is True
    assert psd.attributes["graph_ready"] is True
    assert psd.schema_version == "qphase.product/1"
    assert psd.schema_fingerprint
    assert psd.storage_adapter == "npz/3"
    assert psd.physical_nbytes > 0
    assert psd.sampling_bases
    assert psd.uncertainties
    json.dumps(catalog.model_dump(mode="json"))
