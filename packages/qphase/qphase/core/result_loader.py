"""qphase: Result Loader
---------------------------------------------------------
Restore entry point for persisted job results, used by the scheduler's resume
and input-resolution paths.

The artifact manifest v4 (``artifact_manifest.json`` with
``schema_version="qphase.artifact/4"``) is the only supported on-disk result
format: products are reopened as lazily backed typed datasets through
:func:`qphase.data.store.load_bundle`, without filename guessing and without
``allow_pickle``.

Public API
----------
load_result
    Restore a job result from its artifact directory.
"""

import json
from pathlib import Path

from .errors import QPhaseError
from .protocols import ResultProtocol


def load_result(job_name: str, job_dir: Path) -> ResultProtocol:
    """Restore a job result from the artifact manifest v4 in ``job_dir``.

    Parameters
    ----------
    job_name : str
        Name of the job (used in error messages only).
    job_dir : Path
        Job directory holding the persisted artifact.

    Returns
    -------
    ResultProtocol
        The restored bundle: a concrete bundle such as ``SDEDataBundle`` when
        its resource package registered a bundle adapter, otherwise a
        :class:`~qphase.data.bundle.GenericDataBundle`.

    Raises
    ------
    QPhaseError
        If the job directory does not exist, contains no artifact manifest,
        or the manifest does not declare ``schema_version``
        ``"qphase.artifact/4"``.

    """
    if not job_dir.exists():
        raise QPhaseError(f"Job directory not found: {job_dir}")

    manifest_path = job_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        raise QPhaseError(
            f"job '{job_name}' has no artifact manifest in {job_dir}: job "
            f"results are persisted as artifact manifest v4 "
            f"('{manifest_path.name}' with schema_version "
            "'qphase.artifact/4')"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QPhaseError(
            f"failed to read artifact manifest {manifest_path}: {exc}"
        ) from exc
    schema_version = raw.get("schema_version")
    if schema_version != "qphase.artifact/4":
        raise QPhaseError(
            f"job '{job_name}' artifact manifest {manifest_path} has "
            f"unsupported schema_version {schema_version!r}: expected "
            "'qphase.artifact/4'"
        )

    from ..data.store import load_bundle

    return load_bundle(job_dir)
