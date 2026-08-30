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

from pathlib import Path

from .protocols import ResultProtocol


def load_result(job_name: str, job_dir: Path) -> ResultProtocol:
    """Restore a job result from the artifact manifest v4 in ``job_dir``.

    Parameters
    ----------
    job_name : str
        Name of the job (used for context in error messages only).
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
    qphase.data.errors.ArtifactError
        If the job directory does not exist, contains no artifact manifest,
        or the manifest fails validation. Artifact errors subclass
        :class:`qphase.core.errors.QPhaseError`, so core callers keep one
        error taxonomy.

    """
    from ..data.errors import ArtifactError
    from ..data.store import load_bundle

    try:
        return load_bundle(job_dir)
    except ArtifactError as exc:
        raise type(exc)(f"job '{job_name}': {exc}") from exc
