"""qphase_cam: the historical search-evidence migration input
---------------------------------------------------------
The pre-2.0 CAM package shipped twelve ``qphase.search-evidence/1`` documents.
They are **re-run inputs**, not evidence: they embed the normalized workflows and
the reproducibility contract that Phase 10G will use to recompute a bounded
conclusion, but they carry no result payloads and no audit counters.

Three things follow, and the module makes all three structural:

* The document is validated, not trusted. The schema marker, the four
  reproducibility flags, the workflow count and every source workflow entry must
  agree, or the import fails.
* It is not a :class:`~qphase_cam.evidence.SearchAuditFragment`. A fragment is a
  typed audit with conserving counters; this is a citation of workflows to run
  again. Nothing here derives counters, and no evidence product can be built from
  it — there is no conversion API, and
  :func:`qphase_cam.data.evidence.build_evidence_product` rejects it by type.
* It keeps the source identity it was read from: the path and the SHA-256 of the
  bytes, so a Phase 10G re-run can prove which document produced it and detect
  later edits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MIGRATION_DOCUMENT_SCHEMA",
    "SearchEvidenceMigrationInput",
    "SearchEvidenceMigrationWorkflow",
    "import_search_evidence_document",
]

#: Schema marker of the historical documents this module reads.
MIGRATION_DOCUMENT_SCHEMA: Literal["qphase.search-evidence/1"] = (
    "qphase.search-evidence/1"
)

#: Schema marker of the normalized workflow embedded in a source entry.
WORKFLOW_SCHEMA = "qphase.workflow/2"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SearchEvidenceMigrationWorkflow(BaseModel):
    """One embedded, normalized workflow of a migration document.

    ``source_path`` and ``sha256`` identify the workflow file the document was
    extracted from (the document records those two itself); ``normalized_workflow``
    is the resolved payload a Phase 10G re-run executes, kept verbatim so nothing
    has to be re-derived.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    last_git_commit: str = ""
    normalized_workflow: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_workflow(self) -> SearchEvidenceMigrationWorkflow:
        schema = self.normalized_workflow.get("schema")
        if schema is not None and str(schema) != WORKFLOW_SCHEMA:
            raise ValueError(
                f"workflow {self.source_path!r} declares schema {schema!r}; "
                f"expected {WORKFLOW_SCHEMA!r}"
            )
        jobs = self.normalized_workflow.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(
                f"workflow {self.source_path!r} carries no jobs; it cannot be re-run"
            )
        if not str(self.normalized_workflow.get("id", "")).strip():
            raise ValueError(f"workflow {self.source_path!r} carries no id")
        return self


class SearchEvidenceMigrationInput(BaseModel):
    """One validated historical search-evidence document.

    This is a migration input. It deliberately exposes no counters and no
    accepted candidates, because the document does not contain any: the fields it
    does expose are the ones a re-run needs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_schema: Literal["qphase.search-evidence/1"] = MIGRATION_DOCUMENT_SCHEMA
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=_SHA256_PATTERN)
    model: str = Field(min_length=1)
    status: str = Field(min_length=1)
    conclusion_report: str | None = None
    reproducibility_contract: dict[str, Any] = Field(min_length=1)
    workflow_count: int = Field(ge=1)
    workflows: tuple[SearchEvidenceMigrationWorkflow, ...] = Field(min_length=1)
    workspace_head_at_extraction: str = ""
    generated_at_gate: str = ""

    @model_validator(mode="after")
    def _check_contract(self) -> SearchEvidenceMigrationInput:
        contract = self.reproducibility_contract
        expected = {
            "workflow_semantics_embedded": True,
            "result_payloads_embedded": False,
            "legacy_runtime_compatibility_claimed": False,
        }
        for flag, value in expected.items():
            if flag not in contract:
                raise ValueError(
                    f"reproducibility_contract is missing {flag!r}; this document "
                    "is not a re-run input"
                )
            if bool(contract[flag]) is not value:
                raise ValueError(
                    f"reproducibility_contract.{flag} must be {value}, got "
                    f"{contract[flag]!r}"
                )
        if self.workflow_count != len(self.workflows):
            raise ValueError(
                f"workflow_count={self.workflow_count} but {len(self.workflows)} "
                "workflows are embedded"
            )
        paths = [workflow.source_path for workflow in self.workflows]
        if len(set(paths)) != len(paths):
            raise ValueError(
                "embedded workflows must have unique source paths; duplicated: "
                f"{sorted({path for path in paths if paths.count(path) > 1})}"
            )
        if not str(contract.get("goal", "")).strip():
            raise ValueError("reproducibility_contract.goal must not be blank")
        return self

    # -- derived views --------------------------------------------------------

    @property
    def is_migration_input(self) -> bool:
        """Always true: this object is a re-run input, never evidence."""
        return True

    def workflow_by_source_path(self) -> dict[str, SearchEvidenceMigrationWorkflow]:
        """Return the ``source_path -> workflow`` mapping a re-run walks."""
        return {workflow.source_path: workflow for workflow in self.workflows}


def _read_document(path: Path) -> tuple[dict[str, Any], str]:
    """Return the parsed document and the SHA-256 of its bytes."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not a JSON document: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(document), digest


def import_search_evidence_document(path: str | Path) -> SearchEvidenceMigrationInput:
    """Read and validate one historical ``qphase.search-evidence/1`` document.

    The returned object is a *migration input*: the workflows a Phase 10G re-run
    executes plus the contract that says what a reproduction must achieve. It is
    not a search-audit fragment and it cannot become an evidence product, because
    the document carries no counters to build one from.

    Raises
    ------
    ValueError
        If the schema marker is not ``qphase.search-evidence/1``, the
        reproducibility flags are not the historical ones (semantics embedded,
        no result payloads, no legacy compatibility claim), ``workflow_count``
        disagrees with the embedded workflows, or a source workflow is missing a
        field a re-run needs.

    """
    target = Path(path)
    if not target.exists():
        raise ValueError(f"search-evidence document not found: {target}")
    document, digest = _read_document(target)
    schema = document.get("schema")
    if schema != MIGRATION_DOCUMENT_SCHEMA:
        raise ValueError(
            f"{target} declares schema {schema!r}; expected "
            f"{MIGRATION_DOCUMENT_SCHEMA!r}"
        )
    missing = [
        key
        for key in (
            "model",
            "status",
            "reproducibility_contract",
            "workflow_count",
            "workflows",
        )
        if key not in document
    ]
    if missing:
        raise ValueError(f"{target} is missing required fields: {missing}")
    conclusion = document.get("conclusion_report")
    if conclusion is not None and not isinstance(conclusion, str):
        raise ValueError(
            f"{target}: conclusion_report must be a string or null, got "
            f"{type(conclusion).__name__}"
        )
    return SearchEvidenceMigrationInput(
        source_path=str(target),
        document_sha256=digest,
        model=str(document["model"]),
        status=str(document["status"]),
        conclusion_report=conclusion,
        reproducibility_contract=dict(document["reproducibility_contract"]),
        workflow_count=int(document["workflow_count"]),
        workflows=tuple(
            SearchEvidenceMigrationWorkflow.model_validate(entry)
            for entry in document["workflows"]
        ),
        workspace_head_at_extraction=str(
            document.get("workspace_head_at_extraction", "")
        ),
        generated_at_gate=str(document.get("generated_at_gate", "")),
    )
