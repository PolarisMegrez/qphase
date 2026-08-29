"""Execution tag CLI commands.

``qphase execution tag`` edits the *submission tags* frozen on the persisted
execution record (``qphase.execution/1``); they are the submit-time layer and
can only change while the execution is still queued. After-the-fact
organization uses annotations instead (``--private`` here, or the GUI/service
``tag_execution`` shared annotations).
"""

from __future__ import annotations

import typer

from qphase.core.catalog import ProjectObjectCatalog
from qphase.core.errors import QPhaseError
from qphase.core.persistence import ProjectStateStore
from qphase.core.project import ProjectContext
from qphase.core.tags import (
    canonicalize_tag_syntax,
    freeze_tag_rules,
    load_tag_policy,
    validate_declared_tags,
)

from ._annotations import (
    ADD_OPTION,
    PRIVATE_OPTION,
    REMOVE_OPTION,
    catalog_service,
    fail,
)

app = typer.Typer(help="Manage executions")


@app.command("tag")
def tag_execution(
    execution_id: str = typer.Argument(..., help="Execution id"),
    add: list[str] = ADD_OPTION,
    remove: list[str] = REMOVE_OPTION,
    private: bool = PRIVATE_OPTION,
) -> None:
    """Edit an execution's submission tags (queued only) or private tags."""
    if private:
        try:
            tags = catalog_service().tag_execution(
                execution_id, add=add, remove=remove, private=True
            )
        except (QPhaseError, RuntimeError, ValueError) as exc:
            fail(exc)
        typer.echo(
            f"execution {execution_id} tags=[{', '.join(item.tag for item in tags)}]"
        )
        return
    project = ProjectContext.discover()
    store = ProjectStateStore(project)
    try:
        records = store.load_executions()
        record = next(
            (
                payload
                for payload in records
                if payload.get("execution_id") == execution_id
            ),
            None,
        )
        if record is None:
            raise ValueError(f"unknown execution: {execution_id}")
        if record.get("state") != "queued":
            raise ValueError("submission tags can only be updated while queued")
        removed = {canonicalize_tag_syntax(tag) for tag in remove}
        kept = [
            str(tag)
            for tag in record.get("submission_tags", [])
            if str(tag) not in removed
        ]
        policy = load_tag_policy(project)
        record["submission_tags"] = validate_declared_tags(
            [*kept, *add], "execution", policy
        )
        record["tag_policy_revision"] = policy.revision if policy is not None else None
        record["submission_tag_rules"] = freeze_tag_rules(
            policy, record["submission_tags"]
        )
        store.save_execution(record)
        ProjectObjectCatalog(project).reindex()
    except (QPhaseError, RuntimeError, ValueError) as exc:
        fail(exc)
    typer.echo(f"execution {execution_id} submission_tags={record['submission_tags']}")
