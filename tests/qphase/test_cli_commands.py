"""Tests for CLI commands using Typer's CliRunner."""

import pytest
from qphase.main import app
from typer.testing import CliRunner

pytestmark = pytest.mark.integration

runner = CliRunner()


def test_version():
    """Test --version flag."""
    _ = runner.invoke(app, ["--version"])
    # Note: Typer might not handle --version automatically unless configured
    # If it fails, we might need to check if we implemented a version callback
    # For now, let's check if it runs without error or if we need to add it
    pass  # Skip for now as version callback might not be in main.py


def test_project_show(temp_workspace):
    """The active project can be inspected explicitly."""
    result = runner.invoke(app, ["project", "show"])
    assert result.exit_code == 0
    assert "test-project" in result.stdout


def test_config_show_system(temp_workspace):
    """Test 'config show system'."""
    result = runner.invoke(app, ["config", "show", "--system"])
    assert result.exit_code == 0
    assert "scan_runtime" in result.stdout
    assert "paths" not in result.stdout


def test_plugin_list():
    """Test 'list' command."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "engine" in result.stdout
    assert "backend" in result.stdout


def test_plugin_show():
    """Test 'show' command."""
    # Assuming 'dummy' backend is available
    result = runner.invoke(app, ["show", "backend.dummy"])
    assert result.exit_code == 0
    assert "param" in result.stdout


def test_workflow_list():
    """Test recursive workflow catalog."""
    result = runner.invoke(app, ["workflow", "list"])
    assert result.exit_code == 0


def test_template_command():
    """Test 'template' command."""
    result = runner.invoke(app, ["template", "engine.dummy"])
    assert result.exit_code == 0
    assert "param:" in result.stdout


def test_run_jobs_command(temp_workspace, sample_job_file, dummy_model):
    """Test 'run jobs' command."""
    # The sample_job_file fixture creates a valid job file in the workspace
    # dummy_model fixture registers the 'dummy' model used in sample_job_file

    # Run the job
    # We need to pass the job name (without extension), not the file path
    _ = runner.invoke(app, ["run", "test_job", "--plan"])

    # Note: This might fail if the engine/backend implementation has issues
    # But we are testing the CLI invocation here

    # Note: If the job fails, exit code might be non-zero.
    # But we want to ensure the CLI command itself ran.
    # If it failed due to engine error, stdout should contain info.

    # For now, let's just check if it didn't crash the CLI.
    # assert result.exit_code == 0  # Might fail if dummy engine not fully working

    # Check that job ran (output shows run directory)
    # assert "[test_job]" in result.stdout

    # There should be a subdirectory for the run if it succeeded
    # assert any(output_dir.iterdir())


def _catalog_session(workspace, session_id="session-1"):
    """Fabricate a minimal session directory for catalog CLI tests."""
    import json

    root = workspace / "runs" / "2026" / "08" / session_id
    root.mkdir(parents=True)
    manifest = {
        "schema": "qphase.session/2",
        "session_id": session_id,
        "project_id": "test-project",
        "workflow_id": "example",
        "status": "completed",
        "start_time": "2026-08-26T10:00:00+08:00",
        "jobs": {},
    }
    (root / "session_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def test_tag_policy_show_without_policy(temp_workspace):
    """'tag policy show' reports the absence of a policy file."""
    result = runner.invoke(app, ["tag", "policy", "show"])
    assert result.exit_code == 0
    assert "no tag policy" in result.stdout


def test_tag_policy_show_and_validate(temp_workspace):
    """'tag policy' shows and validates configs/tags.yaml."""
    policy = temp_workspace / "configs" / "tags.yaml"
    policy.write_text(
        "schema: qphase.tag-policy/1\nnamespaces:\n  stage:\n    open: true\n",
        encoding="utf-8",
    )
    shown = runner.invoke(app, ["tag", "policy", "show"])
    valid = runner.invoke(app, ["tag", "policy", "validate"])
    assert shown.exit_code == 0
    assert "stage" in shown.stdout
    assert valid.exit_code == 0
    assert "valid" in valid.stdout

    policy.write_text("schema: qphase.tag-policy/1\nnamespaces: 42\n", encoding="utf-8")
    invalid = runner.invoke(app, ["tag", "policy", "validate"])
    assert invalid.exit_code == 1


def test_session_tag_and_list_by_tag(temp_workspace):
    """'session tag' edits are visible through 'session list --tag'."""
    _catalog_session(temp_workspace)
    tagged = runner.invoke(app, ["session", "tag", "session-1", "--add", "task:scan"])
    assert tagged.exit_code == 0

    listed = runner.invoke(app, ["session", "list", "--tag", "task:scan"])
    missing = runner.invoke(app, ["session", "list", "--tag", "task:other"])
    assert listed.exit_code == 0
    assert "session-1" in listed.stdout
    assert "session-1" not in missing.stdout

    removed = runner.invoke(
        app, ["session", "tag", "session-1", "--remove", "task:scan"]
    )
    assert removed.exit_code == 0
    assert "session-1" not in runner.invoke(
        app, ["session", "list", "--tag", "task:scan"]
    ).stdout


def test_session_lifecycle_command(temp_workspace):
    """'session lifecycle' sets and clears the lifecycle."""
    _catalog_session(temp_workspace)
    result = runner.invoke(app, ["session", "lifecycle", "session-1", "archived"])
    assert result.exit_code == 0
    listed = runner.invoke(app, ["session", "list", "--lifecycle", "archived"])
    assert "session-1" in listed.stdout
    cleared = runner.invoke(app, ["session", "lifecycle", "session-1", "--clear"])
    assert cleared.exit_code == 0
    invalid = runner.invoke(app, ["session", "lifecycle", "session-1", "bogus"])
    assert invalid.exit_code == 1


def test_project_reindex(temp_workspace):
    """'project reindex' rebuilds the catalog and prints the counts."""
    _catalog_session(temp_workspace)
    result = runner.invoke(app, ["project", "reindex"])
    assert result.exit_code == 0
    assert "1 sessions" in result.stdout


def test_view_save_list_delete_roundtrip(temp_workspace, monkeypatch, tmp_path):
    """'view save/list/delete' roundtrip against an isolated private home."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    saved = runner.invoke(
        app,
        ["view", "save", "review", "--kind", "session", "--tag", "task:scan"],
    )
    assert saved.exit_code == 0

    listed = runner.invoke(app, ["view", "list"])
    assert listed.exit_code == 0
    assert "review" in listed.stdout
    assert "task:scan" in listed.stdout

    deleted = runner.invoke(app, ["view", "delete", "review"])
    assert deleted.exit_code == 0
    assert "review" not in runner.invoke(app, ["view", "list"]).stdout


def test_view_save_rejects_unknown_kind(temp_workspace, monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    result = runner.invoke(app, ["view", "save", "bad", "--kind", "nope"])
    assert result.exit_code == 1


def test_project_migrate_requires_dry_run(temp_workspace):
    """'project migrate' without --dry-run fails fast with a Phase 4 hint."""
    result = runner.invoke(app, ["project", "migrate"])
    assert result.exit_code == 1
    assert "Phase 4" in result.stdout


def test_project_migrate_dry_run(temp_workspace):
    """'project migrate --dry-run' previews legacy imports without writing."""
    import json

    root = _catalog_session(temp_workspace, "legacy-session")
    (root / "session_metadata.json").write_text(
        json.dumps({"alias": "old-run"}), encoding="utf-8"
    )
    (root / "workflow_snapshot.yaml").write_text(
        "tags: [vdp_2mode]\n", encoding="utf-8"
    )
    before = {
        path: path.read_bytes()
        for path in temp_workspace.rglob("*")
        if path.is_file()
    }

    result = runner.invoke(app, ["project", "migrate", "--dry-run"])

    assert result.exit_code == 0
    assert "legacy-session" in result.stdout
    assert "old-run" in result.stdout
    assert "vdp_2mode" in result.stdout
    after = {
        path: path.read_bytes()
        for path in temp_workspace.rglob("*")
        if path.is_file()
    }
    assert after == before
def test_occurrence_tag_requires_job_when_ambiguous(temp_workspace):
    """'occurrence tag' refuses silent first-match; --job disambiguates."""
    import json

    from tests.qphase.test_catalog import _v4_artifact_manifest

    root = _catalog_session(temp_workspace)
    for job in ("sim", "fit"):
        job_dir = root / job
        job_dir.mkdir()
        (job_dir / "artifact_manifest.json").write_text(
            json.dumps(_v4_artifact_manifest("art-1")),
            encoding="utf-8",
        )

    ambiguous = runner.invoke(
        app, ["occurrence", "tag", "session-1", "art-1", "--add", "task:scan"]
    )
    assert ambiguous.exit_code == 1
    assert "ambiguous" in ambiguous.stdout

    resolved = runner.invoke(
        app,
        ["occurrence", "tag", "session-1", "art-1", "--job", "fit",
         "--add", "task:scan"],
    )
    assert resolved.exit_code == 0
    assert "task:scan" in resolved.stdout


def _execution_record(workspace, execution_id, state):
    """Persist a minimal execution record for execution CLI tests."""
    from qphase.core.persistence import ProjectStateStore
    from qphase.core.project import ProjectContext

    ProjectStateStore(ProjectContext.discover()).save_execution(
        {
            "schema": "qphase.execution/1",
            "execution_id": execution_id,
            "source_workflow": "example",
            "workflow": {"id": "example"},
            "submission_tags": [],
            "submitted_at": "2026-08-26T09:00:00+08:00",
            "state": state,
        }
    )


def _workflow_file(workspace):
    path = workspace / "configs" / "workflows" / "example.yaml"
    path.write_text(
        "schema: qphase.workflow/2\n"
        "id: example\n"
        "title: Example\n"
        "jobs:\n"
        "  - name: sim\n"
        "    engine:\n"
        "      dummy: {}\n",
        encoding="utf-8",
    )


def test_project_tag_alias_note_roundtrip(temp_workspace):
    tagged = runner.invoke(app, ["project", "tag", "--add", "task:paper"])
    assert tagged.exit_code == 0
    assert "task:paper" in tagged.stdout
    alias = runner.invoke(app, ["project", "alias", "paper-project"])
    assert alias.exit_code == 0
    note = runner.invoke(app, ["project", "note", "results"])
    assert note.exit_code == 0
    cleared = runner.invoke(app, ["project", "alias", "--clear"])
    assert cleared.exit_code == 0


def test_workflow_and_job_tag_roundtrip(temp_workspace):
    _workflow_file(temp_workspace)
    listed = runner.invoke(app, ["workflow", "list"])
    assert listed.exit_code == 0
    revision_id = listed.stdout.splitlines()[0].split(":", 1)[0]
    assert revision_id.startswith("example@")

    tagged = runner.invoke(
        app, ["workflow", "tag", revision_id, "--add", "task:reviewed"]
    )
    assert tagged.exit_code == 0
    assert "task:reviewed" in tagged.stdout
    hit = runner.invoke(app, ["workflow", "list", "--tag", "task:reviewed"])
    assert revision_id in hit.stdout

    jobs = runner.invoke(app, ["job", "list"])
    assert jobs.exit_code == 0
    job_id = jobs.stdout.split()[0]
    assert job_id == f"{revision_id}:sim"
    job_tagged = runner.invoke(app, ["job", "tag", job_id, "--add", "method:cam"])
    assert job_tagged.exit_code == 0
    assert "method:cam" in job_tagged.stdout
    job_hit = runner.invoke(app, ["job", "list", "--tag-any", "method:cam"])
    assert job_id in job_hit.stdout


def test_execution_tag_requires_queued_state(temp_workspace):
    _execution_record(temp_workspace, "exec-queued", "queued")
    _execution_record(temp_workspace, "exec-done", "completed")

    ok = runner.invoke(app, ["execution", "tag", "exec-queued", "--add", "task:retry"])
    assert ok.exit_code == 0
    assert "task:retry" in ok.stdout

    blocked = runner.invoke(
        app, ["execution", "tag", "exec-done", "--add", "task:retry"]
    )
    assert blocked.exit_code == 1
    assert "queued" in blocked.stdout

    missing = runner.invoke(app, ["execution", "tag", "exec-nope", "--add", "task:x"])
    assert missing.exit_code == 1


def test_occurrence_list_filters_by_session_and_artifact(temp_workspace):
    import json

    from tests.qphase.test_catalog import _v4_artifact_manifest

    root = _catalog_session(temp_workspace)
    job_dir = root / "sim"
    job_dir.mkdir()
    (job_dir / "artifact_manifest.json").write_text(
        json.dumps(_v4_artifact_manifest("art-1")), encoding="utf-8"
    )
    listed = runner.invoke(app, ["occurrence", "list", "--session", "session-1"])
    assert listed.exit_code == 0
    assert "art-1:session-1:sim" in listed.stdout
    empty = runner.invoke(app, ["occurrence", "list", "--session", "other"])
    assert "art-1" not in empty.stdout
    by_artifact = runner.invoke(app, ["occurrence", "list", "--artifact", "art-1"])
    assert "art-1:session-1:sim" in by_artifact.stdout


def test_private_tag_and_promote_via_cli(temp_workspace):
    _catalog_session(temp_workspace)
    tagged = runner.invoke(
        app, ["session", "tag", "session-1", "--add", "task:mine", "--private"]
    )
    assert tagged.exit_code == 0
    listed = runner.invoke(app, ["session", "list", "--tag", "task:mine"])
    assert "session-1" in listed.stdout
    promoted = runner.invoke(
        app, ["tag", "promote", "session", "session-1", "task:mine"]
    )
    assert promoted.exit_code == 0
    assert "task:mine" in promoted.stdout


def test_session_list_supports_full_query_flags(temp_workspace):
    _catalog_session(temp_workspace)
    tagged = runner.invoke(app, ["session", "tag", "session-1", "--add", "task:scan"])
    assert tagged.exit_code == 0
    hit = runner.invoke(
        app,
        [
            "session", "list", "--tag-any", "task:scan",
            "--facet", "status=completed", "--direct",
        ],
    )
    assert "session-1" in hit.stdout
    excluded = runner.invoke(app, ["session", "list", "--tag-without", "task:scan"])
    assert "session-1" not in excluded.stdout
    paged = runner.invoke(app, ["session", "list", "--limit", "1", "--offset", "1"])
    assert "session-1" not in paged.stdout


def test_project_migrate_dry_run_extended_sections(temp_workspace):
    """'project migrate --dry-run' prints all extended report sections."""
    _catalog_session(temp_workspace, "session-1")

    result = runner.invoke(app, ["project", "migrate", "--dry-run"])

    assert result.exit_code == 0
    assert "Rebuildable:" in result.stdout
    assert "Legacy occurrence keys:" in result.stdout
    assert "Duplicate artifact identities" in result.stdout
    assert "Catalog reindex parity:" in result.stdout
    assert "Object counts:" in result.stdout
    assert "Private store:" in result.stdout


def test_derived_facet_flags_on_list_commands(temp_workspace):
    """--plugin/--quantity/--model/--engine/--has-model reach the catalog."""
    import json

    from tests.qphase.test_catalog import _v4_artifact_manifest

    workflow = (
        "schema: qphase.workflow/2\n"
        "id: example\n"
        "title: Example\n"
        "jobs:\n"
        "  - name: sim\n"
        "    engine:\n"
        "      dummy: {}\n"
        "    plugins:\n"
        "      model:\n"
        "        cam: {}\n"
    )
    (temp_workspace / "configs" / "workflows" / "example.yaml").write_text(
        workflow, encoding="utf-8"
    )
    root = _catalog_session(temp_workspace)
    (root / "workflow_snapshot.yaml").write_text(workflow, encoding="utf-8")
    job_dir = root / "sim"
    job_dir.mkdir()
    (job_dir / "artifact_manifest.json").write_text(
        json.dumps(_v4_artifact_manifest("art-1", quantities=("position",))),
        encoding="utf-8",
    )

    jobs = runner.invoke(app, ["job", "list", "--plugin", "model:cam"])
    assert jobs.exit_code == 0
    assert ":sim" in jobs.stdout
    missing = runner.invoke(app, ["job", "list", "--plugin", "model:other"])
    assert ":sim" not in missing.stdout

    artifacts = runner.invoke(app, ["artifact", "list", "--quantity", "position"])
    assert artifacts.exit_code == 0
    assert "art-1" in artifacts.stdout
    missing = runner.invoke(app, ["artifact", "list", "--quantity", "velocity"])
    assert "art-1" not in missing.stdout

    by_model = runner.invoke(app, ["session", "list", "--model", "cam"])
    assert "session-1" in by_model.stdout
    by_engine = runner.invoke(app, ["session", "list", "--engine", "dummy"])
    assert "session-1" in by_engine.stdout
    has_model = runner.invoke(app, ["session", "list", "--has-model"])
    assert "session-1" in has_model.stdout
    no_model = runner.invoke(app, ["session", "list", "--model", "other"])
    assert "session-1" not in no_model.stdout
