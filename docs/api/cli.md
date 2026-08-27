---
description: QPhase 2 command-line interface
---

# CLI Reference

The CLI is QPhase's primary and complete automation interface. Commands that
need a Project accept the global selector:

```bash
qphase --project <project-root-or-qphase.toml> <command>
```

Without it, QPhase uses `QPHASE_PROJECT` and then searches upward for
`qphase.toml`.

## Project

```bash
qphase project init [PATH] [--name TEXT] [--force]
qphase project show
qphase project tag [--add TAG] [--remove TAG] [--private]
qphase project alias [TEXT] [--clear]
qphase project note [TEXT] [--clear]
qphase project reindex
qphase project migrate --dry-run
```

`init` writes `qphase.toml`, creates Workflow, plugin, and Session directories,
and generates Project plugin defaults. `show` prints the resolved Project ID,
root, Workflow root, Session root, and defaults path. `tag`/`alias`/`note`
annotate the Project itself. `reindex` rebuilds the object catalog read model
and lists location issues. `migrate --dry-run` previews the Phase 4 history
migration without writing anything.

## Workflow Catalog

```bash
qphase workflow list [--collection NAME] [--tag TAG] [--query TEXT] [--json]
qphase workflow show WORKFLOW_ID
qphase workflow path WORKFLOW_ID
qphase workflow tag WORKFLOW_ID@REVISION [--add TAG] [--remove TAG] [--private]
```

The catalog scans the Project's Workflow root recursively and can filter by
Collection, Tag, or ID/title/path text. Stable Workflow IDs, not filenames, are
the normal invocation contract. Duplicate IDs are errors. `workflow tag`
annotates one content revision of a workflow.

## Catalog & Tags

Sessions, artifacts, occurrences, jobs, and executions are listed and
annotated through the object catalog. Every `list` command accepts the same
filters: `--tag`, `--tag-any`, `--tag-without`, `--tag-descendant`,
`--tag-namespace`, `--facet k=v`, `--range k=low..high`, `--lifecycle`,
`--retention`, `--direct`, `--limit`, and `--offset`.

```bash
qphase session list|tag|lifecycle|retention ...
qphase artifact list|tag|lifecycle|retention ...
qphase occurrence list [--session ID] [--artifact ID]
qphase occurrence tag SESSION_ID ARTIFACT_ID [--add TAG] [--job NAME] [--private]
qphase occurrence retention SESSION_ID ARTIFACT_ID [VALUE] [--clear] [--job NAME]
qphase job list ...
qphase job tag WORKFLOW_ID@REVISION:JOB_NAME [--add TAG] [--remove TAG] [--private]
qphase execution tag EXECUTION_ID [--add TAG] [--remove TAG] [--private]
```

Tag commands write shared annotation documents by default; `--private` keeps
the change in the user-private store. `execution tag` edits the frozen
submission tags, which is only allowed while the execution is queued.

```bash
qphase tag policy show
qphase tag policy validate
qphase tag promote KIND OBJECT_ID TAG
qphase view save NAME --kind KIND [--tag TAG] [--lifecycle X] [--retention Y]
qphase view list
qphase view delete NAME
```

`tag promote` moves a private tag into the shared annotation layer. Saved
views are user-private named catalog filters. See the
[Tags & Catalog](../user_guide/tags.md) guide for the object model, tag
policy, inheritance, and shadowing rules.

## Execute

```bash
qphase run WORKFLOW [OPTIONS]
```

`WORKFLOW` is a stable ID or a YAML path relative to the Workflow root.

| Option | Purpose |
| --- | --- |
| `--plan` | Validate and display logical Jobs, scan summaries, and dependency edges without creating a Session. |
| `--dry-run` | Alias of planning behavior. |
| `--resume-from PATH` | Resume a compatible interrupted Session. |
| `--verbose`, `-v` | Show detailed terminal diagnostics. |
| `--log-file PATH` | Also write complete logs to an explicit path. |
| `--log-json` | Use JSON file logs. |
| `--suppress-warnings` | Suppress captured warnings. |
| `--json` | Emit machine-readable plan or completion output. |

A successful run creates one Session under the Project Session root and prints
its path. Normal CLI output remains concise; complete diagnostics belong to the
Session log.

## Plugins

```bash
qphase list [--category NAME] [--tree] [--parent PATH]
qphase show model.vdp_2mode [backend.numpy ...]
qphase template model.vdp_2mode [OPTIONS]
```

`list --tree` includes declared subplugin classes and implementations. Local
plugins are loaded from the current Project's manifest paths.

## Configuration

```bash
qphase config show [--system]
qphase config set KEY VALUE [--system]
qphase config reset [--system] [--force]
qphase config schema PLUGIN_PATH
qphase config options PARENT/SLOT
```

Without `--system`, `show`, `set`, and `reset` operate on Project plugin
defaults (`configs/defaults.yaml` in the standard layout). With `--system`, they
operate on sparse user machine policy under `~/.qphase/config.yaml`.

## GUI

```bash
qphase gui [--host 127.0.0.1] [--port 8000] [--reload]
```

Only loopback hosts are accepted because the local API has no remote
authentication. Use an SSH tunnel on a server. The GUI uses the same Project,
Workflow, Execution, and Session services as the CLI.
