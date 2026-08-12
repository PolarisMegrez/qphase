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
```

`init` writes `qphase.toml`, creates Workflow, plugin, and Session directories,
and generates Project plugin defaults. `show` prints the resolved Project ID,
root, Workflow root, Session root, and defaults path.

## Workflow Catalog

```bash
qphase workflow list [--collection NAME] [--tag TAG] [--query TEXT] [--json]
qphase workflow show WORKFLOW_ID
qphase workflow path WORKFLOW_ID
```

The catalog scans the Project's Workflow root recursively and can filter by
Collection, Tag, or ID/title/path text. Stable Workflow IDs, not filenames, are
the normal invocation contract. Duplicate IDs are errors.

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
