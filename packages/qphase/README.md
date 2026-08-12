# qphase

`qphase` is the core runtime and primary CLI for project-oriented scientific
workflows. It discovers plugins, validates versioned Workflow documents,
schedules logical Jobs, records Sessions, and exposes the local service API and
GUI. Scientific algorithms live in resource packages such as `qphase-sde` and
`qphase-cam`.

## Concepts

- A **Project** is declared by `qphase.toml` and owns Workflows, local plugins,
  defaults, and Session storage.
- A **Workflow** is a versioned YAML document with a graph of logical Jobs.
- An **Execution** is one attempt to run a Workflow; its persisted record is a
  **Session**.
- Jobs emit typed **Artifacts**. Parameter scans stay inside one logical Job.

## Quick Start

```bash
qphase project init my-research
qphase --project my-research project show
qphase --project my-research workflow list
qphase --project my-research run <workflow-id> --plan
qphase --project my-research run <workflow-id>
```

Use `qphase config show` for Project plugin defaults and `qphase config show
--system` for machine policy. Project paths are defined only by `qphase.toml`;
`~/.qphase/config.yaml` is Project-independent.

## Optional Dependencies

```bash
pip install qphase[standard]
pip install qphase[gui]
```

## License

MIT License.
