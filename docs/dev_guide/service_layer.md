---
description: Service Layer
---

# Service Layer

The service layer is the Python-first application API for QPhase clients. It sits between user interfaces and the core primitives so CLI, GUI, notebooks, and future local APIs can share the same orchestration logic.

```text
client UI -> qphase.service -> qphase.core -> resource packages -> plugins
```

## Design Rules

CLI and GUI are peers. The CLI should parse command-line input and render terminal output; the GUI should manage visual interaction, configuration editing, execution plans, progress, and result browsing. Neither client should wrap the other.

Services return structured objects instead of printing Rich tables or Typer messages. This keeps the same API usable from tests, GUI code, and automation scripts.

The service layer should wrap existing core behavior before introducing new behavior. If a rule already lives in `qphase.core.scheduler`, `qphase.core.config_loader`, or `qphase.core.registry`, the service should delegate to it or extract a shared helper rather than copying the rule.

## Current Facades

`ConfigService` loads system/global/job configuration, normalizes raw job dictionaries, previews merged configuration, and validates job plugin blocks against the registry.

`RegistryService` discovers plugins, returns a catalog, exposes plugin JSON schemas, validates plugin configuration, reports scanable parameters, and reads engine manifests.

`SchedulerService` lists and loads jobs, builds a structured execution plan, runs jobs through the core scheduler, exposes dry-run planning, reads session manifests, and summarizes artifacts.

## Structured Models

Service return values live in `qphase.service.models`. The initial model set includes plugin catalog objects, config validation issues, merged config previews, execution plans, execution plan jobs and edges, artifact summaries, and run handles.

These models are intentionally serializable. A GUI or local HTTP API should be able to convert them to JSON without depending on internal scheduler objects or plugin instances.

## Execution Plans

`SchedulerService.build_plan()` is the shared planning surface for dry-run, GUI preview, and future machine-readable CLI output. It validates jobs, expands parameter scans, reports dependency edges, and returns validation issues without creating a real run session.

Each planned job records its original base job, expanded index, engine name, explicit plugin namespaces, required and optional namespaces from the engine manifest, inherited global defaults for required namespaces, optional namespaces explicitly enabled by the job, scan parameter values, input/output fields, and expected output names. Dependency edges distinguish normal input flow, aggregate input flow, explicit `depends_on`, and output references.

Planning should remain side-effect light. Creating run directories, writing manifests, and instantiating engines belong to execution, not preview.

## Configuration Ownership

System configuration owns runtime paths and core behavior. Global configuration owns user or machine defaults such as backend preferences. Job configuration owns the intent of one workflow. Plugin schemas own plugin-specific validation contracts. Engine manifests own required and optional plugin namespaces.

Global defaults may fill required runtime choices, but optional workflow steps should not run only because a global default exists. The engine manifest and explicit job configuration decide which namespaces participate in a job.

## Implementation Guidance

Keep service methods thin until a shared use case appears in more than one client. Prefer small DTOs over returning raw private scheduler state. If a service needs a private core method in multiple places, extract a core helper with tests before expanding the service API further.
