---
description: GUI and Local API
---

# GUI and Local API

QPhase's GUI work starts with a local API backend. The goal is to expose the same service layer used by the CLI so a future browser UI can browse jobs, inspect plugins, preview execution plans, and manage configuration without calling Typer command functions.

## Install GUI Dependencies

The local API is optional. Install the GUI extra when you want to run or develop it:

```bash
pip install "qphase[gui]"
```

For source checkouts, install the standard development environment:

```bash
uv sync --dev
```

## Start the Local API

The quickest way to start the local GUI is:

```bash
qphase gui
```

This launches the FastAPI backend and serves the built-in web console at `http://127.0.0.1:8000`.

For development, the same API is also exposed as a FastAPI app factory:

```bash
uvicorn qphase.gui.api:create_app --factory --reload
```

Use `qphase gui --host 0.0.0.0 --port 8080` when you need a different bind address.

## Web Console

The built-in console is a lightweight browser UI with six views:

| View | Purpose |
| :--- | :--- |
| Jobs | List available jobs, open YAML-derived job data, plan, and run. |
| Config | View and edit global configuration as JSON. |
| Plugins | Browse registered plugin namespaces and schema availability. |
| Plan | Build execution plans for selected jobs. |
| Run | Start a synchronous local run and capture progress events. |
| Results | Inspect session manifests, run events, artifacts, and JSON/text payloads. |

## Available Endpoints

The first backend slice focuses on read and preview workflows:

| Endpoint | Purpose |
| :--- | :--- |
| `GET /health` | Check that the local API is running. |
| `GET /jobs` | List available job names from configured job directories. |
| `GET /jobs/{name}` | Load a job configuration by name. |
| `POST /jobs/validate` | Validate a raw job object against registry schemas. |
| `POST /plans` | Build an execution plan for selected jobs. |
| `POST /runs` | Run selected jobs synchronously and return a run handle plus results. |
| `GET /plugins` | List plugin catalog entries, optionally filtered by namespace. |
| `GET /plugins/{namespace}/{name}/schema` | Return a plugin JSON schema. |
| `GET /config/global` | Read global configuration. |
| `PUT /config/global` | Save global configuration. |
| `GET /runs/{session_id}` | Read a session manifest. |
| `GET /runs/{session_id}/events` | Read run progress events recorded by this local API process. |
| `GET /runs/{session_id}/artifacts` | List artifacts under a session directory. |
| `GET /artifacts?path=...` | Read a JSON/text artifact payload or return binary metadata. |

Example plan request:

```bash
curl -X POST http://127.0.0.1:8000/plans \
  -H "Content-Type: application/json" \
  -d '{"jobs": ["vdp_sde"]}'
```

Example run request:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"jobs": ["vdp_sde"]}'
```

After a run completes, use the returned `session_id` to inspect events and artifacts:

```bash
curl http://127.0.0.1:8000/runs/SESSION_ID/events
curl http://127.0.0.1:8000/runs/SESSION_ID/artifacts
curl "http://127.0.0.1:8000/artifacts?path=/absolute/path/to/session_manifest.json"
```

## MVP Scope

This MVP is intentionally small. It provides a local web console and API for job browsing, plugin browsing, config editing, plan preview, synchronous local runs, progress events, and result artifact inspection. It does not provide drag-and-drop workflow editing, remote execution, user accounts, or notebook replacement features.
