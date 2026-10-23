---
description: Test Layers & CI Gates
---

# Test Layers & CI Gates

The suite is partitioned by five pytest markers, declared in `pyproject.toml`
and enforced with `--strict-markers`. The goal is that every test states the
layer it belongs to, so a fast gate and a full sweep select from the same tree.

## Markers

| Marker | Meaning |
| :-- | :-- |
| `unit` | Fast, isolated tests: one component, no scheduler run. |
| `integration` | Cross-component / cross-plugin behaviour: scheduler, engines, analyser graphs. |
| `e2e` | Full CLI / workflow runs, including persisted Sessions and Job directories. |
| `gpu` | Requires a CUDA device (CuPy). Auto-skipped when no device is available. |
| `slow` | Long-running cases; deselect them for a fast gate. |

Two rules keep the grid consistent:

- Tests without any layer marker are marked `unit` automatically
  (`tests/conftest.py::pytest_collection_modifyitems`). Explicitly marking a
  file `integration`/`e2e` therefore *removes* it from the fast gate.
- `gpu` and `slow` are orthogonal to the layer. A GPU case inside an
  integration file carries `@pytest.mark.gpu` per test, not per file, and a
  spawn-worker case adds `slow` on top.

## CUDA gating

`tests/conftest.py::cuda_available()` is the single device probe of the suite:
it returns whether CuPy can reach at least one CUDA device. It is used twice:

- the `gpu` marker gate skips every case whose collected keywords contain
  `gpu` when no device exists;
- per-test `@pytest.mark.skipif(not cuda_available(), ...)` guards cases that
  need a device even when the marker is missing.

Never add a private device probe to a test module: import `cuda_available`
from `tests.conftest` instead.

## Local selection

```bash
pytest -m "unit and not gpu"                          # fast gate
pytest -m "integration and not gpu and not slow"      # integration layer
pytest -m "e2e and not gpu"                           # CLI / workflow runs
pytest -m "slow and not gpu"                          # long regression sweep
pytest -m "not gpu"                                   # everything CPU-runnable
pytest                                                # the whole tree
```

## CI jobs

`.github/workflows/ci.yml` runs one job per layer:

| Job | Selection | Interpreter | Timeout |
| :-- | :-- | :-- | :-- |
| `test` | `-m "unit and not gpu"` | 3.11 / 3.12 / 3.13 / 3.14 | 20 min |
| `integration` | `-m "integration and not gpu and not slow"` | 3.13 | 45 min |
| `e2e` | `-m "e2e and not gpu"` (push only, blocking) | 3.13 | 30 min |
| `slow` | `-m "slow and not gpu"` (push only) | 3.13 | 60 min |
| `lint` | `ruff check .` and `mypy packages` | 3.13 | 15 min |

Every job installs the CPU-only dependency group
(`uv sync --no-dev --group dev-cpu`), which carries pytest, ruff, mypy and the
CLI/GUI/resource packages but no CuPy. That keeps the whole matrix installable
on interpreters without a CuPy wheel — and it means the suite always runs the
"CuPy is not installed" branch, so every `gpu` case skips through the device
probe rather than through an import error.

`mypy` currently type-checks `packages` only. Tests are linted by `ruff` but
not type-checked: `mypy packages tests` still reports a registered backlog of
74 errors, so the test-side type gate is suspended.

## GPU CI lane

`.github/workflows/gpu.yml` is the only workflow that actually executes `gpu`
cases; it targets a self-hosted CUDA runner rather than a GitHub-hosted one.
`-m gpu` currently collects 47 parametrized cases across 12 files.

- Runner: a self-hosted runner registered with the labels `self-hosted` and
  `gpu`, on a machine with the NVIDIA driver. CuPy 14+ ships the required CUDA
  components as PyPI wheels (`cupy-cuda12x[ctk]`), so no system CUDA toolkit is
  needed.
- Trigger: `workflow_dispatch` only, for now. With no matching runner, a
  push-triggered job queues for roughly 24 hours before failing, so the `push`
  trigger is added back only after the first manual dispatch passes.
- Install: plain `uv sync` — the default `dev` group, which includes
  `qphase[cupy]`. This lane must never use `dev-cpu`: without CuPy every case
  would skip and the lane would look green while testing nothing.
- Guard: a dedicated step fails the job when CuPy cannot be imported or
  `cupy.cuda.runtime.getDeviceCount()` is 0, so the lane cannot pass by
  skipping everything.
- Matrix: 3.11–3.13. 3.14 joins once `uv.lock` moves from `cupy-cuda12x` 13.6.0
  (cp311–cp313 wheels only) to CuPy 14 (cp314 wheels).
