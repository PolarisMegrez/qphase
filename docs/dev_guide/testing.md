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
| `dev-cpu` | `-m "not gpu"` with the `dev-cpu` group (no CuPy) | 3.13 | 30 min |
| `lint` | `ruff check .` and `mypy packages` | 3.13 | 15 min |

The `dev-cpu` job exists to exercise the "CuPy is not installed" branch: it
installs the CPU-only dependency group and runs the suite without the GPU
packages, so that every `gpu` case is skipped through the device probe rather
than through an import error.
