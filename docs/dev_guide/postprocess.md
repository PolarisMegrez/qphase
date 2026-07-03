---
description: Postprocess Architecture
---

# Postprocess Architecture

SDE postprocessing currently belongs to `qphase_sde`, not to core. It depends on SDE result metadata, PSD analyser payloads, distribution payloads, and scan parameters, so core should only provide scheduling and registry mechanics.

## Current Compatibility Surface

The legacy import path remains available:

```python
from qphase_sde.postprocess import postprocess_run, export_postprocess_bundle
```

The core CLI command `qphase postprocess` also remains as a compatibility facade. It imports `qphase_sde.postprocess` at runtime and preserves the existing CSV outputs.

## Decomposed Categories

New code should prefer the category modules:

| Category | Module | Responsibility |
| :--- | :--- | :--- |
| Aggregator | `qphase_sde.aggregators.scan_psd` | Load saved SDE `.npz` files and extract aligned PSD traces. |
| Fitter | `qphase_sde.fitters.lorentzian` | Fit Lorentzian peak parameters from one PSD trace. |
| Exporter | `qphase_sde.exporters.csv_bundle` | Write `fit_results.csv`, `psd_merged.csv`, and optional distribution bundles. |

These modules are intentionally small wrappers around the proven behavior. The old facade can continue to orchestrate them while callers migrate to explicit categories.

## Scheduler Workflow Engine

`qphase_sde` now exposes a package-level workflow engine:

```text
engine.sde_postprocess = qphase_sde.workflows.postprocess.engine:SDEPostprocessEngine
```

The engine wraps the existing postprocess use case so postprocessing can be represented as a scheduler job. The first implementation operates on saved run directories or `.npz` files via `run_dir`; future versions can add richer in-memory aggregation from upstream scheduler inputs.

Example job shape:

```yaml
name: postprocess
engine:
  sde_postprocess:
    run_dir: runs/example_session
    scan_param: epsilon
    mode: 0
    overwrite: true
```

## Boundaries

Single-result analysis still belongs to analyser plugins. Cross-result merging belongs to aggregators. Curve fitting belongs to fitters. File layout and export formats belong to exporters. The workflow engine composes those pieces for one user-facing postprocess job.
