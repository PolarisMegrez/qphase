---
description: Postprocess Architecture
---

# Postprocess Architecture

Postprocessing is not a separate core command or a separate resource package. It is expressed as a normal scheduler job that uses the SDE engine's `mode: analyze` together with the `analyser.lorentz_fitter` plugin.

## Design Principle

- **Core** (`qphase`) provides generic aggregation and export utilities in `qphase.core.aggregation`.
- **`qphase_sde`** provides the SDE-specific cross-job analyzer `lorentz_fitter`, which fits Lorentzian peaks to aggregated PSD data and writes merged outputs.
- The **`qphase postprocess` CLI command has been removed**. Use `qphase run <workflow.yaml>` instead.

## Workflow Example

```yaml
- name: sim
  save: true
  engine:
    sde:
      t1: 1.0
      dt: 0.01
      n_traj: 2
  model:
    kerr_2mode:
      omega_a: [0.9, 1.1]
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd:
      modes: [0]
      kind: complex

- name: fit
  input: sim
  aggregate_input:
    on: params.omega_a
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

The scheduler will:

1. Expand the `sim` job into one job per value of `omega_a`.
2. Aggregate the expanded results into a single input for the `fit` job.
3. Run `analyser.lorentz_fitter` in `analyze` mode, producing `fit_results.csv` and `psd_merged.csv` in the `fit` job's run directory.

## Output Files

| File | Produced by | Content |
| :--- | :--- | :--- |
| `fit_results.csv` | `lorentz_fitter` | One row per scan value with fitted parameters and covariance-derived standard deviations. |
| `psd_merged.csv` | `lorentz_fitter` | Frequency axis plus PSD and optional PSD SEM columns per scan value. |
| `dist_merged.npz` | `lorentz_fitter` (optional) | Aggregated distribution payloads. |
| `pdist_merged.pkl` | `lorentz_fitter` (optional) | Aggregated polar distribution payloads. |

The NPZ/PKL bundles include `__schema_version__` and `__created_by__` metadata via `qphase.core.aggregation`.

## Boundaries

- Single-result analysis (per-job PSD, peak finding, distributions) belongs to `analyser` plugins.
- Cross-result aggregation, sorting, and schema-versioned exporting belongs to `qphase.core.aggregation`.
- SDE-specific curve fitting and payload extraction belongs to `qphase_sde.analyser.lorentz_fitter`.
