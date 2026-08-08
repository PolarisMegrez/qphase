---
description: Postprocess Architecture
---

# Postprocess Architecture

Postprocessing is not a separate core command or a separate resource package. It is expressed as a normal scheduler job that uses the SDE engine's `mode: analyze` together with the `analyser.lorentz_fitter` plugin.

## Design Principle

- **Core** (`qphase`) provides generic aggregation and export utilities in `qphase.core.aggregation`.
- **`qphase_sde`** provides SDE-specific cross-job analyzers: `lorentz_fitter`
  for PSD peaks and `allan_scaling` for white-FM windows and perturbation
  scaling.
- The **`qphase postprocess` CLI command has been removed**. Use `qphase run <workflow.yaml>` instead.

## Workflow Example

```yaml
- name: sim
  save: true
  scan:
    axes:
      omega_a:
        target: model.kerr_2mode.omega_a
        values: [0.9, 1.1]
  engine:
    sde:
      t1: 1.0
      dt: 0.01
      n_traj: 2
  model:
    kerr_2mode:
      omega_a: 0.9
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
  input:
    from: sim
    mode: dataset
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

The scheduler will:

1. Compile the `sim` scan and let the SDE engine execute it as one logical dataset.
2. Pass that complete dataset once to the `fit` job.
3. Run `analyser.lorentz_fitter` in `analyze` mode, producing `fit_results.csv` and `psd_merged.csv` in the `fit` job's run directory.

## Output Files

| File | Produced by | Content |
| :--- | :--- | :--- |
| `fit_results.csv` | `lorentz_fitter` | One row per scan value with fitted parameters and covariance-derived standard deviations. |
| `psd_merged.csv` | `lorentz_fitter` | Frequency axis plus PSD and optional PSD SEM columns per scan value. |
| `dist_merged.npz` | `lorentz_fitter` (optional) | Aggregated distribution payloads. |
| `pdist_merged.pkl` | `lorentz_fitter` (optional) | Aggregated polar distribution payloads. |
| `allan_points.csv` | `allan_scaling` | Per-point white-FM windows, Allan intensity, uncertainty, frequency, and gates. |
| `allan_scaling.json` | `allan_scaling` | Common tau/epsilon window, bootstrap power-law fits, and normal-form checks. |

The NPZ/PKL bundles include `__schema_version__` and `__created_by__` metadata via `qphase.core.aggregation`.

## Boundaries

- Single-result analysis (per-job PSD, peak finding, distributions) belongs to `analyser` plugins.
- Dataset views, generic aggregation, and schema-versioned exporting belong to core.
- SDE-specific curve fitting and payload extraction belongs to downstream
  `qphase_sde` analyser plugins such as `lorentz_fitter` and `allan_scaling`.
