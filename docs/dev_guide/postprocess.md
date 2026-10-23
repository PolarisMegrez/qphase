---
description: Postprocess Architecture
---

# Postprocess Architecture

Postprocessing is not a separate core command or a separate resource package. It is expressed as a normal scheduler job that uses the SDE engine's `mode: analyze` together with the `analyser.lorentz_fitter` plugin.

## Design Principle

- **Core** (`qphase`) provides generic in-memory aggregation containers
  (`qphase.core.aggregation`) and the typed-product persistence layer.
- **`qphase_sde`** provides SDE-specific cross-job analysers: `lorentz_fitter`
  for PSD peaks and `allan_scaling` for white-FM windows and perturbation
  scaling. Their results are persisted as typed products.
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
      measurement:
        complex_field: {channels: [0], representation: complex}
  input_bindings:
    psd:
      signal: input.trajectories

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
3. Run `analyser.lorentz_fitter` in `analyze` mode, persisting its typed product
   in the `fit` Job directory.

## Output Files

Analysers no longer write CSV/NPZ/PKL side files: every result is persisted as
one typed product in the Job directory's artifact manifest. Load them with
`qphase.data.load_products(job_dir)`.

| Product | Produced by | Content |
| :--- | :--- | :--- |
| `lorentz_fitter` | `lorentz_fitter` | One fit row per scan value with fitted parameters and covariance-derived standard deviations, plus the merged PSD/SEM traces. |
| `allan_scaling` | `allan_scaling` | Per-point white-FM windows, Allan intensity, uncertainty, frequency and gates, plus the common tau/epsilon window and bootstrap power-law fits. |

## Boundaries

- Single-result analysis (per-job PSD, peak finding, distributions) belongs to `analyser` plugins.
- Dataset views, in-memory aggregation containers and typed-product persistence belong to core.
- SDE-specific curve fitting and payload extraction belongs to downstream
  `qphase_sde` analyser plugins such as `lorentz_fitter` and `allan_scaling`.
