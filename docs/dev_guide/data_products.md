---
description: Data Product Contract
---

# Data Product Contract

!!! warning "Experimental — qphase 2.0"
    The contract described on this page is frozen for the qphase/qphase_sde 2.0 upgrade
    and implemented under `qphase.data` and `qphase.core.task_profile`. Field decisions
    listed here are reviewed before any production Result serialization changes.

QPhase 2.0 replaces the untyped `trajectory + dict[str, Any]` result shape with
**data products**: typed containers whose axes, variables, uncertainties and provenance
are described by a machine-readable schema. The core defines the schema language and
three public data kinds; resource packages define quantities, provenance and reducers,
and must not redefine incompatible dataset base classes.

## Data kinds

`DataKind` has exactly three values:

- `time_series` — sampled traces with axes such as `scan`, `trajectory`, `time`,
  `channel`. Requires `t0`+`dt` or an explicit monotonic time coordinate; declares
  dtype, real/complex domain, channel definitions, independent-realization semantics
  and the frequency orientation convention. May carry valid-length/mask, warm-up,
  trajectory-ID and RNG provenance.
- `spectral` — frequency-domain products with axes such as `scan`,
  `trajectory`/`statistic`, `frequency`, `channel`.
- `statistics` — low-dimensional results: moments, distributions, Allan variance,
  first-passage summaries, fit parameters. May contain structured tables, but arbitrary
  unverifiable objects never enter the persisted schema.

## Product schema

A `ProductSchema` is JSON-serializable, strictly extra-forbid, and has a stable hash.
Shapes may be partially unknown at plan time (`AxisSchema.size is None`) but must be
**closed** before materialization.

- `AxisSchema` — `name`, optional `size`, `coordinate` (`regular` or `explicit`),
  `units`, `monotonic`, and the `independent` role flag marking realization axes
  (scan/trajectory) that uncertainty merging relies on.
- `VariableSchema` — `name`, `dtype` (object dtype is forbidden), `value_domain`
  (`real` or `complex`), named `dims` referencing axes, `quantity`, `units`, and
  `constraints` (for example `nonnegative`, or a tensor `symmetry`/`layout`
  descriptor). The **variable**, not the dataset class, decides real vs. complex:
  spectral products are never split into incompatible real/complex result classes.
- `UncertaintySchema` — `target` (the variable it describes), `kind` (`sample_std`,
  `sem`, `confidence_interval`, `covariance`, `other`), `independent_unit` (which
  realization axis the estimate counts over), `confidence` and `count`. Uncertainties
  of complex variables must declare an explicit covariance representation (`real`,
  `real_imag`, `magnitude_phase`, `custom`) — a bare complex "std" is not allowed.
- Matrix/tensor variables use named dimensions plus a symmetry/layout descriptor;
  moment order is an axis or variable attribute, never a new dataset class.

## Spectral quantities

The frozen minimal `SpectralQuantity` set is `fourier_amplitude`,
`power_spectral_density`, `cross_spectral_density` and `coherence`. Spectral products
carry the common attributes frequency units/orientation, sidedness, normalization,
window, estimator and effective degrees of freedom. PSD variables declare
real/nonnegative; cross-spectrum variables declare complex with a Hermitian layout.
Statistics explicitly distinguish mean, sample std, SEM and independent count instead
of relying on key-name conventions.

## Moment families

Statistics products that group related moments declare a `MomentFamilySchema`:
`moment_kind` (`raw`, `central`, `cumulant`, `factorial`), `ordering` (`c_number`,
`normal`, `symmetric`), `maximum_order`, tensor symmetry/layout and a `family_id`.
Moments of one family are stored as one product sharing independent counts and joint
covariance; they split only when orders come from different populations, estimators or
provenance.

## Runtime handles vs. artifacts

In-process transfer, session caching and persistence are separate layers:

- `DataHandleProtocol` — in-process, possibly device-resident buffer with `schema`,
  `device`, `dtype`, `shape`, `nbytes`, `read_only` and `owner`; supports
  `acquire()`/`release()`, `materialize()` and `export_interface()`.
- `DataLeaseProtocol` — reference-counted lifetime; the buffer may be reclaimed only
  after the last consumer releases. Leases declare their consumer, lifetime scope and
  pin state.
- `ArtifactRef` — durable, cross-process reference: artifact id, product schema,
  loader, hash and provenance only.
- `DataMaterializerProtocol` — resource-registered conversion between runtime handles
  and artifact-backed products.

A `DataProduct` is the semantic layer and may be backed by either a runtime handle or
an artifact reference. `ResultProtocol.save()` is not a cross-job transport, and the
artifact store is not a runtime cache. Handle/lease ownership, read-only rules, device
and stream synchronization, copy policy and failure semantics are defined by core;
resource packages never hand raw device buffers to another job outside a lease.

## Product graphs and task profiles

`ProductRequirement` / `ProductDeclaration` describe typed inputs and outputs;
`ProductGraph` is the validated acyclic graph of `ProductNode`s (identified by
fingerprint) that engines compile from plugin declarations.

`EngineTaskProfile` makes plugin requirements conditional on the job:

- `PluginRequirementSet` — required/optional/forbidden plugin classes;
- `InputProductRequirement` / `OutputProductDeclaration` — typed input selectors and
  declared outputs;
- a profile resolver that may only inspect job configuration and input product
  **schemas**, never the underlying large arrays.

This is what allows an `analyze` job to require an input product and analysers without
pretending to own a model or integrator.
