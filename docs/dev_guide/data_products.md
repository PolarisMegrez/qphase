---
description: Data Product Contract
---

# Data Product Contract

!!! info "Phase 0 contract — qphase 2.0"
    This contract is approved and frozen for Phase 1 implementation under
    `qphase.data` and `qphase.core.task_profile`; the Phase 1 audit sync
    (coordinates as typed variables, registry-id artifact references) is a
    draft under review. Production Result serialization still changes only
    through the staged qphase 2.0 migration. The implemented on-disk
    formats are documented in [Artifact Formats](artifact_formats.md).

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
  `trajectory`, `frequency`, `channel`.
- `statistics` — low-dimensional results: moments, distributions, Allan variance,
  first-passage summaries, fit parameters. May contain structured tables, but arbitrary
  unverifiable objects never enter the persisted schema.

## Product schema

A `ProductSchema` is JSON-serializable, strictly extra-forbid, and has a stable hash.
Shapes may be partially unknown at plan time (`AxisSchema.size is None`) but must be
**closed** before artifact persistence. Runtime planning may use open templates;
persisted products never do.

- `AxisSchema` — `name`, `role`, optional `size`, `coordinate` (`regular` or
  `explicit`), `units`, `monotonic`. The `AxisRole` is one of `parameter` (a swept
  scan axis — never a sample ensemble), `realization` (independent trajectories that
  remain in the payload), `coordinate`, `component` and `index`. A realization axis
  must be retained by at least one variable; parameter/grouping coordinates may be
  represented by their coordinate payload or a resource-defined segmented layout.
- `CoordinateSchema` — a named coordinate backed by a typed variable payload
  (`name`, `variable`, `dims`, `role` of `dimension`/`auxiliary`/`parameter`,
  `units`, `monotonic`). Coordinates are ordinary typed variables, so they enter
  the backing, storage and checksum layers automatically; a regular axis may rely
  on `start`/`step` alone, while explicit values are persisted as coordinate
  variables (for example `omega_b[scan]`), never as object arrays.
- `SamplingBasisSchema` — a realization source already reduced out of the payload,
  such as trajectories contributing to a mean PSD. It declares a stable name and
  closes through either a retained realization `source_axis`, a fixed `count`, or an
  integer `count_variable`. This prevents a discarded trajectory dimension from
  masquerading as a sliceable product axis.
- `VariableSchema` — `name`, `dtype` (object dtype is forbidden), `value_domain`
  (`real` or `complex`; checked against the dtype in both directions), named `dims`
  referencing axes, `quantity`, `units`, and `constraints` (for example `nonnegative`
  — real numeric variables only — or a tensor `symmetry`/`layout` descriptor; a
  Hermitian layout requires at least two component/index dims). The **variable**, not
  the dataset class, decides real vs. complex: spectral products are never split into
  incompatible real/complex result classes.
- `UncertaintySchema` — `target` (the variable it describes), `kind` (`sample_std`,
  `sem`, `confidence_interval`, `covariance`, `other`), `sampling_basis` (the
  retained or reduced realization source the estimate counts over), an optional
  resource-defined `scope` identifier (for example
  `conditional`/`sampling`), `confidence` in `(0, 1)` and a positive integer `count`.
  Complex targets must declare `real_imag` or `magnitude_phase` covariance; real
  targets use `real`; there is no `custom` escape hatch. Covariance payloads are typed
  variables referenced by `data_variable` — never metadata dicts.
- Matrix/tensor variables use named dimensions plus a symmetry/layout descriptor;
  moment order is an axis or variable attribute, never a new dataset class.

`monotonic`, `nonnegative` and tensor symmetry/layout are declarative scientific
constraints. Core checks that they are structurally applicable to the declared
axes and dtypes, but does not eagerly scan large payloads to prove them. Producers
or domain-specific validators perform numerical checks when required.

## Spectral quantities

The frozen minimal `SpectralQuantity` set is `fourier_amplitude`,
`power_spectral_density`, `cross_spectral_density` and `coherence`. Spectral products
must carry the mandatory attribute set: frequency units, orientation, sidedness,
normalization, window and estimator (all non-empty), plus optional effective degrees
of freedom. PSD variables declare real/nonnegative; cross-spectrum variables declare
complex with a Hermitian layout. Statistics explicitly distinguish mean, sample std,
SEM and independent count instead of relying on key-name conventions.

## Moment families

The core schema has **no** moment-family field — that domain semantics is owned by the
resource package. `qphase_sde` defines a private, versioned
`SDEMomentFamilySchema` descriptor (`moment_kind`, `ordering`, a fixed `order` index
axis and explicit positive integer `orders`) embedded into the product's
`attributes`. Only moments whose orders share the remaining dims (for example
`moment[scan, order, channel]`) are covered; arbitrary mixed-rank moment tensors are
deliberately not claimed by the first schema version.

## Runtime handles vs. artifacts

In-process transfer, session caching and persistence are separate layers:

- `DataHandleProtocol` — an in-process, possibly device-resident buffer backing
  exactly **one variable** of a product: `variable_schema`, `device`, `dtype`,
  `shape`, `nbytes`, `read_only` and `owner`. The only frozen exchange operation is
  `materialize(target_device, copy_policy)`; implementations must never perform an
  implicit device-to-host copy. Zero-copy export descriptors are a later design and
  are deliberately absent.
- `DataLeaseProtocol` — the consumer-facing lifetime contract (`handle`, `consumer`,
  `scope` of `execution` or `session`, idempotent `release()`). Only the owner closes
  or reclaims the buffer; consumers only release leases. Pinning/eviction policies are
  not part of the frozen surface.
- `RuntimeProductBacking` — a product's runtime backing: one handle per schema
  variable, checked by `validate_backing` (missing/extra variables, full variable
  schema identity, dtype and closed-axis shape mismatches are rejected).
- `ArtifactRef` — durable, cross-process reference carrying identity only: artifact
  id, product name, product schema and the registered storage adapter id. It names
  no code and no filesystem location; no
  provenance, no arrays, no extra fields or cache state.
- `DataMaterializerProtocol` — resource-registered conversion between runtime handles
  and artifact-backed products.

A `DataProduct` is the semantic layer; its backing is either a
`RuntimeProductBacking` or an `ArtifactRef` — the two are distinct types and are never
conflated. `ResultProtocol.save()` is not a cross-job transport, and the artifact
store is not a runtime cache. Handle/lease ownership, read-only rules and failure
semantics are defined by core; resource packages never hand raw device buffers to
another job outside a lease.

## Product graphs and task profiles

`ProductRequirement` / `ProductDeclaration` describe typed inputs and outputs;
`ProductGraph` is the validated acyclic graph of `ProductNode`s. Each node has
a compiler-assigned `node_id`; graph edges use those readable IDs. The graph
does not hash nodes to manufacture identities.

`EngineTaskProfile` makes plugin requirements conditional on the job:

- `PluginRequirementSet` — required/optional/forbidden plugin classes; the three sets
  are pairwise disjoint, validated as registry-style namespaces and stored sorted so
  fingerprints never depend on YAML ordering;
- `InputProductRequirement` / `OutputProductDeclaration` — typed input selectors and
  declared outputs;
- an optional profile resolver receiving a restricted
  `TaskProfileResolutionContext` (normalized job config and input product **schemas**
  only — never handles, payloads, loaders or the scheduler); normalized config must be
  JSON-serializable. The resolver returns a
  **complete** `PluginRequirementSet` that replaces the profile defaults and is
  re-validated with the same invariants.

This is what allows an `analyze` job to require an input product and analysers without
pretending to own a model or integrator — while a model-aware analyser can still turn
`model` into an explicit requirement through its resolver.
