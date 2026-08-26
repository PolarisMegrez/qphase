---
description: Artifact and resource formats of qphase 2.x
---

# Artifact Formats

!!! note "Current v4 format"
    This page describes the approved v4 artifact and NPZ 3.x contracts. The
    authoritative behavior is the implementation under `qphase.data` and
    `qphase_sde`. It is the format-level companion of the [Data Product
    Contract](data_products.md).

A qphase 2.x artifact is a **directory**: one `artifact_manifest.json` plus
payload files written by a registered storage adapter. The reference `npz/3`
adapter writes one NPZ chunk file per variable chunk. Directories are
self-describing, relocatable and inspectable with plain NumPy — restoring
never requires `allow_pickle`.

## Manifest v4

`artifact_manifest.json` is a strictly validated (`extra="forbid"`) JSON
document:

- `schema_version` — the literal `"qphase.artifact/4"`.
- `artifact_id` — stable artifact identifier.
- `created_at` — creation timestamp.
- `bundle` — a `BundleDescriptor`: `type_id` (for example
  `generic.dataset_bundle/1`, `sde.bundle/1`), `adapter_id` selecting the
  **registered** bundle adapter used to rebuild a concrete result (a trusted
  registry id, never a code path), `descriptor_schema`, the adapter-validated
  JSON `descriptor` (for SDE bundles: the scan grid — `shape`,
  `dimension_order`, `axes`, `n_traj_per_point`, optional `combine`), and
  `product_roles` mapping stable semantic roles to job-local product names.
  Labels without a cross-workflow meaning are intentionally omitted.
- `products` — list of product entries, each with:
    - `name` — product name (unique within the artifact);
    - `product_schema` — the full frozen [product schema](data_products.md)
      JSON (`qphase.product/1`): axes, variables, coordinates, sampling
      bases, uncertainties and attributes;
    - `storage` — `adapter` (a registered adapter id such as `npz/3`),
      `descriptor_schema`, the adapter-specific `descriptor`, and a common
      `summary` (per-variable `nbytes`/`chunk_count`) that listings can read
      without opening the adapter;
- `provenance` — JSON-serializable engine/plugin metadata (validated).
- `parents` — artifact ids this artifact derives from (unique).

There is no `loader` field and no `module:attr` reference anywhere in the
format: storage adapters and bundle adapters are resolved through trusted
process-local registries (`register_adapter`, `register_bundle_adapter`).
Manifest paths are validated as safe artifact-relative paths; product names,
parents and bundle role targets must be unique/consistent. Violations raise
typed errors: `ArtifactNotFoundError` (also a `FileNotFoundError`),
`ArtifactUnsupportedError` (unknown schema version), `ArtifactCorruptError`
(parse or structural failures) and `ArtifactAdapterError` (unregistered
adapter).
Registered storage and bundle adapters validate their descriptors while the
manifest is read, without opening payload files. Unknown bundle adapters
remain listable as generic bundles; malformed descriptors owned by known
adapters are corrupt, not merely unavailable. Manifest metadata is strict
JSON: `NaN` and infinity are rejected, while typed numeric payload arrays may
still contain them. For registered storage adapters, manifest validation also
aggregates payload ownership across products: one payload file may hold several
keys of one product, but it cannot be shared by two products.

## NPZ 3.x storage adapter

The reference adapter (`qphase.data.npz`, adapter id `npz/3`, descriptor
schema `npz.product/3`) records per variable in its descriptor:

- `full_shape` and `dtype` of the whole variable;
- `chunk_axis` — the **named** dimension along which the variable is sharded
  (`null` for unsharded variables; any axis may be chosen to meet the byte
  target, not just the first dimension);
- `chunks` — contiguous, non-overlapping, fully covering chunk records with
  `file` (artifact-relative), `key`, `logical_range` (`[start, stop)` along
  `chunk_axis`, `null` when the chunk holds the whole variable), `shape`,
  `dtype`. Ordinary reads verify the actual dtype, shape and exact
  descriptor-wide key set for that payload file. Undeclared keys are
  corruption.

File layout:

- sharded variables: `{stem}__{variable}__{chunk:04d}.npz`, one `"data"` key
  per chunk file;
- unsharded variables: `{stem}__{variable}.npz`;
- `layout="single"` writes one `{stem}.npz` holding every variable of the
  product as keys (no external sharding);
- arrays are stored in their **native dtype** (complex and tensor payloads
  included) — never object arrays; metadata lives only in the manifest JSON.

Writes are transactional: chunks are staged in a `.staging-{token}`
directory, their descriptors are checked before publication, atomically moved
to their final names, and the manifest is published last through an atomic
`os.replace`. Payload bytes are not hashed by the normal writer or reader.
An existing manifest is never overwritten
unless `replace=True`; replacement removes the old payload only after the
new manifest is published, so a failed write leaves the previous artifact
fully readable. A first publish also refuses to replace any pre-existing
payload path when no validated prior manifest owns it.

## Writing and reading

The current scheduler treats each Job directory as the root of exactly one
primary bundle artifact. Job logs, configuration snapshots and exported CSVs
are not artifact payload unless the manifest references them. A future
multi-artifact layout may place separate artifact roots below the Job, but it
must preserve the manifest contracts defined here.

- `save_products(directory, products, *, provenance=None, parents=(),
  artifact_id=None, shard_target_bytes=..., bundle=None, layout="sharded",
  replace=False)` persists typed datasets and returns the written
  `ArtifactManifest`. Artifact-backed datasets are fully materialized first
  (an explicit load, never an implicit one); device-resident payloads are
  copied to the host with an explicit `copy_policy="allow"`. An empty product
  mapping is allowed. Every persisted product schema must be closed. Without an explicit `bundle` a generic bundle
  descriptor is recorded.
- `load_products(directory)` reopens an artifact as **lazily backed**
  datasets: payloads are not read; handles expose shape/dtype/nbytes from the
  validated manifest, and selections such as `point_view` read only the
  chunks they touch. `load_bundle(directory)` additionally rebuilds the
  concrete bundle through the registered bundle adapter (the generic adapter
  yields a `GenericDataBundle`).
- `DirectoryArtifactResolver` remains available for explicitly bound,
  process-local references. `ProjectArtifactResolver` resolves an
  `ArtifactRef` within one Project's Session root without process-global state;
  the scheduler supplies it through `ExecutionContext.artifact_resolver`.
  Resolution is catalog-first with a direct manifest scan as the fresh-truth
  fallback. An `artifact_id` identifies the immutable artifact; each on-disk
  location is a producing occurrence. Without occurrence context the resolver
  returns the location only when it is unique and raises
  `ArtifactAmbiguousError` (listing every location) otherwise — it never picks
  one copy arbitrarily. An `ArtifactRef` carries identity only – artifact
  id, product name, product schema and storage adapter id; it names no code and
  no filesystem location.

Artifact-backed datasets must be materialized with an explicit resolver, for
example `dataset.materialize(resolver=ProjectArtifactResolver(project))`.
There is no implicit process-global resolver on the normal materialization
path; this prevents identical Artifact IDs from crossing Project boundaries.

## SDE data products

`qphase_sde` returns typed `SDEDataBundle`s from every `engine.run()` exit
point and persists them as v4 artifacts with an `sde.bundle/1` bundle
descriptor:

- `trajectories` — a `time_series` product with axes
  `(scan, trajectory, time, channel)` and a `valid_length` variable;
  device-resident arrays stay on device (CuPy payloads are wrapped in
  `BackendArrayHandle`s without copying).
- typed analysis products (`graph_ready=True`) — `psd` persists as a
  `spectral` product with the full mandatory spectral attribute set and
  sampling-basis-backed uncertainties; Allan variance, moment families,
  moment statistics, coherence matrix and coherence carrier persist as
  `statistics` products with declared axes, quantities and per-variable
  uncertainties.
- Every graph-ready scan product carries flattened scan parameter coordinates
  as typed `(scan,)` variables. Sampling coordinates shared by every point
  (frequency, tau, lag and channel) are deduplicated into dimension
  coordinates; point-varying coordinates remain declared auxiliary
  coordinates.
- The `sde/1` bundle adapter cross-checks its scan descriptor against every
  product schema: shape extents are strict positive integers, every product
  carrying a `scan` axis has the flattened bundle scan size, and the stable
  `trajectories`/`primary_spectrum` roles point to compatible time-series and
  spectral products.
- remaining analysers persist through the versioned `legacy_analysis/1`
  bridge (`graph_ready=False`) until Phase 2: numeric leaves become
  variables, nested dicts are flattened with dotted paths, string/JSON-safe
  leaves land in `attributes["payload_meta"]`, ragged per-point leaves
  degrade to meta lists recorded under `per_point_meta`, and unbridgeable
  keys are reported under `dropped_keys`.
- manifest provenance records `engine`, the versioned
  `qphase_sde.provenance/1` record under `sde`, JSON-safe job `meta` (plus
  `meta_dropped`), and the real installed distribution `versions` of `qphase`
  and `qphase_sde`. Its `dt` is the SDE integration step, not a core-wide
  provenance field or the saved time-series sample interval; other resource
  packages define their own numerical provenance schemas.
- importing `qphase_sde.result` registers the `sde/1` bundle adapter, so a
  clean process restores scan bundles (shape, axes, per-point parameter
  views) straight from the v4 manifest; `legacy_result()` renders the
  single-point 1.x view and `point_view` rewrites `metadata["params"]` to
  the point's scan parameters.
- SDE bundle roles expose only stable meanings: `trajectories` when retained
  and `primary_spectrum` when exactly one spectral product exists. Other
  products are selected by kind, quantity and fields.
- Peak candidates and paths are not currently declared as graph-ready public
  products. Their grouped ragged schema and producers are a mandatory Global
  Phase 5A prerequisite to the ProductGraph executor. Until then, legacy PSD
  peak metadata is stored in a `legacy_peaks/1` bridge carrying explicit
  `source_product` and `payload_field` routes, so `legacy_result()` restores
  the original `analysis["psd"]["peaks"]` field without treating it as a
  graph-ready peak product.

## Migrating SDE 1.x results

!!! warning "Temporary major-version transition tool"
    These 1.x-to-2.x utilities are not stable 2.x APIs. Global Phase 4 removes
    them after all retained project data has been migrated and verified. QPhase
    2.x does not promise permanent old-major compatibility.

`qphase_sde.runtime.migrate` converts existing results **one way** to v4:

- `migrate_legacy_result(source, output_dir, *, adapter=None,
  shard_target_bytes=None)` — one `sde_result/1` or `trajectory_set/1` file.
- `migrate_scan_artifact(manifest_path, output_dir, *, adapter=None)` — an
  `sde_scan/2` per-point artifact, streamed per point: each shard is read
  twice (structure pass, chunk pass), every point contributes one chunk per
  variable along the scan axis, and peak memory stays within one shard plus
  one output chunk.

Guarantees: source files are SHA-256 hashed (recorded in the output
manifest's provenance) and **never modified**; the output directory must be
empty and disjoint from the sources; unknown object payloads are rejected at
the npy-header level unless an `adapter` maps them to bridge-compatible
mappings. Migration provenance also records the real distribution versions
of the converting environment. Both functions return a `MigrationReport`
(with `MigrationWarning` entries) and raise `LegacyFormatError` for
unrecognized inputs.

## Service and GUI access

Listings never materialize payloads and never register artifact locations:

- `SchedulerService.list_artifacts(session_dir)` returns one item per current
  v4 manifest-backed artifact with its real UUID `artifact_id`. Ordinary files
  outside artifact directories have no artifact identity and are addressed by
  a project-relative `file_ref`.
- The GUI uses `GET /sessions/{session_id}/artifacts/{artifact_id}` for a
  typed product catalog and `GET /sessions/{session_id}/files/{file_ref}` for
  ordinary text or binary files.
- `SchedulerService.describe_products(path, *, session_dir)` builds an
  `ArtifactProductCatalog` purely from the manifest plus `stat` of the
  manifest-referenced payload files: artifact id, loader (adapter ids),
  total size (referenced files only — stray files in the
  directory are not counted), a `BundleSummary` (type/adapter ids,
  descriptor schema, product roles, unpacked scan shape/combine/axes and
  `n_traj_per_point`), and one `ProductSummary` per product: kind, axes
  (including regular-coordinate `start`/`step`), variables with constraints,
  coordinates, sampling bases, uncertainties, devices, `materializable` with
  a typed `missing_reason` (unregistered adapter or missing payload file),
  logical `nbytes` and physical on-disk `physical_nbytes`, `chunk_count`,
  `schema_version`/`schema_fingerprint`, storage adapter and descriptor
  schema, attributes.
- The GUI exposes it as `GET /sessions/{session_id}/jobs/{job_name}/products`,
  returning the catalog JSON; missing or non-artifact directories answer
  404, while unsupported or corrupt artifacts answer 422.
