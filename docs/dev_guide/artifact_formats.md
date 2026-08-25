---
description: Artifact and resource formats of qphase 2.x
---

# Artifact Formats

!!! info "qphase 2.0 — Phase 1 implementation"
    This page documents the on-disk formats implemented in Phase 1 under
    `qphase.data` (artifact store + NPZ 2.x adapter) and `qphase_sde` (typed data
    bundles + 1.x migration tool). It is the format-level companion of the
    [Data Product Contract](data_products.md).

A qphase 2.x artifact is a **directory**: one `artifact_manifest.json` plus one
NPZ chunk file per variable chunk. Directories are self-describing, relocatable
and inspectable with plain NumPy — restoring never requires `allow_pickle`.

## Manifest v3

`artifact_manifest.json` is a strictly validated (`extra="forbid"`) JSON document:

- `schema_version` — the literal `"qphase.artifact/3"`.
- `artifact_id` — stable artifact identifier.
- `created_at` — creation timestamp.
- `loader` — public restore entry point in `module:attr` syntax; defaults to
  `qphase.data.npz:load_product_backing`. Loaders never use `allow_pickle`.
- `products` — list of product entries, each with:
    - `name` — product name (unique within the artifact);
    - `product_schema` — the frozen [product schema](data_products.md) JSON;
    - `storage` — `adapter` id plus `variables`: a mapping of variable name to
      its **chunk records**;
    - `sha256` — content hash over the product's chunk hashes.
- `provenance` — JSON-serializable engine/plugin metadata (validated).
- `parents` — artifact ids this artifact derives from.
- `content_hash` — SHA-256 over the canonical product/parent listing.

Each chunk record carries `file` (relative to the artifact directory, so
artifacts can be relocated), `key` (always `"data"`), `shape`, `dtype`,
`sha256` (over the C-contiguous payload bytes, **verified on every read**) and
`axis0_range` — the `[start, stop)` slice along the first dimension for
sharded variables, `null` when the chunk holds the whole variable.

## NPZ 2.x layout

The reference storage adapter (`qphase.data.npz`) writes:

- unsharded variables as `{stem}__{variable}.npz`;
- sharded variables as `{stem}__{variable}__{chunk:04d}.npz`, split along the
  first dimension;
- every chunk with a single `"data"` key, in its **native dtype** (complex and
  tensor payloads included) — never object arrays;
- metadata only in the manifest JSON, never inside the NPZ files.

## Writing and reading

- `save_products(directory, products, *, provenance=None, parents=(),
  artifact_id=None, shard_target_bytes=...)` persists typed datasets and
  returns the written `ArtifactManifestV3`. Artifact-backed datasets are fully
  materialized first (an explicit load, never an implicit one); device-resident
  payloads are copied to the host with an explicit `copy_policy="allow"`. An
  empty product mapping is allowed (e.g. a run that only yields scalars).
- `load_products(directory)` reopens an artifact as **lazily backed** datasets:
  payloads are not read; handles expose shape/dtype/nbytes from the manifest,
  and selections such as `point_view` read only the chunks they touch (no full
  concatenation for point access).
- `ArtifactManifestV3.read(directory)` reads and validates a manifest;
  `manifest.product_ref(name)` builds the durable `ArtifactRef` of one product.
- A process-local registry maps product-scoped artifact ids
  (`"{artifact_id}:{product}"`) to artifact directories so `ArtifactRef`-backed
  datasets can resolve their storage. `save_products`/`load_products` populate
  it; cross-process restores must open the artifact directory once before
  dereferencing refs.

## SDE data products

`qphase_sde` returns typed `SDEDataBundle`s from every `engine.run()` exit
point. A bundle exposes `products` (mapping of name to dataset), `provenance`
and `require(...)`, and implements both `ResultProtocol` and
`DatasetResultProtocol`:

- `trajectories` — a `time_series` product with axes
  `(scan, trajectory, time, channel)` and a `valid_length` variable;
  device-resident arrays stay on device (CuPy payloads are wrapped in
  `BackendArrayHandle`s without copying).
- analysis products — persisted through the versioned `legacy_analysis/1`
  bridge: numeric leaves become variables, nested dicts are flattened with
  dotted paths, string/JSON-safe leaves land in `attributes["payload_meta"]`,
  ragged per-point leaves degrade to meta lists recorded under
  `per_point_meta`, and unbridgeable keys are reported under `dropped_keys`.
- `legacy_result()` renders the single-point 1.x view; `point_view` rewrites
  `metadata["params"]` to the point's scan parameters (the legacy
  `SDEScanResult` semantics).

## Migrating SDE 1.x results

`qphase_sde.runtime.migrate` converts existing results **one way** to v3:

- `migrate_legacy_result(source, output_dir, *, adapter=None,
  shard_target_bytes=None)` — one `sde_result/1` or `trajectory_set/1` file.
- `migrate_scan_artifact(manifest_path, output_dir, *, adapter=None)` — an
  `sde_scan/2` per-point artifact, streamed per point: each shard is read twice
  (structure pass, chunk pass), every point contributes one chunk per variable
  along the scan axis, and peak memory stays within one shard plus one output
  chunk.

Guarantees: source files are SHA-256 hashed (recorded in the output manifest's
provenance) and **never modified**; the output directory must be empty and
disjoint from the sources; unknown object payloads are rejected at the
npy-header level unless an `adapter` maps them to bridge-compatible mappings.
Both functions return a `MigrationReport` (with `MigrationWarning` entries)
and raise `LegacyFormatError` for unrecognized inputs.

## Service and GUI access

Listings never materialize payloads:

- `SchedulerService.describe_products(path, *, session_dir)` returns an
  `ArtifactProductCatalog`: artifact id, loader, content hash, total size, and
  one `ProductSummary` per product (kind, axes including regular-coordinate
  `start`/`step`, variables, devices, `nbytes`, `chunk_count`, `sha256`,
  attributes) — all from the manifest and schemas.
- The GUI exposes it as `GET /sessions/{session_id}/jobs/{job_name}/products`,
  returning the catalog JSON; missing or non-v3 directories answer 404.
