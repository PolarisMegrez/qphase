---
description: QPhase release and persisted-data compatibility policy
---

# Versioning and Migration Policy

QPhase releases use `x.y.z`, with separate but coordinated rules for Python/API
compatibility and persisted scientific data.

## Major `x`

A major release may break APIs, Workflow schemas, plugin contracts and persisted
formats. A new major is not required to read or migrate artifacts of an older
major. Reproduction of an old major uses its archived release, environment or
branch; compatibility code is not retained indefinitely in the new runtime.

The 1.x-to-2.x migration utilities were workspace-transition tools, never
stable 2.x APIs. With the project migrated and verified at the 2.0.0 boundary,
their runtime entry points, dedicated fixtures and migration documentation have
been removed.

## Minor `y`

Within one major, public Python APIs remain backward compatible. Persisted data
may evolve through a one-way migration: the newer minor must either read older
minor data directly or provide a verified migrator. Older minors need not read
newer-minor output.

Minor migrators remain available until the next major release, either as a
direct migration from all supported earlier minors or as a verified migration
chain. A migration records source hashes, tool/package versions and warnings,
and never modifies the source artifact in place.

## Patch `z`

Patch releases are bidirectionally compatible at the public contract level:
they do not add required schema fields, change field meaning, alter hash
algorithms or reject data that was valid under another patch of the same
minor. Security checks may reject data that was already invalid under the
frozen contract; a genuine contract tightening requires a minor release.

## Schema versions

Package and schema versions are not interchangeable. Persisted contracts carry
explicit identifiers such as `qphase.product/1`, `qphase.artifact/4`, `npz/3`
and `qphase_sde.provenance/1`. A schema change is classified as patch, minor or
major by compatibility impact, not by matching the package version numerically.

Migration tools are explicit commands or maintenance utilities. Artifact
loading never silently rewrites data, and ordinary plugin discovery never
imports an old-major compatibility layer.

## Wave 6D breaking changes (qphase_sde 2.0)

The Wave 6D migration reworked the PSD/Allan measurement configuration, the
typed-input requirement name, and the product provenance attribute of
`qphase_sde`. All configuration models are `extra="forbid"`, so removed keys
are hard validation errors, and no dual-key compatibility is retained.

1. **PSD/Allan `measurement` union.** `PsdAnalyserConfig` and
   `AllanVarianceConfig` dropped the top-level `kind`/`modes` keys in favour of
   a strictly discriminated `measurement` union. PSD takes
   `measurement: {complex_field: {channels, representation, orientation?}}` or
   `measurement: {sampled_scalar: {channels, ...}}`; Allan takes
   `measurement: {phase_frequency: {channels, orientation, amplitude_floor, min_independent_windows}}`
   or `measurement: {sampled_scalar: {channels, min_nonoverlap_pairs}}`. All
   phase-only knobs (`orientation`, `amplitude_floor`,
   `min_independent_windows`) moved into the `phase_frequency` variant; the
   `sampled_scalar` variants are `extra="forbid"` and reject `orientation`.

   ```yaml
   # before
   analyser:
     psd: {modes: [0], kind: complex, orientation: phase_decreasing}
     allan_variance: {modes: [0], min_independent_windows: 4}

   # after
   analyser:
     psd:
       measurement:
         complex_field:
           channels: [0]
           representation: complex
           orientation: phase_decreasing
     allan_variance:
       measurement:
         phase_frequency:
           channels: [0]
           min_independent_windows: 4
   input_bindings:
     psd: {signal: input.trajectories}
     allan_variance: {signal: input.trajectories}
   ```

2. **Requirement `trajectories` renamed to `signal`.** The typed input
   requirement of both analysers is now named `signal`, so a workflow must bind
   it explicitly:
   `input_bindings: {<analyser_label>: {signal: input.trajectories}}`.
3. **`source_analyser` renamed to `producer_analyser`.** The typed-product
   attribute (and provenance entry) written by `assemble_typed_product` uses
   the new key everywhere; consumers reading `attributes["source_analyser"]`
   must switch to `producer_analyser`.
4. **Typed products carry per-variable units.** `assemble_typed_product`
   gained a `variable_units` parameter, and the spectral analysers annotate
   every numeric leaf at the product boundary. The sampled-scalar PSD variables
   `psd`, `psd_std` and `psd_sem` all carry `<value-units>^2/inverse_time`
   (`1/inverse_time` when the source observable is dimensionless);
   sampled-scalar Allan variance, SEM and per-trajectory leaves carry
   `<value-units>^2` (`1` when dimensionless). Phase-path variance leaves
   declare `inverse_time^2` and the mean-frequency leaves `inverse_time`.
   Source units `"1"` and `""` both normalise to the documented dimensionless
   form. Spectral PSD products expose `convention`, and complex-field products
   additionally `representation`, as top-level attributes.
5. **`kind: modular` removed.** The magnitude PSD of the complex field is
   `representation: magnitude` inside `measurement.complex_field`.
