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

The current 1.x-to-2.x migration utilities are workspace-transition tools, not
stable 2.x APIs. After Global Phase 4 has migrated and verified the project,
their runtime entry points, dedicated fixtures and migration documentation are
removed.

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
explicit identifiers such as `qphase.product/1`, `qphase.artifact/3`, `npz/2`
and `qphase_sde.provenance/1`. A schema change is classified as patch, minor or
major by compatibility impact, not by matching the package version numerically.

Migration tools are explicit commands or maintenance utilities. Artifact
loading never silently rewrites data, and ordinary plugin discovery never
imports an old-major compatibility layer.
