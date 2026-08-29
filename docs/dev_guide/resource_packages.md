---
description: Resource Package Contract
---

# Resource Package Contract

!!! info "Phase 0 contract — qphase 2.0"
    The `qphase.resource/1` contract is approved and frozen for Phase 1 implementation
    under `qphase.resources`.

A **resource package** is the managed asset unit of QPhase: a Python distribution that
bundles exactly one execution engine, its plugin classes, data products and public
contracts. `qphase_sde` and `qphase_cam` are resource packages; `qphase_viz` may adopt
the same contract with a smaller profile.

The authoritative description of a resource package is its
**`ResourcePackageManifest`**, not its source tree. Installed wheels are not guaranteed
to preserve a walkable source layout, so the scheduler, registry, CLI and GUI must be
able to enumerate a package's assets from the manifest and entry points alone. The
directory skeleton below is a development and audit convention, checked by contract
tests and a development-time validator — never a runtime discovery mechanism.

## Resource profiles

Packages compose their obligations from declarative profiles instead of inheriting a
single monolithic skeleton:

| Profile | Required additions on top of `base` |
| --- | --- |
| `base` | `manifest.py`, `engine.py`, `config.py`, `state.py`, `result.py`, `errors.py`, `contracts/` |
| `compute` | `planning.py`, `runtime/` |
| `simulation` | `model.py` |

Profiles are composable: `qphase_sde` declares `base + compute + simulation`, while a
visualization package can declare `base` alone and is not forced to create a
meaningless `model.py`.

- `__init__.py` exports only the version, the manifest and stable public types. It must
  not eagerly import concrete plugins.
- `manifest.py` is the single declaration site of the package's
  `ResourcePackageManifest`.
- `engine.py` is the only execution entry point the scheduler uses for the package.
- `config.py` holds package-level shared configuration and task profiles; it does not
  swallow concrete plugin configuration.
- `state.py` defines the package runtime and recoverable state protocol.
- `result.py` defines bundle/result adapters and the named data products.
- `errors.py` defines stable error codes and their mapping to the core error report.
- `contracts/` holds package-specific public protocols, quantities and capabilities.
- `planning.py` (`compute`) compiles resolved plugins and input products into an
  execution plan; `runtime/` (`compute`) holds package-private arena views, batch/tile
  and execution helpers. Plugins must not depend on concrete runtime schedulers.
- `model.py` (`simulation`) is the stable entry point for model protocols and
  capabilities.

Standard optional asset directories — `math/`, `serialization/`, `_native/` — must be
declared in the manifest with a purpose and a `public`/`private` visibility. Any other
package-specific directory must also be declared. Root-level catch-all modules such as
`utils.py` or a vaguely scoped `core/` are not allowed.

## Plugin class directories

Every plugin class occupies one root-level namespace directory (for example
`integrator/`, `analyser/`, `peak_finder/`). The directory contains at least
`__init__.py` and `base.py` (the public contract); complex classes may add
`config.py`, `result.py` or `contracts.py`.

- Concrete plugins live only inside their owning plugin-class directory.
- Parent/child plugin relations are expressed by the manifest slot graph, not by
  directory nesting.
- A concrete plugin must not directly import or construct a concrete plugin from
  another directory.

## The resource manifest

`ResourcePackageManifest` (schema `qphase.resource/1`) declares at least:

- `resource_id`, `schema_version`, `package_version`;
- the unique engine reference and the declared resource profiles;
- plugin classes: namespace, protocol, config schema reference, entry-point namespace;
- public data products, quantities and materializers;
- backend/device/optional-dependency capabilities;
- compatibility range and a deterministic asset fingerprint.

The manifest does **not** duplicate `EngineManifest` or concrete `PluginManifest`:
engine task requirements stay with `EngineManifest`, child slots and concrete
configuration stay with `PluginManifest`. The resource manifest only stores stable
references; the `ResourcePackageCatalog` resolves all three and cross-validates them
(exactly one engine per package, entry points matching namespaces, acyclic child
graph, resolvable task profiles, importable public schemas). Project, session and
artifact manifests keep independent schemas and namespaces and are never mixed with
resource manifests.

## Discovery and overlays

Resource packages register their manifest in the existing `qphase` entry-point group
under the name `resource.<id>`; core does not add a parallel discovery group.
`qphase list/show/config` and the GUI consume the catalog only.

Project-local and third-party concrete plugins register as **catalog overlays** onto
plugin classes declared by the resource manifest. They are not written back into the
manifest and are not required to adopt the full package skeleton. Overlay provenance
(`package`, `project_overlay`, `third_party`), compatibility and resolved-job
fingerprints are recorded so that a resolved job snapshot can distinguish
package-owned assets from overlays.

## Asset fingerprint

The resource asset fingerprint is derived from the canonicalized manifest, the package
version and the entry-point descriptors. It must not depend on source absolute paths,
file modification times or directory traversal order, so that the same installed
distribution always produces the same fingerprint.

## Development-time validation

`qphase.resources.validation` provides validators used by contract tests and release
checks:

- source-layout validation: required profile modules exist, concrete plugins sit in
  their declared class directory, optional asset directories are declared;
- manifest validation: schema conformance, unique engine, fingerprint stability;
- entry-point validation, scoped by ownership: `partition_entry_points` first splits
  the global `qphase` group into package-owned, project-overlay and third-party
  descriptors by distribution; `validate_package_entry_points` then checks only the
  descriptors owned by the package's own distribution (exactly one `engine.*`, exactly
  one `resource.<id>`, declared namespaces only), so co-installed SDE/CAM packages and
  backend plugins never trigger false engine-count or unknown-namespace findings;
  `validate_overlay_entry_points` applies a separate, narrower policy to project
  overlays — attributed to packages by namespace, they must never occupy the reserved
  `resource.*`/`engine.*` namespaces. Third-party descriptors are provenance-labeled
  via `classify_origin` and validated against their own distribution's manifest.

Runtime and wheel installations treat the manifest and entry points as the only source
of truth; source-tree walking is never used at runtime.
