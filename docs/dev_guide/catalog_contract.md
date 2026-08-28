---
description: Catalog, identity and annotation development contracts
---

# Catalog Contract

This page fixes the contracts that the tag/annotation/catalog subsystem
guarantees. Code that touches these areas must preserve them; changes to the
contracts themselves require a schema or read-model version bump.

## Identity Rules

- A path is a **location**, never an identity. Sessions and artifacts may move
  or be duplicated; their ids do not change.
- **Artifact** identity is the immutable `artifact_id` from the
  `qphase.artifact/4` manifest. The catalog never fabricates an id: a location
  whose manifest is unreadable or unsupported indexes no artifact row and is
  surfaced as a `corrupt`/`unsupported` location issue.
- **Occurrence** identity is the triple `artifact_id:session_id:job_name`.
  Occurrence annotation keys inside `session_annotations.json` are
  `job_name:artifact_id`, so two occurrences of one artifact in a session
  never collide. Legacy bare-artifact keys are migration input, flagged by
  `qphase project migrate --dry-run`.
- `:` is the reserved identity separator: `JobConfig.name` and the manifest
  `artifact_id` reject it at validation time. Existing job names or artifact
  ids containing it are migration input, listed by
  `qphase project migrate --dry-run`.
- **Workflow revision** identity is `workflow_id@revision` where `revision`
  is a content hash of the workflow document; **job** identity extends it to
  `workflow_id@revision:job_name`. Revisions are rebuilt deterministically
  from `configs/workflows` files, session snapshots and execution records —
  the same content always yields the same revision id.
- **Execution** identity is the `execution_id` of the persisted
  `qphase.execution/1` record.

## Provenance Freezing

- Declared workflow/job tags are frozen per session into
  `workflow_snapshot.yaml` plus `tag_snapshot.yaml` (canonical tags, stable
  assignment ids, the policy revision that validated them). Later edits to
  the workflow file or the policy never rewrite a past session's provenance.
- Submission tags are frozen on the execution record together with
  `tag_policy_revision`; they are mutable only while the execution is queued.
- Every annotation `TagAssignment` freezes `policy_revision` **and the
  minimal namespace rule** (`inherit`/`cardinality`/`objects`) at write time.
  Effective-tag resolution prefers the frozen rule; assignments without one
  (legacy documents, rule-less namespaces, private tags) fall back to the
  policy current at read time. Assignments are immutable: editing a tag
  removes one assignment and adds a new one, so an effective tag can always
  cite a stable `assignment_id`.
- Declared tags have no annotation document; their assignment ids are derived
  deterministically (`sha256` over the declaration identity, including the
  workflow revision computed by the shared
  `qphase.core.workflow.workflow_revision`). A workflow edit therefore yields
  fresh assignment ids instead of colliding with the previous revision's;
  historical sidecar ids are trusted as-is.

## The Catalog Is a Rebuildable Read Model

- The catalog (`<project>/.qphase/object_catalog.sqlite`, read-model schema
  `qphase.catalog/4`) is a pure function of disk truth: manifests, snapshots,
  execution records, workflow files, the tag policy and the annotation
  documents. It may be deleted at any time and rebuilt with
  `qphase project reindex`.
- Derived facet tables keep common filters in SQL: the `job_plugins` and
  `artifact_quantities` side tables, `sessions.workflow_revision_id`, and
  per-occurrence `engine`/`model` taken from the session's frozen workflow
  snapshot. `CatalogQuery` exposes them as kind-checked filters (`plugin`,
  `quantity`, `model`/`engine`/`has_model`); a filter used with the wrong
  object kind raises `ValueError`.
- Reads probe a cheap fingerprint (project root; manifest/record counts and
  newest mtimes; annotation document counts and newest mtime; workflow file
  count and mtime; tag policy mtime); a mismatch triggers one rebuild before
  serving. State flips of a running job therefore surface on the *next*
  catalog query — the accepted cost of a derived read model.
- A corrupt, schema-mismatched or foreign database is rebuilt from disk
  truth instead of serving empty results. The `meta` table binds the database
  to `project_id`, so a copied project never reads another project's catalog.
- Concurrent rebuilds of the same project are serialized by a cross-process
  sibling lock file (plus an in-process per-path lock); each rebuild
  populates a temporary database that atomically replaces the live one, so
  a failed rebuild never deletes the previous read model.

## Annotation Documents and Locking

- Three document types hold shared annotations:
  `session_annotations.json` (`qphase.session-annotations/1`, also carries
  per-occurrence annotations), `artifact_annotations.json`
  (`qphase.artifact-annotations/1`) and `.qphase/project_annotations.json`
  (`qphase.project-annotations/1`, for the project itself plus workflow
  revisions, jobs and executions).
- Writes are optimistic-concurrency checked (`revision` must match) and the
  read-check-write cycle is serialized by a cross-process lock in a sibling
  `<name>.lock` file (blocking `msvcrt`/`fcntl` lock, released by the OS if
  the holder crashes). A revision conflict surfaces as a `RuntimeError`;
  callers reload and retry.
- Annotation writes never touch artifact manifests or payloads.
- User-private state (private tags, private alias/note, saved views, project
  locations) lives in `~/.qphase/gui/<project_id>.sqlite` and is overlaid at
  read time. Private state never enters the shared documents or the catalog.

## Artifact vs. Occurrence

- Artifact mutations (`tag_artifact`, lifecycle) require exactly one indexed
  location; several locations raise `ArtifactAmbiguousError` instead of
  silently picking the first.
- Occurrence mutations address one producing context; when an artifact occurs
  in several jobs of a session, the job name is required.
- A later occurrence of one artifact whose identity facets disagree with the
  first indexed occurrence is a `conflict` location issue; the artifact row
  stays first-seen.

## Explicit Non-Goals

Per the Phase 3 audit, the subsystem deliberately does not:

- introduce Campaign/Study objects;
- build a second catalog or an archive directory tree;
- write private annotations into shared truth;
- grow a generic ORM, event sourcing, or automatic resource scheduling;
- restore artifact payload hashes;
- pull Phase 6 worker/GUI-renderer work into this phase;
- fabricate artifact ids for corrupt files or silently return an empty
  catalog.
