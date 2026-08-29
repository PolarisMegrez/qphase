---
description: Tags, annotations and the project object catalog
---

# Tags & Catalog

Tags are QPhase's organization layer: they label projects, workflows,
executions, sessions, artifacts and artifact occurrences so you can find,
filter and group them later. Tags never change what a run computes — they are
metadata about objects, stored outside the immutable run records. The
**catalog** is the searchable index that makes those labels queryable.

## Object Model

Seven object kinds carry tags. Each has a stable identity:

| Kind | Identity | What it is |
| --- | --- | --- |
| Project | `project_id` | The whole project rooted at `qphase.toml`. |
| Workflow revision | `workflow_id@revision` | One content revision of a workflow document. |
| Job | `workflow_id@revision:job_name` | One logical job of a workflow revision. |
| Execution | `execution_id` | One submitted run of a workflow. |
| Session | `session_id` | One persisted run directory under the session root. |
| Artifact | `artifact_id` | One immutable result bundle (identity, not location). |
| Occurrence | `artifact_id:session_id:job_name` | One materialization of an artifact inside a session job. |

An artifact is an identity; an occurrence is one place that identity was
produced. The same artifact may occur in several sessions — annotations on the
artifact follow the identity, annotations on an occurrence stay local to that
producing context.

Because `:` joins these identity shapes, job names and artifact ids must
never contain it; new writes are rejected at validation time and existing
data is flagged by `qphase project migrate --dry-run`.

## Tag Syntax and Namespaces

A tag has the form `namespace:value` or `namespace:path/to/value`, all
lowercase. The namespace matches `[a-z][a-z0-9_]*`; each path segment matches
`[a-z0-9][a-z0-9_.-]*`. Tags are canonicalized on write, so `Task:Scan` and
`task:scan` are the same tag.

Hierarchical paths are first-class: querying `purpose:paper` with descendant
matching also finds `purpose:paper/fig3`.

## Tag Policy

The optional project tag policy lives in `configs/tags.yaml`
(schema `qphase.tag-policy/1`):

```yaml
schema: qphase.tag-policy/1
namespaces:
  stage:
    cardinality: one        # at most one stage:* value per object
    values: [q1, q2]        # closed vocabulary
  task:
    open: true              # any value allowed
  model:
    aliases: {vdp: vdp_2mode}
    objects: [workflow, session]   # restricted applicability
    inherit: false           # never flows down the object hierarchy
reserved_namespaces: [system]
retention_inherits_to_occurrences: true
```

Per-namespace rules:

- `cardinality`: `many` (default) or `one`. In a `one` namespace, a nearer
  assignment **shadows** a farther one (see below).
- `values`: closed list of allowed values. Omit and set `open: true` for a
  free-form namespace.
- `aliases`: map alternative spellings to the canonical value.
- `objects`: restrict which object kinds may carry the namespace.
- `inherit`: when `false`, the tag never flows from workflow to job, session
  or occurrence.
- `reserved_namespaces`: rejected everywhere (reserved for future built-ins).
- `retention_inherits_to_occurrences`: whether a session's retention applies
  to its occurrences by default (default `true`).

Without a policy file there is no governance: tags are only
syntax-canonicalized. The policy has a content-hash **revision**; every
tag assignment freezes the revision that validated it, so historical
provenance survives later policy edits. Each assignment also freezes the
minimal namespace rule that governs it (`inherit`, `cardinality`,
`objects`): effective-tag resolution of a historical assignment follows the
rule as it was at write time, while assignments written before rule
freezing (or without a governing rule, such as private tags) fall back to
the current policy. Inspect the policy with
`qphase tag policy show` and check it with `qphase tag policy validate`.

## The Four Tag Scopes

Tags enter the system through four scopes, listed from farthest to nearest:

1. **Declared tags** (workflow/job): written in the workflow YAML or a job
   definition. When a workflow runs, the resolved declared tags are frozen
   into the session's `workflow_snapshot.yaml` / `tag_snapshot.yaml` — later
   edits to the workflow file never rewrite history.
2. **Submission tags** (execution): supplied at submit time and frozen on the
   execution record together with their minimal namespace rules. They can
   only be edited while the execution is still queued (`qphase execution
   tag`).
3. **Shared annotations**: written to annotation documents inside the project
   (`session_annotations.json`, `artifact_annotations.json`,
   `.qphase/project_annotations.json`). They are shared with everyone who
   opens the project and belong in version control.
4. **User-private tags**: stored in your per-user database under
   `~/.qphase/gui/`, never in the project. Use `--private` on any tag
   command. Private tags participate in queries for you alone and can be
   promoted into shared annotations with `qphase tag promote`.

## Inheritance and Shadowing

Tags flow down the hierarchy: project → workflow revision → job, and
workflow/execution → session → artifact occurrence. Namespaces marked
`inherit: false` stay on the object where they were declared. Whether one
historical assignment inherits or shadows is decided by the namespace rule
frozen on it (see *Tag Policy*), never re-decided by a later policy edit.

Within one object, nearer scopes win in `cardinality: one` namespaces: a
session annotation shadows the workflow-declared value of the same namespace,
and a private tag shadows both. Shadowed tags remain visible with provenance
in the API but are hidden from default listings and do not match queries.

`lifecycle` never inherits — it describes the object itself. `retention`
flows from session to occurrence when the policy allows it, and an occurrence
can override it locally. Setting a session's retention freezes the policy's
inherit switch with it, so later policy edits never re-decide history;
sessions written before this contract fall back to the current policy until
the Phase 4 migration backfills them.

## Lifecycle and Retention

Lifecycle and retention are typed fields, not tags:

- **Lifecycle** (`active`, `reference`, `superseded`, `archived`) marks where
  an object stands in your workflow. `archived` means cold storage: the object
  stays queryable but is considered retired from active use.
- **Retention** (`transient`, `preserve`, `evidence`, `pinned`) states how
  long the data must be kept. `evidence` and `pinned` feed the built-in
  *paper-evidence* virtual folder.

```bash
qphase session lifecycle <session-id> reference
qphase session retention <session-id> evidence
qphase artifact lifecycle <artifact-id> archived
qphase occurrence retention <session-id> <artifact-id> pinned [--job NAME]
```

## Querying from the CLI

Every list command accepts the same filter set:

```bash
qphase session list --tag task:scan --tag-without task:wip \
    --tag-any method:cam --tag-any method:fpgen \
    --tag-descendant purpose:paper --tag-namespace model \
    --facet status=completed --range start_time=2026-08-01.. \
    --lifecycle active --retention evidence \
    --direct --limit 50 --offset 0
```

- `--tag` requires an effective tag (repeatable, AND). `--tag-any` matches if
  any of the repeated values is present. `--tag-without` excludes.
- `--tag-descendant` matches the tag or any path below it;
  `--tag-namespace` matches the whole namespace.
- `--facet k=v` filters object facets; `--range k=low..high` filters ranges
  (either bound may be empty).
- `--direct` ignores inherited tags and matches only direct assignments.

List commands also expose the derived-facet shortcuts as kind-checked flags:
`--plugin` for jobs (matches any declared plugin), `--quantity` for artifacts
(matches any produced quantity), and `--model`/`--engine`/`--has-model` for
sessions, resolved through the jobs of the session's workflow revision. The
same filters exist on `CatalogQuery` and as HTTP query parameters on the GUI
catalog route; each rejects the wrong object kind.

The object groups:

```bash
qphase project tag --add task:paper [--private]
qphase project alias "paper project" [--clear]
qphase project note "results for the paper" [--clear]
qphase workflow list [--collection NAME] [--query TEXT] [--json]
qphase workflow tag <workflow_id@revision> --add task:reviewed [--private]
qphase job list / qphase job tag <workflow_id@revision:job_name> --add ...
qphase execution tag <execution-id> --add ... [--private]   # queued only (shared)
qphase session list|tag|lifecycle|retention ...
qphase artifact list|tag|lifecycle|retention ...
qphase occurrence list [--session ID] [--artifact ID]
qphase occurrence tag <session-id> <artifact-id> --add ... [--job NAME]
```

Occurrence annotations are keyed by producing job; when one artifact occurs in
several jobs of a session, `--job` is required to disambiguate.

Move a private tag into the shared layer:

```bash
qphase tag promote <kind> <object-id> <tag>
```

## Virtual Folders and Saved Views

Built-in virtual folders group sessions by meaning: `by-model`,
`paper-evidence`, `diagnostics`, `superseded`, `cold-storage`. `by-model`
lists sessions whose workflow revision declares any model plugin; to narrow
down to one concrete model, use the `model` query filter instead. Saved
views are user-private named filters:

```bash
qphase view save review --kind session --tag task:scan --lifecycle active
qphase view list
qphase view delete review
```

## Reindex and Location Issues

The catalog is a SQLite read model at `.qphase/object_catalog.sqlite`,
rebuilt from disk truth. A cheap fingerprint probe rebuilds it automatically
when manifests or workflow files change; `qphase project reindex` forces a
rebuild and prints per-kind counts. Artifact locations that cannot be indexed
are reported as location issues: `unsupported` (unknown manifest schema),
`corrupt` (unreadable manifest), or `conflict` (two occurrences of one
artifact identity disagree on its facets). Issues are listed by
`qphase project reindex` and never silently dropped.

## Migration Boundary

The formal history migration ships in Phase 4. Today you can preview it
without writing anything:

```bash
qphase project migrate --dry-run
```

The report covers legacy alias/note imports, invalid snapshot tags,
rebuildable workflow revisions and jobs, legacy occurrence-key conversion
(convertible vs. ambiguous), duplicate artifact identities, existing job
names and artifact ids containing the reserved `:` separator, invalid
annotation documents, annotation assignments missing policy provenance,
catalog reindex parity (a full-row multiset comparison per table, not just
counts), a private-store summary, and the per-kind object
counts the migration will act on.
