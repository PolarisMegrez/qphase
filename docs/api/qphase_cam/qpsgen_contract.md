# qpsgen Integration Contract

`qphase_cam` uses qpsgen as its symbolic dynamics and exact-reduction provider.
qpsgen remains a separately distributed project. QPhase must not copy its source
into `packages/` or depend on qpsgen implementation modules.

## Supported Runtime

The reviewed contract is:

- qpsgen release series: `0.5.x`
- model schema: `2.0`
- moment dynamics API: `1.0`
- reduction API: `1.1`
- state layouts: `hermitian-declared-index-v1` and
  `hermitian-normal-anomalous-declared-index-v2`

The committed workspace and published QPhase packages resolve the declared
`qpsgen>=0.5,<0.6` dependency from the package index. Contributors developing
both repositories together may temporarily install qpsgen in editable mode,
but must not commit a local path override.

## Model Provider

A CAM model that supports symbolic bifurcation analysis implements
`cam_qpsgen_dynamics()` and returns the public `qpsgen.CovarianceDynamics` type.
The returned model must satisfy these conditions:

- its state IDs use QPhase canonical Hermitian order: diagonal entries, upper
  triangular real entries, then upper triangular imaginary entries;
- state indices are contiguous;
- its parameter names exactly match `model.params`;
- it exposes a state matrix and one of the supported state layouts;
- frequency parameters use the `real` domain and constrained physical
  parameters declare their qpsgen domain explicitly.

Model modules may use names exported by `qpsgen.__all__` to construct dynamics.
They must not import `qpsgen.covariance`, `qpsgen.numerical`, or other internal
modules.

## CAM Adapter Boundary

All solvers, postprocessors, and tests consume `QPSGenDynamicsAdapter`; the raw
`CovarianceDynamics` object is private. The adapter owns:

- runtime and layout validation;
- parameter ordering and conversion;
- compiled NumPy RHS, state Jacobian, parameter Jacobian, and state matrix;
- exact directional derivatives and high-precision callables;
- regular linear-reduction search, plans, and materialization;
- symbolic state/parameter coordinates and expressions needed by CAM tests;
- qpsgen provenance and closure diagnostics.

Code outside `qphase_cam.core.qpsgen` must not access `_dynamics`. Reduction
plans and materialized reductions may cross the adapter because CAM's reduction
engine consumes their documented public mathematical fields.

## Numerical Shapes

For state size `n`, parameter count `p`, and state-matrix shape `(m, m)`, the
compiled NumPy contract is:

| Callable | Input | Output |
| --- | --- | --- |
| `rhs` | `(..., n)`, `(..., p)` | `(..., n)` |
| `jacobian` | `(..., n)`, `(..., p)` | `(..., n, n)` |
| `parameter_jacobian` | `(..., n)`, `(..., p)` | `(..., n, p)` |
| `state_matrix` | `(..., n)`, `(..., p)` | `(..., m, m)` |

Reduction search must return `ReductionSearchResult` with candidates, coverage,
truncation reasons, and `manifest()`. A selected candidate must support
`linear_reduce(candidate=...)`; its plan must support fraction-free
`materialize()` and expose the fields used by `qphase_cam.core.reduction`.
Every candidate exposes a stable, serializable `chart_id`
(`ret:<retained_ids>|eq:<retained_equations>`) that distinguishes equation
partitions sharing the same retained variables, enabling cross-chart
deduplication and provenance. The search result also carries
`rejected_partitions` entries (per-partition rejection reason and retained
coordinates), and `manifest()` reports `rejected_partition_count` and
`materialization_skipped_oversized` (materialization skipped when the
eliminated block exceeds dimension 3). Materialization errors are diagnostics,
not partition rejections: `materialization_failures` records them by `chart_id`
and `manifest()` reports `materialization_failure_count`.
Search covers regular affine-elimination branches only. Singular branches are
not implied to be absent when coverage is exhaustive.

## Change Procedure

When qpsgen changes its public API:

1. Change qpsgen's package version and the relevant schema/API version constant.
2. Update and run `tests/qphase_cam/test_qpsgen_contract.py` against that qpsgen
   revision. An unexplained compatibility-test failure blocks integration.
3. Run `uv run python tools/generate_qpsgen_api_snapshot.py` and review the new
   source revision, signatures, and fields in `reports/qpsgen_api_snapshot.md`.
4. Update `QPSGenDynamicsAdapter` first, then this document. Solver and
   postprocessor code must continue to use only the adapter.
5. Run `uv run pytest tests/qphase_cam` before accepting the new contract.

The snapshot is diagnostic evidence, not a substitute for executable
compatibility tests. Do not loosen version checks merely to accept a new qpsgen
revision.
