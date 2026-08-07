# fpgen Integration Contract

`qphase_cam` uses fpgen as its symbolic dynamics and exact-reduction provider.
fpgen remains a separate local project. QPhase must not copy its package source
into `packages/` or depend on fpgen implementation modules.

## Supported Runtime

The reviewed contract is:

- fpgen release series: `0.5.x`
- model schema: `2.0`
- moment dynamics API: `1.0`
- reduction API: `1.1`
- state layouts: `hermitian-declared-index-v1` and
  `hermitian-normal-anomalous-declared-index-v2`

The QPhase workspace resolves fpgen through the editable local source declared
in `[tool.uv.sources]`. `qphase_cam` declares `fpgen>=0.5,<0.6`, but neither
workspace-only package is currently published to PyPI.

## Model Provider

A CAM model that supports symbolic bifurcation analysis implements
`cam_fpgen_dynamics()` and returns the public `fpgen.CovarianceDynamics` type.
The returned model must satisfy these conditions:

- its state IDs use QPhase canonical Hermitian order: diagonal entries, upper
  triangular real entries, then upper triangular imaginary entries;
- state indices are contiguous;
- its parameter names exactly match `model.params`;
- it exposes a state matrix and one of the supported state layouts;
- frequency parameters use the `real` domain and constrained physical
  parameters declare their fpgen domain explicitly.

Model modules may use names exported by `fpgen.__all__` to construct dynamics.
They must not import `fpgen.covariance`, `fpgen.numerical`, or other internal
modules.

## CAM Adapter Boundary

All solvers, postprocessors, and tests consume `FPGenDynamicsAdapter`; the raw
`CovarianceDynamics` object is private. The adapter owns:

- runtime and layout validation;
- parameter ordering and conversion;
- compiled NumPy RHS, state Jacobian, parameter Jacobian, and state matrix;
- exact directional derivatives and high-precision callables;
- regular linear-reduction search, plans, and materialization;
- symbolic state/parameter coordinates and expressions needed by CAM tests;
- fpgen provenance and closure diagnostics.

Code outside `qphase_cam.core.fpgen` must not access `_dynamics`. Reduction
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

When fpgen changes its public API:

1. Change fpgen's package version and the relevant schema/API version constant.
2. Update and run `tests/qphase_cam/test_fpgen_contract.py` against that fpgen
   revision. An unexplained compatibility-test failure blocks integration.
3. Run `uv run python tools/generate_fpgen_api_snapshot.py` and review the new
   source revision, signatures, and fields in `reports/fpgen_api_snapshot.md`.
4. Update `FPGenDynamicsAdapter` first, then this document. Solver and
   postprocessor code must continue to use only the adapter.
5. Run `uv run pytest tests/qphase_cam` before accepting the new contract.

The snapshot is diagnostic evidence, not a substitute for executable
compatibility tests. Do not loosen version checks merely to accept a new fpgen
revision.
