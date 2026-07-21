# Coherent-Amplitude Matrices

The workspace-only `qphase_cam` package solves the steady-state matrix equation

\[
\mathcal{L}(R)=-iH(R)R+iRH(R)^\dagger+D(R)=0.
\]

Use `engine.cam` with one backend, one CAM-capable model, one `cam_solver`, and
zero or more `cam_postprocessor` plugins. Complete examples are available in
`configs/jobs/vdp_2mode_cam.yaml`, `kerr_2mode_cam.yaml`,
`crosskerr_2mode_cam.yaml`, and `kerr_3mode_cam.yaml`.

## Parameter Scans

CAM uses the core `ScanSpec`. Model parameters remain scalar in the plugin
configuration, while named axes target those parameters explicitly:

```yaml
scan:
  combine: cartesian
  axes:
    omega_a:
      target: model.vdp_2mode.omega_a
      linspace: {start: -0.3, stop: 0.3, num: 101}
    gamma_b:
      target: model.vdp_2mode.gamma_b
      linspace: {start: 0.2, stop: 1.1, num: 101}
```

`values`, `linspace`, and `logspace` axes are supported, with Cartesian or
zipped combination. Axis declaration order defines the Cartesian result shape.
The example is one logical job with shape `(101, 101)`, not 10,201 scheduler
jobs.

`multistability` and `batched_newton` consume the grid through their native tile
or batch strategies. `steady_state` and `deflation` use the CAM pointwise scan
helper. `continuation` rejects an external `ScanSpec` because its continuation
coordinate and adaptive steps define a different topology.

## Solvers

| Solver | Multiple states | Backend | Jacobian | Intended use |
| --- | --- | --- | --- | --- |
| `steady_state` | No | NumPy | Optional for root; unused by Cholesky | One state from one initial guess. `auto` tries root, then positive-semidefinite Cholesky. |
| `multistability` | Yes | NumPy | Optional | Recommended automatic multi-state search. Runs root/Cholesky from many guesses, clusters converged states, and enforces model capacity. |
| `deflation` | Yes | NumPy | Required | Repels Newton iterations from roots already found. Useful when ordinary multi-start repeatedly returns the same root. |
| `batched_newton` | Yes, if multiple guesses are supplied | NumPy or CuPy | Required | High-throughput parameter scans with a prepared seed set. Its default single identity guess normally finds at most one state. |
| `continuation` | One traced sheet | NumPy | Required | Follows one pseudo-arclength sheet through folds from a known initial state; it is not an automatic all-state search. |

`batched_newton`, `deflation`, and `continuation` require a model Jacobian.
Ordinary root and Cholesky solves can run without one. Finite-difference
Jacobians are disabled unless explicitly enabled by a supporting plugin schema.
Models may additionally implement `cam_residual_vector` and
`cam_jacobian_vector` in the canonical CAM coordinate order. These optional
callbacks avoid matrix reconstruction in root-solver hot paths; the solver
falls back to the standard H/D and Jacobian capabilities when they are absent.

For an unknown multistable system, start with `multistability`. Set
`guess_bounds: auto` to infer diagonal-balance scales and sample bounded plus
heavy-tailed Hermitian guesses. This is automatic guess generation, not a proof
that every root has been found. Once representative states are known, use them
as the guess set for `batched_newton` when scanning many points, or as the start
of `continuation` when sheet topology and folds matter. Use `deflation` when
multi-start remains dominated by an already known root.

`method: root` searches unconstrained Hermitian matrices and may return
non-physical mathematical roots. `method: cholesky` enforces
positive-semidefinite states but can miss non-PSD roots and may have different
basins of attraction.

For large scans, `tile_workers` is the requested process count and `n_tiles`
controls the bounded number of scan tasks. Keep `n_tiles` larger than the worker
count so work remains balanced; the VDP 101 x 101 job uses 24 requested workers
and 288 tiles, matching the pre-migration scanner. On Windows, the solver limits
BLAS threads and may reduce the effective process count according to available
physical memory plus `SystemConfig.scan_runtime.resources`. If a spawned pool is
terminated by memory pressure, it retries with fewer workers instead of losing
the logical job immediately. Result metadata records requested/effective worker
counts, tile count, and retries.

The multistability scan follows a continuation-assisted search rather than
treating points as unrelated roots. With `guess_bounds: auto`, it merges bounds
from representative corners and the grid center, performs a full-model global
seed search, and adds those states to every point. Each spatial tile then starts
near its center and propagates already converged cardinal-neighbor states. Empty
points receive `retry_guesses`; after the first pass, solution-count jumps are
revisited with `refine_guesses` plus surrounding states. `n_guesses` counts the
fresh random guesses at each point; explicit, global, and neighbor guesses are
additional. Analytic or symbolic Jacobians are used by default and their
callbacks are reused across all root attempts at a parameter point. Once the
model's declared solution capacity has been reached, the solver stops after
`capacity_patience` successful duplicate convergences (default 10); failed
guesses do not advance that counter.

## Backend Support

The CAM engine supports a CuPy backend only through `batched_newton`. The VDP2,
Kerr2, cross-Kerr2, and Kerr3 analytic Jacobians are backend-aware.
The other solvers explicitly reject CuPy because they depend on SciPy root
finding or CPU pseudo-arclength logic.

| Component | NumPy | CuPy |
| --- | --- | --- |
| `steady_state` | Yes | No |
| `multistability` | Yes, including process tiles | No |
| `deflation` | Yes | No |
| `batched_newton` | Yes | Yes |
| `continuation` | Yes | No |
| Rayleigh/Hamiltonian/physicality postprocessing | Yes | Yes, with result arrays transferred to CPU |
| Jacobian spectrum | Yes | Yes |
| Bifurcation refinement | Yes, on continuation output | No |

`batched_newton` transfers converged states back to NumPy when constructing
`CAMResult`; persistence and most postprocessing are therefore CPU-side even
when Newton iterations and Jacobian solves run on the GPU.

## Physicality

For every valid, converged solution, the `physicality` postprocessor evaluates:

1. Hermiticity: `R` must agree with `R^dagger` within
   `hermitian_tolerance`.
2. Positive semidefiniteness: the minimum eigenvalue from `eigvalsh(R)` must be
   at least `-psd_tolerance`.
3. Equation accuracy: the stored Liouvillian residual must not exceed
   `residual_tolerance`.

`is_physical` is true only when all three conditions hold. The matrix is tested
for positive **semidefiniteness**, not strict positive definiteness: zero
eigenvalues are allowed for `R = alpha alpha^dagger`, and small negative values
within `psd_tolerance` are treated as numerical error. The output also records
`is_hermitian`, `is_positive_semidefinite`, `minimum_state_eigenvalue`, and
`residual_within_tolerance` separately. It does not currently test trace
normalization or positive semidefiniteness of `D(R)`.

## Results

Every result uses the model's fixed solution capacity. `valid_mask` and
`solution_count` identify populated slots. Within each parameter point, slots
are sorted by `real(R[0,0])` unless the model overrides the key. A slot is not a
global branch identifier and no continuity between neighboring scan points is
implied.

The preferred scalar frequency is `rayleigh_frequency`. The complete complex
Hamiltonian spectrum is stored as `hamiltonian_eigenvalues`, with its real part
in `mode_frequencies`. The package does not emit an ambiguous `omega` field.

The logical array shape is `scan_shape + (capacity, n_modes, n_modes)`. NPZ
files contain the complete fixed-capacity result, while the companion CSV
flattens parameters, independent solution slots, Hermitian coordinates,
residuals, frequencies, stability, and physicality fields. Large results may be
stored as a bounded set of shards; `artifact_manifest.json` records the layout
and `CAMResult.load_dataset` restores the same logical shape.
