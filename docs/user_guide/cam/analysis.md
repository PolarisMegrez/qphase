# Coherent-Amplitude Matrices

The workspace-only `qphase_cam` package solves the steady-state matrix equation

\[
\mathcal{L}(R)=-iH(R)R+iRH(R)^\dagger+D(R)=0.
\]

Use `engine.cam` with one backend, one CAM-capable model, one `cam_solver`, and
zero or more `cam_postprocessor` plugins. Complete examples are available in
`configs/jobs/vdp_2mode_cam.yaml`, `kerr_2mode_cam.yaml`,
`crosskerr_2mode_cam.yaml`, and `kerr_3mode_cam.yaml`. Higher-order equilibrium
search is demonstrated by `configs/jobs/vdp_2mode_bifurcation.yaml`.

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
coordinate and adaptive steps define a different topology. `bifurcation`
interprets an external scan as a set of independent search cases inside one
logical job. Its scan axes may target fixed model parameters, including the base
value of the perturbation parameter, but must not overlap the adaptive
`controls` searched inside each case.

## Solvers

| Solver | Multiple states | Backend | Jacobian | Intended use |
| --- | --- | --- | --- | --- |
| `steady_state` | No | NumPy | Optional for root; unused by Cholesky | One state from one initial guess. `auto` tries root, then positive-semidefinite Cholesky. |
| `multistability` | Yes | NumPy | Optional | Recommended automatic multi-state search. Runs root/Cholesky from many guesses, clusters converged states, and enforces model capacity. |
| `deflation` | Yes | NumPy | Required | Repels Newton iterations from roots already found. Useful when ordinary multi-start repeatedly returns the same root. |
| `batched_newton` | Yes, if multiple guesses are supplied | NumPy or CuPy | Required | High-throughput parameter scans with a prepared seed set. Its default single identity guess normally finds at most one state. |
| `continuation` | One traced sheet | NumPy | Required | Follows one pseudo-arclength sheet through folds from a known initial state; it is not an automatic all-state search. |
| `bifurcation` | Variable candidates | NumPy | Exact fpgen dynamics | Jointly searches states and control parameters for equilibrium roots of multiplicity 2-4. |

`batched_newton`, `deflation`, and `continuation` require a model Jacobian.
Ordinary root and Cholesky solves can run without one. Finite-difference
Jacobians are disabled unless explicitly enabled by a supporting plugin schema.
Models may additionally implement `cam_residual_vector` and
`cam_jacobian_vector` in the canonical CAM coordinate order. These optional
callbacks avoid matrix reconstruction in root-solver hot paths; the solver
falls back to the standard H/D and Jacobian capabilities when they are absent.

### Higher-Order Equilibrium Bifurcations

`cam_solver.bifurcation` is a solver, not a postprocessor. Its `controls`
define the adaptive search domain within one case. An optional outer `ScanSpec`
creates independent cases for fixed model parameters without expanding them
into scheduler jobs. The required `target` subplugin currently supports
`equilibrium_multiplicity` with `order: 2`, `3`, or `4`; exactly `order - 1`
control parameters must be supplied. One scalar `perturbation.parameter` is
also required. It selects the physical parameter varied after the critical
point is located; it may also be one of the controls. All other parameters are
held fixed during classification.

The default `strategy.auto` evaluates every available scalar linear reduction
and unions those candidates with a full bordered Lyapunov-Schmidt search. Small
eliminated blocks use fraction-free equations; large blocks use implicit
condensed derivatives without expanding a large rational expression. This is
more expensive than selecting one reduction, but avoids making the selected
order parameter an undocumented coverage assumption. `strategy.reduced` uses
all available scalar paths without the full search, while `strategy.full`
requires the bordered path. Bifurcation discovery currently supports domain
and upstream `seeds`; a continuation discovery option is not advertised until
it performs actual branch tracing.

The default `bifurcation_classifier.scaling_signature` expands the reduced
equation in the center coordinate `x` and the selected perturbation `epsilon`.
For a lower-Newton-edge balance

```text
a epsilon^k x^m + b x^n = 0
```

it records the named signature `(n,k,m)` and the exact rational exponent
`k/(n-m)`. Sublinear response is therefore `k < n-m`; a transcritical normal
form has `(2,1,1)` and exponent one. Classification is for the complete normal
or augmented CAM state matrix. A model-specific experimental readout may
cancel its leading matrix coefficient and is a separate postprocessing task.

Float64 solves only discover candidates. Accepted candidates are refined from
the exact fpgen expressions with mpmath, starting at `verification.initial_digits`
and increasing up to `max_digits`. A `verified` candidate has passed the
high-precision root conditions, complete CAM residual, regularity,
non-degeneracy, and physicality checks. This is not interval certification and
does not prove that no other candidates exist. Result metadata records search
coverage and fpgen provenance. Coverage is reported separately as structural
reduction coverage, finite numerical-search coverage, physical-domain filtering,
and treatment of singular reduction paths. These fields are an audit trail, not
a completeness certificate. Models may implement `cam_bifurcation_scales()` to
provide physically meaningful state scales for seed generation and reduced-root
normalization; otherwise the solver uses unit scales.

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
| `bifurcation` | Yes | No |
| Rayleigh/Hamiltonian/physicality postprocessing | Yes | Yes, with result arrays transferred to CPU |
| Jacobian spectrum | Yes | Yes |

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

For nonlinear models, the CAM equation is a moment closure: nonlinear moments
such as `E[|alpha|^2 alpha alpha^dagger]` are represented using `R`. A locally
stable, physical CAM root therefore does not by itself establish global
confinement or the existence of a normalizable stationary SDE distribution.

## Results

Every result uses the model's fixed solution capacity. `valid_mask` and
`solution_count` identify populated slots. Within each parameter point, slots
are sorted by `real(R[0,0])` unless the model overrides the key. A slot is not a
global branch identifier and no continuity between neighboring scan points is
implied.

`rayleigh_frequency` is the scalar matrix-weighted average
`Re Tr(HR) / Tr(R)`. It is not generally a PSD peak: when `R` contains more
than one mode, distinct modal frequencies can cancel in this average. The
complete complex Hamiltonian spectrum is stored as `hamiltonian_eigenvalues`,
with its real part in `mode_frequencies`. Neither field proves that the
underlying stochastic dynamics has a globally stationary distribution. The
package does not emit an ambiguous `omega` field.

The logical array shape is `scan_shape + (capacity, n_modes, n_modes)`. NPZ
files contain the complete fixed-capacity result, while the companion CSV
flattens parameters, independent solution slots, Hermitian coordinates,
residuals, frequencies, stability, and physicality fields. Large results may be
stored as a bounded set of shards; `artifact_manifest.json` records the layout
and `CAMResult.load_dataset` restores the same logical shape.

`CAMBifurcationResult` instead has a variable candidate axis. Schema 2 stores a
candidate table plus a branch-response table linked by `candidate_index`. The
branch table contains local branch indices, `(n,k,m)`, exact exponent numerator
and denominator, perturbation side, amplitude, and the complete leading state
matrix coefficient. `to_candidate_table()`, `to_branch_table()`, and
`branch_view()` avoid direct dependence on NPZ object internals. Candidate CSV
includes canonical state coordinates and numerical diagnostics; branch CSV
includes scalar branch diagnostics, while complete matrix coefficients remain
in NPZ. Candidate and local branch indices are not global branch identifiers.

An outer bifurcation scan returns `CAMBifurcationScanResult`. It stores named
case axes, one flattened candidate table, and `candidate_offsets` so zero- and
multi-candidate cases remain distinguishable. One NPZ and companion `cases`,
`candidates`, and optional `branches` CSV files are written for the logical job;
no per-case run directories are created.
