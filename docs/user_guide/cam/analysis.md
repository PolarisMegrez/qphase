# Coherent-Amplitude Matrices

The workspace-only `qphase_cam` package solves the steady-state matrix equation

\[
\mathcal{L}(R)=-iH(R)R+iRH(R)^\dagger+D(R)=0.
\]

Use `engine.cam` with one backend, one CAM-capable model, one `cam_solver`, and
zero or more `cam_postprocessor` plugins. Complete examples are available in
`configs/workflows/vdp_2mode/vdp_2mode_cam.yaml`,
`configs/workflows/kerr_2mode/kerr_2mode_cam.yaml`, and the corresponding model
Collections. Higher-order equilibrium search is demonstrated by
`configs/workflows/vdp_2mode/vdp_2mode_bifurcation.yaml`.

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

Each control accepts `sampling: linear|log` for discovery seeds. `linear` is
the default. `log` requires a strictly positive lower bound and is appropriate
when a threshold may span several decades, such as an induced weak-coupling
rate of order `g^2/Delta`. Sampling changes only seed placement; refinement
continues to solve for the physical control values with the configured bounds.

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

For large scans, `tile_workers` is the requested process count. When neither
`n_tiles` nor `tile_size` is configured, the solver targets four times the worker
count (with a minimum of 16 tasks), so ordinary jobs do not need a static tile
setting. Override the partition only after a workload-specific benchmark. On
Windows, the solver limits BLAS threads and may reduce the effective process
count according to available physical memory plus
`SystemConfig.scan_runtime.resources`. If a spawned pool is terminated by memory
pressure, it retries with fewer workers instead of losing the logical job
immediately. Result metadata records requested/effective worker counts, tile
count, and retries.

The multistability scan follows a continuation-assisted search rather than
treating points as unrelated roots. With `guess_bounds: auto`, it merges bounds
from representative corners and the grid center, performs a full-model global
seed search, and adds those states to every point. Each spatial tile then starts
near its center, runs a denser seed search there, and propagates already
converged cardinal-neighbor states. Empty points receive `retry_guesses`. After
the tile pass, points with fewer solutions than a neighbor are revisited with
`refine_guesses` plus surrounding states; recovered states are propagated across
tile boundaries until no further solution count increases. `n_guesses` counts
the fresh random guesses at each ordinary point; explicit, global, tile-center,
and neighbor guesses are additional. Analytic or symbolic Jacobians are used by
default and their callbacks are reused across all root attempts at a parameter
point. Once the model's declared solution capacity has been reached, the solver
stops after `capacity_patience` successful duplicate convergences (default 10);
failed guesses do not advance that counter.

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
| Local bifurcation response validation | Yes | No |
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

`CAMResult.solution_order()` constructs a non-mutating per-point ordering view.
It uses `real(R[0,0])` by default, accepts a custom callable of `(state, params)`,
and represents invalid slots with `-1`. Consumers can choose another matrix
element or derived scalar without changing the stored canonical slots.

`rayleigh_frequency` is the scalar matrix-weighted average
`Re Tr(HR) / Tr(R)`. It is not generally a PSD peak: when `R` contains more
than one mode, distinct modal frequencies can cancel in this average. The
complete complex Hamiltonian spectrum is stored as `hamiltonian_eigenvalues`,
with its real part in `mode_frequencies`. Neither field proves that the
underlying stochastic dynamics has a globally stationary distribution. The
package does not emit an ambiguous `omega` field.

The optional `coherence_pole_spectrum` postprocessor diagnoses poles of the
closed CAM correlation

```text
G_W(tau) = Tr[W exp(-i H tau) R].
```

It supports bare-mode projectors, the incoherent trace (`W=I`), and named fixed
coherent channels (`W=l l^dagger`). For each measurement it selects the
minimum-decay pole whose residue exceeds `relative_residue_floor` relative to
the strongest residue in that channel. The result contains the selected complex
eigenvalue, frequency, decay rate, residue, relative residue, validity mask,
and eigenvector condition number.

```yaml
cam_postprocessor:
  coherence_pole_spectrum:
    modes: [0, 1, 2]
    include_trace: true
    channels:
      bright: ["0j", "1+0j", "1+0j"]
      dark: ["0j", "1+0j", "-1+0j"]
    relative_residue_floor: 0.001
```

This is an independent CAM diagnostic, not a replacement for
`rayleigh_frequency`. The Rayleigh quotient is the closed correlation's
weighted phase slope at `tau=0+`; the selected coherence pole describes its
long-time component. They need not agree in a multi-pole channel. A workflow
must select either quantity according to the declared measurement semantics;
the package does not silently substitute one for the other.

The `finite_delay_carrier` postprocessor provides the continuous detector
interpolation between these two descriptions. For every configured channel and
rate `kappa`, it evaluates

```text
Omega(kappa) = integral exp(-2*kappa*tau)
                 Im[conj(G) dG/dtau] dtau
               / integral exp(-2*kappa*tau) |G|^2 dtau
```

analytically from all closed-CAM poles and residues. It also outputs the exact
zero-delay value and the generalized Rayleigh quotient; their residual is a
numerical consistency diagnostic.

```yaml
cam_postprocessor:
  finite_delay_carrier:
    modes: [0, 1, 2]
    include_trace: true
    detector_rates: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
```

As `kappa` increases, the result approaches the Rayleigh quotient. At small
`kappa`, long-lived visible poles receive more weight. The matching SDE
analyser computes the same detector functional, but its zero-delay limit is the
true SDE instantaneous carrier and may differ from CAM under closure failure.

### Rayleigh Frequencies and Coherence Poles

Let `W` be a fixed positive-semidefinite measurement matrix. `W=I` is an
incoherent trace measurement, a bare-mode projector selects one mode, and
`W=l l^dagger` describes the coherent readout `c=l^dagger alpha`. For a fixed
CAM state and a diagonalizable Hamiltonian,

\[
H=\sum_j\lambda_j r_j l_j^\dagger,\qquad
l_j^\dagger r_k=\delta_{jk},
\]

the closed correlation has the exact expansion

\[
G_{W,\mathrm{CAM}}^{(1)}(\tau)
=\operatorname{Tr}\!\left[W e^{-iH\tau}R\right]
=\sum_j A_j e^{-i\lambda_j\tau},\qquad
A_j=l_j^\dagger R W r_j.
\]

With the `phase_decreasing` convention, its phase slope at `tau=0+` is

\[
\omega_W^{R}
=\operatorname{Re}
\frac{\operatorname{Tr}(W H R)}{\operatorname{Tr}(W R)}
=\operatorname{Re}\frac{\sum_j A_j\lambda_j}{\sum_j A_j}.
\]

The trace case `W=I` is the value stored by `rayleigh_frequency`. If pole zero
is the long-time pole selected by the visibility rule, the difference is
exactly

\[
\omega_W^{R}-\operatorname{Re}\lambda_0
=\operatorname{Re}
\frac{\sum_{j\ne0}A_j(\lambda_j-\lambda_0)}{\sum_jA_j}.
\]

When `abs(A_0) > sum(abs(A_j), j != 0)`, a sufficient bound is

\[
\left|\omega_W^{R}-\operatorname{Re}\lambda_0\right|
\le
\frac{\sum_{j\ne0}|A_j|\,|\lambda_j-\lambda_0|}
{|A_0|-\sum_{j\ne0}|A_j|}.
\]

This is a statement about the closed CAM propagator only. For a stationary Ito
process with drift `-i H(R_t) alpha_t`, define

\[
G_{W,\mathrm{SDE}}^{(1)}(\tau)
=\operatorname{Tr}\!\left[W\,\mathbb{E}
\{\alpha(t+\tau)\alpha(t)^\dagger\}\right].
\]

Adapted mean-zero noise gives the exact short-time frequency

\[
\omega_{W,\mathrm{SDE}}(0^+)
=\operatorname{Re}
\frac{\mathbb{E}\,\operatorname{Tr}[W H(R_t)R_t]}
{\mathbb{E}\,\operatorname{Tr}(W R_t)}.
\]

It is generally not obtained by replacing the expectations with
`H(R_CAM) R_CAM`. Its long-time poles belong to the complete Markov generator,
not to `H(R_CAM)`. A useful diagnostic decomposition is therefore

\[
\omega_{W,\mathrm{SDE}}^{\mathrm{long}}-\omega_W^{R}
=\left(\omega_{W,\mathrm{SDE}}^{\mathrm{long}}
-\operatorname{Re}\lambda_0\right)
+\left(\operatorname{Re}\lambda_0-\omega_W^{R}\right).
\]

The second term is the closed-CAM multi-pole correction. The first contains
stochastic self-energy, closure, and finite-noise corrections and must be
estimated from stochastic data. Exact agreement at all lags is sufficient if
the stochastic correlation can be written as
`exp(-i omega_R tau) F(tau)` with real positive `F`. Symmetric zero-mean
Gaussian phase modulation can approximate this condition, but global phase
symmetry alone does not guarantee it.

Interpret a CAM/SDE frequency correspondence only within a stated domain. At a
minimum, verify stationarity; an isolated visible pole; adequate decay-rate
separation; a bounded eigenvector condition number; and robustness to the
frequency band and lag interval. Multi-state switching, weak pole residue,
nearly defective eigensystems, non-Gaussian frequency fluctuations, or an
unresolved observation time can invalidate a single-pole interpretation.
Statistical evidence should use a measurement and estimator fixed independently
of the CAM target, independent trajectories or seeds, trajectory-level
bootstrap uncertainties, and an explicit equivalence tolerance. A failure to
reject unequal frequencies is not proof of an exact identity.

The optional `petermann_spectrum` postprocessor stores aligned eigensystems for
three distinct matrices. `hamiltonian_petermann_factors` belongs to `H(R)`;
`bogoliubov_*` belongs to the doubled fluctuation block matrix; and
`monodromy_*` belongs to that doubled matrix shifted into the rotating frame
defined by `rayleigh_frequency`. The last object is a monodromy matrix, not the
canonical-coordinate CAM Jacobian stored by `jacobian_spectrum`. Eigenvalues are
ordered by real part and then imaginary part, with Petermann factors reordered
identically. Models opt into this analysis through
`cam_bogoliubov_interaction(state, params)`.

The logical array shape is `scan_shape + (capacity, n_modes, n_modes)`. NPZ
files contain the complete fixed-capacity result, while the companion CSV
flattens parameters, independent solution slots, Hermitian coordinates,
residuals, frequencies, stability, and physicality fields. Large results may be
stored as a bounded set of shards; `artifact_manifest.json` records the layout
and `CAMResult.load_dataset` restores the same logical shape.

`CAMBifurcationResult` instead has a variable candidate axis. Schema 3 stores a
candidate table plus a branch-response table linked by `candidate_index`. The
branch table contains local branch indices, `(n,k,m)`, exact exponent numerator
and denominator, perturbation side, amplitude, and the complete leading state
matrix coefficient. `to_candidate_table()`, `to_branch_table()`, and
`branch_view()` avoid direct dependence on NPZ object internals. Candidate CSV
includes canonical state coordinates and numerical diagnostics; branch CSV
includes scalar branch diagnostics, while complete matrix coefficients remain
in NPZ. Candidate and local branch indices are not global branch identifiers.

High-precision verification distinguishes `multiplicity_residual_norm`, the
residual of the repeated-root equations, from `verified_full_residual_norm`, the
residual of the complete CAM dynamics after high-precision reconstruction.
Canonical state and search-unknown decimal strings are retained for audit and
for initializing local response solves; ordinary `states` remain float64.

The optional `local_response_validation` postprocessor fixes all critical
controls and changes only the configured perturbation parameter. For each real
local branch it solves the complete CAM residual on a logarithmic epsilon grid,
then records convergence, branch continuity, physicality, Jacobian stability,
the complete-state response exponent, and the Rayleigh-frequency response.
It also evaluates the largest Hamiltonian Petermann factor at the critical
state and at every converged branch point. The response table contains
`critical_hamiltonian_petermann_max`, `hamiltonian_petermann_max`,
`hamiltonian_petermann_effective_exponent`, and
`hamiltonian_petermann_fit_exponent`. The exponent convention is
`K_H ~ |epsilon|^(-q_K)`, so a value consistent with zero means that the
Hamiltonian Petermann factor has no resolved critical scaling. This quantity
is distinct from the Jacobian critical-mode condition number reported by
`stochastic_validity`.
`rayleigh_visibility < 1e-3` is reported as `weak_projection`: the scalar readout
may look linear over a practical finite window even when its nonzero asymptotic
term has the same sublinear exponent as the complete state. Validation status
never removes or reclassifies a mathematical candidate. Results are stored in
NPZ and an additional `*_responses.csv` table. Candidate-level convergence,
continuity, physicality, and stability fractions are stored separately in
`*_response_summary.csv`; this distinguishes stability exactly at a multiple
root from stability throughout the sampled perturbation window.

The optional `stochastic_validity` postprocessor estimates whether finite
sample-matrix noise is likely to mask the deterministic local response. For a
normal Hermitian CAM with zero anomalous diffusion it pushes `D(R)` into the
canonical coordinates of the instantaneous matrix `R=alpha alpha^dagger`,
projects that covariance onto the left critical mode, and combines it with the
classified additive normal form `(n,1,0)`. It reports critical-mode
conditioning, the noncritical spectral gap, projected noise, a confining-form
check, the characteristic fluctuation scale, and an `epsilon_crossover`.
Supplying `probe_epsilon` additionally labels the point as `noise_dominated` or
`response_dominated`.

This is a regime diagnostic, not a linewidth predictor. Its noise process
inherits the fpgen moment closure. Every row records the representation, FPE
exactness, moment-closure policy, and noise semantics. Unsupported augmented,
anomalous-noise, oscillatory-critical-mode, or non-additive signatures produce
an explicit status and do not reject the bifurcation candidate. Results are
stored in NPZ and `*_stochastic_validity.csv`.

An outer bifurcation scan returns `CAMBifurcationScanResult`. It stores named
case axes, one flattened candidate table, and `candidate_offsets` so zero- and
multi-candidate cases remain distinguishable. One NPZ and companion `cases`,
`candidates`, and optional `branches` CSV files are written for the logical job;
no per-case directories or Session manifest entries are created.

Large bifurcation campaigns may isolate ordinary per-case numerical failures:

```yaml
engine:
  cam:
    case_failure_policy: record  # abort (default) or record
```

`record` emits an empty case with `case_status`, exception type and message,
flat/index coordinates, and scanned parameter values. It does not suppress
`MemoryError`, cancellation, corrupt checkpoints, or process termination.
Chunk persistence and resume remain core `SystemConfig.scan_runtime.checkpoint`
policies; the CAM engine does not discover or override system configuration.
