# qphase_cam

`qphase_cam` is a workspace-only QPhase resource package registered as
`engine.cam`.

## Plugin Namespaces

- `cam_solver`: `steady_state`, `multistability`, `deflation`,
  `batched_newton`, `continuation`, and `bifurcation`.
- `cam_postprocessor`: `rayleigh_frequency`, `hamiltonian_spectrum`,
  `coherence_pole_spectrum`, `finite_delay_carrier`, `jacobian_spectrum`, `physicality`,
  `local_response_validation`, and `stochastic_validity`.

`cam_solver.bifurcation` owns four subplugin slots:
`bifurcation_target`, `bifurcation_strategy`, `bifurcation_discovery`, and
`bifurcation_classifier`.
Its first target is equilibrium multiplicity of order 2-4. It uses qpsgen exact
dynamics for scalar reduction or a full bordered-system fallback and writes typed
data products, not a result object: one case is a ragged `candidates` product
(candidate identity, control values, the physical and Jacobian diagnostics as JSON
classification/audit columns, and the branch table), and an outer parameter grid
adds `case_parameters`, `case_metadata`, `candidate_offsets` and
`group_scan_index`, so an empty case stays visible instead of disappearing.

A consumer reads that product through
`qphase_cam.postprocessor.candidates.candidate_view`, which exposes the case
metadata, the case parameters, the perturbation design and the
`scaling_signatures` branch table; `candidate_parameters` and `candidate_state`
take that view. The fixed-capacity result of `steady_state`/`batched_newton` is the
`state`/`state_scan` product instead: a `scan` axis, a fixed `solution` capacity and
typed parameter columns. Both products are read with `qphase.data.load_bundle` and
the Artifact v4 manifest. The pre-2.0 `CAMResult`, `CAMBifurcationResult` and
`CAMBifurcationScanResult` payloads are not runtime types: the package root does not
export them, and they are reachable only through `qphase_cam.data.legacy` (one-time
historical import) and `qphase_cam.data.compare` (export comparison).

All plugin implementations inherit the public base class in their namespace and
declare a strict Pydantic `config_schema`.

## External Symbolic Contract

The workspace-only qpsgen integration is isolated behind
`QPSGenDynamicsAdapter`. See the [qpsgen integration contract](qpsgen_contract.md)
for the supported versions, numerical layouts, reduction API, and upgrade
procedure.

## Scan Runtime

`engine.cam` receives the core `ParameterGrid` directly. Native batch solvers
retain their resource-specific tile or NumPy/CuPy execution strategies;
pointwise solvers use the shared CAM helper. A finished job persists typed products
— `state`/`state_scan` for fixed-capacity solutions and `candidates` for ragged
candidate sets — as an Artifact v4 bundle (npz/3 products plus an
`artifact_manifest.json`); `qphase.data.load_bundle` reads them back, and there is no
scheduler point job and no per-result loader.

Bifurcation control domains remain adaptive solver inputs. An external
`ScanSpec` may vary fixed model parameters across independent cases in one
logical job, but its axes cannot overlap those controls. Ragged scan products use
`candidate_offsets` to preserve empty cases, so a candidate's case comes from the
offsets alone and `state_rows`/`candidate_view` need no second index. One
perturbation parameter is required. Classified reduced candidates expose their
branch-response table in the candidate's `classification["scaling_signatures"]`: the
named `(n,k,m)` orders per signature plus the complete leading CAM state-matrix
coefficients of every local branch.

`engine.cam.case_failure_policy` controls independent bifurcation-scan case
failures. The default `abort` preserves fail-fast behavior; `record` stores an
empty case with structured error metadata and continues. Core checkpoint and
resume policy remains part of `SystemConfig`.
