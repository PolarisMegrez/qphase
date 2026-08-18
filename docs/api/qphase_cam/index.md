# qphase_cam

`qphase_cam` is a workspace-only QPhase resource package registered as
`engine.cam`.

## Plugin Namespaces

- `cam_solver`: `steady_state`, `multistability`, `deflation`,
  `batched_newton`, `continuation`, and `bifurcation`.
- `cam_postprocessor`: `rayleigh_frequency`, `hamiltonian_spectrum`,
  `coherence_pole_spectrum`, `jacobian_spectrum`, `physicality`,
  `local_response_validation`, and `stochastic_validity`.

`cam_solver.bifurcation` owns four subplugin slots:
`bifurcation_target`, `bifurcation_strategy`, `bifurcation_discovery`, and
`bifurcation_classifier`.
Its first target is equilibrium multiplicity of order 2-4. It uses fpgen exact
dynamics for scalar reduction or a full bordered-system fallback and returns a
`CAMBifurcationResult`, not a fixed-capacity `CAMResult`. An outer parameter
grid returns a ragged `CAMBifurcationScanResult`.

All plugin implementations inherit the public base class in their namespace and
declare a strict Pydantic `config_schema`.

## External Symbolic Contract

The workspace-only fpgen integration is isolated behind
`FPGenDynamicsAdapter`. See the [fpgen integration contract](fpgen_contract.md)
for the supported versions, numerical layouts, reduction API, and upgrade
procedure.

## Scan Runtime

`engine.cam` receives the core `ParameterGrid` directly. Native batch solvers
retain their resource-specific tile or NumPy/CuPy execution strategies;
pointwise solvers use the shared CAM helper. `CAMResult` implements the dataset
protocol with named axes and fixed solution capacity, and persists as a single
or sharded logical artifact without scheduler point jobs.

Bifurcation control domains remain adaptive solver inputs. An external
`ScanSpec` may vary fixed model parameters across independent cases in one
logical job, but its axes cannot overlap those controls. Ragged scan results use
`candidate_offsets` to preserve empty cases and support point views, NPZ, and
case/candidate/branch CSV output. One perturbation parameter is required.
Classified reduced candidates expose a branch-response table with named
`(n,k,m)` signatures and complete leading CAM state-matrix coefficients.

`engine.cam.case_failure_policy` controls independent bifurcation-scan case
failures. The default `abort` preserves fail-fast behavior; `record` stores an
empty case with structured error metadata and continues. Core checkpoint and
resume policy remains part of `SystemConfig`.
