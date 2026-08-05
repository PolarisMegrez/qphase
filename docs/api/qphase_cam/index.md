# qphase_cam

`qphase_cam` is a workspace-only QPhase resource package registered as
`engine.cam`.

## Plugin Namespaces

- `cam_solver`: `steady_state`, `multistability`, `deflation`,
  `batched_newton`, `continuation`, and `bifurcation`.
- `cam_postprocessor`: `rayleigh_frequency`, `hamiltonian_spectrum`,
  `jacobian_spectrum`, and `physicality`.

`cam_solver.bifurcation` owns four subplugin slots:
`bifurcation_target`, `bifurcation_strategy`, `bifurcation_discovery`, and
`bifurcation_classifier`.
Its first target is equilibrium multiplicity of order 2-4. It uses fpgen exact
dynamics for scalar reduction or a full bordered-system fallback and returns a
`CAMBifurcationResult`, not a fixed-capacity `CAMResult`.

All plugin implementations inherit the public base class in their namespace and
declare a strict Pydantic `config_schema`.

## Scan Runtime

`engine.cam` receives the core `ParameterGrid` directly. Native batch solvers
retain their resource-specific tile or NumPy/CuPy execution strategies;
pointwise solvers use the shared CAM helper. `CAMResult` implements the dataset
protocol with named axes and fixed solution capacity, and persists as a single
or sharded logical artifact without scheduler point jobs.

Bifurcation control domains are adaptive solver inputs and are intentionally
incompatible with an external `ScanSpec`. Candidate results use one variable
candidate axis and support `point_view((candidate,))`, NPZ, and CSV output.
One perturbation parameter is required. Classified reduced candidates expose a
branch-response table with named `(n,k,m)` signatures and complete leading CAM
state-matrix coefficients.
