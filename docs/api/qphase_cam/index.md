# qphase_cam

`qphase_cam` is a workspace-only QPhase resource package registered as
`engine.cam`.

## Plugin Namespaces

- `cam_solver`: `steady_state`, `multistability`, `deflation`,
  `batched_newton`, and `continuation`.
- `cam_postprocessor`: `rayleigh_frequency`, `hamiltonian_spectrum`,
  `jacobian_spectrum`, `physicality`, and `bifurcation`.

All plugin implementations inherit the public base class in their namespace and
declare a strict Pydantic `config_schema`.

## Scan Runtime

`engine.cam` receives the core `ParameterGrid` directly. Native batch solvers
retain their resource-specific tile or NumPy/CuPy execution strategies;
pointwise solvers use the shared CAM helper. `CAMResult` implements the dataset
protocol with named axes and fixed solution capacity, and persists as a single
or sharded logical artifact without scheduler point jobs.
