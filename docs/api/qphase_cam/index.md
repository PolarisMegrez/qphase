# qphase_cam

`qphase_cam` is a workspace-only QPhase resource package registered as
`engine.cam`.

## Plugin Namespaces

- `cam_solver`: `steady_state`, `multistability`, `deflation`,
  `batched_newton`, and `continuation`.
- `cam_postprocessor`: `rayleigh_frequency`, `hamiltonian_spectrum`,
  `jacobian_spectrum`, `physicality`, and `bifurcation`.
- `batch_planner.cam` and `result_splitter.cam_scan_splitter` integrate
  independent parameter points with the scheduler.

All plugin implementations inherit the public base class in their namespace and
declare a strict Pydantic `config_schema`.
