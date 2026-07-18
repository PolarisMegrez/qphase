# Coherent-Amplitude Matrices

The workspace-only `qphase_cam` package solves the steady-state matrix equation

\[
\mathcal{L}(R)=-iH(R)R+iRH(R)^\dagger+D(R)=0.
\]

Use `engine.cam` with one backend, one CAM-capable model, one `cam_solver`, and
zero or more `cam_postprocessor` plugins. Complete examples are available in
`configs/jobs/vdp_2mode_cam.yaml`, `kerr_2mode_cam.yaml`, and
`kerr_3mode_cam.yaml`.

## Solvers

- `steady_state` provides SciPy root and positive-semidefinite Cholesky paths.
- `multistability` performs bounded or heavy-tailed multi-start searches.
- `deflation` repels Newton iterations from already discovered roots.
- `batched_newton` runs independent parameter points on NumPy or CuPy.
- `continuation` traces a pseudo-arclength sequence on NumPy.

`batched_newton`, `deflation`, and `continuation` require a model Jacobian.
Ordinary root and Cholesky solves can run without one. Finite-difference
Jacobians are disabled unless explicitly enabled by a supporting plugin schema.

## Results

Every result uses the model's fixed solution capacity. `valid_mask` and
`solution_count` identify populated slots. Within each parameter point, slots
are sorted by `real(R[0,0])` unless the model overrides the key. A slot is not a
global branch identifier and no continuity between neighboring scan points is
implied.

The preferred scalar frequency is `rayleigh_frequency`. The complete complex
Hamiltonian spectrum is stored as `hamiltonian_eigenvalues`, with its real part
in `mode_frequencies`. The package does not emit an ambiguous `omega` field.

NPZ files contain the complete fixed-capacity result. The companion CSV flattens
parameters, independent solution slots, Hermitian matrix coordinates, residuals,
frequencies, stability, and physicality fields for inspection.
