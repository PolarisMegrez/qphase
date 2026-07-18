# qphase_cam

`qphase_cam` is the workspace-only coherent-amplitude matrix resource package for
QPhase. It solves

```text
L(R) = -i H(R) R + i R H(R)^dagger + D(R) = 0
```

through `engine.cam`. The engine requires one `backend`, one CAM-capable `model`,
and one `cam_solver`; any number of `cam_postprocessor` plugins may be enabled.
Explicit core `ScanSpec` axes are consumed inside the CAM engine, so large scans
remain one logical job and one fixed-capacity dataset.

The package is intentionally not published to PyPI and does not depend on
`qphase_sde`. Project-local model plugins may expose both SDE and CAM capabilities
while sharing one parameter schema and one physical definition.

See `docs/user_guide/cam/analysis.md` and the jobs under `configs/jobs/*_cam.yaml`.
