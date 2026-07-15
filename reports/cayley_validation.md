# Cayley-Maruyama validation

Date: 2026-07-15

The checks below used the model and numerical settings from the VDP production
job, but reduced the scan to one or two parameter points. The full 31-point,
100-trajectory production scan was not run.

## Deterministic frequency

Settings: `D=0`, `dt=0.1`, one trajectory, fused CuPy chunk kernel with
`chunk_steps=128`. Frequency was measured from the phase increment after the
transient rather than from a PSD.

| omega_a | dtype | continuous | measured | error vs continuous |
|---:|:---|---:|---:|---:|
| 0.001 | complex64 | 0.063331156 | 0.063336038 | +4.882e-6 |
| 0.001 | complex128 | 0.063331156 | 0.063330944 | -2.117e-7 |
| 0.100 | complex64 | 0.329819605 | 0.329789768 | -2.984e-5 |

At `omega_a=0.1`, the measured result differed from the Cayley discrete
dispersion prediction by `5.606e-8`. At `omega_a=0.001`, the `complex128`
result differed from that prediction by `5.940e-15`; the larger `complex64`
residual is therefore accumulated roundoff rather than an integration bias.

## Short stochastic PSD

Settings: `omega_a=0.1`, `D=1`, `dt=0.1`, `t1=50000`, 20 trajectories,
seed 42, and the production candidate fused chunk kernel. This is about 0.65%
of the trajectory-step count of the 31-point, 100-trajectory candidate job.

| save_stride | samples | abs(center) | linewidth | R2 |
|---:|---:|---:|---:|---:|
| 20 | 25001 | 0.329722972 | 1.264e-4 | 0.845086 |
| 40 | 12501 | 0.329761626 | 6.503e-5 | 0.997997 |

The center difference was `3.865e-5`, below the short run's angular-frequency
bin spacing of approximately `1.257e-4`. Both centers eliminate the roughly
`0.005` Euler frequency offset. The linewidth comparison is not conclusive at
this shortened duration because the narrow line is near the FFT resolution.
