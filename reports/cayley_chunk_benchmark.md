# Cayley-Maruyama Chunk Kernel Benchmark

Date: 2026-07-14

Device: NVIDIA T1000, 4096 MiB, driver 582.08

Workload:

- 15 scan points x 100 trajectories = 1500 independent trajectories
- 512 fixed steps
- `dt=0.1`, `complex64`
- identical pre-generated Wiener increments for every comparison
- no trajectory saves or FFT in the timed region

| path | wall time (s) | steps/s | speedup |
|---|---:|---:|---:|
| fused single-step | 0.111420 | 4,595 | 1.0x |
| chunk 64 | 0.002383 | 214,837 | 46.75x |
| chunk 128 | 0.001326 | 386,182 | 84.04x |
| chunk 256 | 0.001082 | 473,329 | 103.00x |

This isolates integration kernel launch and parameter-preparation overhead. A full
engine run also includes random-number generation, progress reporting, trajectory
saves, and PSD analysis, so end-to-end speedups will be lower. `chunk_steps=128`
is the initial production candidate; 64 and 256 remain benchmark options.
