# CuPy Overhead Diagnosis Report

Configuration: n_time=1000, dt=0.001, model=vdp_2mode, seed=42

## 1. Wall-clock time by backend and trajectory count

| n_traj | numpy (s) | cupy (s) | cupy/numpy |
|--------|-----------|----------|------------|
| 100 | 0.0923 | 1.4105 | 15.27x |
| 500 | 0.1658 | 1.2017 | 7.25x |
| 1000 | 0.2642 | 1.4546 | 5.50x |
| 5000 | 0.9471 | 1.1967 | 1.26x |
| 10000 | 2.4458 | 1.2622 | 0.52x |

## 2. CuPy timing breakdown (host-side timers)

| n_traj | total (s) | drift (us/step) | diffusion (us/step) | contract (us/step) | other (us/step) |
|--------|-----------|-----------------|---------------------|--------------------|-----------------|
| 100 | 1.4105 | 283.19 | 329.90 | 481.05 | 316.36 |
| 500 | 1.2017 | 258.16 | 302.40 | 429.38 | 211.74 |
| 1000 | 1.4546 | 318.44 | 375.21 | 510.96 | 249.93 |
| 5000 | 1.1967 | 258.30 | 310.67 | 427.04 | 200.67 |
| 10000 | 1.2622 | 277.66 | 332.80 | 448.82 | 202.91 |

## 3. CuPy percentage breakdown

| n_traj | drift % | diffusion % | contract % | other % |
|--------|---------|-------------|------------|---------|
| 100 | 20.1 | 23.4 | 34.1 | 22.4 |
| 500 | 21.5 | 25.2 | 35.7 | 17.6 |
| 1000 | 21.9 | 25.8 | 35.1 | 17.2 |
| 5000 | 21.6 | 26.0 | 35.7 | 16.8 |
| 10000 | 22.0 | 26.4 | 35.6 | 16.1 |

## 4. Memory & launch estimates (CuPy only)

| n_traj | peak device bytes | est. kernel launches |
|--------|-------------------|----------------------|
| 100 | 6656 | 25000 |
| 500 | 32256 | 25000 |
| 1000 | 64000 | 25000 |
| 5000 | 320000 | 25000 |
| 10000 | 640000 | 25000 |

## 5. Interpretation checklist

- If **other %** is large at small `n_traj`, host-side Python loop overhead dominates.
- If **contract %** is large, `matmul`/`einsum` launch cost dominates; kernelized contraction helps.
- If **diffusion %** is large, the per-step Python diffusion function is the bottleneck; fused drift+diffusion kernel helps.
- If **peak device bytes** grows linearly with `n_traj`, memory allocation churn is significant; pre-allocated buffers help.

## 6. Raw JSON

```json
[
  {
    "backend": "numpy",
    "n_traj": 100,
    "wall_seconds": 0.092342700008885,
    "success": true,
    "breakdown": {
      "total": 0.092342700008885,
      "drift": 0.010350700074923225,
      "diffusion": 0.014123299974016845,
      "contract": 0.04685380002774764,
      "n_steps": 1000,
      "peak_device_bytes": null
    }
  },
  {
    "backend": "numpy",
    "n_traj": 500,
    "wall_seconds": 0.16578730000765063,
    "success": true,
    "breakdown": {
      "total": 0.16578730000765063,
      "drift": 0.014224000115063973,
      "diffusion": 0.018605700141051784,
      "contract": 0.09240450036304537,
      "n_steps": 1000,
      "peak_device_bytes": null
    }
  },
  {
    "backend": "numpy",
    "n_traj": 1000,
    "wall_seconds": 0.264239500000258,
    "success": true,
    "breakdown": {
      "total": 0.264239500000258,
      "drift": 0.02074670021829661,
      "diffusion": 0.024625499921967275,
      "contract": 0.151176899598795,
      "n_steps": 1000,
      "peak_device_bytes": null
    }
  },
  {
    "backend": "numpy",
    "n_traj": 5000,
    "wall_seconds": 0.9470807000034256,
    "success": true,
    "breakdown": {
      "total": 0.9470807000034256,
      "drift": 0.05586969984869938,
      "diffusion": 0.058544099927530624,
      "contract": 0.5824436001857975,
      "n_steps": 1000,
      "peak_device_bytes": null
    }
  },
  {
    "backend": "numpy",
    "n_traj": 10000,
    "wall_seconds": 2.4458239000086905,
    "success": true,
    "breakdown": {
      "total": 2.4458239000086905,
      "drift": 0.11485529974743258,
      "diffusion": 0.11951790000603069,
      "contract": 1.6393752001895336,
      "n_steps": 1000,
      "peak_device_bytes": null
    }
  },
  {
    "backend": "cupy",
    "n_traj": 100,
    "wall_seconds": 1.4104925999999978,
    "success": true,
    "breakdown": {
      "total": 1.4104925999999978,
      "drift": 0.28319440031191334,
      "diffusion": 0.32989500007533934,
      "contract": 0.48104709998006,
      "n_steps": 1000,
      "peak_device_bytes": 6656
    }
  },
  {
    "backend": "cupy",
    "n_traj": 500,
    "wall_seconds": 1.2016891999955988,
    "success": true,
    "breakdown": {
      "total": 1.2016891999955988,
      "drift": 0.2581605000887066,
      "diffusion": 0.30240029985725414,
      "contract": 0.42938480028533377,
      "n_steps": 1000,
      "peak_device_bytes": 32256
    }
  },
  {
    "backend": "cupy",
    "n_traj": 1000,
    "wall_seconds": 1.4545513999910327,
    "success": true,
    "breakdown": {
      "total": 1.4545513999910327,
      "drift": 0.31844480024301447,
      "diffusion": 0.3752132002991857,
      "contract": 0.5109648001089226,
      "n_steps": 1000,
      "peak_device_bytes": 64000
    }
  },
  {
    "backend": "cupy",
    "n_traj": 5000,
    "wall_seconds": 1.1966762999945786,
    "success": true,
    "breakdown": {
      "total": 1.1966762999945786,
      "drift": 0.2582978997816099,
      "diffusion": 0.3106683996738866,
      "contract": 0.42704370005230885,
      "n_steps": 1000,
      "peak_device_bytes": 320000
    }
  },
  {
    "backend": "cupy",
    "n_traj": 10000,
    "wall_seconds": 1.2621819000050891,
    "success": true,
    "breakdown": {
      "total": 1.2621819000050891,
      "drift": 0.2776589999557473,
      "diffusion": 0.3327986997901462,
      "contract": 0.4488167002418777,
      "n_steps": 1000,
      "peak_device_bytes": 640000
    }
  },
  {
    "backend": "torch",
    "n_traj": 100,
    "wall_seconds": 0.0,
    "success": false,
    "breakdown": null
  },
  {
    "backend": "torch",
    "n_traj": 500,
    "wall_seconds": 0.0,
    "success": false,
    "breakdown": null
  },
  {
    "backend": "torch",
    "n_traj": 1000,
    "wall_seconds": 0.0,
    "success": false,
    "breakdown": null
  },
  {
    "backend": "torch",
    "n_traj": 5000,
    "wall_seconds": 0.0,
    "success": false,
    "breakdown": null
  },
  {
    "backend": "torch",
    "n_traj": 10000,
    "wall_seconds": 0.0,
    "success": false,
    "breakdown": null
  }
]
```
