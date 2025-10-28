# QPhaseSDE v0.1.0

QPhaseSDE is a configuration-driven framework for simulating complex-valued stochastic differential equations (SDEs) in quantum optics. It supports multi-trajectory simulations, time-series data output, and automatic generation of phase portraits and power spectral density (PSD) plots — all managed through a simple YAML configuration file.

**Authors: Yu Xue‑Hao and Qiao Cong‑Feng (University of Chinese Academy of Sciences, UCAS)**

## Feature overview

- Multi‑trajectory SDE engine (Itô) for complex modes
- Euler–Maruyama solver (built‑in); Milstein alias (currently same as EM)
- Gaussian noise (independent or correlated)
- Reproducible runs: master seed, per‑trajectory seeds, config snapshot, manifest, NPZ time‑series
- Visualizations
  - Phase portraits: Re–Im and |.|–|.| views
  - PSD: complex (two‑sided) and modular (one‑sided); averaged across trajectories

## Install and use (Windows PowerShell)

The quickest way to try QPhaseSDE:

```powershell
# Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the core and CLI packages (editable mode for local use)
pip install -e packages\QPhaseSDE
pip install -e packages\QPhaseSDE_cli

# Install a YAML loader
python -m pip install ruamel.yaml   # or: python -m pip install pyyaml

# Run the example configuration
qps run sde --config configs\vdp_run.yaml
```

Results are written to `runs/<timestamp>_<id>/`:
- `time_series/` — NPZ time‑series per initial condition (IC)
- `figures/` — images for requested plots
- `psd/` — PSD data (NPZ) when enabled
- `config_snapshot/` and `manifest.json` — provenance and metadata

Re‑render figures later without recomputing trajectories:

```powershell
qps analyze phase --from-run runs\<run_id>
qps analyze psd   --from-run runs\<run_id>
```

## A minimal YAML guide

A configuration has three parts: `model`, `profile`, and `run`.

```yaml
model:
  module: models.vdp_two_mode
  function: build_sde
  params: { omega_a: 0.005, omega_b: 0.0, gamma_a: 2.0, gamma_b: 1.0, Gamma: 0.01, g: 0.5, D: 1.0 }
  ic: ["7.0+0.0j", "0.0-7.0j"]      # single vector; you can also provide a list of vectors
  noise: { kind: independent }

profile:
  backend: numpy
  solver: euler
  save:
    root: runs
    save_every: 20
    save_timeseries: true
    save_psd_complex: false
    save_psd_modular: false
  visualization:
    psd:
      convention: symmetric         # or pragmatic
      x_scale: linear               # x/y axis scales: linear or log
      y_scale: log

run:
  time: { dt: 0.001, steps: 200000 }
  trajectories: { n_traj: 10, master_seed: 42 }
  visualization:
    phase_portrait:
      - kind: Re_Im
        modes: [0]
      - kind: abs_abs
        modes: [0, 1]
        t_range: [10.0, 200.0]
    psd:
      - kind: complex
        modes: [0, 1]
        xlim: [-0.5, 0.5]
        t_range: [20.0, 100.0]
```

Notes for YAML users
- `model.ic` can be a single vector (broadcast to all trajectories) or a list of IC vectors; internally it is normalized to a list of vectors.
- `profile.save` decides what to persist: time‑series and/or PSD data.
- PSD “convention” sets normalization and frequency axis: symmetric/unitary (ω) or pragmatic (f).
- For full details, see `docs/config_spec_en.md` and `docs/config_spec_zh.md`.

## Project structure

- `packages/QPhaseSDE` — core library (engine, protocols, backends, integrators, IO, visualizers)
  - `core/` — minimal protocols, registry, and integration engine
  - `backends/` — backend registrations (NumPy built‑in)
  - `integrators/` — Euler–Maruyama (Milstein alias)
  - `visualizers/` — Spec (Pydantic) → Renderer → Service; PSD and phase portraits
  - `analysis/` — reusable analysis (e.g., PSD computation)
  - `io/` — results saving/loading; snapshots
- `packages/QPhaseSDE_cli` — Typer‑based CLI (`qps`) for running and analyzing
- `models/` — example models (e.g., `vdp_two_mode.py`)
- `configs/` — example YAMLs (run and PSD)
- `runs/` — generated outputs
- `docs/` — configuration specs and user guides (EN/ZH)
- `tests/` — local smoke tests

Internals rely on a central registry (`QPhaseSDE.core.registry.registry`) to create pluggable components by name (e.g., `visualization:psd`). The visualizer service validates specs, slices time windows, merges styles, dispatches renderers, and saves images.

## Notes

- Python ≥ 3.9 recommended. Requires NumPy and Matplotlib. Install `ruamel.yaml` or `PyYAML` to use YAML configs.
- Storage guard: before saving time‑series, the CLI estimates disk usage and aborts if it exceeds a default 1 GiB limit. Override with `--max-storage-gb`.
- Reproducibility: runs are reproducible for the same backend and package version (and device type where applicable). Cross‑backend equivalence is not guaranteed.
- PSD data are saved per IC under `psd/icXX/psd_{kind}_{convention}.npz` when enabled.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.