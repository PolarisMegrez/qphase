# QPhaseSDE User Guide (v0.1.1)

Audience: Physicists with basic Python familiarity who want to run stochastic simulations of complex-valued mode dynamics and visualize results.

## Install (Windows PowerShell)

We recommend a virtual environment:

1. Create and activate a venv
2. Upgrade pip
3. Install the core and CLI packages in editable mode
4. Install a YAML loader (ruamel.yaml preferred)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e packages\QPhaseSDE
pip install -e packages\QPhaseSDE_cli
python -m pip install ruamel.yaml
```

## Your first run

Use the provided example model (two-mode Van der Pol) and config:

```powershell
qps run sde --config configs\vdp_run.yaml
```

This creates a new folder under `runs/` with:
- `config_snapshot/` — exact inputs used
- `time_series/` — NPZ time series (per IC)
- `figures/` — per-IC images for requested plots
- `manifest.json` — run metadata

Tip: `model.ic` can be a single vector or a list of vectors. When multiple ICs are provided, each IC is simulated independently.

## Visualizations

Two classes of built-in plots are available. You can request multiple figures in one run.

- Phase portraits (`run.visualization.phase_portrait`):
  - `kind: re_im` with 1 mode index → Re(α) vs Im(α)
  - `kind: abs_abs` with 2 mode indices → |α_i| vs |α_j|
  - Optional `t_range: [t_start, t_end]` selects the time window
  - Style under `profile.visualization.phase_portrait.{re_im|abs_abs}`

- Power spectral density (PSD) (`run.visualization.psd`):
  - `kind: complex` (FFT of complex signal) or `modular` (FFT of |signal|)
  - `modes: [...]` — one or more modes per figure
  - Optional `xlim: [xmin, xmax]` and `t_range: [t_start, t_end]`
  - Global PSD style under `profile.visualization.psd`:
    - `convention: symmetric|unitary|pragmatic`
    - `x_scale: linear|log`, `y_scale: linear|log`
  - Data are averaged across trajectories. For complex PSD we plot two-sided; for modular PSD we plot one-sided.

## Saving controls and disk guard

In `profile.save`:
- `save_timeseries`: save time series NPZ per IC
- `save_psd_complex`: save complex-PSD NPZ per IC
- `save_psd_modular`: save modular-PSD NPZ per IC

Before saving time series, the CLI estimates disk usage and aborts if the projected size exceeds 1 GiB by default. Override with `--max-storage-gb`.

## Re-rendering from saved results

You can re-generate figures without recomputing trajectories:

```powershell
qps analyze phase --from-run runs\<run_id>
qps analyze psd --from-run runs\<run_id>
```

Optionally, override specs on the fly (phase example):

```powershell
qps analyze phase --from-run runs\<run_id> --use-snapshot false --specs-json "[{\"kind\":\"re_im\",\"modes\":[0]}]"
```

## Troubleshooting

- Missing YAML loader: install `ruamel.yaml` or `PyYAML`.
- Large disk usage: lower `save_every`, disable `save_timeseries`, or raise `--max-storage-gb` if you are sure.
- Long runs: reduce `steps`, `n_traj`, or narrow `t_range` for plotting.

## Next steps

- Inspect and modify `models/vdp_two_mode.py` or add your own model exposing `build_sde(params)`.
- Explore the registry to discover available components:
  - `integrator: euler, milstein (alias of euler)`
  - `backend: numpy`
  - `visualization: phase_portrait, psd`
