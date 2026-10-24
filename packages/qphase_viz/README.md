# qphase-viz

**Plotting Utilities for QPhase**

`qphase-viz` renders typed QPhase data bundles. Plotters select products by
stable bundle role, data kind, quantity, and required fields.

## Features

- **Phase Portraits**:
    - **Re-Im Plane**: Visualize distributions in complex phase space.
    - **Abs-Abs Plane**: Analyze amplitude correlations.
    - **Marginal Distributions**: Project dynamics onto axes.
- **Spectral Analysis**:
    - **Power Spectral Density (PSD)**: Plot typed PSD products.
    - **Log/Linear Scaling**: Flexible axis options.
- **Time Series**:
    - **Trajectory Evolution**: Plot stochastic paths and averages.
    - **Confidence Intervals**: Visualize variance/std-dev.

## Installation

```bash
pip install qphase-viz
```

## Usage

### As a QPhase Plugin
Add a visualization job that consumes a typed upstream bundle:

```yaml
schema: qphase.workflow/2
id: typed_viz
title: Typed visualization
jobs:
  - name: plot_phase
    input: {from: my_simulation, mode: dataset}
    engine:
      viz: {output_dir: figures, format: png}
    visualizer:
      phase_plane:
        plots: [{channel_x: 0, mode: hist2d}]
      power_spectrum:
        plots: [{channels: [0], scale: log}]
```

### Python Usage
```python
from qphase_viz.engine import VizEngine
from qphase_viz.plotters.phase import PhasePlanePlotter

engine = VizEngine(
    plugins={"visualizer": PhasePlanePlotter(plots=[{"channel_x": 0}])}
)
result = engine.run(bundle)  # bundle implements DatasetResultProtocol
```

Time-series and phase-plane plotters require a typed field-amplitude
time-series product. The spectrum plotter requires an existing typed PSD
product: it never recomputes a spectrum from raw trajectories, so run a PSD
analysis (for example the `psd` analyser) first and feed its typed output to
the viz engine.
Parameter evolution consumes explicit scan coordinates and typed peak or
trajectory products. Raw arrays, trajectory containers, and payload dicts are
not public inputs.

## Styles and editable output

Figures render with a named style: `publication` (default, STIX typography),
`sm` (the same typography one size tier down), or `sans` (Arial with sans
math). `publication` and `sm` export text as editable text by default
(`svg.fonttype='none'`, `pdf.fonttype=42`), so an SVG can be opened and its
labels edited in Inkscape. A plot spec may override the style for its own
figure, and `style_overrides` always wins over the style table:

```python
from qphase_viz import VizEngine, VizEngineConfig
from qphase_viz.plotters.evolution import TimeSeriesPlotter

engine = VizEngine(
    config=VizEngineConfig(
        output_dir="figures",
        format="svg",
        style="publication",          # or "sm" / "sans"
        editable=True,                # svg.fonttype='none', pdf.fonttype=42
        style_overrides={"axes.grid": False},
    ),
    plugins={
        "visualizer": TimeSeriesPlotter(
            plots=[
                {
                    "channels": [0],
                    "filename": "narrow",
                    "style": "sm",                      # per-figure style
                    "style_overrides": {"axes.labelsize": 12},
                    "annotations": [
                        {
                            "text": r"$\gamma/\kappa = 0.1$",
                            "x": 0.02,
                            "y": 0.95,
                            "coords": "axes fraction",
                        }
                    ],
                }
            ]
        )
    },
)
```

Priority for every figure: single-plot setting, then the engine setting, then
the `publication` default. Styling is applied in a local Matplotlib rc
context, so rendering never changes the process-wide `rcParams`. A direct
`plotter.plot(data, output_dir, format)` call uses the `publication` defaults.

### PSD across a scan

`SpectralScanPlotter` draws a multi-point PSD scan as one panel per mode,
sharing the frequency axis. The scan coordinate is always named explicitly and
the plotter never averages across scan points:

```yaml
visualizer:
  spectral_scan:
    plots:
      - scan_parameter: omega       # coordinate stored on the PSD product
        channels: [0, 1, 2]         # one panel each
        scan_indices: [0, 10, 20]   # or scan_coordinates: [0.12, 0.24]
        log_scale: true
        label_template: "{name}={value:.6g}"
```

## License

MIT License
