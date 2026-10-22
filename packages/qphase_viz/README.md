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

## License

MIT License
