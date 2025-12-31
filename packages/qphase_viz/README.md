# qphase-viz

**Plotting Utilities for QPhase**

`qphase-viz` provides a set of plotting tools for visualizing data from quantum phase-space simulations. It is designed to work with `qphase` but can also be used independently to generate figures.

## Features

- **Phase Portraits**:
    - **Re-Im Plane**: Visualize distributions in complex phase space.
    - **Abs-Abs Plane**: Analyze amplitude correlations.
    - **Marginal Distributions**: Project dynamics onto axes.
- **Spectral Analysis**:
    - **Power Spectral Density (PSD)**: Compute spectra with windowing.
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
Add a visualization job to your `qphase` configuration:

```yaml
jobs:
  - name: "plot_phase"
    type: "viz"
    dependencies: ["my_simulation"]
    config:
      plots:
        - kind: "phase_plane"
          mode: 0
          style: "density"
        - kind: "spectrum"
          source: "output.npy"
```

### Standalone Usage
```python
from qphase_viz.engine import VizEngine
from qphase_viz.config import PhasePlaneConfig

# Configure and run
config = PhasePlaneConfig(mode=0)
engine = VizEngine()
engine.plot(data, config)
```

## License

MIT License
