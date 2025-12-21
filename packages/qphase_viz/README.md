# qphase-viz

The visualization plugin for **QPhase**.

This package provides specialized plotting tools for analyzing quantum phase-space simulations, including phase portraits and power spectral density (PSD) analysis.

## Features

- **Phase Portraits**: Visualize trajectories in Re-Im or Abs-Abs planes.
- **PSD Analysis**: Compute and plot Power Spectral Densities.
- **Integration**: Seamlessly integrates with `qphase` CLI for automated report generation.

## Installation

```bash
pip install qphase-viz
```

## Usage

Configure visualization in your `qphase` YAML config:

```yaml
visualizer:
  phase_portrait:
    - kind: Re_Im
      modes: [0]
```

## License

This project is licensed under the MIT License.
