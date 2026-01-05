# qphase — A Modular Toolkit for Phase-Space Simulation in Quantum Optics

`qphase` is a small command-line tool for configuring and running phase-space simulations in quantum optics. It serves as the main entry point for the project, handling configuration loading and job execution.

## Features

- **CLI Interface (`qphase`)**: A simple command-line interface to run simulations and analysis tasks.
- **Plugin Support**: Loads external packages (like `qphase-sde` and `qphase-viz`) to extend functionality.
- **Session Management**: Basic job tracking with support for resuming interrupted runs and dry-run validation.
- **Configuration**: Uses YAML/JSON files to define simulation parameters and workflows.
- **Parameter Scanning**: Supports simple parameter sweeps (Cartesian and Zipped).

## Installation

```bash
pip install qphase
# Optional: Install standard backends (Numba, PyTorch)
pip install qphase[standard]
```

## Quick Start

1.  **Initialize a project**:
    ```bash
    qphase init
    ```

2.  **Run a simulation**:
    ```bash
    qphase run my_simulation
    ```

3.  **List available jobs**:
    ```bash
    qphase run --list
    ```

## Project Structure

This repository contains the following packages:
- **`qphase`**: The main CLI and scheduler.
- **`qphase-sde`**: The numerical solver for SDEs.
- **`qphase-viz`**: Plotting utilities for simulation data.

## License

MIT License
