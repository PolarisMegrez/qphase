---
layout: default
title: Home
nav_order: 1
---
# QPhase

**QPhase** is a lightweight, modular simulation toolset designed for physics research. It was built to solve a common problem in scientific computing: the repetitive rewriting of "boilerplate" code for every new experiment.

## The Problem it Solves
In typical physics research (e.g., simulating SDEs or quantum systems), we often find ourselves writing the same supporting code over and over again:
- How do I pass parameters to this script? (Argparse/Config files)
- How do I save the results so I don't overwrite yesterday's data?
- How do I run this on a GPU instead of a CPU?
- How do I scan parameter $X$ from 0 to 10?

**QPhase** separates these "operational" concerns from the actual "physics". It provides a stable **Shell** that handles configuration, data saving, and parallel execution, allowing you to write only the **Kernel**—the equations of motion or physical model.

## Key Features
- **Focus on Physics**: You write the model (e.g., `dx/dt = ...`), and the framework handles the integration loop, progress bars, and file I/O.
- **Reproducibility**: Every run automatically saves a snapshot of the exact configuration used, so you never lose track of what parameters produced a specific plot.
- **Hardware Switching**: Switch between NumPy (CPU) and PyTorch/CuPy (GPU) just by changing a line in the config file, without rewriting your model.
- **Parameter Sweeps**: Define a list of parameters in the config, and QPhase automatically generates and runs the batch jobs.

## Documentation Structure
- **[User Guide](user_guide/quick_start.md)**: For users who want to run simulations. Focuses on installation, configuration (YAML), and running jobs.
- **[Developer Guide](dev_guide/architecture.md)**: For writing new models or understanding the internal logic. Explains the plugin system and design choices in detail.
- **[API Reference](api/core.md)**: Technical details of the core classes.

## Project Status

This is a personal research project currently in active development. It is designed to be flexible enough for my own research needs in quantum phase-space simulations, but structured enough to be useful for others facing similar computational challenges.
