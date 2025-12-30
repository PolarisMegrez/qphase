---
description: "Modular Quantum Phase-Space Simulation Framework"
---

# QPhase - Modular Quantum Phase-Space Simulation Framework

[Get Started](user_guide/quick_start.md){ .md-button .md-button--primary } [View on GitHub](https://github.com/PolarisMegrez/qphase){ .md-button }

---

## Overview

**QPhase** is a lightweight, modular simulation toolset designed for physics research. It solves a common problem in scientific computing: the repetitive rewriting of "boilerplate" code for every new experiment.

!!! note "The Problem"
    In typical physics research, we often rewrite the same supporting code:

    *   How do I pass parameters? (Argparse/Config)
    *   How do I save results safely?
    *   How do I switch to GPU?
    *   How do I scan parameters?

**QPhase** separates these "operational" concerns from the actual "physics". It provides a stable **Shell** that handles configuration, data saving, and parallel execution, allowing you to write only the **Kernel**—the equations of motion.

---

## Key Features

### Focus on Physics
You write the model (e.g., `dx/dt = ...`), and the framework handles the integration loop, progress bars, and file I/O.

### Reproducibility
Every run automatically saves a snapshot of the exact configuration used. Never lose track of what parameters produced a specific plot.

### Hardware Switching
Switch between **NumPy** (CPU) and **PyTorch/CuPy** (GPU) just by changing a line in the config file, without rewriting your model.

### Parameter Sweeps
Define a list of parameters in the config, and QPhase automatically generates and runs the batch jobs.

---

## Quick Start

### 1. Installation

```bash
pip install qphase
```

### 2. Run a Simulation

```bash
qphase run config.yaml
```

---

## Documentation

| Section | Description |
|:--------|:------------|
| [**User Guide**](user_guide/quick_start.md) | Installation, configuration (YAML), and running jobs. |
| [**Developer Guide**](dev_guide/architecture.md) | Plugin system, architecture, and extending the framework. |
| [**API Reference**](api/core.md) | Technical details of the core classes. |

---

## Project Status

This is a personal research project currently in active development. It is designed to be flexible enough for my own research needs in quantum phase-space simulations, but structured enough to be useful for others facing similar computational challenges.
