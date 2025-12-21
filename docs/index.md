---
layout: default
title: Home
nav_order: 1
---

# QPhase

**QPhase** is a modular, high-performance computational framework designed for theoretical physics simulations. It provides a robust, plugin-based architecture that decouples numerical methods from physical models, enabling researchers to focus on scientific discovery rather than software infrastructure.

## Key Features

*   **Plugin-Driven Architecture**: A flexible registry system that manages models, integrators, and backends as independent plugins.
*   **Unified Interface**: A consistent Command Line Interface (CLI) for managing simulations, configuration, and extensions.
*   **Reproducibility**: Automatic configuration snapshots and structured output management ensure every result is traceable.
*   **Hardware Agnostic**: Seamlessly switch between CPU (NumPy) and GPU (CuPy/Torch) backends without modifying model code.
*   **Parameter Scanning**: Built-in support for parallel parameter sweeps and batch job execution.

## Why QPhase?

Scientific computing often involves repetitive boilerplate code for input parsing, data management, and parallelization. QPhase abstracts these concerns into a stable "shell," allowing the "kernel"—your physics code—to remain clean and focused.

Whether you are simulating Stochastic Differential Equations (SDEs) or exploring quantum phase-space representations, QPhase provides the tooling to:
1.  **Define** complex simulation campaigns via simple YAML configurations.
2.  **Execute** jobs efficiently with automatic dependency resolution.
3.  **Extend** functionality through a standardized plugin protocol.

## Getting Started

*   **[User Guide](user_guide/quick_start.md)**: Learn how to install QPhase and run your first simulation.
*   **[Configuration](user_guide/configuration.md)**: Master the YAML-based configuration system.
*   **[Developer Guide](dev_guide/architecture.md)**: Understand the internal architecture and how to build custom plugins.

## Project Status

QPhase is currently in active development. While the core SDE functionality is stable, new modules for quantum phase-space methods are planned for future releases.
