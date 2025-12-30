---
description: Architecture Overview
---

# Architecture Overview

This document details the architectural design of the QPhase framework. It is intended for developers seeking to understand the internal mechanisms, design patterns, and structural decisions that govern the system.

## Design Philosophy

QPhase is architected to address the specific challenges of scientific computing: reproducibility, modularity, and hardware agnosticism. Unlike ad-hoc scripting, where simulation logic is tightly coupled with infrastructure code (I/O, configuration, parallelization), QPhase enforces a strict separation of concerns.

### The Shell-Kernel Dichotomy

The framework is conceptually divided into two distinct layers:

1.  **The Shell (Infrastructure Layer)**:
    *   **Responsibility**: Manages the operational lifecycle of a simulation. This includes configuration parsing, dependency injection, job scheduling, resource management, and result persistence.
    *   **Characteristics**: Generic, physics-agnostic, and stable. It provides the "runtime environment" for simulations.

2.  **The Kernel (Domain Layer)**:
    *   **Responsibility**: Encapsulates the scientific logic. This includes physical models (Hamiltonians, Drift/Diffusion vectors), numerical integrators, and backend implementations.
    *   **Characteristics**: Domain-specific, modular, and extensible. Users primarily interact with this layer by implementing plugins.

## Core Concepts

To understand how QPhase operates, it is essential to distinguish between three fundamental concepts: the **Job**, the **Engine**, and the **Plugin**.

### 1. The Job (The "Intent")
A **Job** represents a single, atomic execution request. It answers the question: *"What simulation do I want to run?"*
*   **Definition**: A Job is defined entirely by its configuration (a resolved YAML document). It contains all the parameters needed to reproduce a simulation.
*   **Isolation**: Each Job runs in its own isolated directory (`runs/{timestamp}_{job_name}/`). This ensures that side effects (like file I/O) from one simulation do not contaminate another.
*   **Lifecycle**: A Job is created by the Scheduler (often by expanding a parameter scan), executed, and then finalized when its results are saved.

### 2. The Engine (The "Workflow")
An **Engine** is a special type of plugin that defines the *lifecycle* of a simulation. It answers the question: *"How should the simulation proceed?"*
*   **Role**: The Engine acts as the "main loop" or orchestrator.
    *   The `sde` engine runs a time-stepping loop for stochastic differential equations.
    *   The `viz` engine runs a data processing and plotting pipeline.
*   **Orchestration**: The Engine does not perform the low-level physics or math itself. Instead, it requests other plugins (like Models or Backends) to do the actual work.

### 3. The Plugin (The "Building Block")
A **Plugin** is a modular component that implements a specific capability. Plugins are the "Lego blocks" that the Engine assembles to build a simulation.
*   **Model**: Defines the physical system (e.g., drift and diffusion vectors).
*   **Backend**: Provides the computational primitives (e.g., NumPy for CPU, PyTorch for GPU).
*   **Integrator**: Implements the numerical solver (e.g., Euler-Maruyama).
*   **Analyser**: Processes raw simulation data into metrics.

### The Relationship: Dependency Injection
The power of QPhase lies in how these components connect. You do not write code to wire them together; the **Scheduler** does it for you based on the configuration.

1.  **Selection**: The **Job** configuration selects an **Engine** (e.g., `engine: sde`).
2.  **Declaration**: The **Engine** declares what it needs via a **Manifest** (e.g., "I require a `model` and a `backend`").
3.  **Injection**: The **Scheduler** reads the Manifest, looks up the requested plugins in the Job config, instantiates them via the **Registry**, and *injects* them into the Engine's constructor.

## Core Architectural Patterns

### 1. Registry Pattern & Dependency Injection

To achieve modularity, QPhase avoids hardcoded dependencies. Instead, it utilizes a **Registry Pattern** combined with **Dependency Injection**.

*   **Registry**: A centralized singleton (`RegistryCenter`) that maintains a dynamic mapping of component names (strings) to their implementations (classes/factories). This allows components to be selected at runtime via configuration files.
*   **Dependency Injection**: When an `Engine` is instantiated, it does not instantiate its dependencies (Model, Backend) directly. Instead, the `Scheduler` resolves these dependencies via the Registry and injects them into the Engine's constructor. This inversion of control facilitates testing and component swapping.

### 2. Backend Abstraction (Tensor Dispatching)

A critical requirement for modern scientific computing is hardware portability (CPU vs. GPU). QPhase addresses this through the **Backend Abstraction**.

*   **Problem**: Direct usage of libraries like `numpy` or `torch` couples the simulation code to a specific hardware backend.
*   **Solution**: The framework defines a `BackendBase` Protocol that specifies a standard interface for tensor operations.
*   **Implementation**: Concrete implementations (`NumpyBackend`, `TorchBackend`) wrap the underlying libraries. The simulation kernel interacts exclusively with the abstract interface (conventionally named `xp`), allowing the underlying execution engine to be swapped via configuration without code changes.

### 3. Structural Subtyping (Protocols)

QPhase leverages Python's `typing.Protocol` (PEP 544) for interface definitions rather than Abstract Base Classes (ABCs).

*   **Rationale**: This enforces **Structural Subtyping** (Duck Typing) rather than Nominal Subtyping. A class is considered a valid plugin if it implements the required methods, regardless of its inheritance hierarchy.
*   **Benefit**: This reduces coupling between user code and the framework core. Researchers can develop plugins without importing framework-specific base classes, simplifying distribution and testing.

### 4. Explicit Dependency Contracts (Manifests)

To ensure robustness in a loosely coupled system, QPhase employs **Explicit Dependency Contracts**.

*   **Problem**: "Blind" dependency injection can lead to runtime errors if an Engine requires a plugin (e.g., a specific Model type) that the user failed to configure.
*   **Solution**: Engines declare their dependencies statically via an `EngineManifest`.
*   **Mechanism**: The `Scheduler` validates the Job Configuration against the Engine's Manifest *before* execution begins, ensuring all required plugins are present and correctly typed.

## Execution Lifecycle

The execution of a simulation follows a deterministic lifecycle managed by the `Scheduler`:

1.  **Initialization**: The CLI entry point initializes the application context and loads the system configuration.
2.  **Discovery**: The Registry scans entry points and local directories to populate the component catalog.
3.  **Configuration Resolution**: Job configurations are loaded, validated against Pydantic schemas, and merged with global defaults.
4.  **Validation**: The Scheduler validates job dependencies against the target Engine's `EngineManifest`.
5.  **Job Expansion**: Parameter scans are processed, expanding high-level job definitions into a list of atomic execution units (`JobConfig`).
6.  **Execution Loop**:
    *   **Isolation**: A unique run directory is provisioned.
    *   **Snapshotting**: The resolved configuration is serialized to `config_snapshot.json` for reproducibility.
    *   **Instantiation**: The `Engine` and its dependencies are instantiated via the Registry.
    *   **Simulation**: The Engine's `run()` method executes the physics loop.
    *   **Persistence**: Results are serialized and flushed to disk.

## Directory Structure

The project follows a monorepo structure managed by `uv` workspaces.

### Source Code (`packages/`)
*   `qphase/`: **The Core Framework (Shell)**.
    *   `core/`: Scheduler, Registry, Configuration, Protocols.
    *   `commands/`: CLI implementation.
*   `qphase_sde/`: **Standard Engine**.
    *   Implements SDE solvers (Euler-Maruyama, SRK).
    *   Contains standard physics models (Kerr Cavity, VdP).
*   `qphase_viz/`: **Visualization Engine**.
    *   Handles plotting and data post-processing.

### Runtime Artifacts (`runs/`)
When simulations are executed, QPhase organizes outputs hierarchically:

```text
runs/
├── 2025-12-29T10-00-00Z_scan_job_0/   # Job 1 (chi=0.1)
│   ├── config_snapshot.json           # Full config for this specific point
│   └── results.h5                     # Simulation data
├── 2025-12-29T10-00-05Z_scan_job_1/   # Job 2 (chi=0.2)
│   ├── config_snapshot.json
│   └── results.h5
└── ...
```
