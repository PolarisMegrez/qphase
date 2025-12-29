---
layout: default
title: Architecture Overview
parent: Developer Guide
nav_order: 1
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

*   `packages/qphase/core/`: **The Shell**. Contains the Scheduler, Registry, Configuration system, and Protocol definitions.
*   `packages/qphase_sde/`: **Reference Engine**. A standard implementation of a Stochastic Differential Equation solver.
*   `packages/qphase_viz/`: **Visualization Engine**. A modular plotting system.
