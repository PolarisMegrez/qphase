# qphase

**qphase** is a modular numerical simulation framework designed for small-scale scientific computing. It provides a structured workflow for defining simulation problems, organizing computational tasks, and integrating numerical implementations. The framework aims to separate domain-specific modeling from repetitive control logic and infrastructure code, emphasizing explicit configuration, minimal assumptions about underlying physics, and extensibility through well-defined interfaces.

## Key Features

*   **Configuration Management**: Simulations are specified through structured YAML or JSON configuration files. The system supports parameter scanning and schema validation based on Pydantic models, ensuring robust input handling.
*   **Task Scheduling**: A built-in scheduler manages simulation jobs, controlling execution order and performing consistent result collection.
*   **Backend Abstraction**: Numerical backends are accessed through abstract interfaces. This allows different computational libraries (such as NumPy, Numba, or PyTorch) to be used interchangeably, enabling performance optimization and basic parallel execution without changing simulation logic.
*   **Plugin Architecture**: Solvers, backends, and analysis routines are integrated via a protocol-based plugin system. This keeps extensions decoupled from the core framework, allowing functionality to be added independently.

## Documentation

*   **[User Guide](user_guide/getting_started.md)**: Instructions for installation, configuration, and running simulations.
*   **[Developer Guide](dev_guide/index.md)**: Details on internal architecture, plugin development, and the registry system.
*   **[API Reference](api/index.md)**: Technical documentation of the `qphase` package and core modules.
