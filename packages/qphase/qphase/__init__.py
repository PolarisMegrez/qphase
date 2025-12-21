"""qphase - Simulation Control Layer
======================
The control layer serves as the central orchestration engine for QPhase simulations,
decoupling algorithmic implementations from execution management. It provides a
unified command-line interface (``qps``), a robust plugin system for dynamic
resource loading (backends, integrators, models), and a hierarchical configuration
system based on declarative YAML. This package manages the full lifecycle of
simulation jobs, from parameter validation and dependency resolution to execution
scheduling and result persistence, ensuring reproducibility and extensibility
across different computational environments.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 0.1.3 (Dec 2025)
"""

__version__ = "0.1.3"

__all__ = ["__version__"]
