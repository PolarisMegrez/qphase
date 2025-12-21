"""qphase: core subpackage
---------------------------------------------------------
This subpackage implements the foundational architecture of the control layer,
providing the essential building blocks for plugin management and job orchestration.
It includes the ``RegistryCenter`` for dynamic plugin discovery and instantiation,
the ``Scheduler`` for serial job execution with dependency resolution and parameter
scanning, and the Pydantic-based configuration models (``JobConfig``, ``SystemConfig``)
that ensure type safety and validation. Additionally, it defines the unified
exception hierarchy and logging infrastructure used throughout the framework.

Public API
----------
``JobConfig``, ``JobList`` : Job configuration and batch job definitions
``Scheduler``, ``JobResult``, ``JobProgressUpdate`` : Task scheduling and execution
``RegistryCenter``, ``registry`` : Plugin discovery, registration, and instantiation
``SystemConfig`` : System-level configuration model
``load_system_config``, ``save_user_config``, ``get_system_param`` : System config I/O
``load_global_config``, ``save_global_config`` : Global plugin configuration I/O
``merge_configs``, ``get_config_for_job``, ``list_available_jobs`` : Config utilities
``QPhaseError`` and subclasses : Unified exception hierarchy
``get_logger``, ``configure_logging`` : Logging utilities
"""

from .config import JobConfig, JobList
from .config_loader import (
    get_config_for_job,
    get_system_param,
    list_available_jobs,
    load_global_config,
    load_jobs_from_files,
    merge_configs,
    save_global_config,
)
from .errors import (
    QPhaseCLIError,
    QPhaseConfigError,
    QPhaseError,
    QPhaseIOError,
    QPhasePluginError,
    QPhaseRuntimeError,
    QPhaseSchedulerError,
    configure_logging,
    get_logger,
)
from .registry import RegistryCenter, registry
from .scheduler import JobProgressUpdate, JobResult, Scheduler
from .system_config import SystemConfig, load_system_config, save_user_config

__all__ = [
    # Errors & Logging
    "QPhaseError",
    "QPhaseConfigError",
    "QPhaseIOError",
    "QPhasePluginError",
    "QPhaseSchedulerError",
    "QPhaseRuntimeError",
    "QPhaseCLIError",
    "get_logger",
    "configure_logging",
    # Registry
    "registry",
    "RegistryCenter",
    # Scheduler
    "Scheduler",
    "JobResult",
    "JobProgressUpdate",
    # Config
    "JobConfig",
    "JobList",
    # System config
    "SystemConfig",
    "load_system_config",
    "save_user_config",
    "get_system_param",
    # Config loader
    "load_global_config",
    "load_jobs_from_files",
    "save_global_config",
    "merge_configs",
    "get_config_for_job",
    "list_available_jobs",
]
