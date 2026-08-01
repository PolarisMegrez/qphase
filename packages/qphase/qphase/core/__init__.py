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
``JobConfig``, ``JobList`` : Job configuration and batch job definitions.
``Scheduler``, ``JobResult``, ``ProgressSnapshot`` : Task scheduling and execution.
``RegistryCenter``, ``registry`` : Plugin discovery, registration, and instantiation.
``SystemConfig`` : System-level configuration model.
``load_system_config``, ``save_user_config``, ``get_system_param`` : System config I/O.
``load_global_config``, ``save_global_config`` : Global plugin configuration I/O.
``merge_configs``, ``get_config_for_job``, ``list_available_jobs`` : Config utilities.
``QPhaseError`` : Unified exception hierarchy base class.
``get_logger``, ``configure_logging``
    Logging utilities.
"""

from .config import InputSpec, JobConfig, JobList
from .config_loader import (
    get_config_for_job,
    get_system_param,
    list_available_jobs,
    load_global_config,
    load_jobs_from_files,
    merge_configs,
    save_global_config,
)
from .dataset import DatasetResultProtocol, DatasetSaveReport
from .error_report import ErrorReport, ErrorSummary
from .errors import (
    ErrorCode,
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
from .execution import ExecutionContext
from .progress import (
    ProgressEvent,
    ProgressReporter,
    ProgressSnapshot,
    ProgressTracker,
)
from .registry import RegistryCenter, registry
from .scan import ParameterGrid, ScanAxisSpec, ScanSpec
from .scheduler import JobResult, Scheduler
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
    "ErrorCode",
    "get_logger",
    "configure_logging",
    # Error reporting
    "ErrorReport",
    "ErrorSummary",
    # Progress
    "ProgressEvent",
    "ProgressReporter",
    "ProgressSnapshot",
    "ProgressTracker",
    # Registry
    "registry",
    "RegistryCenter",
    # Scheduler
    "Scheduler",
    "JobResult",
    # Config
    "JobConfig",
    "JobList",
    "InputSpec",
    "ScanSpec",
    "ScanAxisSpec",
    "ParameterGrid",
    "ExecutionContext",
    "DatasetResultProtocol",
    "DatasetSaveReport",
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
