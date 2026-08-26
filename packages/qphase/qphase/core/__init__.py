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
``JobConfig``, ``WorkflowSpec`` : Logical job and workflow definitions.
``Scheduler``, ``JobResult``, ``ProgressSnapshot`` : Task scheduling and execution.
``RegistryCenter``, ``registry`` : Plugin discovery, registration, and instantiation.
``SystemConfig`` : System-level configuration model.
``load_system_config``, ``save_user_config``, ``get_system_param`` : System config I/O.
``ProjectContext`` : Explicit project identity and portable path boundary.
``QPhaseError`` : Unified exception hierarchy base class.
``get_logger``, ``configure_logging``
    Logging utilities.
"""

from .compiler import CompiledJob, CompiledWorkflow, WorkflowCompiler
from .config import InputSpec, JobConfig, WorkflowSpec
from .config_loader import (
    get_config_for_job,
    get_system_param,
    load_project_defaults,
    merge_configs,
    save_project_defaults,
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
from .execution import (
    BackendRuntimeSnapshot,
    CancellationController,
    ExecutionContext,
    HardwareSnapshot,
    ResourceSnapshot,
)
from .persistence import EventStoreProtocol, ProjectStateStore, SessionStoreProtocol
from .plugin_graph import PluginGraphResolver, ResolvedPluginNode
from .progress import (
    ProgressEvent,
    ProgressReporter,
    ProgressSnapshot,
    ProgressTracker,
)
from .project import ProjectContext, ProjectManifest, ProjectPaths
from .protocols import PluginManifest, SubpluginSlot
from .registry import RegistryCenter, RegistryView, registry
from .scan import ParameterGrid, ScanAxisSpec, ScanSpec
from .scheduler import JobResult, Scheduler
from .system_config import (
    SystemConfig,
    SystemConfigStore,
    load_system_config,
    reset_user_config,
    save_user_config,
)
from .workflow import WorkflowCatalog, WorkflowReference, load_workflow

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
    "RegistryView",
    "WorkflowCompiler",
    "CompiledWorkflow",
    "CompiledJob",
    "PluginGraphResolver",
    "ResolvedPluginNode",
    "SessionStoreProtocol",
    "EventStoreProtocol",
    "ProjectStateStore",
    "PluginManifest",
    "SubpluginSlot",
    # Scheduler
    "Scheduler",
    "JobResult",
    # Config
    "JobConfig",
    "WorkflowSpec",
    "InputSpec",
    "ScanSpec",
    "ScanAxisSpec",
    "ParameterGrid",
    "ExecutionContext",
    "CancellationController",
    "HardwareSnapshot",
    "BackendRuntimeSnapshot",
    "ResourceSnapshot",
    "DatasetResultProtocol",
    "DatasetSaveReport",
    "ProjectContext",
    "ProjectManifest",
    "ProjectPaths",
    "WorkflowCatalog",
    "WorkflowReference",
    # System config
    "SystemConfig",
    "SystemConfigStore",
    "load_system_config",
    "reset_user_config",
    "save_user_config",
    "get_system_param",
    # Config loader
    "load_project_defaults",
    "load_workflow",
    "save_project_defaults",
    "merge_configs",
    "get_config_for_job",
]
