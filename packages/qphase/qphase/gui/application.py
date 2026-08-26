"""Application-level composition for the local QPhase console."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from qphase.core.project import ProjectContext
from qphase.core.registry import registry as global_registry
from qphase.core.system_config import SystemConfig, load_system_config
from qphase.service import (
    ConfigService,
    ExecutionManager,
    ProjectService,
    RegistryService,
    SchedulerService,
)


@dataclass
class ApplicationContext:
    project: ProjectContext
    system_config: SystemConfig
    config: ConfigService
    registry: RegistryService
    scheduler: SchedulerService
    executions: ExecutionManager
    project_service: ProjectService

    @classmethod
    def create(
        cls,
        system_config: SystemConfig | None = None,
        project: ProjectContext | None = None,
    ) -> ApplicationContext:
        project_context = project or ProjectContext.discover()
        system = system_config or load_system_config()
        for directory in [project_context.workflow_root, *project_context.plugin_dirs]:
            path = Path(directory).expanduser().resolve()
            for candidate in (path, path.parent):
                value = str(candidate)
                if candidate.exists() and value not in sys.path:
                    sys.path.insert(0, value)
        project_registry = global_registry.snapshot()
        registry = RegistryService(
            registry_center=project_registry,
            system_config=system,
            project=project_context,
        )
        registry.discover(include_local=True)
        scheduler = SchedulerService(
            system,
            project=project_context,
            registry_center=project_registry,
        )
        return cls(
            project_context,
            system,
            ConfigService(project_context, system),
            registry,
            scheduler,
            ExecutionManager(scheduler),
            ProjectService(project_context),
        )

    def close(self) -> None:
        self.executions.close()
