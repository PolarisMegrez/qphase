"""Job Expansion Logic
===================

Handles the expansion of parameter scans into multiple job configurations.

Public API
----------
JobExpander
    Class for expanding jobs based on parameter scans.
"""

from __future__ import annotations

import copy
from itertools import product
from typing import Any

from .config import JobConfig
from .errors import QPhaseConfigError
from .registry import RegistryCenter


class JobExpander:
    """Expands jobs based on scanable parameters.

    Parameters
    ----------
    registry : RegistryCenter
        The registry center to look up scanable parameters.

    """

    def __init__(self, registry: RegistryCenter):
        self.registry = registry

    def expand(self, job: JobConfig, method: str = "cartesian") -> list[JobConfig]:
        """Expand a job into multiple jobs based on scanable parameters.

        Parameters
        ----------
        job : JobConfig
            Job configuration to expand
        method : str, optional
            Expansion method ('cartesian' or 'zipped'), by default "cartesian"

        Returns
        -------
        list[JobConfig]
            List of expanded job configurations

        Raises
        ------
        QPhaseConfigError
            If method is invalid or parameters have mismatched lengths (zipped)

        """
        scanable_params = self._detect_scanable_params(job)

        if not scanable_params:
            return [job]

        if method == "cartesian":
            return self._cartesian_expand(job, scanable_params)
        elif method == "zipped":
            return self._zipped_expand(job, scanable_params)
        else:
            raise QPhaseConfigError(
                f"Invalid parameter scan method '{method}'. "
                "Must be 'cartesian' or 'zipped'."
            )

    def _detect_scanable_params(self, job: JobConfig) -> dict[str, list[Any]]:
        """Detect parameters marked as scanable in job configuration."""
        scanable_params = {}

        if not self.registry:
            return {}

        # Check engine plugin
        engine_name = job.get_engine_name()
        if engine_name and engine_name in job.engine:
            engine_config = job.engine[engine_name]
            scanable_info = self.registry.get_scanable_params("engine", engine_name)

            for param_name, is_scanable in scanable_info.items():
                if is_scanable and param_name in engine_config:
                    value = engine_config[param_name]
                    if isinstance(value, list) and len(value) > 0:
                        param_path = f"engine.{engine_name}.{param_name}"
                        scanable_params[param_path] = value

        # Check other plugin types
        if job.plugins:
            for plugin_type, plugin_config in job.plugins.items():
                if isinstance(plugin_config, dict):
                    for plugin_name, config in plugin_config.items():
                        scanable_info = self.registry.get_scanable_params(
                            plugin_type, plugin_name
                        )
                        for param_name, is_scanable in scanable_info.items():
                            if is_scanable and param_name in config:
                                value = config[param_name]
                                if isinstance(value, list) and len(value) > 0:
                                    param_path = (
                                        f"{plugin_type}.{plugin_name}.{param_name}"
                                    )
                                    scanable_params[param_path] = value

        return scanable_params

    def _cartesian_expand(
        self, job: JobConfig, scanable_params: dict[str, list[Any]]
    ) -> list[JobConfig]:
        """Expand job using cartesian product of all scanable parameters."""
        param_names = list(scanable_params.keys())
        param_values = list(scanable_params.values())

        combinations = list(product(*param_values))

        expanded_jobs = []
        for combo in combinations:
            new_job = self._copy_job_config(job)
            for param_name, param_value in zip(param_names, combo, strict=True):
                self._set_nested_param(new_job, param_name, param_value)
            expanded_jobs.append(new_job)

        return expanded_jobs

    def _zipped_expand(
        self, job: JobConfig, scanable_params: dict[str, list[Any]]
    ) -> list[JobConfig]:
        """Expand job using zipped (aligned) expansion of scanable parameters."""
        param_names = list(scanable_params.keys())
        param_values = list(scanable_params.values())

        lengths = [len(v) for v in param_values]
        if len(set(lengths)) > 1:
            raise QPhaseConfigError(
                f"Job '{job.name}' has scanable parameters with different "
                f"lengths: "
                f"{dict(zip(param_names, lengths, strict=True))}. "
                f"For zipped expansion, all scanable parameters must have "
                f"the same length."
            )

        num_jobs = lengths[0]
        expanded_jobs = []

        for i in range(num_jobs):
            new_job = self._copy_job_config(job)
            for param_name, param_list in zip(param_names, param_values, strict=True):
                self._set_nested_param(new_job, param_name, param_list[i])
            expanded_jobs.append(new_job)

        return expanded_jobs

    def _copy_job_config(self, job: JobConfig) -> JobConfig:
        """Create a deep copy of a job configuration."""
        return copy.deepcopy(job)

    def _set_nested_param(self, job: JobConfig, param_path: str, value: Any) -> None:
        """Set a parameter value using dot notation for nested parameters."""
        parts = param_path.split(".")

        if len(parts) >= 3 and parts[0] == "engine":
            engine_name = parts[1]
            param_name = parts[2]
            if engine_name not in job.engine:
                job.engine[engine_name] = {"name": engine_name}
            job.engine[engine_name][param_name] = value

        elif len(parts) >= 3:
            plugin_type = parts[0]
            plugin_name = parts[1]
            param_name = parts[2]

            if job.plugins is None:
                job.plugins = {}
            if plugin_type not in job.plugins:
                job.plugins[plugin_type] = {}
            if plugin_name not in job.plugins[plugin_type]:
                job.plugins[plugin_type][plugin_name] = {}

            job.plugins[plugin_type][plugin_name][param_name] = value
