"""Job Configuration Models
========================

Defines the Pydantic models that structure job configurations, including the
``JobConfig`` for individual task specification and ``JobList`` for batch execution
containers. These models provide built-in validation, default value handling, and
support for parameter scanning specifications.

Public API
----------
JobConfig
    Configuration for a single job with engine, plugins, and parameters.
JobList
    Container for multiple JobConfig instances.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import QPhaseConfigError
from .system_config import SystemConfig


class JobConfig(BaseModel):
    """Configuration for a single job in the qphase pipeline.

    A job represents a single unit of work to be executed by a
    resource package (e.g., an SDE simulation, a visualization task).

    Attributes
    ----------
    name : str
        Unique name for this job.
    engine : dict[str, Any]
        Engine configuration (must include 'name' field).
    system : SystemConfig | None
        System configuration (overrides global if provided).
    plugins : dict[str, dict[str, Any]]
        Plugin configurations by type (backend, integrator, etc.).
    params : dict[str, Any]
        Job-specific parameters.
    input : str | None
        Input data source (upstream job name or file path).
    output : str | None
        Output destination (downstream job name or filename without extension).
    tags : list[str]
        Tags for job categorization.

    """

    # Basic job information
    name: str = Field(..., description="Unique name for this job")

    # Engine configuration (raw dictionary)
    # Must include 'name' field specifying the engine type (e.g., sde, viz)
    # Will be validated at load time using registry config schemas (plugin.engine.*)
    engine: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine configuration (must include 'name' field)",
    )

    # System configuration (can override global defaults)
    system: SystemConfig | None = Field(
        default=None,
        description="System configuration (overrides global if provided)",
    )

    # Dynamic plugin configurations (raw dictionaries)
    # Will be validated at load time using registry config schemas
    plugins: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Plugin configurations by type (backend, integrator, etc.)",
    )

    # Job-specific parameters
    # This is a flexible field that can contain any parameters
    # specific to the job (model parameters, time settings, etc.)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Job-specific parameters",
    )

    # Input data source (optional)
    # Can be a job name (for dependency) or a file path
    input: str | None = Field(
        default=None,
        description="Input data source (upstream job name or file path)",
    )

    # Output destination (optional)
    # Can be a job name (for passing to downstream job) or a file path
    # (filename only, no extension)
    # If not specified, scheduler will auto-save using job name as filename
    output: str | None = Field(
        default=None,
        description="Output destination (downstream job name or filename "
        "without extension)",
    )

    # Tags for categorization and filtering
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for job categorization",
    )

    # Job dependencies (for future workflow support)
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of job names this job depends on",
    )

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    def __init__(self, **data):
        """Initialize JobConfig and validate plugins.

        Parameters
        ----------
        **data : Any
            Keyword arguments for the configuration.

        """
        super().__init__(**data)
        self._validated_plugins: dict[str, Any] = {}

        # Validate plugins after initialization
        self._validate_plugins()

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that the engine configuration is valid.

        The engine configuration should be a dictionary with engine name as key
        and engine config as value, e.g., {"sde": {"t_end": 10.0}}.

        Parameters
        ----------
        v : dict[str, Any]
            The engine configuration dictionary.

        Returns
        -------
        dict[str, Any]
            The validated engine configuration.

        Raises
        ------
        QPhaseConfigError
            If the configuration is invalid.

        """
        if not isinstance(v, dict):
            raise QPhaseConfigError("Engine configuration must be a dictionary")

        # Check if engine config is provided
        if not v:
            raise QPhaseConfigError("Engine configuration cannot be empty")

        # Validate that exactly one engine is specified
        if len(v) > 1:
            engine_names = ", ".join(v.keys())
            raise QPhaseConfigError(
                f"Job can only use one engine, but multiple were specified: "
                f"{engine_names}"
            )

        # Get the engine name (the only key)
        engine_name = list(v.keys())[0]

        # Validate engine name format
        if (
            not engine_name
            or not str(engine_name)
            .replace("_", "")
            .replace("-", "")
            .replace(".", "")
            .isalnum()
        ):
            raise QPhaseConfigError(
                f"Engine name '{engine_name}' must be alphanumeric (with _ or - or .)"
            )

        # Return a copy with normalized (lowercase) engine name
        v = v.copy()
        normalized_name = str(engine_name).lower()
        engine_config = v.pop(engine_name)  # Remove the old key
        v[normalized_name] = engine_config  # Re-add with normalized name

        return v

    def _validate_plugins(self):
        """Validate all plugin configurations using registry schemas.

        Supports nested format: {plugin_type: {plugin_name: config}}
        This is the standard format used in global.yaml.

        This method validates each plugin configuration against its registered
        schema and populates the _validated_plugins dict with strong-typed config
        objects.

        Raises
        ------
        ValueError
            If a plugin type is not registered or config is invalid

        """
        from qphase.core.registry import registry

        validated = {}

        for plugin_type, config_data in self.plugins.items():
            if not isinstance(config_data, dict):
                raise QPhaseConfigError(
                    f"Plugin config for '{plugin_type}' must be a dictionary"
                )

            # Nested format: {plugin_name: config, ...}
            # Validate each plugin individually
            type_validated = {}
            for plugin_name, plugin_config in config_data.items():
                if not isinstance(plugin_config, dict):
                    raise QPhaseConfigError(
                        f"Plugin config for '{plugin_type}:{plugin_name}' "
                        "must be a dictionary"
                    )

                # Convert to flat format with name
                flat_config = dict(plugin_config)
                flat_config["name"] = plugin_name

                try:
                    type_validated[plugin_name] = registry.validate_plugin_config(
                        plugin_type, flat_config
                    )
                except Exception as e:
                    raise QPhaseConfigError(
                        f"Invalid configuration for plugin "
                        f"'{plugin_type}:{plugin_name}': {e}"
                    ) from e

            validated[plugin_type] = type_validated

        self._validated_plugins = validated

    def get_plugin_config(self, plugin_type: str) -> Any | None:
        """Get validated plugin configuration for a specific plugin type.

        Parameters
        ----------
        plugin_type : str
            Type of plugin (backend, integrator, noise, etc.)

        Returns
        -------
        Any | None
            Validated plugin configuration object if found, None otherwise.
            The return type depends on the registered config schema.

        """
        return self._validated_plugins.get(plugin_type)

    def get_all_plugin_configs(self) -> dict[str, Any]:
        """Get all validated plugin configurations as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping plugin type names to their validated configurations.

        """
        return dict(self._validated_plugins)

    def get_engine_name(self) -> str:
        """Get the engine name from the engine configuration.

        The engine configuration is a dictionary where the key is the engine name
        and the value is the engine configuration.

        Returns
        -------
        str
            The engine name (e.g., 'sde', 'viz')

        """
        if not self.engine:
            return ""
        return list(self.engine.keys())[0]

    def merge_with_system_config(self, global_system: SystemConfig) -> SystemConfig:
        """Merge job's system config with global system config.

        Job-specific system config takes precedence over global.

        Parameters
        ----------
        global_system : SystemConfig
            Global system configuration

        Returns
        -------
        SystemConfig
            Merged system configuration

        """
        if self.system is None:
            return global_system

        # Merge: job system overrides global system
        merged_dict = global_system.model_dump()
        merged_dict.update(self.system.model_dump(exclude_unset=True))

        return SystemConfig(**merged_dict)


class JobList(BaseModel):
    """Configuration for a list of jobs to be executed.

    This model represents a collection of jobs that can be executed
    together as part of a session or workflow.

    Attributes
    ----------
    jobs : list[JobConfig]
        List of job configurations.
    system : SystemConfig | None
        Global system configuration (applied to all jobs unless overridden).
    name : str | None
        Name of this job list/workflow.
    description : str | None
        Description of this job list/workflow.

    """

    # List of jobs
    jobs: list[JobConfig] = Field(..., description="List of job configurations")

    # Global system configuration (applied to all jobs unless overridden)
    system: SystemConfig | None = Field(
        default=None,
        description="Global system configuration",
    )

    # Workflow metadata
    name: str | None = Field(
        default=None,
        description="Name of this job list/workflow",
    )

    description: str | None = Field(
        default=None,
        description="Description of this job list/workflow",
    )

    model_config = ConfigDict(extra="allow")
