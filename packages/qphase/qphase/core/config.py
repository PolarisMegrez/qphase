"""qphase: Job Configuration Models
---------------------------------------------------------
Defines the Pydantic models that structure job configurations, including the
``JobConfig`` for individual task specification and ``WorkflowSpec`` for an
executable workflow. These models provide validation, default value handling, and
support for parameter scanning specifications.

Public API
----------
JobConfig
    Configuration for a single job with engine, plugins, and parameters.
WorkflowSpec
    Versioned workflow containing one or more logical jobs.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import QPhaseConfigError
from .scan import ScanSpec
from .system_config import SystemConfig
from .utils import deep_merge_dicts


class InputSpec(BaseModel):
    """Structured upstream data selection for one logical job."""

    from_: str = Field(alias="from", min_length=1)
    mode: str = Field(default="dataset", pattern="^(dataset|map)$")
    select: dict[str, Any] | None = None
    group_by: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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
    input: InputSpec | None = Field(
        default=None,
        description="Structured upstream job or external dataset input",
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

    scan: ScanSpec | None = Field(
        default=None,
        description="Explicit parameter scan owned by this logical job",
    )

    # Save control (optional)
    # Controls whether to save the result to disk.
    # True: Save using default filename (output or name)
    # False: Do not save to disk (memory only)
    # String: Save using this specific filename
    # None: Follow system default (auto_save_results)
    save: bool | str | None = Field(
        default=None,
        description="Control result saving behavior (True/False/Filename)",
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

    @model_validator(mode="before")
    @classmethod
    def reject_removed_workflow_syntax(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "aggregate_input" in data:
            raise QPhaseConfigError(
                "aggregate_input was removed; use input: {from, mode, group_by}"
            )
        if isinstance(data.get("input"), str):
            raise QPhaseConfigError(
                "string input syntax was removed; use input: {from: <source>, "
                "mode: dataset|map}"
            )
        if "parameter_scan" in data:
            raise QPhaseConfigError(
                "job parameter_scan was removed; define explicit job.scan axes"
            )
        runtime_shortcuts = {
            "storage",
            "storage_layout",
            "resources",
            "checkpoint",
            "scan_runtime",
        } & set(data)
        if runtime_shortcuts:
            fields = ", ".join(sorted(runtime_shortcuts))
            raise QPhaseConfigError(
                f"job runtime field(s) {fields} are not supported; configure "
                "job.system.scan_runtime using the SystemConfig schema"
            )
        return data

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the job name avoids the catalog id separator.

        ``:`` joins catalog object ids (``artifact_id:session_id:job_name``)
        and occurrence annotation keys (``job_name:artifact_id``), so job
        names must never contain it.
        """
        if ":" in v:
            raise ValueError(
                f"job name {v!r} must not contain ':'; it is reserved as the "
                "catalog object-id separator"
            )
        return v

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that the engine configuration is valid.

        The engine configuration should be a dictionary with engine name as key
        and engine config as value, e.g.,
        {"sde": {"t1": 10.0, "dt": 0.01, "n_traj": 16}}.

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
        merged_dict = deep_merge_dicts(
            global_system.model_dump(),
            self.system.model_dump(exclude_unset=True),
        )

        return SystemConfig(**merged_dict)


class WorkflowSpec(BaseModel):
    """A versioned workflow containing one or more logical jobs."""

    schema_: Literal["qphase.workflow/2"] = Field(alias="schema")
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1)
    description: str | None = None
    collection: str | None = None
    tags: list[str] = Field(default_factory=list)
    jobs: list[JobConfig] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
