"""qphase: Protocol Definitions
---------------------------------------------------------
Defines the structural contracts (Protocols) that underpin the plugin architecture.
It specifies the interfaces for configuration models (``PluginConfigBase``), plugin
implementations (``PluginBase``), execution engines (``EngineBase``), and result
containers (``ResultProtocol``), enabling type checking and documentation while
supporting duck typing for resource packages.

Public API
----------
PluginConfigBase
    Base Pydantic model for plugin configuration.
PluginBase
    Protocol for plugin implementation classes.
EngineBase
    Protocol for engine classes with run() method.
ResultProtocol
    Protocol for serializable result containers.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# Deprecated callback retained for one compatibility cycle. New engines report
# natural work counts through ``ExecutionContext.progress``.
LegacyProgressCallback = Callable[[float | None, float | None, str, str | None], None]
ProgressCallback = LegacyProgressCallback


@dataclass(frozen=True)
class SubpluginSlot:
    """A named child-plugin selection owned by one parent plugin."""

    namespace: str
    cardinality: Literal["one", "optional", "many"] = "one"
    default: str | None = None
    protocol: str | None = None
    allowed: frozenset[str] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate slot cardinality and namespace invariants."""
        if not self.namespace.strip():
            raise ValueError("subplugin namespace must not be empty")
        if self.cardinality == "many" and self.default is not None:
            raise ValueError("a many-valued subplugin slot cannot have one default")
        if self.cardinality == "optional" and self.default is not None:
            raise ValueError("an optional subplugin slot cannot have a default")


@dataclass
class PluginManifest:
    """Relationships between a parent plugin and its child-plugin slots."""

    subplugins: Mapping[str, SubpluginSlot] = field(default_factory=dict)
    schema_version: str = "1.0"


@dataclass
class EngineManifest(PluginManifest):
    """Manifest declaring engine dependencies.

    Attributes
    ----------
    required_plugins : set[str]
        Required plugin types (e.g., {'backend', 'model'}).
    optional_plugins : set[str]
        Optional plugin types (e.g., {'integrator', 'analyzer'}).
    defaults : dict[str, str]
        Default plugin implementations (e.g., {'integrator': 'euler_maruyama'}).

    """

    # Required plugin types (e.g., {'backend', 'model'}).
    # Use an empty set if the engine does not enforce any required plugins.
    required_plugins: set[str] = field(default_factory=set)
    # Optional plugin types (e.g., {'integrator', 'analyzer'})
    optional_plugins: set[str] = field(default_factory=set)
    # Default plugin implementations (e.g., {'integrator': 'euler_maruyama'})
    defaults: dict[str, str] = field(default_factory=dict)
    # Plugins required when the job provides an upstream input (e.g. analyze mode).
    # If non-empty, scheduler will validate against these namespaces instead of
    # ``required_plugins`` when ``job.input`` is set.
    input_plugins: set[str] = field(default_factory=set)


class PluginConfigBase(BaseModel):
    """Base configuration class for all plugins.

    All plugin configuration classes should inherit from this class.
    This is a minimal base class that provides Pydantic validation
    and serialization capabilities.

    Plugin configurations are simple parameter containers that are
    passed to plugin __init__ methods. They do not contain plugin
    metadata like name or description.
    """

    # Pydantic v2 configuration: allow extra fields by default to be
    # tolerant to user-provided / future fields in plugin configs.
    model_config = ConfigDict(extra="allow")


@runtime_checkable
class PluginBase(Protocol):
    """Protocol for QPhase plugins.

    Plugins are components loaded by the Engine to perform specific tasks.
    They must define:
    - name: ClassVar[str] - Unique identifier for the plugin
    - description: ClassVar[str] - Human-readable description (can be empty)
    - config_schema: ClassVar[type[Any]] - Configuration schema class
    - __init__(config, **kwargs) - Initialize with config instance
    """

    # Plugin metadata (must be defined as class variables)
    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[Any]]

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        """Initialize the plugin with a validated configuration object."""
        ...


# Protocol definition for result objects
# This allows any object that implements data, metadata, and save() to be used as
# a result object
@runtime_checkable
class ResultProtocol(Protocol):
    """Protocol for result objects."""

    @property
    def data(self) -> Any: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def label(self) -> Any: ...

    def save(self, path: str | Path) -> None: ...


@runtime_checkable
class EngineBase(PluginBase, Protocol):
    """Protocol for the main application engine.

    The Engine is responsible for managing plugins, configuring the environment,
    and executing the main computational workflow. It follows the Plugin pattern
    for configuration but adds a `run` method.

    The Engine is the entry point for a Resource Package. It is instantiated
    by the Scheduler via the Registry.
    """

    manifest: ClassVar[EngineManifest]

    def __init__(self, config: Any, plugins: dict[str, Any], **kwargs: Any) -> None:
        """Initialize the Engine.

        Parameters
        ----------
        config : Any
            The validated Engine configuration object (Pydantic model).
        plugins : Dict[str, Any]
            A dictionary of instantiated plugins (backend, integrator, etc.).
        **kwargs : Any
            Additional keyword arguments for future extensibility.

        """
        ...

    def run(
        self,
        data: Any | None = None,
        *,
        context: Any | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> ResultProtocol:
        """Execute the main computational task and return the result.

        Parameters
        ----------
        data : Any | None
            Input data from upstream jobs or external sources.
            Can be a Python object (in-memory transfer) or a Path (file transfer).
        context : ExecutionContext | None, optional
            Scheduler-owned runtime services and an optional parameter grid.
            Engines report work through ``context.progress``.
        progress_cb : ProgressCallback | None, optional
            Deprecated percent-based compatibility callback. New engines must
            consume ``context`` instead of estimating durations themselves.

        Returns
        -------
        ResultProtocol
            The result of the computation.

        """
        ...
