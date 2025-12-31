"""Protocol Definitions
====================

Defines the structural contracts (Protocols) that underpin the plugin architecture.
It specifies the interfaces for configuration models (``PluginConfigBase``), plugin
implementations (``PluginBase``), execution engines (``EngineBase``), and result
containers (``ResultBase``), enabling type checking and documentation while
supporting duck typing for resource packages.

Public API
----------
PluginConfigBase
    Base Pydantic model for plugin configuration.
PluginBase
    Protocol for plugin implementation classes.
EngineBase
    Protocol for engine classes with run() method.
ResultBase
    Base class for serializable result containers.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# Self type for factory methods
_R = TypeVar("_R", bound="ResultBase")

# Progress callback signature
# args: percent (0.0-1.0), total_duration_estimate (s), message, stage
ProgressCallback = Callable[[float | None, float | None, str, str | None], None]


@dataclass
class EngineManifest:
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

    # Required plugin types (e.g., {'backend', 'model'})
    required_plugins: set[str]
    # Optional plugin types (e.g., {'integrator', 'analyzer'})
    optional_plugins: set[str] = field(default_factory=set)
    # Default plugin implementations (e.g., {'integrator': 'euler_maruyama'})
    defaults: dict[str, str] = field(default_factory=dict)


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


class ResultBase(BaseModel):
    """Base class for application results with standardized I/O.

    Results must be able to save themselves to disk and load themselves back.

    Notes
    -----
    The save() and load() methods should treat the path as a filename without
    extension. The implementation is responsible for adding the appropriate
    file extension based on the chosen format.

    This class can be used as a concrete base class for implementing results.
    For type checking against the result interface, use
    isinstance(obj, ResultProtocol).

    """

    data: Any = Field(description="The actual output data from engine execution")

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the result"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "allow"

    def save(self, path: str | Path) -> None:
        """Save the result to disk.

        Parameters
        ----------
        path : str | Path
            Path where result should be saved, without file extension.
            The implementation should add the appropriate extension based on
            the
            chosen file format (e.g., '.json', '.npz', '.h5').
            For example, if path is '/results/simulation',
            save to '/results/simulation.json'.

        """
        raise NotImplementedError("Subclasses must implement save()")

    @classmethod
    def load(cls: type[_R], path: str | Path) -> _R:
        """Load the result from disk.

        Parameters
        ----------
        path : str | Path
            Path where result was previously saved, without file extension.
            The implementation should try common extensions or use the same
            extension that was used during save().

        Returns
        -------
        _R
            Loaded result instance

        """
        raise NotImplementedError("Subclasses must implement load()")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"data_type={type(self.data).__name__}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )


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
        progress_cb: ProgressCallback | None = None,
    ) -> ResultProtocol:
        """Execute the main computational task and return the result.

        Parameters
        ----------
        data : Any | None
            Input data from upstream jobs or external sources.
            Can be a Python object (in-memory transfer) or a Path (file transfer).
        progress_cb : ProgressCallback | None, optional
            Callback for progress reporting.
            Signature: (percent, total_duration_estimate, message, stage) -> None
            - percent: float | None (0.0-1.0)
            - total_duration_estimate: float | None (seconds)
            - message: str (status message)
            - stage: str | None (e.g., "warmup", "sampling")

        Returns
        -------
        ResultProtocol
            The result of the computation.

        """
        ...
