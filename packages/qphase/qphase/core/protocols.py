"""qphase: Protocol Definitions
---------------------------------------------------------
Defines the structural contracts (Protocols) that underpin the plugin architecture.
It specifies the interfaces for configuration models (``PluginConfigBase``), plugin
implementations (``PluginBase``), execution engines (``EngineBase``), and result
containers (``ResultBase``), enabling type checking and documentation while
supporting duck typing for resource packages.

Public API
----------
``PluginConfigBase`` : Base Pydantic model for plugin configuration
``PluginBase`` : Protocol for plugin implementation classes
``EngineBase`` : Protocol for engine classes with run() method
``ResultBase`` : Base class for serializable result containers

Notes
-----
- These protocols define structural contracts for type checking and documentation
- Duck typing is supported; resource packages need not inherit from them at runtime

"""

from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Self type for factory methods
_R = TypeVar("_R", bound="ResultBase")


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

    @classmethod
    def from_raw(
        cls: type["PluginConfigBase"], raw: Any | None = None
    ) -> "PluginConfigBase":
        """Normalize and validate `raw` into an instance of this config class.

        Accepts:
        - None -> produce default instance (using defaults)
        - dict-like -> validate and construct
        - already an instance of cls -> returned as-is

        Raises:
        - pydantic.ValidationError on invalid input (preserved)

        """
        # 1) None -> produce defaults
        if raw is None:
            return cls.model_validate({})

        # 2) already the correct model instance
        if isinstance(raw, cls):
            return raw

        # 3) validate dict-like input
        try:
            return cls.model_validate(raw)
        except ValidationError:
            # let callers handle the ValidationError; preserve traceback
            raise


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

    def run(self, data: Any | None = None) -> ResultBase | Any:
        """Execute the main computational task and return the result.

        Parameters
        ----------
        data : Any | None
            Input data from upstream jobs or external sources.
            Can be a Python object (in-memory transfer) or a Path (file transfer).

        Returns
        -------
        ResultBase | Any
            The result of the computation.

        """
        ...

    # ===== OPTIONAL PROGRESS METHODS =====
    # These methods are OPTIONAL. Engine packages may implement them
    # to provide progress reporting, but are NOT required to do so.
    # Engine packages should NOT import anything from qphase.core.

    def get_progress(self) -> dict[str, Any] | None:
        """Get current progress state.

        OPTIONAL: If implemented, should return a dict with progress information.

        Expected dict keys (all optional):
        - 'percent': float (0.0-100.0)
        - 'message': str - Current status message
        - 'stage': str - Current stage name
        - 'total_duration_estimate': float - Estimated total duration
          (including elapsed time)

        Returns
        -------
        dict[str, Any] | None
            Progress state dict, or None if progress reporting not supported.

        """
        return None
