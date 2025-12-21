"""qphase: Core Utilities
---------------------------------------------------------
Provides a collection of shared utility functions used throughout the control layer.
This includes robust YAML parsing (with fallback), deep dictionary merging and
copying for configuration management, and helper functions for Pydantic schema
introspection.

Public API
----------
``load_yaml_file`` : Load YAML with error handling and fallback parser
``deep_merge_dicts``, ``deep_copy`` : Dictionary manipulation utilities
``extract_defaults_from_schema`` : Get default values from Pydantic model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ruamel.yaml import YAML

    _ruamel_yaml: Any = YAML(typ="safe")
except ImportError:
    import yaml as _ruamel_yaml  # type: ignore[import-untyped,no-redef]

from .errors import QPhaseConfigError, QPhaseIOError


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load YAML file using available parser with error handling.

    Parameters
    ----------
    path : Path
        Path to the YAML file

    Returns
    -------
    Dict[str, Any]
        Loaded YAML data as dictionary

    Raises
    ------
    QPhaseIOError
        If file doesn't exist or can't be parsed

    """
    if not path.exists():
        raise QPhaseIOError(f"File not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            if hasattr(_ruamel_yaml, "safe_load"):
                return dict(_ruamel_yaml.safe_load(f) or {})
            elif hasattr(_ruamel_yaml, "load"):
                return dict(_ruamel_yaml.load(f) or {})
            else:
                raise RuntimeError("No YAML parser available")
    except Exception as e:
        raise QPhaseConfigError(f"Failed to parse YAML file {path}: {e}") from e


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override values taking precedence.

    Parameters
    ----------
    base : Dict[str, Any]
        Base dictionary
    override : Dict[str, Any]
        Override dictionary

    Returns
    -------
    Dict[str, Any]
        Merged dictionary

    """
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def deep_copy(data: Any) -> Any:
    """Deep copy a data structure.

    Parameters
    ----------
    data : Any
        Data to copy

    Returns
    -------
    Any
        Deep copy of the data

    """
    if isinstance(data, dict):
        return {key: deep_copy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [deep_copy(item) for item in data]
    else:
        return data


def extract_defaults_from_schema(schema: type[Any]) -> dict[str, Any]:
    """Extract default values from a Pydantic schema.

    Parameters
    ----------
    schema : Type[Any]
        Pydantic model class (BaseModel subclass).

    Returns
    -------
    Dict[str, Any]
        Dictionary of field names to their default values.
        Fields without defaults are omitted.

    """
    defaults: dict[str, Any] = {}

    if not hasattr(schema, "model_fields"):
        return defaults

    for field_name, field_info in schema.model_fields.items():
        # Check if field has a default value (not PydanticUndefined)
        default_val = field_info.default
        if default_val is not None and "PydanticUndefined" not in repr(default_val):
            defaults[field_name] = default_val
        elif field_info.default_factory is not None:
            try:
                defaults[field_name] = field_info.default_factory()
            except Exception:
                pass

    return defaults
