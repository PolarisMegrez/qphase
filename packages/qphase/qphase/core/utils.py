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

from pydantic_core import PydanticUndefined
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from .errors import QPhaseConfigError, QPhaseIOError

_ruamel_yaml: Any = YAML(typ="safe")


def load_yaml(path: Path) -> dict[str, Any]:
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
            return dict(_ruamel_yaml.load(f) or {})
    except Exception as e:
        raise QPhaseConfigError(f"Failed to parse YAML file {path}: {e}") from e


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save YAML using available library."""
    try:
        from ruamel.yaml import YAML

        y = YAML()
        with open(path, "w", encoding="utf-8") as f:
            y.dump(data, f)
    except Exception as e:
        raise QPhaseIOError(f"Failed to save config to {path}: {e}") from e


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


def schema_to_yaml_map(
    model_cls: type[Any], existing_values: dict[str, Any], plugin_name: str
) -> CommentedMap:
    """Convert a Pydantic model class to a CommentedMap for YAML output.

    Merges existing values from global config if provided. The 'name' field is
    excluded as it's implicit in the nested config structure.

    Parameters
    ----------
    model_cls : type
        Pydantic model class with model_fields
    existing_values : dict[str, Any]
        Existing values from global.yaml to merge
    plugin_name : str
        Plugin name (unused, kept for API compatibility)

    Returns
    -------
    CommentedMap
        YAML-compatible map with field descriptions as comments

    """
    data = CommentedMap()

    for field_name, field in model_cls.model_fields.items():
        # Determine value
        if field_name in existing_values:
            value = existing_values[field_name]
        elif field.default is not PydanticUndefined:
            value = field.default
        elif field.default_factory is not None:
            try:
                value = field.default_factory()
            except Exception:
                value = "<generated>"
        else:
            type_str = str(field.annotation).replace("typing.", "")
            value = f"[REQUIRED] {type_str}"

        data[field_name] = value

        # Add comment
        if field.description:
            comment = field.description
            data.yaml_add_eol_comment(comment, field_name)

    return data
