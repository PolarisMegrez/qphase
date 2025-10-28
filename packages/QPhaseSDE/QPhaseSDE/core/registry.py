from __future__ import annotations

"""Unified registry and factory center for all pluggable modules.

Design goals:
- Single source of truth for registration across namespaces (integrator, noise_model, backend, visualization)
- Self-registration via decorator when modules are imported
- Unified factory: create("namespace:name", **kwargs)
- Namespace isolation and explicit naming
- Lazy and configurable loading via dotted-path registration
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple

from .errors import (
    ConfigError,
    ConflictError,
    ImportBackendError,
    ImportVisualizerError,
)


Namespace = str
Name = str
FullName = str
Builder = Callable[..., Any]


@dataclass
class _Entry:
    kind: str  # "callable" | "dotted"
    builder: Optional[Builder] = None
    target: Optional[str] = None  # dotted path like "pkg.mod:Class" or "pkg.mod.func"
    meta: Dict[str, Any] = field(default_factory=dict)  # optional metadata


class RegistryCenter:
    """Central registry for all plugin types.

    registry[namespace][name] -> _Entry
    """

    VALID_NAMESPACES = {"integrator", "noise_model", "backend", "visualization", "default"}

    def __init__(self) -> None:
        self._tables: Dict[Namespace, Dict[Name, _Entry]] = {}

    # --------------------------- utilities ---------------------------
    @staticmethod
    def _split(full_name: FullName) -> Tuple[Namespace, Name]:
        if ":" in full_name:
            ns, nm = full_name.split(":", 1)
            return ns.strip().lower(), nm.strip().lower()
        # default namespace when omitted
        return "default", full_name.strip().lower()

    def _ensure_ns(self, namespace: Namespace) -> Dict[Name, _Entry]:
        ns = namespace.strip().lower()
        if ns not in self.VALID_NAMESPACES:
            # Permit ad-hoc namespaces but warn via ConfigError on access
            # Here we accept to enable external plugins, but still separate tables
            self.VALID_NAMESPACES.add(ns)
        return self._tables.setdefault(ns, {})

    # --------------------------- registration ---------------------------
    def register(self, namespace: Namespace, name: Name, builder: Builder, *, overwrite: bool = False, **meta: Any) -> None:
        ns = namespace.strip().lower()
        nm = name.strip().lower()
        table = self._ensure_ns(ns)
        if not overwrite and nm in table:
            raise ConflictError(f"[600] Duplicate registration: {ns}:{nm}")
        full_meta = dict(meta or {})
        full_meta.setdefault("registered_at", datetime.now(UTC).isoformat())
        full_meta.setdefault("builder_type", self._infer_builder_type(builder))
        full_meta.setdefault("delayed_import", False)
        table[nm] = _Entry(kind="callable", builder=builder, target=None, meta=full_meta)

    def register_lazy(self, namespace: Namespace, name: Name, target: str, *, overwrite: bool = False, **meta: Any) -> None:
        """Register by dotted path without importing until create().

        target examples:
        - "pkg.module:ClassName"
        - "pkg.module:function_name"
        - "pkg.module.ClassName" (dot form also supported)
        """
        ns = namespace.strip().lower()
        nm = name.strip().lower()
        table = self._ensure_ns(ns)
        if not overwrite and nm in table:
            raise ConflictError(f"[601] Duplicate lazy registration: {ns}:{nm}")
        full_meta = dict(meta or {})
        full_meta.setdefault("registered_at", datetime.now(UTC).isoformat())
        full_meta.setdefault("builder_type", "dotted")
        full_meta.setdefault("delayed_import", True)
        full_meta.setdefault("module_path", target)
        table[nm] = _Entry(kind="dotted", builder=None, target=str(target), meta=full_meta)

    # --------------------------- decorators ---------------------------
    def decorator(self, namespace: Namespace, name: Name, **meta: Any):
        """Class/function decorator to auto-register on import."""
        def _wrap(obj: Any):
            self.register(namespace, name, obj, **meta)
            return obj
        return _wrap

    # Provide a shorter alias name
    def register_decorator(self, namespace: Namespace, name: Name, **meta: Any):
        return self.decorator(namespace, name, **meta)

    # --------------------------- factory ---------------------------
    def create(self, full_name: FullName, /, **kwargs: Any) -> Any:
        ns, nm = self._split(full_name)
        table = self._tables.get(ns, {})
        entry = table.get(nm)
        if entry is None:
            raise ConfigError(f"[602] Unknown registry key: {ns}:{nm}")
        if entry.kind == "callable":
            assert entry.builder is not None
            # Optionally return the callable itself without invoking
            if entry.meta.get("return_callable"):
                return entry.builder
            return entry.builder(**kwargs)
        # dotted path import
        assert entry.target is not None
        try:
            obj = self._import_target(entry.target)
        except Exception as e:
            if ns == "backend":
                raise ImportBackendError(f"[604] Failed to import backend '{nm}' from '{entry.target}': {e}")
            if ns == "visualization":
                raise ImportVisualizerError(f"[605] Failed to import visualizer '{nm}' from '{entry.target}': {e}")
            raise
        if entry.meta.get("return_callable"):
            return obj
        return obj(**kwargs) if callable(obj) else obj

    def _import_target(self, target: str) -> Any:
        # Support both module:attr and module.attr forms
        module_name: str
        attr_name: Optional[str] = None
        if ":" in target:
            module_name, attr_name = target.split(":", 1)
        else:
            # split last dot as attribute
            if "." in target:
                parts = target.rsplit(".", 1)
                module_name = parts[0]
                attr_name = parts[1]
            else:
                module_name = target
                attr_name = None
        mod = import_module(module_name)
        if attr_name is None:
            return mod
        if not hasattr(mod, attr_name):
            raise ConfigError(f"[603] Target '{target}' not found")
        return getattr(mod, attr_name)

    # --------------------------- introspection ---------------------------
    def list(self, namespace: Optional[Namespace] = None) -> Dict[str, Any]:
        if namespace is None:
            # map of ns -> list of names
            return {ns: sorted(list(tbl.keys())) for ns, tbl in self._tables.items()}
        ns = namespace.strip().lower()
        table = self._tables.get(ns, {})
        return {
            name: {
                "kind": ("callable" if e.kind == "callable" else "dotted"),
                **(e.meta or {}),
            }
            for name, e in table.items()
        }

    # --------------------------- helpers ---------------------------
    @staticmethod
    def _infer_builder_type(obj: Any) -> str:
        try:
            if callable(obj):
                # class vs function detection
                return "class" if hasattr(obj, "__mro__") else "function"
        except Exception:
            pass
        return type(obj).__name__.lower()


# Global singleton
registry = RegistryCenter()


# Convenience export: decorator function
def register(namespace: Namespace, name: Name, **meta: Any):  # decorator form
    return registry.decorator(namespace, name, **meta)


def register_lazy(namespace: Namespace, name: Name, target: str, **meta: Any) -> None:
    registry.register_lazy(namespace, name, target, **meta)
