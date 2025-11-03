"""
QPhaseSDE: Registry
-------------------
Lightweight, centralized registries for classes and functions across QPhaseSDE
namespaces (integrator, noise_model, backend, visualization) with a unified
factory interface.

Behavior
--------
- Provide a single source of truth per namespace using keys of the form
  "namespace:name"; support eager registration and dotted-path lazy
  registration; the factory resolves entries and returns either the callable
  itself or an instantiated object according to metadata.

Notes
-----
- Error codes and import/lookup semantics are documented in the corresponding
  function/method docstrings. Dotted import supports both `module:attr` and
  `module.attr` forms.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple
from .errors import (
    QPSConfigError,
    QPSRegistryError,
)

__all__ = [
    "RegistryCenter",
    "registry",
    "register",
    "register_lazy",
]

Namespace = str
Name = str
FullName = str
Builder = Callable[..., Any]

@dataclass
class _Entry:
    """Internal record describing a registry entry.

    Stores either a direct builder (callable) or a dotted import target with
    associated metadata. Not part of the public API.
    """
    kind: str  # "callable" | "dotted"
    builder: Optional[Builder] = None
    target: Optional[str] = None  # dotted path like "pkg.mod:Class" or "pkg.mod.func"
    meta: Dict[str, Any] = field(default_factory=dict)  # optional metadata

class RegistryCenter:
    """Central registry for plugin types with factory-style lookup.

    Maintains per-namespace tables that map names to either callables or dotted
    import targets. Provides registration, lazy registration, decorator helpers,
    factory creation, and introspection.

    Attributes
    ----------
    VALID_NAMESPACES : set[str]
        Known namespaces. Ad-hoc namespaces are permitted and added on demand.

    Methods
    -------
    register(namespace, name, builder, *, overwrite=False, **meta) -> None
        Register a callable builder immediately. Raises QPSRegistryError on duplicates
        (- [400]).
    register_lazy(namespace, name, target, *, overwrite=False, **meta) -> None
        Register a dotted path for deferred import. Raises QPSRegistryError on duplicates
        (- [401]).
    decorator(namespace, name, **meta) -> Callable
        Return a decorator that registers the decorated object on import.
    create(full_name, /, **kwargs) -> Any
        Resolve and construct entries given "namespace:name". Raises QPSConfigError
        on unknown keys (- [404]) and import errors for registered targets
        (- [402]).
    list(namespace=None) -> Dict[str, Any]
        List available entries with metadata for a namespace or all.

    Examples
    --------
    >>> rc = RegistryCenter()
    >>> rc.register("default", "adder", lambda x, y: x + y, return_callable=True)
    >>> add = rc.create("default:adder")
    >>> add(1, 2)
    3
    """

    VALID_NAMESPACES = {"integrator", "noise_model", "backend", "visualization", "default"}

    def __init__(self) -> None:
        self._tables: Dict[Namespace, Dict[Name, _Entry]] = {}

    # --------------------------- utilities ---------------------------
    @staticmethod
    def _split(full_name: FullName) -> Tuple[Namespace, Name]:
        """Split a full key into namespace and name.

        Parameters
        ----------
        full_name : str
            Key in the form ``"namespace:name"`` or just ``"name"``.

        Returns
        -------
        tuple[str, str]
            A pair ``(namespace, name)`` normalized to lowercase; when the
            namespace is omitted, ``"default"`` is used.
        """
        if ":" in full_name:
            ns, nm = full_name.split(":", 1)
            return ns.strip().lower(), nm.strip().lower()
        # default namespace when omitted
        return "default", full_name.strip().lower()

    def _ensure_ns(self, namespace: Namespace) -> Dict[Name, _Entry]:
        """Ensure a namespace table exists and return it.

        Ad-hoc namespaces are permitted and added on demand to allow external
        plugins while keeping tables isolated.
        """
        ns = namespace.strip().lower()
        if ns not in self.VALID_NAMESPACES:
            # Permit ad-hoc namespaces but warn via QPSConfigError on access
            # Here we accept to enable external plugins, but still separate tables
            self.VALID_NAMESPACES.add(ns)
        return self._tables.setdefault(ns, {})

    # --------------------------- registration ---------------------------
    def register(self, namespace: Namespace, name: Name, builder: Builder, *, overwrite: bool = False, **meta: Any) -> None:
        """Register a callable builder immediately under a namespace.

        Parameters
        ----------
        namespace : str
            Target namespace (e.g., ``"integrator"``, ``"backend"``).
        name : str
            Public key under which to register the builder (case-insensitive).
        builder : Callable[..., Any]
            Class or function that constructs/returns the registered object.
        overwrite : bool, default False
            When False, duplicate keys raise a conflict; when True, existing
            entries are replaced.
        **meta : Any
            Optional metadata stored with the entry (e.g., ``return_callable``,
            ``tags``). Some fields are auto-filled (``registered_at``,
            ``builder_type``, ``delayed_import``).

        Raises
        ------
        QPSRegistryError
            - [400] Duplicate registration for the same ``namespace:name`` when
              ``overwrite`` is False.

        Examples
        --------
        >>> rc = RegistryCenter()
        >>> rc.register("default", "adder", lambda x, y: x + y, return_callable=True)
        >>> add = rc.create("default:adder")
        >>> add(1, 2)
        3
        """
        ns = namespace.strip().lower()
        nm = name.strip().lower()
        table = self._ensure_ns(ns)
        if not overwrite and nm in table:
            raise QPSRegistryError(f"[400] Duplicate registration: {ns}:{nm}")
        full_meta = dict(meta or {})
        full_meta.setdefault("registered_at", datetime.now(UTC).isoformat())
        full_meta.setdefault("builder_type", self._infer_builder_type(builder))
        full_meta.setdefault("delayed_import", False)
        table[nm] = _Entry(kind="callable", builder=builder, target=None, meta=full_meta)

    def register_lazy(self, namespace: Namespace, name: Name, target: str, *, overwrite: bool = False, **meta: Any) -> None:
        """Register by dotted path without importing until ``create()``.

        Parameters
        ----------
        namespace : str
            Target namespace (e.g., ``"integrator"``, ``"backend"``).
        name : str
            Public key under which to register the target (case-insensitive).
        target : str
            Dotted path to the builder, e.g. ``"pkg.module:ClassName"``,
            ``"pkg.module:function_name"`` or ``"pkg.module.ClassName"``.
        overwrite : bool, default False
            When False, duplicate keys raise a conflict; when True, existing
            entries are replaced.
        **meta : Any
            Optional metadata stored with the entry. The field
            ``delayed_import=True`` is implied for lazy registrations.

        Raises
        ------
        QPSRegistryError
            - [401] Duplicate lazy registration for the same ``namespace:name``
              when ``overwrite`` is False.

        Examples
        --------
        >>> rc = RegistryCenter()
        >>> rc.register_lazy("default", "Adder", "mylib.adders:Adder")
        >>> cls = rc.create("default:Adder")  # doctest: +SKIP
        """
        ns = namespace.strip().lower()
        nm = name.strip().lower()
        table = self._ensure_ns(ns)
        if not overwrite and nm in table:
            raise QPSRegistryError(f"[401] Duplicate lazy registration: {ns}:{nm}")
        full_meta = dict(meta or {})
        full_meta.setdefault("registered_at", datetime.now(UTC).isoformat())
        full_meta.setdefault("builder_type", "dotted")
        full_meta.setdefault("delayed_import", True)
        full_meta.setdefault("module_path", target)
        table[nm] = _Entry(kind="dotted", builder=None, target=str(target), meta=full_meta)

    # --------------------------- decorators ---------------------------
    def decorator(self, namespace: Namespace, name: Name, **meta: Any):
        """Return a decorator that registers the object on import."""
        def _wrap(obj: Any):
            self.register(namespace, name, obj, **meta)
            return obj
        return _wrap

    # Provide a shorter alias name
    def register_decorator(self, namespace: Namespace, name: Name, **meta: Any):
        """Alias of ``decorator`` for convenience."""
        return self.decorator(namespace, name, **meta)

    # --------------------------- factory ---------------------------
    def create(self, full_name: FullName, /, **kwargs: Any) -> Any:
        """Resolve and construct entries given ``"namespace:name"`` keys.

        Parameters
        ----------
        full_name : str
            Key of the form ``"namespace:name"``. If the namespace is omitted,
            the default namespace ``"default"`` is assumed.
        **kwargs : Any
            Keyword arguments forwarded to the builder/constructor when the
            entry refers to a callable object and ``return_callable`` is not set.

        Returns
        -------
        Any
            - If the entry metadata includes ``return_callable=True``, returns
              the callable itself without invoking it.
            - Otherwise, returns the instantiated object (or the imported
              object for dotted targets that are not callable).

        Raises
        ------
        QPSConfigError
            - [404] Unknown registry key.
            - [403] Target not found in the imported module.
        QPSRegistryError
            - [402] Failed to import a registered target from its dotted path.

        Examples
        --------
        >>> rc = RegistryCenter()
        >>> rc.register("default", "adder", lambda x, y: x + y, return_callable=True)
        >>> add = rc.create("default:adder")
        >>> add(2, 3)
        5
        """
        ns, nm = self._split(full_name)
        table = self._tables.get(ns, {})
        entry = table.get(nm)
        if entry is None:
            raise QPSConfigError(f"[404] Unknown registry key: {ns}:{nm}")
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
            raise QPSRegistryError(f"[402] Failed to import {ns} '{nm}' from '{entry.target}': {e}")
        if entry.meta.get("return_callable"):
            return obj
        return obj(**kwargs) if callable(obj) else obj

    def _import_target(self, target: str) -> Any:
        """Import a dotted target supporting ``module:attr`` or ``module.attr``.

        Raises
        ------
        QPSConfigError
            - [403] Target not found in the imported module.
        """
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
            raise QPSConfigError(f"[403] Target '{target}' not found")
        return getattr(mod, attr_name)

    # --------------------------- introspection ---------------------------
    def list(self, namespace: Optional[Namespace] = None) -> Dict[str, Any]:
        """List available entries with metadata.

        Parameters
        ----------
        namespace : str or None, default None
            When provided, list entries only for the given namespace; otherwise
            return a mapping for all namespaces.

        Returns
        -------
        dict[str, Any]
            A mapping of entry names to their metadata for a single namespace,
            or a mapping of namespace to sorted entry-name lists when
            ``namespace`` is None.
        """
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
        """Infer a human-friendly builder type string.

        Returns "class" for classes, "function" for callables without an
        ``__mro__`` attribute, or the lowercase type name otherwise.
        """
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
    """Decorator form registration for a namespace.

    Parameters
    ----------
    namespace : str
        Target namespace (e.g., "integrator", "backend").
    name : str
        Public key under which to register the object.
    **meta : Any
        Optional metadata stored with the entry (e.g., return_callable, tags).

    Returns
    -------
    Callable
        A decorator that registers the decorated object on import.

    Raises
    ------
    QPSRegistryError
        - [400] Duplicate registration for the same namespace/name.

    Examples
    --------
    >>> @register("default", "hello")
    ... def hello():
    ...     return "world"

    See Also
    --------
    RegistryCenter.register
        Immediate (eager) registration of a callable builder.
    RegistryCenter.decorator
        Underlying decorator helper used by this convenience function.
    RegistryCenter.create
        Resolve and construct registered entries.
    """
    return registry.decorator(namespace, name, **meta)

def register_lazy(namespace: Namespace, name: Name, target: str, **meta: Any) -> None:
    """Register by dotted path without importing until first use.

    Parameters
    ----------
    namespace : str
        Target namespace (e.g., "integrator", "backend").
    name : str
        Public key under which to register the target.
    target : str
        Dotted path like "pkg.mod:ClassName" or "pkg.mod.func" ("module.attr" also
        supported).
    **meta : Any
        Optional metadata (e.g., delayed_import=True is implied).

    Raises
    ------
    QPSRegistryError
        - [401] Duplicate lazy registration for the same namespace/name.

    Examples
    --------
    >>> register_lazy("default", "Adder", "mylib.adders:Adder")

    See Also
    --------
    RegistryCenter.register_lazy
        Lazy registration storing a dotted path for deferred import.
    RegistryCenter.create
        Triggers import/instantiation of lazy entries on first use.
    """
    registry.register_lazy(namespace, name, target, **meta)
