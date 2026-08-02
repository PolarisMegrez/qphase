"""Recursive validation and construction of parent/child plugin graphs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import QPhaseConfigError, QPhasePluginError
from .protocols import PluginManifest, SubpluginSlot
from .utils import deep_merge_dicts

__all__ = [
    "PluginGraphResolver",
    "ResolvedPluginNode",
    "merge_plugin_config",
]


@dataclass
class ResolvedPluginNode:
    """One validated node in a recursively instantiated plugin graph."""

    namespace: str
    name: str
    config: Any
    raw_config: dict[str, Any]
    instance: Any | None = None
    children: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return f"{self.namespace}.{self.name}"


class PluginGraphResolver:
    """Resolve child selections declared by plugin manifests."""

    def __init__(self, registry: Any, *, max_depth: int = 8) -> None:
        self.registry = registry
        self.max_depth = max_depth

    def resolve(
        self,
        namespace: str,
        name: str,
        raw_config: Mapping[str, Any] | None,
        *,
        instantiate: bool = True,
        parent_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolvedPluginNode:
        """Validate and optionally instantiate a complete plugin graph."""
        namespace = namespace.strip().lower()
        name = name.strip().lower()
        return self._resolve(
            namespace,
            name,
            dict(raw_config or {}),
            instantiate=instantiate,
            parent_kwargs=dict(parent_kwargs or {}),
            ancestry=(),
            display_path=f"{namespace}.{name}",
        )

    def _resolve(
        self,
        namespace: str,
        name: str,
        raw_config: dict[str, Any],
        *,
        instantiate: bool,
        parent_kwargs: dict[str, Any],
        ancestry: tuple[tuple[str, str], ...],
        display_path: str,
    ) -> ResolvedPluginNode:
        identity = (namespace, name)
        if identity in ancestry:
            chain = " -> ".join(
                f"{ns}.{nm}" for ns, nm in (*ancestry, identity)
            )
            raise QPhaseConfigError(f"subplugin cycle detected: {chain}")
        if len(ancestry) >= self.max_depth:
            raise QPhaseConfigError(
                f"subplugin graph exceeds maximum depth {self.max_depth} at "
                f"{display_path}"
            )

        plugin_class = self.registry.get_plugin_class(namespace, name)
        normalizer = getattr(plugin_class, "normalize_plugin_config", None)
        if callable(normalizer):
            raw_config = dict(normalizer(dict(raw_config)))
        manifest = _plugin_manifest(plugin_class)
        slot_names = set(manifest.subplugins)
        parent_raw = {
            key: value for key, value in raw_config.items() if key not in slot_names
        }
        schema = self.registry.get_plugin_schema(namespace, name)
        if schema is None:
            raise QPhaseConfigError(
                f"No configuration schema found for '{namespace}:{name}'"
            )
        try:
            config = schema.model_validate(parent_raw)
        except Exception as exc:
            raise QPhaseConfigError(f"invalid config at {display_path}: {exc}") from exc

        children: dict[str, Any] = {}
        expanded_raw = dict(config.model_dump())
        for slot_name, slot in manifest.subplugins.items():
            resolved, expanded = self._resolve_slot(
                slot,
                raw_config.get(slot_name),
                instantiate=instantiate,
                ancestry=(*ancestry, identity),
                display_path=f"{display_path}/{slot_name}",
            )
            if expanded is not None:
                expanded_raw[slot_name] = expanded
            if resolved is not None:
                children[slot_name] = resolved

        instance = None
        if instantiate:
            kwargs = dict(parent_kwargs)
            kwargs["config"] = config
            if manifest.subplugins:
                kwargs["subplugins"] = {
                    slot: _instances(value) for slot, value in children.items()
                }
            try:
                instance = self.registry.create(f"{namespace}:{name}", **kwargs)
            except Exception as exc:
                raise QPhasePluginError(
                    f"failed to instantiate plugin {display_path}: {exc}"
                ) from exc

        return ResolvedPluginNode(
            namespace=namespace,
            name=name,
            config=config,
            raw_config=expanded_raw,
            instance=instance,
            children=children,
        )

    def _resolve_slot(
        self,
        slot: SubpluginSlot,
        selection: Any,
        *,
        instantiate: bool,
        ancestry: tuple[tuple[str, str], ...],
        display_path: str,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        if selection is None:
            if slot.default is not None:
                selection = {slot.default: {}}
            elif slot.cardinality == "one":
                raise QPhaseConfigError(
                    f"{display_path}: expected exactly one child, got none"
                )
            else:
                return None, None
        if not isinstance(selection, Mapping):
            raise QPhaseConfigError(f"{display_path}: child selection must be a map")

        entries = list(selection.items())
        if slot.cardinality in {"one", "optional"} and len(entries) != 1:
            raise QPhaseConfigError(
                f"{display_path}: expected exactly one child, got {len(entries)}"
            )
        if slot.cardinality == "many" and not entries:
            return OrderedDict(), {}

        resolved: OrderedDict[str, ResolvedPluginNode] = OrderedDict()
        expanded: OrderedDict[str, dict[str, Any]] = OrderedDict()
        protocol = _load_protocol(self.registry, slot.protocol)
        for child_name, child_raw in entries:
            child_name = str(child_name).strip().lower()
            if slot.allowed is not None and child_name not in slot.allowed:
                raise QPhaseConfigError(
                    f"{display_path}.{child_name}: child is not allowed"
                )
            if not isinstance(child_raw, Mapping):
                raise QPhaseConfigError(
                    f"{display_path}.{child_name}: config must be a map"
                )
            node = self._resolve(
                slot.namespace,
                child_name,
                dict(child_raw),
                instantiate=instantiate,
                parent_kwargs={},
                ancestry=ancestry,
                display_path=f"{display_path}.{child_name}",
            )
            if protocol is not None and instantiate:
                if not isinstance(node.instance, protocol):
                    raise QPhaseConfigError(
                        f"{display_path}.{child_name}: does not satisfy "
                        f"protocol {slot.protocol}"
                    )
            resolved[child_name] = node
            expanded[child_name] = node.raw_config

        if slot.cardinality == "many":
            return resolved, dict(expanded)
        only_name = next(iter(resolved))
        return resolved[only_name], dict(expanded)


def merge_plugin_config(
    registry: Any,
    namespace: str,
    name: str,
    inherited: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge parent config while treating child selections as typed slots."""
    base = dict(inherited or {})
    incoming = dict(override or {})
    plugin_class = registry.get_plugin_class(namespace, name)
    manifest = _plugin_manifest(plugin_class)
    slot_names = set(manifest.subplugins)
    merged = deep_merge_dicts(
        {key: value for key, value in base.items() if key not in slot_names},
        {key: value for key, value in incoming.items() if key not in slot_names},
    )
    for slot_name, slot in manifest.subplugins.items():
        old = base.get(slot_name)
        new = incoming.get(slot_name, _MISSING)
        if new is _MISSING:
            if old is not None:
                merged[slot_name] = old
            continue
        if new is None:
            if slot.cardinality != "many":
                merged[slot_name] = None
            continue
        if not isinstance(new, Mapping):
            merged[slot_name] = new
            continue
        if slot.cardinality == "many":
            selected = OrderedDict(old or {})
            for child_name, child_override in new.items():
                if child_override is None:
                    selected.pop(child_name, None)
                    continue
                selected[child_name] = _merge_child(
                    registry,
                    slot.namespace,
                    child_name,
                    selected.get(child_name),
                    child_override,
                )
            merged[slot_name] = dict(selected)
            continue
        old_items = list(old.items()) if isinstance(old, Mapping) else []
        new_items = list(new.items())
        same_child = (
            len(old_items) == 1
            and len(new_items) == 1
            and old_items[0][0] == new_items[0][0]
        )
        if same_child:
            child_name, child_override = new_items[0]
            merged[slot_name] = {
                child_name: _merge_child(
                    registry,
                    slot.namespace,
                    child_name,
                    old_items[0][1],
                    child_override,
                )
            }
        else:
            merged[slot_name] = dict(new)
    return merged


def _merge_child(
    registry: Any,
    namespace: str,
    name: str,
    inherited: Any,
    override: Any,
) -> Any:
    if not isinstance(inherited, Mapping) or not isinstance(override, Mapping):
        return override
    return merge_plugin_config(registry, namespace, str(name), inherited, override)


def _plugin_manifest(plugin_class: Any) -> PluginManifest:
    manifest = getattr(plugin_class, "manifest", None)
    return manifest if isinstance(manifest, PluginManifest) else PluginManifest()


def _load_protocol(registry: Any, dotted: str | None) -> Any | None:
    return None if dotted is None else registry._import_target(dotted)


def _instances(value: Any) -> Any:
    if isinstance(value, ResolvedPluginNode):
        return value.instance
    if isinstance(value, Mapping):
        return OrderedDict((name, node.instance) for name, node in value.items())
    return value


_MISSING = object()
