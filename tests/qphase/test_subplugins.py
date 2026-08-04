from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest
from pydantic import BaseModel
from qphase.core.errors import QPhaseConfigError
from qphase.core.plugin_graph import PluginGraphResolver, merge_plugin_config
from qphase.core.protocols import PluginManifest, SubpluginSlot
from qphase.core.registry import RegistryCenter
from qphase.service import RegistryService


class EmptyConfig(BaseModel):
    pass


class ValueConfig(BaseModel):
    value: int = 1


@runtime_checkable
class ChildContract(Protocol):
    def compute(self) -> int: ...


class GoodChild:
    config_schema = ValueConfig

    def __init__(self, config):
        self.config = config

    def compute(self) -> int:
        return self.config.value


class BadChild:
    config_schema = EmptyConfig

    def __init__(self, config):
        self.config = config


class Parent:
    config_schema = ValueConfig
    manifest = PluginManifest(
        subplugins={
            "worker": SubpluginSlot(
                namespace="worker",
                default="good",
                protocol=f"{__name__}:ChildContract",
            )
        }
    )

    def __init__(self, config, *, subplugins):
        self.config = config
        self.worker = subplugins["worker"]


def make_registry() -> RegistryCenter:
    result = RegistryCenter()
    result.register("parent", "main", Parent)
    result.register("worker", "good", GoodChild)
    result.register("worker", "bad", BadChild)
    return result


def test_resolver_applies_default_and_constructs_parent_graph():
    node = PluginGraphResolver(make_registry()).resolve(
        "parent", "main", {"value": 4}
    )

    assert node.instance.config.value == 4
    assert node.instance.worker.compute() == 1
    assert node.raw_config["worker"] == {"good": {"value": 1}}


def test_resolver_validates_cardinality_and_protocol():
    resolver = PluginGraphResolver(make_registry())
    with pytest.raises(QPhaseConfigError, match="expected exactly one child"):
        resolver.resolve(
            "parent", "main", {"worker": {"good": {}, "bad": {}}}
        )
    with pytest.raises(QPhaseConfigError, match="does not satisfy"):
        resolver.resolve("parent", "main", {"worker": {"bad": {}}})


def test_registry_config_validation_accepts_declared_child_slots():
    config = make_registry().validate_plugin_config(
        "parent",
        {
            "name": "main",
            "value": 4,
            "worker": {"good": {"value": 7}},
        },
    )

    assert config.value == 4


def test_merge_replaces_different_single_child_and_merges_same_child():
    registry = make_registry()
    inherited = {"value": 2, "worker": {"good": {"value": 3}}}

    same = merge_plugin_config(
        registry,
        "parent",
        "main",
        inherited,
        {"worker": {"good": {"value": 8}}},
    )
    assert same["worker"] == {"good": {"value": 8}}

    replaced = merge_plugin_config(
        registry,
        "parent",
        "main",
        inherited,
        {"worker": {"bad": {}}},
    )
    assert replaced["worker"] == {"bad": {}}


def test_resolver_rejects_recursive_plugin_cycles():
    class Recursive:
        config_schema = EmptyConfig
        manifest = PluginManifest(
            subplugins={
                "next": SubpluginSlot(namespace="loop", default="recursive")
            }
        )

        def __init__(self, config, *, subplugins):
            self.config = config
            self.subplugins = subplugins

    registry = RegistryCenter()
    registry.register("loop", "recursive", Recursive)

    with pytest.raises(QPhaseConfigError, match="cycle detected"):
        PluginGraphResolver(registry).resolve("loop", "recursive", {})


def test_registry_service_projects_flat_children_as_parent_tree():
    service = RegistryService(make_registry())

    tree = service.get_plugin_tree("parent")
    assert len(tree) == 1
    assert tree[0].path == "parent.main"
    assert tree[0].slots[0].name == "worker"
    assert [item.name for item in tree[0].slots[0].options] == ["good", "bad"]

    namespace, name, path = service.resolve_path("parent.main/worker.good")
    assert (namespace, name, path) == (
        "worker",
        "good",
        "parent.main/worker.good",
    )
    schema = service.get_composite_schema("parent.main")
    assert "good" in schema["subplugins"]["worker"]["options"]
    template = service.build_template(
        "parent.main", selections={"worker": "good"}
    )
    assert template["worker"]["good"]["value"] == 1
