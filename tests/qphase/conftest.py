"""Fixtures scoped to core-package (``qphase``) tests.

Dummy plugin registration is autouse but limited to this directory so that
``tests/qphase_sde`` and ``tests/qphase_viz`` never see these registrations
when run standalone.
"""

import pytest
from qphase.core.registry import registry

_DUMMY_NAMESPACES = ("engine", "backend", "model")


@pytest.fixture(autouse=True)
def register_dummy_plugins():
    """Register dummy plugins for testing, removing them again afterwards."""
    from tests.plugins.dummy_plugin import DummyPlugin

    for namespace in _DUMMY_NAMESPACES:
        registry.register(
            namespace=namespace, name="dummy", builder=DummyPlugin, overwrite=True
        )
    yield
    # The registry is a global singleton and has no public unregister API;
    # drop exactly the entries added above so nothing leaks into other tests.
    for namespace in _DUMMY_NAMESPACES:
        registry._tables.get(namespace, {}).pop("dummy", None)
