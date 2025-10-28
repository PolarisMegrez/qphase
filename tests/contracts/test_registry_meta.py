from __future__ import annotations

from QPhaseSDE.core.registry import registry


def test_registry_list_includes_metadata():
    backends = registry.list("backend")
    assert isinstance(backends, dict)
    # numpy backend is registered lazily in backends/__init__.py
    assert "numpy" in backends
    meta = backends["numpy"]
    assert "kind" in meta and "delayed_import" in meta
