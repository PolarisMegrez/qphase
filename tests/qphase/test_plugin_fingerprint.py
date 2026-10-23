"""Tests for plugin fingerprinting and its per-process distribution cache."""

from __future__ import annotations

import importlib.metadata

import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.execution import _packages_distributions, plugin_fingerprint


class WorkspaceOnlyPlugin:
    """Plugin class that belongs to no installed distribution."""


def _uncached_distribution_version(instance) -> str | None:
    """Resolve the distribution version exactly as the fingerprinter does."""
    top_level = type(instance).__module__.partition(".")[0]
    for distribution in importlib.metadata.packages_distributions().get(top_level, []):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@pytest.fixture(autouse=True)
def fresh_distribution_cache():
    """Keep the process-wide distribution scan from leaking between tests."""
    _packages_distributions.cache_clear()
    yield
    _packages_distributions.cache_clear()


def test_repeated_fingerprints_are_equal_but_distinct():
    """The scan is cached, the fingerprint dict is not shared."""
    first = plugin_fingerprint(NumpyBackend())
    second = plugin_fingerprint(NumpyBackend())

    assert first == second
    assert first is not second
    assert _packages_distributions.cache_info().hits >= 1


@pytest.mark.parametrize("plugin", [NumpyBackend(), WorkspaceOnlyPlugin()])
def test_fingerprint_matches_the_uncached_resolution(plugin):
    """Caching the scan must not change what the fingerprint resolves to."""
    fingerprint = plugin_fingerprint(plugin)

    assert fingerprint == {
        "class": f"{type(plugin).__module__}:{type(plugin).__qualname__}",
        "distribution_version": _uncached_distribution_version(plugin),
    }


def test_workspace_plugin_reports_no_distribution(monkeypatch):
    """A class outside every installed distribution resolves to None."""
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions", lambda: {"qphase": []}
    )

    assert plugin_fingerprint(WorkspaceOnlyPlugin())["distribution_version"] is None


def test_distribution_scan_runs_once_per_process(monkeypatch):
    """Two fingerprints cost one scan; cache_clear forces a rescan."""
    calls = 0
    real = importlib.metadata.packages_distributions

    def counting_scan():
        nonlocal calls
        calls += 1
        return real()

    monkeypatch.setattr(importlib.metadata, "packages_distributions", counting_scan)

    plugin_fingerprint(NumpyBackend())
    plugin_fingerprint(WorkspaceOnlyPlugin())
    assert calls == 1

    _packages_distributions.cache_clear()
    plugin_fingerprint(NumpyBackend())
    assert calls == 2
