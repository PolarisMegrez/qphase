"""Collection-hook invariants: layer markers come from markers, not path names.

``tests/qphase_sde/e2e/`` is a directory, not a layer declaration, yet pytest
adds the ``e2e`` keyword to every item collected below it. The auto-``unit``
rule in ``tests/conftest.py`` must therefore read real markers. These cases
drive the hook with stand-in items, so no CUDA device is ever probed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from tests import conftest


class _Item:
    """Collected-item stand-in exposing keywords, real markers, added markers."""

    def __init__(
        self,
        nodeid: str,
        *,
        keywords: Iterable[str] = (),
        marks: Sequence[str] = (),
    ) -> None:
        self.nodeid = nodeid
        self.keywords = dict.fromkeys(keywords, 0)
        self._marks = [getattr(pytest.mark, name).mark for name in marks]

    def get_closest_marker(self, name: str):
        return next((mark for mark in reversed(self._marks) if mark.name == name), None)

    def add_marker(self, marker) -> None:
        self._marks.append(marker.mark)

    def marker_names(self) -> set[str]:
        return {mark.name for mark in self._marks}


def _forbid_probe(monkeypatch) -> list[bool]:
    """Replace the device probe and return the list recording its calls."""
    calls: list[bool] = []

    def probe() -> bool:
        calls.append(True)
        return False

    monkeypatch.setattr(conftest, "cuda_available", probe)
    return calls


def test_directory_keyword_is_not_a_layer_marker(monkeypatch):
    """A file under ``e2e/`` is unit unless it declares a real layer marker."""
    probed = _forbid_probe(monkeypatch)
    item = _Item(
        "tests/qphase_sde/e2e/test_models.py::test_vdp_equations",
        keywords=("e2e", "tests", "qphase_sde", "test_models", "test_vdp_equations"),
    )

    conftest.pytest_collection_modifyitems([item])

    assert item.get_closest_marker("unit") is not None
    assert item.get_closest_marker("e2e") is None
    assert probed == []


@pytest.mark.parametrize("layer", ["integration", "e2e"])
def test_explicit_layer_marker_is_preserved(monkeypatch, layer):
    """An explicit layer declaration is never rewritten to unit."""
    _forbid_probe(monkeypatch)
    item = _Item(
        f"tests/qphase_sde/e2e/test_models.py::test_{layer}_case",
        keywords=("e2e", "tests", "qphase_sde", layer),
        marks=(layer,),
    )

    conftest.pytest_collection_modifyitems([item])

    assert item.marker_names() == {layer}


def test_gpu_keyword_without_marker_neither_probes_nor_skips(monkeypatch):
    """A ``gpu`` path token is not the marker: no probe, no skip, gets unit."""
    probed = _forbid_probe(monkeypatch)
    item = _Item("tests/qphase/gpu/test_case.py::test_case", keywords=("gpu",))

    conftest.pytest_collection_modifyitems([item])

    assert probed == []
    assert item.marker_names() == {"unit"}


def test_gpu_marker_probes_once_and_skips_without_device(monkeypatch):
    """A real ``gpu`` marker still gates the probe and the skip."""
    probed = _forbid_probe(monkeypatch)
    item = _Item(
        "tests/qphase_sde/e2e/test_models.py::test_gpu_case",
        marks=("gpu",),
    )

    conftest.pytest_collection_modifyitems([item])

    assert probed == [True]
    assert item.marker_names() == {"gpu", "skip"}
