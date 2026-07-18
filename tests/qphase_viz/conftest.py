"""Fixtures for ``qphase_viz`` tests: headless matplotlib rendering."""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
