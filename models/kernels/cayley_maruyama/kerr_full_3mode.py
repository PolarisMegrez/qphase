"""CuPy Cayley-Maruyama provider for fully connected three-mode Kerr."""

from __future__ import annotations

from .kerr_3mode import Kerr3ModeCayleyCuPyKernel


class KerrFull3ModeCayleyCuPyKernel(Kerr3ModeCayleyCuPyKernel):
    """Expose the complete 3x3 Kerr Cayley operator to the full model."""
