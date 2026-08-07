"""CuPy Euler-Maruyama provider for fully connected three-mode Kerr."""

from __future__ import annotations

from .kerr_3mode import Kerr3ModeEulerCuPyKernel


class KerrFull3ModeEulerCuPyKernel(Kerr3ModeEulerCuPyKernel):
    """Expose the complete three-mode Kerr terms operator to the full model."""
