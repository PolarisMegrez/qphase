"""fpgen-generated Cayley-Maruyama kernel for the collective Kerr dimer."""

from .fpgen import FPGenCayleyCuPyKernel

__all__ = ["CollectiveKerr2ModeCayleyCuPyKernel"]


class CollectiveKerr2ModeCayleyCuPyKernel(FPGenCayleyCuPyKernel):
    """CuPy fused provider generated from the collective Kerr fpgen model."""

    kernel_slug = "collective_kerr_2mode"
    mode_count = 2
