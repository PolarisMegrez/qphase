"""qpsgen-generated Cayley-Maruyama kernel for the collective Kerr dimer."""

from .qpsgen import QPSGenCayleyCuPyKernel

__all__ = ["CollectiveKerr2ModeCayleyCuPyKernel"]


class CollectiveKerr2ModeCayleyCuPyKernel(QPSGenCayleyCuPyKernel):
    """CuPy fused provider generated from the collective Kerr qpsgen model."""

    kernel_slug = "collective_kerr_2mode"
    mode_count = 2
