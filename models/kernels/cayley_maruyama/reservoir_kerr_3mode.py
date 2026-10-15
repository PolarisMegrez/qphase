"""qpsgen-generated Cayley-Maruyama kernel for the reservoir Kerr trimer."""

from .qpsgen import QPSGenCayleyCuPyKernel

__all__ = ["ReservoirKerr3ModeCayleyCuPyKernel"]


class ReservoirKerr3ModeCayleyCuPyKernel(QPSGenCayleyCuPyKernel):
    """CuPy fused provider generated from the reservoir Kerr qpsgen model."""

    kernel_slug = "reservoir_kerr_3mode"
    mode_count = 3
