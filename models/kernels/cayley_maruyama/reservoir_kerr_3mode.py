"""fpgen-generated Cayley-Maruyama kernel for the reservoir Kerr trimer."""

from .fpgen import FPGenCayleyCuPyKernel

__all__ = ["ReservoirKerr3ModeCayleyCuPyKernel"]


class ReservoirKerr3ModeCayleyCuPyKernel(FPGenCayleyCuPyKernel):
    """CuPy fused provider generated from the reservoir Kerr fpgen model."""

    kernel_slug = "reservoir_kerr_3mode"
    mode_count = 3
