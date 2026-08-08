"""fpgen-generated Cayley-Maruyama kernel for the collective-loss trimer."""

from .fpgen import FPGenCayleyCuPyKernel

__all__ = ["CollectiveLossKerr3ModeCayleyCuPyKernel"]


class CollectiveLossKerr3ModeCayleyCuPyKernel(FPGenCayleyCuPyKernel):
    """CuPy fused provider generated from the collective-loss fpgen model."""

    kernel_slug = "collective_loss_kerr_3mode"
    mode_count = 3
