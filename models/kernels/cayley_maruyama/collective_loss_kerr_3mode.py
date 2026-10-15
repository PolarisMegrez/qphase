"""qpsgen-generated Cayley-Maruyama kernel for the collective-loss trimer."""

from .qpsgen import QPSGenCayleyCuPyKernel

__all__ = ["CollectiveLossKerr3ModeCayleyCuPyKernel"]


class CollectiveLossKerr3ModeCayleyCuPyKernel(QPSGenCayleyCuPyKernel):
    """CuPy fused provider generated from the collective-loss qpsgen model."""

    kernel_slug = "collective_loss_kerr_3mode"
    mode_count = 3
