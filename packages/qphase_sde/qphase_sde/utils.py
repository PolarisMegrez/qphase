"""qphase_sde: Utilities
---------------------------------------------------------
Common utility functions for SDE simulation.
"""

from typing import Any

from qphase.backend.base import BackendBase

__all__ = ["expand_complex_noise_backend", "resolve_mode_columns"]


def resolve_mode_columns(data: Any, modes: list[int]) -> list[int]:
    """Map physical mode indices to stored trajectory columns."""
    meta = getattr(data, "meta", None)
    mode_indices = meta.get("mode_indices") if isinstance(meta, dict) else None
    if mode_indices is None:
        return list(modes)

    mapping = {int(mode): index for index, mode in enumerate(mode_indices)}
    missing = [mode for mode in modes if mode not in mapping]
    if missing:
        raise ValueError(
            f"requested modes {missing} were not recorded; available modes are "
            f"{list(mapping)}"
        )
    return [mapping[mode] for mode in modes]


def expand_complex_noise_backend(Lc: Any, backend: BackendBase) -> Any:
    """Expand complex-basis diffusion matrix to an equivalent real basis.

    .. deprecated::
        Use :func:`qphase_sde.ops.expand_complex_noise` instead. This wrapper
        is kept for backward compatibility.

    Parameters
    ----------
    Lc : Any
        Complex diffusion matrix.
    backend : BackendBase
        Backend to use for operations.

    Returns
    -------
    Any
        Expanded diffusion matrix in real basis (but complex dtype).

    """
    from qphase_sde import ops

    return ops.expand_complex_noise(Lc, backend)
