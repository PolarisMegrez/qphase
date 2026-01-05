"""qphase: Array Namespace Utilities
---------------------------------------------------------
Helpers to select a NumPy-like array namespace for model code and to convert
arrays to NumPy when needed, keeping models backend-agnostic.

Behavior
--------
- Expose ``get_xp`` and ``convert_to_numpy`` for common array conversions and a small
    device-aware torch shim when appropriate; see function docstrings for exact
    behaviors and supported operations.

Public API
----------
``get_xp`` : Select a NumPy-like array namespace for a given array.
``convert_to_numpy`` : Convert an array to a NumPy array.

Notes
-----
- Avoids importing heavy libraries when not needed; detection is best-effort.

"""

from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = [
    "get_xp",
    "convert_to_numpy",
]


def get_xp(arr: Any):
    """Select a NumPy-like array namespace for a given array.

    Returns the most appropriate array namespace for ``arr`` to keep model code
    backend-agnostic. If ``arr`` is a CuPy array, returns ``cupy``. If ``arr``
    is a PyTorch tensor, returns a tiny shim exposing a minimal NumPy-like API
    backed by PyTorch ops. Otherwise, returns ``numpy``.

    Parameters
    ----------
    arr : Any
        An array-like object. Supported types include ``numpy.ndarray``,
        ``cupy.ndarray`` and ``torch.Tensor``. Other inputs will fall back to
        ``numpy``.

    Returns
    -------
    Any
        - ``cupy`` module when ``arr`` is a CuPy array.
        - A ``types.SimpleNamespace`` shim when ``arr`` is a PyTorch tensor. The
          shim provides: ``abs``, ``sqrt``, ``clip``, ``zeros``, ``empty``,
          ``empty_like``, ``asarray`` and ``concatenate``.
        - ``numpy`` module otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> xp = get_xp(np.zeros((2, 2)))
    >>> xp is np
    True

    >>> # With a torch tensor (if torch is installed), returns a small shim
    >>> import torch
    >>> t = torch.ones(2, 2)
    >>> xp = get_xp(t)
    >>> xp.zeros((1,))  # allocated on the same device as `t`
    tensor([0.])

    """
    # CuPy detection
    try:
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp
    except Exception:
        pass
    # PyTorch detection with a tiny shim for clip
    try:
        import torch as th

        if hasattr(th, "Tensor") and isinstance(arr, th.Tensor):

            def _clip(x, a_min=None, a_max=None):
                # torch.clamp doesn't accept None the same way as numpy; handle cases
                if a_min is None and a_max is None:
                    return x
                if a_min is None:
                    return th.clamp(x, max=a_max)
                if a_max is None:
                    return th.clamp(x, min=a_min)
                return th.clamp(x, min=a_min, max=a_max)

            return SimpleNamespace(
                abs=th.abs,
                sqrt=th.sqrt,
                clip=_clip,
                zeros=lambda shape, dtype=None: th.zeros(
                    shape, dtype=dtype, device=x_device(arr)
                ),
                full=lambda shape, fill_value, dtype=None: th.full(
                    shape, fill_value, dtype=dtype, device=x_device(arr)
                ),
                empty_like=th.empty_like,
                empty=lambda shape, dtype=None: th.empty(
                    shape, dtype=dtype, device=x_device(arr)
                ),
                asarray=th.as_tensor,
                concatenate=lambda arrays, axis=-1: th.cat(arrays, dim=axis),
            )
    except Exception:
        pass
    import numpy as np  # fallback

    return np


def x_device(arr: Any) -> Any:
    """Return device for a torch tensor; otherwise 'cpu'."""
    try:
        import torch as th

        if isinstance(arr, th.Tensor) and arr.device is not None:
            return arr.device
    except Exception:
        pass
    return "cpu"


def convert_to_numpy(x: Any) -> np.ndarray:
    """Convert a torch/cupy/numpy array to a NumPy ``ndarray``.

    Converts common backend arrays into a CPU NumPy array for downstream
    processing or serialization. For unknown inputs, performs a best-effort
    ``numpy.asarray``.

    Parameters
    ----------
    x : Any
        Array-like to convert. Supports ``torch.Tensor``, ``cupy.ndarray``, and
        ``numpy.ndarray``.

    Returns
    -------
    np.ndarray
        A ``numpy.ndarray`` view or copy of ``x`` when possible; otherwise the
        result of ``numpy.asarray(x)``.

    Examples
    --------
    >>> import numpy as np
    >>> convert_to_numpy(np.array([1, 2, 3])).dtype.kind in {'i', 'u'}
    True

    >>> import torch
    >>> t = torch.tensor([1.0, 2.0])
    >>> a = convert_to_numpy(t)
    >>> isinstance(a, np.ndarray)
    True

    """
    try:
        import torch as th

        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass

    # Duck typing for CuPy
    if hasattr(x, "get") and callable(x.get):
        try:
            return x.get()
        except Exception:
            pass

    try:
        import numpy as np

        return np.asarray(x)
    except Exception:
        return x
