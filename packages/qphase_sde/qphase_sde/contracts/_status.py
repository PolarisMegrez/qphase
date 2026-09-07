"""qphase_sde: Shared status-code coercion for contract row models.

The peak and coherence-frequency row models layer bit flags of an
``IntFlag`` enum into one status code, each table enforcing its own mask.
This module holds the single coercion rule shared by all row models:

- booleans are rejected explicitly — ``bool`` and ``numpy.bool_`` are
  integers, but a truth value is never a status code;
- any integer is accepted via the integer protocol (the ``__index__``
  method covers Python ``int`` and NumPy integer scalars; floats and
  strings carry no ``__index__`` and fail);
- the value's bits must stay inside the table mask — bitwise, because
  per-table masks are not necessarily contiguous (the peak path mask
  ``0b101`` excludes ``CENTER_IDENTIFIABLE = 0b010``).
"""

from __future__ import annotations

from enum import IntFlag
from typing import Any, TypeVar

import numpy as np

__all__ = ["coerce_status_flags"]

StatusFlagT = TypeVar("StatusFlagT", bound=IntFlag)


def coerce_status_flags(
    value: Any, *, enum: type[StatusFlagT], mask: int, label: str
) -> StatusFlagT:
    """Coerce a raw status input to the ``enum`` flag under ``mask``.

    Parameters
    ----------
    value : Any
        Raw input of a row-model status field. Integers (Python ``int``,
        NumPy integer scalars, ``IntFlag`` members) are accepted through
        the integer protocol; booleans, floats and strings are rejected.
    enum : type[StatusFlagT]
        The ``IntFlag`` enum the row model stores.
    mask : int
        Bit mask of the flags the table may layer; the check is bitwise.
    label : str
        Table name used in error messages.

    Returns
    -------
    StatusFlagT
        The coerced enum member.

    Raises
    ------
    ValueError
        On booleans, non-integer inputs and out-of-mask bits.

    """
    # Booleans are integers — reject them before the integer protocol.
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"status must layer {enum.__name__} flags, not a boolean")
    try:
        raw = int(value.__index__())
    except AttributeError:
        raise ValueError(
            f"status must be an integer of {enum.__name__} flags, got "
            f"{type(value).__name__}"
        ) from None
    if raw < 0 or raw & ~mask:
        raise ValueError(
            f"status {raw} sets bits outside the {label} {enum.__name__} mask"
        )
    return enum(raw)
