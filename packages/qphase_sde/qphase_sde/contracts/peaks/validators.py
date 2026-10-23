"""Validators for the peak candidate, fit and path payloads.

Every validator takes the grouped payload a plugin produced, coerces it into
arrays and returns the same payload with numpy columns; they raise
``ValueError`` on a shape, length or measurement-basis violation and never
mutate their input.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .models import (
    OPTIONAL_CAPABILITY_FIELDS,
    PEAK_CANDIDATE_STATUS_MASK,
    PEAK_PATH_STATUS_MASK,
)


def _offsets_array(name: str, offsets: Any) -> np.ndarray:
    array = np.asarray(offsets)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    if len(array) == 0 or array[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if len(array) > 1 and np.any(np.diff(array) < 0):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    return array

def _index_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    return array

def _status_code_array(name: str, codes: Any, *, mask: int, label: str) -> np.ndarray:
    array = _index_array(name, codes)
    # Bitwise, not a range check: the per-table masks are not necessarily
    # contiguous (the path mask 0b101 excludes CENTER_IDENTIFIABLE = 0b010).
    if len(array) and (
        int(array.min()) < 0 or bool(np.any(np.bitwise_and(array, ~mask)))
    ):
        raise ValueError(
            f"{name} values set bits outside the {label} PeakStatus mask (0b{mask:b})"
        )
    return array

def _stable_labels(field: str, values: Iterable[str], *, noun: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field} must be a list of strings, not one string")
    result = tuple(values)
    if not result:
        raise ValueError(f"{field} must list at least one {noun}")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field} must hold non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result

CHANNEL_MEASUREMENT_KINDS: tuple[str, ...] = (
    "bare",
    "coherent",
    "incoherent",
    "general",
)

_MEASUREMENT_KIND_SET = frozenset(CHANNEL_MEASUREMENT_KINDS)

_BASIS_RTOL = 1e-6

_BASIS_ATOL = 1e-9

def _real_pair_array(
    name: str, values: Any, entry_count: int, entry_desc: str, context: str
) -> np.ndarray:
    """Validate a flattened real ``[re, im]`` pair array; return float64."""
    array = np.asarray(values)
    if array.shape != (entry_count, 2):
        raise ValueError(
            f"{name} must hold one [re, im] pair per {entry_desc} entry "
            f"flattened row-major, expected shape ({entry_count}, 2)"
            f"{context}, got {array.shape}"
        )
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must hold real [re, im] pairs")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array

def _check_channel_mode_weights(
    weights: Any, channel_count: int, mode_count: int
) -> np.ndarray:
    return _real_pair_array(
        "channel_mode_weights",
        weights,
        channel_count * mode_count,
        "(channel, mode)",
        f" for {channel_count} channels x {mode_count} modes",
    )

def _check_channel_mode_matrix(
    matrix: Any, channel_count: int, mode_count: int
) -> np.ndarray:
    return _real_pair_array(
        "channel_mode_matrix",
        matrix,
        channel_count * mode_count * mode_count,
        "(channel, mode, mode)",
        f" for {channel_count} channels x {mode_count} modes",
    )

def _check_measurement_kinds(kinds: Any, channel_count: int) -> tuple[str, ...]:
    if isinstance(kinds, str):
        raise ValueError(
            "channel_measurement_kinds must be a list of strings, not one string"
        )
    result: tuple[str, ...] = tuple(kinds)
    if len(result) != channel_count:
        raise ValueError(
            "channel_measurement_kinds must hold exactly one entry per "
            f"channel ({channel_count}), got {len(result)}"
        )
    unknown = [kind for kind in result if kind not in _MEASUREMENT_KIND_SET]
    if unknown:
        raise ValueError(
            "channel_measurement_kinds entries must be one of "
            f"{sorted(_MEASUREMENT_KIND_SET)}, got {unknown}"
        )
    return result

def _check_measurement_basis(
    channel_names: tuple[str, ...],
    mode_count: int,
    kinds: tuple[str, ...],
    weights: np.ndarray | None,
    matrices: np.ndarray | None,
) -> None:
    """Enforce the per-channel representation/kind agreement.

    Exactly one representation is populated per channel: the rank-1
    readout vector ``l`` of ``W = l l†`` (``channel_mode_weights``) for
    ``bare``/``coherent`` channels, the Hermitian measurement matrix ``W``
    (``channel_mode_matrix``) for ``incoherent``/``general`` channels. A
    representation block of a channel of the other kind must be identically
    zero, so a channel can never populate both.
    """
    channel_count = len(channel_names)
    vectors = (
        None
        if weights is None
        else (weights[:, 0] + 1j * weights[:, 1]).reshape(channel_count, mode_count)
    )
    w_matrices = (
        None
        if matrices is None
        else (matrices[:, 0] + 1j * matrices[:, 1]).reshape(
            channel_count, mode_count, mode_count
        )
    )
    for index, (name, kind) in enumerate(zip(channel_names, kinds, strict=True)):
        vector = None if vectors is None else vectors[index]
        matrix = None if w_matrices is None else w_matrices[index]
        if kind in ("bare", "coherent"):
            if vector is None:
                raise ValueError(
                    f"{kind} channel {name!r} requires channel_mode_weights: "
                    "the rank-1 readout vector l of W = l l^dagger"
                )
            if matrix is not None and bool(np.any(matrix != 0.0)):
                raise ValueError(
                    f"channel {name!r} populates both channel_mode_weights and "
                    "channel_mode_matrix: exactly one representation per channel"
                )
            if kind == "bare":
                norms = np.abs(vector)
                unit = np.isclose(norms, 1.0, rtol=_BASIS_RTOL, atol=_BASIS_ATOL)
                zero = norms <= _BASIS_ATOL
                if int(np.count_nonzero(unit)) != 1 or not bool(np.all(unit | zero)):
                    raise ValueError(
                        f"bare channel {name!r} must carry exactly one unit "
                        "weight with all other weights zero (W = e_m e_m^dagger)"
                    )
            elif not bool(np.any(vector != 0.0)):
                raise ValueError(
                    f"coherent channel {name!r} must carry a non-zero readout "
                    "vector l (W = l l^dagger)"
                )
        else:
            if matrix is None:
                raise ValueError(
                    f"{kind} channel {name!r} requires channel_mode_matrix: "
                    "the Hermitian measurement matrix W acting on the mode "
                    "vector"
                )
            if vector is not None and bool(np.any(vector != 0.0)):
                raise ValueError(
                    f"channel {name!r} populates both channel_mode_weights and "
                    "channel_mode_matrix: exactly one representation per channel"
                )
            scale = float(np.max(np.abs(matrix)))
            if scale == 0.0:
                raise ValueError(
                    f"{kind} channel {name!r} must carry a non-zero "
                    "measurement matrix W"
                )
            hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
            tolerance = _BASIS_RTOL * scale
            if hermiticity > tolerance:
                raise ValueError(
                    f"channel_mode_matrix of channel {name!r} must be "
                    "Hermitian (W = W^dagger)"
                )
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
            eigen_tolerance = tolerance
            if float(np.min(eigenvalues)) < -eigen_tolerance:
                raise ValueError(
                    f"channel_mode_matrix of channel {name!r} must be "
                    "positive semidefinite (non-negative, W >= 0) for a "
                    "power measurement"
                )
            if kind == "incoherent":
                off_diagonal = matrix.copy()
                np.fill_diagonal(off_diagonal, 0.0)
                if bool(np.any(np.abs(off_diagonal) > tolerance)):
                    raise ValueError(
                        f"incoherent channel {name!r} must carry a diagonal "
                        "W: an incoherent sum over the modes (W = I for a "
                        "plain trace)"
                    )
                diagonal = np.diagonal(matrix)
                if bool(np.any(np.abs(diagonal.imag) > tolerance)) or bool(
                    np.any(diagonal.real < -tolerance)
                ):
                    raise ValueError(
                        f"incoherent channel {name!r} must carry real "
                        "non-negative diagonal entries"
                    )

def validate_candidate_table(
    candidate_offsets: Any,
    group_scan_index: Any,
    group_channel_index: Any,
    columns: Mapping[str, Any],
    *,
    scan_count: int,
    channel_names: Iterable[str],
    mode_labels: Iterable[str] | None = None,
    channel_measurement_kinds: Iterable[str] | None = None,
    channel_mode_weights: Any | None = None,
    channel_mode_matrix: Any | None = None,
    optional_capabilities: Iterable[str] = OPTIONAL_CAPABILITY_FIELDS,
) -> None:
    """Validate the group structure and columns of a candidate table.

    ``group_scan_index`` and ``group_channel_index`` hold one integer entry
    per (scan, channel) group; ``candidate_offsets`` must start at 0, be
    monotonically non-decreasing, hold exactly ``group_count + 1`` entries
    and end at the candidate count. Group indices must reference existing
    scan points (``[0, scan_count)``) and channels
    (``[0, len(channel_names))``), and each (scan, channel) pair owns at
    most one group — there is no implicit replicate dimension.

    ``channel_names`` is the mandatory stable index-to-label mapping
    (single-channel tables list one name). The measurement basis is
    optional as a whole, but when either representation is present,
    ``mode_labels`` and ``channel_measurement_kinds`` are required with it:
    a channel measurement is a Hermitian matrix ``W`` acting on the mode
    vector (the measured correlation is the quadratic form ``α†Wα``), and
    the kind of each channel — ``bare``, ``coherent``, ``incoherent`` or
    ``general`` — selects the representation it carries.

    Exactly one representation is populated per channel.
    ``channel_mode_weights`` holds the rank-1 readout vector ``l`` of
    ``W = l l†``: one finite ``[re, im]`` pair per (channel, mode) entry,
    flattened row-major, so exactly ``len(channel_names) * len(mode_labels)``
    pairs. It is allowed only for ``bare``/``coherent`` channels — a
    ``bare`` channel carries exactly one unit weight with all others zero
    (``W = e_m e_m†``), a ``coherent`` channel any finite non-zero ``l``
    of the coherent readout ``c = l†α``. ``channel_mode_matrix`` holds the
    full Hermitian ``W``: one finite ``[re, im]`` pair per
    (channel, mode, mode) entry, flattened row-major, so exactly
    ``len(channel_names) * len(mode_labels)**2`` pairs. It is required for
    ``incoherent``/``general`` channels — ``W`` must be Hermitian, and an
    ``incoherent`` channel additionally carries a diagonal ``W`` with real
    non-negative entries: a trace is the incoherent sum ``W = I`` over the
    modes, which is generally NOT rank-1 and cannot be written as a
    coherent readout. A channel populating both representations — or a
    representation contradicting its kind — is rejected.

    The required columns ``location``/``intensity``/``status_code`` must be
    present, and every column must hold exactly one entry per candidate.
    ``status_code`` values must stay within the candidate
    :class:`PeakStatus` mask (:data:`PEAK_CANDIDATE_STATUS_MASK`) —
    ``PATH_UNIQUE`` is a path-product bit and fails here. Each present
    optional capability must be paired with a ``<name>_valid`` mask column
    of the same length holding only 0/1 values.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    offsets = _offsets_array("candidate_offsets", candidate_offsets)
    scan_index = _index_array("group_scan_index", group_scan_index)
    channel_index = _index_array("group_channel_index", group_channel_index)
    names = _stable_labels("channel_names", channel_names, noun="channel")
    modes = (
        None
        if mode_labels is None
        else _stable_labels("mode_labels", mode_labels, noun="mode")
    )
    if scan_count < 0:
        raise ValueError("scan_count must be nonnegative")
    if len(scan_index) != len(channel_index):
        raise ValueError(
            "group_scan_index and group_channel_index must hold exactly one "
            "entry per group"
        )
    group_count = len(scan_index)
    if len(offsets) != group_count + 1:
        raise ValueError(
            f"candidate_offsets must hold group_count + 1 "
            f"({group_count + 1}) entries, got {len(offsets)}"
        )
    if group_count:
        if int(scan_index.min()) < 0 or int(scan_index.max()) >= scan_count:
            raise ValueError(
                f"group_scan_index references scan points outside [0, {scan_count})"
            )
        if int(channel_index.min()) < 0 or int(channel_index.max()) >= len(names):
            raise ValueError(
                f"group_channel_index references channels outside [0, {len(names)})"
            )
        pairs = set(zip(scan_index.tolist(), channel_index.tolist(), strict=True))
        if len(pairs) != group_count:
            raise ValueError(
                "duplicate (group_scan_index, group_channel_index) pairs: "
                "exactly one group per scan/channel pair"
            )
    if channel_mode_weights is None and channel_mode_matrix is None:
        if channel_measurement_kinds is not None:
            raise ValueError(
                "channel_measurement_kinds requires a measurement "
                "representation: channel_mode_weights and/or "
                "channel_mode_matrix"
            )
    else:
        if modes is None:
            raise ValueError(
                "the measurement basis (channel_mode_weights/"
                "channel_mode_matrix) requires mode_labels: the basis "
                "indexes the mode axis"
            )
        if channel_measurement_kinds is None:
            raise ValueError(
                "the measurement basis (channel_mode_weights/"
                "channel_mode_matrix) requires channel_measurement_kinds: "
                "the kind selects the representation each channel carries"
            )
        kinds = _check_measurement_kinds(channel_measurement_kinds, len(names))
        weights = (
            None
            if channel_mode_weights is None
            else _check_channel_mode_weights(
                channel_mode_weights, len(names), len(modes)
            )
        )
        matrices = (
            None
            if channel_mode_matrix is None
            else _check_channel_mode_matrix(channel_mode_matrix, len(names), len(modes))
        )
        _check_measurement_basis(names, len(modes), kinds, weights, matrices)
    candidate_count = int(offsets[-1])

    for required in ("location", "intensity", "status_code"):
        if required not in columns:
            raise ValueError(f"candidate table misses required column {required!r}")
    for name, values in columns.items():
        column = np.asarray(values)
        if column.ndim != 1 or len(column) != candidate_count:
            raise ValueError(
                f"column {name!r} must be one-dimensional with exactly one "
                f"entry per candidate ({candidate_count})"
            )
    _status_code_array(
        "status_code",
        columns["status_code"],
        mask=PEAK_CANDIDATE_STATUS_MASK,
        label="candidate",
    )
    for capability in optional_capabilities:
        if capability not in columns:
            continue
        mask_name = f"{capability}_valid"
        if mask_name not in columns:
            raise ValueError(
                f"optional capability {capability!r} requires a "
                f"{mask_name!r} mask column; NaN is not a valid mask"
            )
        mask = np.asarray(columns[mask_name])
        if not np.issubdtype(mask.dtype, np.integer) or not set(np.unique(mask)) <= {
            0,
            1,
        }:
            raise ValueError(f"mask column {mask_name!r} must hold only 0/1 values")

def validate_fit_table(
    candidate_row: Any,
    parameter_code: Any,
    value: Any,
    *,
    candidate_count: int,
    parameter_names: Iterable[str] = (),
) -> None:
    """Validate a materialized peak fit-parameter table.

    The three columns must be one-dimensional with equal length; every
    ``candidate_row`` must reference an existing candidate, and every
    ``parameter_code`` must stay within ``parameter_names`` when the
    code-to-name mapping is given.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    rows = _index_array("candidate_row", candidate_row)
    codes = _index_array("parameter_code", parameter_code)
    values = np.asarray(value)
    if values.ndim != 1:
        raise ValueError("value must be one-dimensional")
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("value must be a real numeric column")
    if not (len(rows) == len(codes) == len(values)):
        raise ValueError(
            "candidate_row, parameter_code and value must have equal length"
        )
    if candidate_count < 0:
        raise ValueError("candidate_count must be nonnegative")
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= candidate_count):
        raise ValueError(
            f"candidate_row references rows outside [0, {candidate_count})"
        )
    names = tuple(parameter_names)
    if len(codes) and int(codes.min()) < 0:
        raise ValueError("parameter_code must be nonnegative")
    if names and len(codes) and int(codes.max()) >= len(names):
        raise ValueError(
            f"parameter_code references parameters outside [0, {len(names)})"
        )

def validate_path_table(
    path_offsets: Any,
    candidate_row: Any,
    scan_position: Any,
    candidate_count: int,
    *,
    path_status_code: Any,
) -> None:
    """Validate a materialized peak path table.

    ``path_offsets`` must be legal offsets whose terminal value equals the
    number of path members; ``candidate_row`` and ``scan_position`` must have
    equal length, and every candidate row must reference an existing
    candidate. ``path_status_code`` must hold one path-level
    :class:`PeakStatus` code per path, within
    :data:`PEAK_PATH_STATUS_MASK` — ``CENTER_IDENTIFIABLE`` is
    candidate-level and fails here. Paths never modify the candidate body.

    Raises
    ------
    ValueError
        On any contract violation.

    """
    offsets = _offsets_array("path_offsets", path_offsets)
    rows = np.asarray(candidate_row)
    positions = np.asarray(scan_position)
    if rows.ndim != 1 or positions.ndim != 1:
        raise ValueError("candidate_row and scan_position must be 1-D")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("candidate_row must have an integer dtype")
    if len(rows) != len(positions):
        raise ValueError("candidate_row and scan_position must have equal length")
    if int(offsets[-1]) != len(rows):
        raise ValueError(
            f"the terminal path offset ({int(offsets[-1])}) must equal the "
            f"path member count ({len(rows)})"
        )
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= candidate_count):
        raise ValueError(
            f"candidate_row references rows outside [0, {candidate_count})"
        )
    status = _status_code_array(
        "path_status_code",
        path_status_code,
        mask=PEAK_PATH_STATUS_MASK,
        label="path",
    )
    if len(status) != len(offsets) - 1:
        raise ValueError(
            f"path_status_code must hold exactly one entry per path "
            f"({len(offsets) - 1}), got {len(status)}"
        )

def validate_confidence_bounds(
    confidence_lower: Any,
    confidence_upper: Any,
    valid_mask: Any | None = None,
) -> None:
    """Validate that confidence lower bounds never exceed upper bounds.

    Bounds are compared elementwise; when ``valid_mask`` is given, only
    masked-in entries are checked.

    Raises
    ------
    ValueError
        On shape mismatch or an inverted interval.

    """
    lower = np.asarray(confidence_lower, dtype=np.float64)
    upper = np.asarray(confidence_upper, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError(
            "confidence_lower and confidence_upper must have the same shape"
        )
    if valid_mask is not None:
        mask = np.asarray(valid_mask).astype(bool)
        if mask.shape != lower.shape:
            raise ValueError("valid_mask must match the bounds shape")
        lower = lower[mask]
        upper = upper[mask]
    if np.any(lower > upper):
        raise ValueError("confidence lower bound exceeds upper bound")
