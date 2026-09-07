"""Contract tests for the grouped, channel-aware peak candidate schema.

Freezes the Phase 5A peak layout: candidates group by scan point x channel
(one group per pair, each holding 0/1/N candidates), channel labels and the
mode-indexed measurement basis live in schema attributes (never in payload
columns), the single ``status_code`` column layers the per-table
:class:`PeakStatus` masks, and a materialized candidate product survives a
clean-process artifact roundtrip.
"""

import json
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.data import AxisRole, ProductSchema, StatisticsDataset, save_products
from qphase_sde.contracts import peaks

CHANNEL_NAMES = ["mode_0", "mode_1"]
MODE_LABELS = ["mode_0", "mode_1"]
#: Identity basis over two bare modes, flattened row-major over
#: (channel x mode): channel 0 reads mode 0, channel 1 reads mode 1.
CHANNEL_MODE_WEIGHTS = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]

#: Variables kept by the concrete product of the roundtrip test. Uncertainty
#: data variables (conditional/sampling location std) must stay because the
#: template's uncertainty entries reference them; ``width`` stands in for an
#: exercised optional capability; the remaining optional columns are dropped,
#: as finders without those capabilities are allowed to do.
_KEPT_VARIABLES = {
    "group_scan_index",
    "group_channel_index",
    "candidate_offsets",
    "location",
    "intensity",
    "status_code",
    "conditional_location_std",
    "conditional_location_std_valid",
    "sampling_location_std",
    "sampling_location_std_valid",
    "width",
    "width_valid",
}


def _grouped_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """3 scan points x 2 channels with 2/0/1/3/0/1 candidates per group."""
    offsets = np.array([0, 2, 2, 3, 6, 6, 7], dtype=np.int64)
    group_scan_index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    group_channel_index = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    candidate_count = int(offsets[-1])
    columns: dict[str, Any] = {
        "location": np.linspace(0.1, 0.7, candidate_count),
        "intensity": np.arange(candidate_count, dtype=np.float64) + 1.0,
        # Candidate codes layer COMPUTED|CENTER_IDENTIFIABLE only (mask 3).
        "status_code": np.array([3, 1, 3, 1, 2, 0, 3], dtype=np.int64),
        "width": np.full(candidate_count, 0.1),
        "width_valid": np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.int8),
    }
    return offsets, group_scan_index, group_channel_index, columns


def _validate(offsets, group_scan_index, group_channel_index, columns) -> None:
    peaks.validate_candidate_table(
        offsets,
        group_scan_index,
        group_channel_index,
        columns,
        scan_count=3,
        channel_names=CHANNEL_NAMES,
    )


def test_grouped_schema_template_layout():
    """The template declares the grouped axes, index columns and attributes."""
    schema = peaks.PEAK_PRODUCT
    assert not schema.is_closed  # plan-time-open template
    roles = {axis.name: axis.role for axis in schema.axes}
    assert roles == {
        "scan": AxisRole.PARAMETER,
        "group": AxisRole.INDEX,
        "group_offset": AxisRole.INDEX,
        "candidate": AxisRole.INDEX,
    }

    group_scan = schema.variable("group_scan_index")
    assert group_scan.dims == ("group",)
    assert np.dtype(group_scan.dtype).kind == "i"
    assert schema.variable("group_channel_index").dims == ("group",)
    # One offsets entry per group plus the terminal offset: the dedicated
    # offsets axis mirrors the path/path_offset pattern of PEAK_PATH_PRODUCT.
    assert schema.variable("candidate_offsets").dims == ("group_offset",)
    assert schema.variable("status_code").dims == ("candidate",)

    # Channel labels and the measurement basis are stable attribute
    # mappings, never payload columns.
    variable_names = {variable.name for variable in schema.variables}
    assert "channel_names" not in variable_names
    assert "mode_labels" not in variable_names
    for variable in schema.variables:
        assert np.dtype(variable.dtype).kind in "fi", variable.name
    attributes = schema.attributes
    assert attributes["channel_names"] == ["channel_0"]
    assert attributes["mode_labels"] == ["mode_0"]
    assert attributes["group_axis"] == "group"
    assert attributes["group_scan_variable"] == "group_scan_index"
    assert attributes["group_channel_variable"] == "group_channel_index"
    assert attributes["offsets_variable"] == "candidate_offsets"
    # status_bits is per-table: exactly the flags each product may layer.
    assert attributes["status_bits"] == {
        "computed": 1,
        "center_identifiable": 2,
    }
    assert peaks.PEAK_PATH_PRODUCT.attributes["status_bits"] == {
        "computed": 1,
        "path_unique": 4,
    }


def test_estimand_capability_lives_in_valid_masks_not_status():
    """Each optional capability has a ``<name>_valid`` mask column; the
    status flags carry no capability bit.
    """
    variable_names = {variable.name for variable in peaks.PEAK_PRODUCT.variables}
    for name in peaks.OPTIONAL_CAPABILITY_FIELDS:
        assert name in variable_names
        assert f"{name}_valid" in variable_names
    assert peaks.PEAK_PRODUCT.attributes["valid_mask_suffix"] == "_valid"
    assert "capability" not in peaks.PeakCandidate.model_fields


def test_peak_status_flag_semantics():
    """One flag enum serves both tables; each table enforces its own mask."""
    assert int(peaks.PeakStatus.COMPUTED) == 1
    assert int(peaks.PeakStatus.CENTER_IDENTIFIABLE) == 2
    assert int(peaks.PeakStatus.PATH_UNIQUE) == 4
    assert peaks.PEAK_STATUS_MASK == 7
    # Candidates layer computation and center identifiability only;
    # PATH_UNIQUE belongs to path products.
    assert peaks.PEAK_CANDIDATE_STATUS_MASK == 3
    # Paths layer computation and path ambiguity only; CENTER_IDENTIFIABLE
    # is meaningless on a path (a path has no center of its own).
    assert peaks.PEAK_PATH_STATUS_MASK == 5
    assert peaks.PeakStatus(0) is peaks.PeakStatus.NONE
    combined = peaks.PeakStatus.COMPUTED | peaks.PeakStatus.PATH_UNIQUE
    assert combined & peaks.PeakStatus.COMPUTED
    assert not combined & peaks.PeakStatus.CENTER_IDENTIFIABLE


def test_peak_candidate_and_path_status_flags_roundtrip():
    """Pydantic rows carry the layered flags and reject out-of-mask bits."""
    candidate = peaks.PeakCandidate(
        location=1.25,
        intensity=42.0,
        status=peaks.PeakStatus.COMPUTED | peaks.PeakStatus.CENTER_IDENTIFIABLE,
    )
    assert int(candidate.status) == 3
    payload = json.loads(json.dumps(candidate.model_dump(mode="json")))
    assert payload["status"] == 3
    assert peaks.PeakCandidate.model_validate(payload) == candidate

    # The default asserts computation only; identifiability is opt-in.
    default = peaks.PeakCandidate(location=1.0, intensity=2.0)
    assert default.status is peaks.PeakStatus.COMPUTED

    path = peaks.PeakPathResult(
        path_id="p0",
        candidate_rows=[0, 3],
        scan_positions=[0, 1],
        status=peaks.PeakStatus.COMPUTED | peaks.PeakStatus.PATH_UNIQUE,
    )
    assert path.status & peaks.PeakStatus.PATH_UNIQUE
    payload = json.loads(json.dumps(path.model_dump(mode="json")))
    assert peaks.PeakPathResult.model_validate(payload) == path
    # The default path asserts computation only; uniqueness is opt-in.
    default_path = peaks.PeakPathResult(
        path_id="p1", candidate_rows=[0], scan_positions=[0]
    )
    assert default_path.status is peaks.PeakStatus.COMPUTED

    # PATH_UNIQUE is a path-product bit and fails on candidates;
    # CENTER_IDENTIFIABLE is candidate-level and fails on paths.
    for bad_status in (4, 5, 7, 8, -1):
        with pytest.raises(ValidationError, match="PeakStatus mask"):
            peaks.PeakCandidate(location=1.0, intensity=2.0, status=bad_status)
    for bad_status in (2, 3, 7, 8, -1):
        with pytest.raises(ValidationError, match="PeakStatus mask"):
            peaks.PeakPathResult(
                path_id="p",
                candidate_rows=[0],
                scan_positions=[0],
                status=bad_status,
            )


def test_grouped_candidate_table_validates():
    """Groups with 0/1/N candidates over 3 scan points x 2 channels pass."""
    offsets, group_scan_index, group_channel_index, columns = _grouped_table()
    _validate(offsets, group_scan_index, group_channel_index, columns)
    # The measurement basis, when present, holds one [re, im] pair per
    # (channel, mode) entry flattened row-major.
    peaks.validate_candidate_table(
        offsets,
        group_scan_index,
        group_channel_index,
        columns,
        scan_count=3,
        channel_names=CHANNEL_NAMES,
        mode_labels=MODE_LABELS,
        channel_mode_weights=CHANNEL_MODE_WEIGHTS,
    )


def test_empty_candidate_table_validates():
    """A table with zero groups and zero candidates is legal."""
    peaks.validate_candidate_table(
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        {
            "location": np.zeros(0),
            "intensity": np.zeros(0),
            "status_code": np.zeros(0, dtype=np.int64),
        },
        scan_count=3,
        channel_names=CHANNEL_NAMES,
    )


def test_candidate_offsets_contract():
    """Offsets: start 0, monotone, group_count + 1 entries, terminal count."""
    offsets, gsi, gci, columns = _grouped_table()

    with pytest.raises(ValueError, match="start at 0"):
        _validate(np.array([1, 2, 2, 3, 6, 6, 7]), gsi, gci, columns)
    with pytest.raises(ValueError, match="non-decreasing"):
        _validate(np.array([0, 3, 2, 3, 6, 6, 7]), gsi, gci, columns)
    with pytest.raises(ValueError, match="integer dtype"):
        _validate(np.array([0.0, 2.0, 2.0, 3.0, 6.0, 6.0, 7.0]), gsi, gci, columns)
    with pytest.raises(ValueError, match=r"group_count \+ 1"):
        # One offsets entry missing for the six groups.
        _validate(np.array([0, 2, 2, 3, 6, 7]), gsi, gci, columns)
    with pytest.raises(ValueError, match="one entry per candidate"):
        # Terminal offset (7) disagrees with the column length (6).
        short = {name: np.asarray(values)[:6] for name, values in columns.items()}
        _validate(offsets, gsi, gci, short)
    with pytest.raises(ValueError, match="required column"):
        _validate(offsets, gsi, gci, {})


def test_group_indices_must_stay_in_range():
    """Group scan/channel indices reference existing scan points/channels."""
    offsets, gsi, gci, columns = _grouped_table()

    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        _validate(offsets, np.array([0, 0, 1, 1, 2, 3]), gci, columns)
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        _validate(offsets, np.array([0, 0, 1, 1, 2, -1]), gci, columns)
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        _validate(offsets, gsi, np.array([0, 1, 0, 1, 0, 2]), columns)
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        _validate(offsets, gsi, np.array([0, 1, 0, 1, 0, -1]), columns)
    with pytest.raises(ValueError, match="one entry per group"):
        _validate(offsets, gsi[:5], gci, columns)
    with pytest.raises(ValueError, match="integer dtype"):
        _validate(offsets, gsi.astype(np.float64), gci, columns)


def test_one_group_per_scan_channel_pair():
    """Duplicate (scan, channel) groups are rejected: no replicate dimension."""
    offsets = np.array([0, 1, 2], dtype=np.int64)
    columns = {
        "location": np.array([0.1, 0.2]),
        "intensity": np.array([1.0, 2.0]),
        "status_code": np.array([1, 3], dtype=np.int64),
    }
    # Two groups claiming scan point 0 / channel 0: rejected.
    with pytest.raises(ValueError, match="one group per scan/channel pair"):
        peaks.validate_candidate_table(
            offsets,
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            columns,
            scan_count=3,
            channel_names=CHANNEL_NAMES,
        )
    # The same scan point on two different channels is legal.
    peaks.validate_candidate_table(
        offsets,
        np.array([0, 0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        columns,
        scan_count=3,
        channel_names=CHANNEL_NAMES,
    )


def test_measurement_basis_attributes_are_stable():
    """channel_names/mode_labels: unique non-empty strings; weights closed."""
    offsets, gsi, gci, columns = _grouped_table()

    for bad, match in (
        ([], "at least one"),
        (["mode_0", "mode_0"], "unique"),
        (["mode_0", ""], "non-empty strings"),
        (["mode_0", 1], "non-empty strings"),
        ("mode_0", "list of strings"),
    ):
        with pytest.raises(ValueError, match=match):
            peaks.validate_candidate_table(
                offsets, gsi, gci, columns, scan_count=3, channel_names=bad
            )
        with pytest.raises(ValueError, match=match):
            peaks.validate_candidate_table(
                offsets,
                gsi,
                gci,
                columns,
                scan_count=3,
                channel_names=CHANNEL_NAMES,
                mode_labels=bad,
            )

    # Weights must close over (channel x mode): 2 channels x 2 modes = 4.
    with pytest.raises(ValueError, match=r"\[re, im\] pair per \(channel, mode\)"):
        peaks.validate_candidate_table(
            offsets,
            gsi,
            gci,
            columns,
            scan_count=3,
            channel_names=CHANNEL_NAMES,
            mode_labels=MODE_LABELS,
            channel_mode_weights=[[1.0, 0.0], [0.0, 1.0]],
        )
    # The nested (channel, mode, 2) form is rejected: only the flattened
    # row-major list of pairs is JSON-clean and shape-unambiguous.
    with pytest.raises(ValueError, match="flattened row-major"):
        peaks.validate_candidate_table(
            offsets,
            gsi,
            gci,
            columns,
            scan_count=3,
            channel_names=CHANNEL_NAMES,
            mode_labels=MODE_LABELS,
            channel_mode_weights=np.array(
                [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        peaks.validate_candidate_table(
            offsets,
            gsi,
            gci,
            columns,
            scan_count=3,
            channel_names=CHANNEL_NAMES,
            mode_labels=MODE_LABELS,
            channel_mode_weights=[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.inf, 0.0]],
        )
    # Weights index the mode axis: they are meaningless without mode labels.
    with pytest.raises(ValueError, match="requires mode_labels"):
        peaks.validate_candidate_table(
            offsets,
            gsi,
            gci,
            columns,
            scan_count=3,
            channel_names=CHANNEL_NAMES,
            channel_mode_weights=CHANNEL_MODE_WEIGHTS,
        )


def test_multi_mode_bare_bright_dark_readout_basis():
    """A 3-mode basis expresses bare, bright and dark readout channels."""
    mode_labels = ["mode_0", "mode_1", "mode_2"]
    channel_names = ["bare_0", "bright", "dark"]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # Readout vectors l of c = l^dagger alpha, flattened row-major over
    # (channel x mode): bare mode 0, bright (1,1,0)/sqrt(2), dark
    # (1,-1,0)/sqrt(2).
    weights = [
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [inv_sqrt2, 0.0],
        [inv_sqrt2, 0.0],
        [0.0, 0.0],
        [inv_sqrt2, 0.0],
        [-inv_sqrt2, 0.0],
        [0.0, 0.0],
    ]
    offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    columns = {
        "location": np.array([0.3, 0.5, 0.4]),
        "intensity": np.array([2.0, 4.0, 0.1]),
        "status_code": np.array([3, 3, 1], dtype=np.int64),
    }
    peaks.validate_candidate_table(
        offsets,
        np.array([0, 0, 0], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        columns,
        scan_count=1,
        channel_names=channel_names,
        mode_labels=mode_labels,
        channel_mode_weights=weights,
    )
    # Every readout vector is normalized; bright and dark are orthogonal.
    vectors = np.asarray(weights)
    vectors = vectors[:, 0] + 1j * vectors[:, 1]
    bare, bright, dark = vectors.reshape(3, 3)
    assert np.isclose(np.linalg.norm(bare), 1.0)
    assert np.isclose(np.linalg.norm(bright), 1.0)
    assert np.isclose(np.linalg.norm(dark), 1.0)
    assert np.isclose(np.vdot(bright, dark), 0.0)
    # Dropping one (channel, mode) pair breaks the flattened closure.
    with pytest.raises(ValueError, match=r"\(channel, mode\)"):
        peaks.validate_candidate_table(
            offsets,
            np.array([0, 0, 0], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            columns,
            scan_count=1,
            channel_names=channel_names,
            mode_labels=mode_labels,
            channel_mode_weights=weights[:-1],
        )


def test_candidate_status_codes_stay_within_candidate_mask():
    """PATH_UNIQUE, unknown bits and negative codes fail candidate tables."""
    offsets, gsi, gci, columns = _grouped_table()

    for bad_code in (4, 5, 7, 8, -1):
        columns["status_code"] = np.array([3, 1, 3, 1, 2, 0, bad_code], dtype=np.int64)
        with pytest.raises(ValueError, match="PeakStatus mask"):
            _validate(offsets, gsi, gci, columns)
    columns["status_code"] = np.full(7, 3, dtype=np.float64)
    with pytest.raises(ValueError, match="integer dtype"):
        _validate(offsets, gsi, gci, columns)


def test_optional_capability_requires_valid_mask():
    """Optional columns need an explicit 0/1 '<name>_valid' mask column."""
    offsets, gsi, gci, columns = _grouped_table()
    _validate(offsets, gsi, gci, columns)

    without_mask = {k: v for k, v in columns.items() if k != "width_valid"}
    with pytest.raises(ValueError, match="mask column"):
        _validate(offsets, gsi, gci, without_mask)
    bad_mask = {**columns, "width_valid": np.full(7, 7, dtype=np.int8)}
    with pytest.raises(ValueError, match="0/1"):
        _validate(offsets, gsi, gci, bad_mask)


def test_fit_table_validation():
    """Fit rows are keyed by candidate_row within the candidate count."""
    peaks.validate_fit_table(
        [0, 0, 3],
        [0, 1, 0],
        [0.5, 0.1, 1.2],
        candidate_count=7,
        parameter_names=["gamma", "shift"],
    )
    with pytest.raises(ValueError, match="equal length"):
        peaks.validate_fit_table([0], [0, 1], [0.5], candidate_count=7)
    with pytest.raises(ValueError, match=r"outside \[0, 7\)"):
        peaks.validate_fit_table([7], [0], [0.5], candidate_count=7)
    with pytest.raises(ValueError, match=r"outside \[0, 7\)"):
        peaks.validate_fit_table([-1], [0], [0.5], candidate_count=7)
    with pytest.raises(ValueError, match="nonnegative"):
        peaks.validate_fit_table([0], [-1], [0.5], candidate_count=7)
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        peaks.validate_fit_table(
            [0], [2], [0.5], candidate_count=7, parameter_names=["gamma", "shift"]
        )
    with pytest.raises(ValueError, match="real numeric"):
        peaks.validate_fit_table(
            [0],
            [0],
            np.array([1.0j]),
            candidate_count=7,
        )


def test_path_table_validation():
    """Path offsets, candidate rows, positions and status codes agree."""
    peaks.validate_path_table(
        path_offsets=[0, 2, 5],
        candidate_row=[0, 3, 1, 2, 4],
        scan_position=[0, 1, 0, 1, 2],
        candidate_count=5,
        path_status_code=[5, 1],
    )
    with pytest.raises(ValueError, match="equal length"):
        peaks.validate_path_table([0, 2], [0, 3], [0], 5, path_status_code=[5])
    with pytest.raises(ValueError, match="terminal path offset"):
        peaks.validate_path_table([0, 3], [0, 3], [0, 1], 5, path_status_code=[5])
    with pytest.raises(ValueError, match="outside"):
        peaks.validate_path_table([0, 2], [0, 9], [0, 1], 5, path_status_code=[5])
    with pytest.raises(ValueError, match="PeakStatus mask"):
        # Unknown bit.
        peaks.validate_path_table([0, 2], [0, 1], [0, 1], 5, path_status_code=[8])
    with pytest.raises(ValueError, match="PeakStatus mask"):
        # CENTER_IDENTIFIABLE is candidate-level and fails on paths.
        peaks.validate_path_table([0, 2], [0, 1], [0, 1], 5, path_status_code=[2])
    with pytest.raises(ValueError, match="one entry per path"):
        peaks.validate_path_table([0, 2], [0, 1], [0, 1], 5, path_status_code=[5, 5])


def test_uncertainty_scopes_stay_separate():
    """conditional/sampling/path_model_selection scopes stay distinct."""
    uncertainties = {u.scope: u for u in peaks.PEAK_PRODUCT.uncertainties}
    assert set(uncertainties) == {"conditional", "sampling"}
    assert uncertainties["conditional"].data_variable == "conditional_location_std"
    sampling = uncertainties["sampling"]
    assert sampling.data_variable == "sampling_location_std"
    assert sampling.sampling_basis == "trajectory"
    assert peaks.PEAK_PATH_PRODUCT.attributes["uncertainty_scope"] == (
        "path_model_selection"
    )


def _closed_schema(*, scan_count: int, group_count: int, candidate_count: int):
    """Close the open template for a concrete multi-scan multi-channel run."""
    payload = peaks.PEAK_PRODUCT.model_dump(mode="json")
    sizes = {
        "scan": scan_count,
        "group": group_count,
        "group_offset": group_count + 1,
        "candidate": candidate_count,
    }
    for axis in payload["axes"]:
        axis["size"] = sizes[axis["name"]]
    for basis in payload["sampling_bases"]:
        basis["count"] = 8
    payload["variables"] = [
        variable
        for variable in payload["variables"]
        if variable["name"] in _KEPT_VARIABLES
    ]
    payload["attributes"]["channel_names"] = list(CHANNEL_NAMES)
    payload["attributes"]["mode_labels"] = list(MODE_LABELS)
    payload["attributes"]["channel_mode_weights"] = CHANNEL_MODE_WEIGHTS
    return ProductSchema.model_validate(payload)


#: Clean-process probe body: the artifact is reopened without any in-process
#: state, the candidate table is re-validated from the persisted schema
#: attributes and the payload is echoed as JSON for the parent to compare.
#: The probe runs twice with different first imports — ``qphase.data`` and
#: ``qphase_sde.contracts.peaks`` — and both entry orders must initialize
#: cleanly.
_PROBE_BODY = """
import json
import sys

from qphase.data import load_bundle
from qphase_sde.contracts import peaks

bundle = load_bundle(sys.argv[1])
product = bundle.products["peaks"]
schema = product.schema
columns = {
    variable.name: product.handle(variable.name).materialize()
    for variable in schema.variables
    if variable.dims == ("candidate",)
}
offsets = product.handle("candidate_offsets").materialize()
group_scan_index = product.handle("group_scan_index").materialize()
group_channel_index = product.handle("group_channel_index").materialize()
peaks.validate_candidate_table(
    offsets,
    group_scan_index,
    group_channel_index,
    columns,
    scan_count=int(sys.argv[2]),
    channel_names=schema.attributes["channel_names"],
    mode_labels=schema.attributes["mode_labels"],
    channel_mode_weights=schema.attributes["channel_mode_weights"],
)
print(
    json.dumps(
        {
            "candidate_offsets": offsets.tolist(),
            "group_scan_index": group_scan_index.tolist(),
            "group_channel_index": group_channel_index.tolist(),
            "location": columns["location"].tolist(),
            "intensity": columns["intensity"].tolist(),
            "status_code": columns["status_code"].tolist(),
            "width_valid": columns["width_valid"].tolist(),
            "channel_names": schema.attributes["channel_names"],
            "mode_labels": schema.attributes["mode_labels"],
            "channel_mode_weights": schema.attributes["channel_mode_weights"],
            "status_bits": schema.attributes["status_bits"],
            "uncertainty_scopes": [u.scope for u in schema.uncertainties],
        }
    )
)
"""


@pytest.mark.parametrize(
    "first_import",
    ["import qphase.data", "import qphase_sde.contracts.peaks"],
    ids=["qphase-data-first", "peaks-first"],
)
def test_candidate_table_clean_process_roundtrip(tmp_path, first_import):
    """A materialized candidate product roundtrips through a clean process."""
    offsets, gsi, gci, columns = _grouped_table()
    candidate_count = int(offsets[-1])
    schema = _closed_schema(
        scan_count=3, group_count=6, candidate_count=candidate_count
    )
    arrays = {
        "group_scan_index": gsi,
        "group_channel_index": gci,
        "candidate_offsets": offsets,
        "location": columns["location"],
        "intensity": columns["intensity"],
        "status_code": columns["status_code"],
        "conditional_location_std": np.full(candidate_count, 0.01),
        "conditional_location_std_valid": np.ones(candidate_count, dtype=np.int8),
        "sampling_location_std": np.full(candidate_count, 0.02),
        "sampling_location_std_valid": np.ones(candidate_count, dtype=np.int8),
        "width": columns["width"],
        "width_valid": columns["width_valid"],
    }
    dataset = StatisticsDataset.from_arrays(schema, arrays, owner="test")
    # The same table validates against the in-process arrays before saving.
    _validate(offsets, gsi, gci, columns)
    save_products(tmp_path, {"peaks": dataset})

    probe = tmp_path / "probe.py"
    probe.write_text(f"{first_import}  # noqa: F401\n" + _PROBE_BODY, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(probe), str(tmp_path), "3"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload == {
        "candidate_offsets": offsets.tolist(),
        "group_scan_index": gsi.tolist(),
        "group_channel_index": gci.tolist(),
        "location": columns["location"].tolist(),
        "intensity": columns["intensity"].tolist(),
        "status_code": columns["status_code"].tolist(),
        "width_valid": columns["width_valid"].tolist(),
        "channel_names": CHANNEL_NAMES,
        "mode_labels": MODE_LABELS,
        "channel_mode_weights": CHANNEL_MODE_WEIGHTS,
        "status_bits": {
            "computed": 1,
            "center_identifiable": 2,
        },
        "uncertainty_scopes": ["conditional", "sampling"],
    }
