"""Continuity-based ridge tracker for grouped peak candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

import numpy as np
from pydantic import Field

from ..contracts.peaks import PeakPathResult, PeakStatus
from .base import BasePeakTrackerConfig, PeakTrackerProtocol

__all__ = ["RidgePeakTracker", "RidgePeakTrackerConfig"]


class RidgePeakTrackerConfig(BasePeakTrackerConfig):
    """Configuration for a data-only continuity tracker."""

    frequency_scale: float = Field(1.0, gt=0.0)
    continuity_weight: float = Field(1.0, ge=0.0)
    huber_delta: float = Field(1.0, gt=0.0)
    path_count: int = Field(1, ge=1, le=8)
    ambiguity_cost_delta: float = Field(
        1.0,
        ge=0.0,
        description="Minimum runner-up cost gap required to assert PATH_UNIQUE",
    )


def _huber_loss(value: float, delta: float) -> float:
    magnitude = abs(value)
    if magnitude <= delta:
        return 0.5 * magnitude**2
    return delta * (magnitude - 0.5 * delta)


def _score_evidence(value: float) -> float:
    return -float(np.log(max(value, 1.0e-12)))


class RidgePeakTracker(PeakTrackerProtocol):
    """Track smooth candidate paths independently for each channel.

    The tracker uses only candidate locations and intensities. It does not
    interpret a path as a physical branch; that distinction remains in the
    peak-path product contract.
    """

    name: ClassVar[str] = "ridge"
    description: ClassVar[str] = "Continuity-based scan-axis ridge tracker"
    config_schema: ClassVar[type[RidgePeakTrackerConfig]] = RidgePeakTrackerConfig
    required_inputs: ClassVar[tuple[str, ...]] = ("candidates",)

    def track(
        self,
        candidate_offsets: np.ndarray,
        group_scan_index: np.ndarray,
        group_channel_index: np.ndarray,
        columns: Mapping[str, np.ndarray],
        *,
        scan_count: int,
        channel_names: Sequence[str],
    ) -> list[PeakPathResult]:
        offsets = np.asarray(candidate_offsets)
        scan_indices = np.asarray(group_scan_index)
        channel_indices = np.asarray(group_channel_index)
        locations = np.asarray(columns["location"], dtype=float)
        intensities = np.asarray(columns["intensity"], dtype=float)
        if offsets.ndim != 1 or len(offsets) != len(scan_indices) + 1:
            raise ValueError("candidate_offsets must have one terminal entry")
        if len(locations) != int(offsets[-1]) or len(intensities) != len(locations):
            raise ValueError("candidate columns do not match candidate_offsets")

        config = cast(RidgePeakTrackerConfig, self.config)
        scores = np.asarray(columns.get("score", intensities), dtype=float)
        if len(scores) != len(locations):
            raise ValueError("score column does not match candidate_offsets")
        paths: list[PeakPathResult] = []
        for channel in range(len(channel_names)):
            groups: dict[int, list[int]] = {}
            for group, (scan, group_channel) in enumerate(
                zip(scan_indices, channel_indices, strict=True)
            ):
                if int(group_channel) == channel:
                    start, stop = int(offsets[group]), int(offsets[group + 1])
                    groups.setdefault(int(scan), []).extend(range(start, stop))
            if not groups:
                continue
            scan_layers = [
                (scan, rows) for scan, rows in sorted(groups.items()) if rows
            ]
            states: dict[int, list[tuple[tuple[int, ...], float]]] = {
                row: [((row,), _score_evidence(scores[row]))]
                for row in scan_layers[0][1]
            }
            # Keep the requested paths plus one runner-up throughout the
            # dynamic program.  The runner-up is needed for the final
            # ambiguity test even when the public output asks for one path;
            # pruning only at the final layer loses paths that merge there.
            internal_path_limit = max(2, config.path_count)
            for _, rows in scan_layers[1:]:
                next_states: dict[int, list[tuple[tuple[int, ...], float]]] = {}
                for target in rows:
                    options: list[tuple[tuple[int, ...], float]] = []
                    for source, source_paths in states.items():
                        transition = config.continuity_weight * _huber_loss(
                            (locations[target] - locations[source])
                            / config.frequency_scale,
                            config.huber_delta,
                        )
                        evidence = _score_evidence(scores[target])
                        options.extend(
                            (path + (target,), cost + transition + evidence)
                            for path, cost in source_paths
                        )
                    options.sort(key=lambda item: item[1])
                    unique: list[tuple[tuple[int, ...], float]] = []
                    seen: set[tuple[int, ...]] = set()
                    for path, cost in options:
                        if path in seen:
                            continue
                        seen.add(path)
                        unique.append((path, cost))
                        if len(unique) == internal_path_limit:
                            break
                    next_states[target] = unique
                states = next_states
            complete_all = sorted(
                (path for state in states.values() for path in state),
                key=lambda item: item[1],
            )
            complete = complete_all[: config.path_count]
            runner_up_gap = (
                complete_all[1][1] - complete_all[0][1]
                if len(complete_all) > 1
                else math.inf
            )
            path_is_unique = runner_up_gap > config.ambiguity_cost_delta
            for path_index, (candidate_rows, _) in enumerate(complete):
                status = PeakStatus.COMPUTED
                if path_index == 0 and path_is_unique:
                    status |= PeakStatus.PATH_UNIQUE
                paths.append(
                    PeakPathResult(
                        path_id=f"{self.name}:{channel_names[channel]}:{path_index}",
                        candidate_rows=list(candidate_rows),
                        scan_positions=[scan for scan, _ in scan_layers],
                        status=status,
                    )
                )
        return paths
