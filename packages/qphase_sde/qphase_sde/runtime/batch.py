"""In-memory views over fused SDE scan results."""

from __future__ import annotations

from typing import Any

from qphase_sde.result import SDEResult
from qphase_sde.state import TrajectorySet


class SDEResultSplitter:
    """Create one point view without creating scheduler jobs or directories."""

    @staticmethod
    def point_view(
        result: SDEResult,
        *,
        index: int,
        scan_count: int,
        params: dict[str, Any],
        job_name: str | None = None,
    ) -> SDEResult:
        if index < 0 or index >= scan_count:
            raise IndexError(index)
        trajectory = result.trajectory
        point_trajectory = None
        if trajectory is not None:
            total = int(trajectory.data.shape[0])
            if total % scan_count != 0:
                raise ValueError(
                    f"trajectory count {total} is not divisible by scan count "
                    f"{scan_count}"
                )
            per_point = total // scan_count
            start = index * per_point
            stop = start + per_point
            point_trajectory = TrajectorySet(
                data=trajectory.data[start:stop],
                t0=trajectory.t0,
                dt=trajectory.dt,
                meta=dict(trajectory.meta),
            )
        analysis = {
            name: value[index]
            if isinstance(value, list) and len(value) == scan_count
            else value
            for name, value in result.analysis.items()
        }
        meta = dict(result.meta)
        meta["params"] = params
        meta["scan_index"] = index
        if job_name is not None:
            meta["job_name"] = job_name
        return SDEResult(point_trajectory, analysis=analysis, meta=meta)
