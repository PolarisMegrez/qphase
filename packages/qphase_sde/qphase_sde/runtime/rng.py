"""qphase_sde: Grouped RNG streams for fused and batched SDE runs.

Every trajectory's random stream derives from (master seed, absolute scan
point, logical group index); these helpers build those streams and draw from
them. Nothing here imports an analyser or the engine class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.backend.base import BackendBase


@dataclass(frozen=True)
class _GroupedRNG:
    """Independent RNG handles, each driving a fixed run of trajectories.

    ``counts`` gives the per-handle trajectory counts when groups are not
    uniform (a short tail group ends a scan point's ensemble); ``None``
    means every handle drives ``group_size`` trajectories.
    """

    handles: tuple[Any, ...]
    group_size: int
    counts: tuple[int, ...] | None = None

def _rng_group_seed(master_seed: int, point_index: int, group_index: int) -> int:
    """Derive one logical-group stream seed from (master, scan point, group)."""
    sequence = np.random.SeedSequence(
        [int(master_seed), int(point_index), int(group_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])

def _rng_group_counts(group_size: int, n_traj: int) -> tuple[int, ...]:
    """Per-group trajectory counts covering ``n_traj`` rows, tail group last."""
    group_count = -(-n_traj // group_size)
    return (group_size,) * (group_count - 1) + (
        n_traj - (group_count - 1) * group_size,
    )

def _fused_rng_groups(
    master_seed: int,
    point_start: int,
    point_stop: int,
    group_size: int,
    n_traj_per_point: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Point-major (seeds, counts) for a fused span of scan points.

    Every trajectory's random stream derives from (master seed, absolute
    scan point, logical group index), so fused, tiled, and
    trajectory-batched executions of the same configuration draw identical
    streams.
    """
    point_counts = _rng_group_counts(group_size, n_traj_per_point)
    seeds: list[int] = []
    counts: list[int] = []
    for point in range(point_start, point_stop):
        seeds.extend(
            _rng_group_seed(master_seed, point, group)
            for group in range(len(point_counts))
        )
        counts.extend(point_counts)
    return tuple(seeds), tuple(counts)

class _AdaptiveNoiseRNG:
    """Generator-style facade the engine hands to adaptive integrators.

    Adaptive integrators draw noise internally through the ``normal(size=...)``
    API declared by the integrator contract; this facade routes the draw
    through the engine's grouped-stream machinery so per-group provenance is
    preserved without exposing the private ``_GroupedRNG`` container to
    integrator or backend code.
    """

    def __init__(self, backend: Any, rng: Any) -> None:
        self._backend = backend
        self._rng = rng

    def normal(self, size: Any = None) -> Any:
        shape = tuple(size) if size is not None else ()
        out = self._backend.empty(shape, np.float64)
        draw_standard_normal_into(self._backend, self._rng, out)
        return out

def draw_standard_normal_into(backend: BackendBase, rng: Any, out: Any) -> Any:
    """Fill a cached noise array, including stable grouped RNG slices."""
    if not isinstance(rng, _GroupedRNG):
        fill = getattr(backend, "randn_into", None)
        if callable(fill):
            return fill(rng, out)
        out[...] = backend.randn(rng, out.shape, dtype=out.dtype)
        return out

    trajectory_axis = 1 if out.ndim == 3 else 0
    counts = rng.counts
    if counts is None:
        counts = (rng.group_size,) * len(rng.handles)
    if len(counts) != len(rng.handles):
        raise ValueError("grouped RNG counts do not match the handle count")
    expected = sum(counts)
    if int(out.shape[trajectory_axis]) != expected:
        raise ValueError("grouped RNG shape does not match the fused scan tile")
    fill = getattr(backend, "randn_into", None)
    start = 0
    for handle, count in zip(rng.handles, counts, strict=True):
        selection = [slice(None)] * out.ndim
        selection[trajectory_axis] = slice(start, start + count)
        start += count
        target = out[tuple(selection)]
        if callable(fill):
            fill(handle, target)
        else:
            target[...] = backend.randn(handle, target.shape, dtype=target.dtype)
    return out
