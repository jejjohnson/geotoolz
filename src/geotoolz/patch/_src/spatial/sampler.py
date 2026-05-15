"""`SpatialSampler` — where in the field to place anchors.

A `SpatialSampler` yields anchors; the `SpatialGeometry` then turns each anchor
into backend-specific indices. Five samplers cover the common cases:

- `SpatialRegularStride` — the canonical lattice (sliding-window inference).
- `SpatialJitteredStride` — regular grid with per-anchor uniform jitter.
- `SpatialRandom` — N uniformly-random anchors (training-time augmentation).
- `SpatialPoissonDisk` — well-spaced random anchors via Bridson's algorithm.
- `SpatialExplicit` — caller-supplied anchors (event-triggered, station list, …).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from geotoolz.patch._src.domains import GridDomain, PointDomain, VectorDomain
from geotoolz.patch._src.spatial.geometry import (
    SpatialGeometry,
    _is_raster_domain,
)


class SpatialSampler:
    """Base for anchor-placement strategies.

    Subclasses implement `anchors(domain, geometry) -> Iterable[Anchor]`.
    The geometry is passed so size-dependent samplers (`SpatialRegularStride`,
    `SpatialJitteredStride`) can compute valid anchor ranges.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterable[Any]:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class SpatialRegularStride(SpatialSampler):
    """Regular lattice — step along each axis.

    On a raster, ``step`` is in pixels; the anchors are upper-left
    corners. On a grid, ``step`` is per declared dim in coord order.

    Args:
        step: Stride. A scalar broadcasts; a sequence is per-axis.
    """

    step: int | tuple[int, ...]

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        if _is_raster_domain(domain):
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            sh, sw = self._broadcast(2)
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            for r in range(0, max(h - ph + 1, 1), sh):
                for c in range(0, max(w - pw + 1, 1), sw):
                    yield (r, c)
            return
        if isinstance(domain, GridDomain):
            dims = list(domain.coords)
            steps = self._broadcast(len(dims))
            size = getattr(geometry, "size", tuple([1] * len(dims)))
            lens = [len(domain.coords[d]) for d in dims]
            ranges = [
                range(0, max(L - int(p) + 1, 1), int(s))
                for L, p, s in zip(lens, size, steps, strict=True)
            ]
            for idxs in _ndrange(ranges):
                yield dict(zip(dims, idxs, strict=True))
            return
        raise NotImplementedError(
            f"SpatialRegularStride doesn't support {type(domain).__name__} domains."
        )

    def _broadcast(self, n: int) -> tuple[int, ...]:
        if isinstance(self.step, int):
            return tuple([self.step] * n)
        return tuple(int(s) for s in self.step)

    def get_config(self) -> dict[str, Any]:
        step = list(self.step) if not isinstance(self.step, int) else self.step
        return {"step": step}


@dataclass(eq=False)
class SpatialJitteredStride(SpatialSampler):
    """`SpatialRegularStride` + per-anchor uniform jitter (training augmentation).

    Args:
        step: As for `SpatialRegularStride`.
        jitter: SpatialMax jitter in step-units (0.0 = no jitter, 0.5 = ± half a step).
        seed: Optional integer seed for reproducible draws.
    """

    step: int | tuple[int, ...]
    jitter: float = 0.5
    seed: int | None = None

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        base = SpatialRegularStride(step=self.step)
        if _is_raster_domain(domain):
            sh, sw = base._broadcast(2)
            for r, c in base.anchors(domain, geometry):
                dr = int(rng.uniform(-self.jitter, self.jitter) * sh)
                dc = int(rng.uniform(-self.jitter, self.jitter) * sw)
                yield (max(0, r + dr), max(0, c + dc))
            return
        if isinstance(domain, GridDomain):
            dims = list(domain.coords)
            steps = base._broadcast(len(dims))
            for anchor in base.anchors(domain, geometry):
                out: dict[str, int] = {}
                for d, s in zip(dims, steps, strict=True):
                    dj = rng.uniform(-self.jitter, self.jitter) * s
                    out[d] = max(0, int(anchor[d] + dj))
                yield out
            return
        raise NotImplementedError(
            f"SpatialJitteredStride doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        step = list(self.step) if not isinstance(self.step, int) else self.step
        return {"step": step, "jitter": self.jitter, "seed": self.seed}


@dataclass(eq=False)
class SpatialRandom(SpatialSampler):
    """N uniformly-random anchors over the domain's placement space.

    Args:
        n_samples: Number of anchors to draw.
        seed: Optional integer seed for reproducible draws.
    """

    n_samples: int
    seed: int | None = None

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        if _is_raster_domain(domain):
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            rs = rng.integers(0, max(h - ph + 1, 1), size=self.n_samples)
            cs = rng.integers(0, max(w - pw + 1, 1), size=self.n_samples)
            for r, c in zip(rs, cs, strict=True):
                yield (int(r), int(c))
            return
        if isinstance(domain, GridDomain):
            dims = list(domain.coords)
            size = getattr(geometry, "size", tuple([1] * len(dims)))
            for _ in range(self.n_samples):
                yield {
                    d: int(rng.integers(0, max(len(domain.coords[d]) - int(p) + 1, 1)))
                    for d, p in zip(dims, size, strict=True)
                }
            return
        if isinstance(domain, PointDomain | VectorDomain):
            n = (
                len(domain.coords)
                if isinstance(domain, PointDomain)
                else len(domain.geometry)
            )
            idx = rng.integers(0, n, size=self.n_samples)
            for i in idx:
                yield int(i)
            return
        raise NotImplementedError(
            f"SpatialRandom doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return {"n_samples": self.n_samples, "seed": self.seed}


@dataclass(eq=False)
class SpatialPoissonDisk(SpatialSampler):
    """Well-spaced random anchors via Bridson's algorithm (raster + point).

    Anchors are returned in arbitrary order; no two anchors are closer
    than ``min_dist`` in pixel (raster) or coord (point) units.

    Args:
        min_dist: Minimum allowed separation between anchors.
        max_tries: Bridson parameter — attempts per active anchor.
        seed: Optional integer seed.
    """

    min_dist: float
    max_tries: int = 30
    seed: int | None = None

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        if _is_raster_domain(domain):
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            yield from _bridson_2d(
                (max(h - ph + 1, 1), max(w - pw + 1, 1)),
                self.min_dist,
                self.max_tries,
                rng,
            )
            return
        if isinstance(domain, PointDomain):
            yield from _bridson_subset(
                domain.coords, self.min_dist, self.max_tries, rng
            )
            return
        raise NotImplementedError(
            f"SpatialPoissonDisk doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "min_dist": self.min_dist,
            "max_tries": self.max_tries,
            "seed": self.seed,
        }


@dataclass(eq=False)
class SpatialExplicit(SpatialSampler):
    """Caller-supplied anchors — the universal escape hatch.

    Args:
        anchors_: Sequence of anchors. Type must match what the
            `SpatialGeometry` expects on the target domain.
    """

    anchors_: Iterable[Any] = field(default_factory=list)

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        yield from self.anchors_

    def get_config(self) -> dict[str, Any]:
        return {"n_anchors": len(list(self.anchors_))}


def _ndrange(ranges: list[range]) -> Iterator[tuple[int, ...]]:
    """Cartesian product over a list of ranges, yielded as tuples."""
    if not ranges:
        yield ()
        return
    head, *tail = ranges
    for h in head:
        for rest in _ndrange(tail):
            yield (h, *rest)


def _bridson_2d(
    shape: tuple[int, int],
    min_dist: float,
    k: int,
    rng: np.random.Generator,
) -> Iterator[tuple[int, int]]:
    """Bridson Poisson-disk sampling on a 2-D integer grid.

    Returns integer anchors ``(r, c)`` within ``shape``; pairwise distance
    is at least ``min_dist`` in Euclidean pixel units.
    """
    H, W = shape
    if H <= 0 or W <= 0:
        return
    cell = min_dist / np.sqrt(2)
    gh = int(np.ceil(H / cell))
    gw = int(np.ceil(W / cell))
    grid: np.ndarray = np.full((gh, gw), -1, dtype=int)
    samples: list[tuple[float, float]] = []
    active: list[int] = []

    def emit(p: tuple[float, float]) -> tuple[int, int]:
        samples.append(p)
        idx = len(samples) - 1
        grid[int(p[0] / cell), int(p[1] / cell)] = idx
        active.append(idx)
        return int(p[0]), int(p[1])

    yield emit((float(rng.uniform(0, H)), float(rng.uniform(0, W))))

    while active:
        ai = rng.integers(0, len(active))
        anchor = samples[active[ai]]
        found = False
        for _ in range(k):
            theta = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(min_dist, 2 * min_dist)
            cand = (anchor[0] + r * np.cos(theta), anchor[1] + r * np.sin(theta))
            if not (0 <= cand[0] < H and 0 <= cand[1] < W):
                continue
            gi, gj = int(cand[0] / cell), int(cand[1] / cell)
            ok = True
            for di in (-2, -1, 0, 1, 2):
                for dj in (-2, -1, 0, 1, 2):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < gh and 0 <= nj < gw and grid[ni, nj] >= 0:
                        other = samples[grid[ni, nj]]
                        d2 = (cand[0] - other[0]) ** 2 + (cand[1] - other[1]) ** 2
                        if d2 < min_dist**2:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                yield emit(cand)
                found = True
                break
        if not found:
            active.pop(ai)


def _bridson_subset(
    coords: np.ndarray,
    min_dist: float,
    k: int,
    rng: np.random.Generator,
) -> Iterator[int]:
    """Bridson-flavoured subset selection from an existing point cloud.

    Walks ``coords`` in random order, accepts each candidate if it's at
    least ``min_dist`` from every previously-accepted point.
    """
    from scipy.spatial import cKDTree

    n = len(coords)
    if n == 0:
        return
    order = rng.permutation(n)
    accepted: list[int] = []
    accepted_coords: list[np.ndarray] = []
    for i in order:
        cand = coords[i]
        if accepted_coords:
            tree = cKDTree(np.asarray(accepted_coords))
            d, _ = tree.query(cand, k=1)
            if d < min_dist:
                continue
        accepted.append(int(i))
        accepted_coords.append(cand)
        yield int(i)
