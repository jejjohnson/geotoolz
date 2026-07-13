"""`SpatialSampler` — where in the field to place anchors.

A `SpatialSampler` yields anchors; the `SpatialGeometry` then turns each anchor
into backend-specific indices. Five samplers cover the common cases:

- `SpatialRegularStride` — the canonical lattice (sliding-window inference).
- `SpatialJitteredStride` — regular grid with per-anchor uniform jitter.
- `SpatialRandom` — N uniformly-random anchors (training-time augmentation).
- `SpatialPoissonDisk` — well-spaced random anchors via Bridson's algorithm.
- `SpatialExplicit` — caller-supplied anchors (event-triggered, station list, …).
- `SpatialExplicitCoords` — caller-supplied world coordinates, optionally in a
  foreign CRS, with a chip centred on each (event / plume catalogues, …).
- `SpatialAlongTrack` — anchors along an ordered track, optionally resampled
  to a fixed along-track spacing (altimetry ground tracks, flight lines, …).

Coordinate-consuming samplers (`SpatialAlongTrack`, `SpatialExplicitCoords`)
accept a ``crs=`` for anchors expressed in a CRS other than the domain's —
the coordinates are reprojected to the domain CRS before the pixel mapping.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Literal

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.domains import GridDomain, PointDomain, VectorDomain
from geopatcher._src.exceptions import IncompleteScanConfiguration
from geopatcher._src.spatial.geometry import (
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
        check_full_scan: If ``True``, raise `IncompleteScanConfiguration`
            at anchor time when ``(domain_len - patch_size) % step != 0``
            on any axis — i.e. when the chosen ``(size, step)`` would
            silently drop a partial tile at the trailing edge. Off by
            default to preserve the existing "drop the partial" behaviour;
            opt in for xrpatcher-style strict-tiling workloads. The
            temporal counterpart is `divide_evenly` in `time/stencils.py`.
    """

    step: int | tuple[int, ...]
    check_full_scan: bool = False

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        if self.check_full_scan:
            self._assert_full_scan(domain, geometry)
        boundary = getattr(geometry, "boundary", "drop")
        if _is_raster_domain(domain):
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            sh, sw = self._broadcast(2)
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            # "drop": stop where the full patch still fits in-domain.
            # "pad"/"shrink"/"raise": extend up to the last anchor that
            # still falls inside the domain (overflow is the geometry /
            # patcher's responsibility from here on).
            if boundary == "drop":
                stop_h, stop_w = max(h - ph + 1, 1), max(w - pw + 1, 1)
            else:
                stop_h, stop_w = h, w
            for r in range(0, stop_h, sh):
                for c in range(0, stop_w, sw):
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

    def _assert_full_scan(self, domain: Any, geometry: SpatialGeometry) -> None:
        """Raise `IncompleteScanConfiguration` if any axis would drop a tile."""
        if _is_raster_domain(domain):
            lens = (int(domain.shape[-2]), int(domain.shape[-1]))
            sizes = tuple(int(s) for s in getattr(geometry, "size", (1, 1)))
            steps = self._broadcast(2)
            axes = ("row", "col")
        elif isinstance(domain, GridDomain):
            dims = list(domain.coords)
            lens = tuple(len(domain.coords[d]) for d in dims)
            sizes = tuple(
                int(s) for s in getattr(geometry, "size", tuple([1] * len(dims)))
            )
            steps = self._broadcast(len(dims))
            axes = tuple(dims)
        else:
            # Domains without a dense axis structure (Point/Vector) — the
            # check doesn't apply; defer to the per-domain anchor logic.
            return
        for axis, length, size, step in zip(axes, lens, sizes, steps, strict=True):
            remainder = (length - size) % step
            if remainder != 0:
                raise IncompleteScanConfiguration(
                    f"Incomplete scan on axis {axis!r}: "
                    f"(length - size) % step = ({length} - {size}) % {step} "
                    f"= {remainder} ≠ 0. Adjust size/step or set "
                    "check_full_scan=False to allow the trailing partial tile."
                )

    def _broadcast(self, n: int) -> tuple[int, ...]:
        if isinstance(self.step, int):
            return tuple([self.step] * n)
        return tuple(int(s) for s in self.step)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialJitteredStride(SpatialSampler):
    """`SpatialRegularStride` + per-anchor uniform jitter (training augmentation).

    Args:
        step: As for `SpatialRegularStride`.
        jitter: SpatialMax jitter in step-units (0.0 = no jitter, 0.5 = ± half a step).
        seed: Integer seed for reproducible draws. When set, two
            samplers with the same configuration return bit-identical
            anchors across calls and across instances (the contract
            tested in ``tests/test_determinism.py`` for issue #18).
            ``None`` (the default) re-seeds from OS entropy on every
            call — anchors will differ between calls.
    """

    step: int | tuple[int, ...]
    jitter: float = 0.5
    seed: int | None = None

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        base = SpatialRegularStride(step=self.step)
        boundary = getattr(geometry, "boundary", "drop")
        if _is_raster_domain(domain):
            sh, sw = base._broadcast(2)
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            # Clamp jittered anchors so the patch still fits in the
            # field — except when the geometry's boundary policy invites
            # overflow (then the patcher handles the edge).
            if boundary == "drop":
                rmax = max(h - ph, 0)
                cmax = max(w - pw, 0)
            else:
                rmax = h - 1
                cmax = w - 1
            for r, c in base.anchors(domain, geometry):
                dr = int(rng.uniform(-self.jitter, self.jitter) * sh)
                dc = int(rng.uniform(-self.jitter, self.jitter) * sw)
                yield (min(rmax, max(0, r + dr)), min(cmax, max(0, c + dc)))
            return
        if isinstance(domain, GridDomain):
            dims = list(domain.coords)
            steps = base._broadcast(len(dims))
            size = getattr(geometry, "size", tuple([1] * len(dims)))
            lens = [len(domain.coords[d]) for d in dims]
            maxes = {
                d: max(L - int(p), 0) for d, L, p in zip(dims, lens, size, strict=True)
            }
            for anchor in base.anchors(domain, geometry):
                out: dict[str, int] = {}
                for d, s in zip(dims, steps, strict=True):
                    dj = rng.uniform(-self.jitter, self.jitter) * s
                    out[d] = min(maxes[d], max(0, int(anchor[d] + dj)))
                yield out
            return
        raise NotImplementedError(
            f"SpatialJitteredStride doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialRandom(SpatialSampler):
    """N uniformly-random anchors over the domain's placement space.

    Args:
        n_samples: Number of anchors to draw.
        seed: Integer seed for reproducible draws. When set, two
            samplers with the same configuration return bit-identical
            anchors across calls and across instances (the contract
            tested in ``tests/test_determinism.py`` for issue #18).
            ``None`` (the default) re-seeds from OS entropy on every
            call — anchors will differ between calls.
    """

    n_samples: int
    seed: int | None = None

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        boundary = getattr(geometry, "boundary", "drop")
        if _is_raster_domain(domain):
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            # "drop": draw only from the fit-only range so the patch is
            # fully in-domain. Non-"drop": draw across the whole domain
            # — the patcher / geometry handles edge overflow.
            if boundary == "drop":
                rhi, chi = max(h - ph + 1, 1), max(w - pw + 1, 1)
            else:
                rhi, chi = h, w
            rs = rng.integers(0, rhi, size=self.n_samples)
            cs = rng.integers(0, chi, size=self.n_samples)
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
        return config_from_fields(self)


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
        boundary = getattr(geometry, "boundary", "drop")
        if _is_raster_domain(domain):
            h, w = int(domain.shape[-2]), int(domain.shape[-1])
            size = getattr(geometry, "size", (1, 1))
            ph, pw = int(size[-2]), int(size[-1])
            # See SpatialRandom — "drop" restricts to the fit-only
            # region, non-"drop" lets Bridson cover the whole domain.
            if boundary == "drop":
                region = (max(h - ph + 1, 1), max(w - pw + 1, 1))
            else:
                region = (h, w)
            yield from _bridson_2d(region, self.min_dist, self.max_tries, rng)
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
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialExplicit(SpatialSampler):
    """Caller-supplied anchors — the universal escape hatch.

    Args:
        anchors_: Sequence of anchors. Type must match what the
            `SpatialGeometry` expects on the target domain. Accepts any
            iterable at construction time; materialised to a list in
            ``__post_init__`` so ``anchors()`` and ``get_config()`` can
            both walk it independently (the previous Iterable typing
            meant ``get_config()`` consumed one-shot iterators and
            silently left ``anchors()`` empty afterwards).
    """

    anchors_: Iterable[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.anchors_, list):
            self.anchors_ = list(self.anchors_)

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        yield from self.anchors_

    def get_config(self) -> dict[str, Any]:
        return {"n_anchors": len(self.anchors_)}


@dataclass(eq=False)
class SpatialAlongTrack(SpatialSampler):
    """Anchors along an ordered track, optionally resampled to fixed spacing.

    The track is an ordered polyline of ``(x, y)`` coordinates in the
    domain's CRS (an altimetry ground track, a flight line, a ship
    transect, …). With ``spacing`` set, the track is resampled to points
    at a fixed along-track distance (linear interpolation along the
    cumulative Euclidean arc length, in coordinate units); with
    ``spacing=None`` the original vertices are used as-is.

    On a raster domain, each track point maps through the inverse affine
    to a pixel and the yielded anchor is the upper-left corner that
    **centres** the geometry's patch on that pixel; track points falling
    outside the raster are skipped, and (under the default ``"drop"``
    boundary) anchors are clamped so the patch stays in-domain. On a
    `PointDomain`, the ``(x, y)`` coordinates themselves are yielded —
    ready for `SpatialKNNGraph` / `SpatialRadiusGraph` neighborhoods.

    Args:
        track: Ordered ``(N, 2)`` array of ``(x, y)`` coordinates. Also
            accepts a `geopandas.GeoDataFrame` / `GeoSeries` of points
            or a `shapely.LineString` — anything exposing ``.geometry``,
            ``.x`` / ``.y``, or ``.coords``.
        spacing: Along-track resampling distance in coordinate units, or
            ``None`` to anchor at the original vertices. Requires at
            least two distinct vertices when set.
        crs: CRS the ``track`` coordinates are expressed in. When set and
            different from ``domain.crs``, the track is reprojected to the
            domain CRS before anchoring; the ``spacing`` resample then runs
            in domain-CRS units. ``None`` (default) assumes the track is
            already in the domain's CRS (bit-identical to the old path).
        polar_guard: Behaviour when a *geographic* ``crs`` places points
            beyond ±80° latitude or a track straddles the ±180°
            antimeridian — reprojection is unreliable there. ``"warn"``
            (default) emits a `RuntimeWarning`, ``"raise"`` errors,
            ``"ignore"`` is silent.
    """

    track: Any
    spacing: float | None = None
    crs: Any | None = None
    polar_guard: Literal["warn", "raise", "ignore"] = "warn"

    def __post_init__(self) -> None:
        self.track = _track_coords(self.track)
        if self.spacing is not None and self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        _validate_polar_guard(self.polar_guard)

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        track = self.track
        if self.crs is not None:
            track = _to_domain_crs(
                track, self.crs, _domain_crs(domain), self.polar_guard
            )
        points = self._resampled(track)
        if _is_raster_domain(domain):
            yield from _raster_center_anchors(points, domain, geometry)
            return
        if isinstance(domain, PointDomain):
            for x, y in points:
                yield (float(x), float(y))
            return
        raise NotImplementedError(
            f"SpatialAlongTrack doesn't support {type(domain).__name__} domains."
        )

    def _resampled(self, track: np.ndarray | None = None) -> np.ndarray:
        """Return track vertices, resampled to `spacing` when it is set."""
        pts = np.asarray(self.track if track is None else track, dtype=float)
        if self.spacing is None:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = seg > 0
        # Collapse zero-length segments so the arc-length axis is
        # strictly increasing for np.interp.
        pts = np.concatenate([pts[:1], pts[1:][keep]])
        if len(pts) < 2:
            raise ValueError(
                "SpatialAlongTrack with spacing needs at least two distinct "
                "track vertices."
            )
        dist = np.concatenate([[0.0], np.cumsum(seg[keep])])
        total = float(dist[-1])
        n_steps = int(np.floor(total / self.spacing + 1e-9))
        s = np.arange(n_steps + 1, dtype=float) * self.spacing
        return np.column_stack(
            [np.interp(s, dist, pts[:, 0]), np.interp(s, dist, pts[:, 1])]
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "n_points": len(self.track),
            "spacing": self.spacing,
            "crs": None if self.crs is None else str(self.crs),
            "polar_guard": self.polar_guard,
        }


@dataclass(eq=False)
class SpatialExplicitCoords(SpatialSampler):
    """Caller-supplied world coordinates, one centred chip per coordinate.

    The coordinate analogue of `SpatialExplicit` (which passes
    backend-native anchors through untouched): each ``(x, y)`` is a world
    coordinate, optionally in a foreign ``crs``, and the yielded anchor is
    the upper-left corner that **centres** the geometry's patch on the
    pixel that coordinate lands in — the same geo→pixel→centred-UL contract
    as `SpatialAlongTrack`. Coordinates outside the raster are skipped.
    On a `PointDomain`, the (reprojected) ``(x, y)`` is yielded directly.

    Args:
        coords: Ordered ``(N, 2)`` array of ``(x, y)`` world coordinates,
            or any track-like object (`GeoDataFrame` / `GeoSeries` /
            `LineString`) `SpatialAlongTrack` accepts.
        crs: CRS the coordinates are in. When set and different from
            ``domain.crs`` they are reprojected to the domain CRS before
            the pixel mapping. ``None`` (default) assumes the domain's CRS.
        polar_guard: Same geographic-edge guard as `SpatialAlongTrack`.

    A single centred read without the patcher is also available via
    `georeader.read.read_from_center_coords(reader, xy, shape,
    crs_center_coords=...)`.
    """

    coords: Any
    crs: Any | None = None
    polar_guard: Literal["warn", "raise", "ignore"] = "warn"

    def __post_init__(self) -> None:
        self.coords = _track_coords(self.coords)
        _validate_polar_guard(self.polar_guard)

    def anchors(self, domain: Any, geometry: SpatialGeometry) -> Iterator[Any]:
        points = self.coords
        if self.crs is not None:
            points = _to_domain_crs(
                points, self.crs, _domain_crs(domain), self.polar_guard
            )
        if _is_raster_domain(domain):
            yield from _raster_center_anchors(points, domain, geometry)
            return
        if isinstance(domain, PointDomain):
            for x, y in points:
                yield (float(x), float(y))
            return
        raise NotImplementedError(
            f"SpatialExplicitCoords doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "n_coords": len(self.coords),
            "crs": None if self.crs is None else str(self.crs),
            "polar_guard": self.polar_guard,
        }


def _raster_center_anchors(
    points: Any, domain: Any, geometry: SpatialGeometry
) -> Iterator[tuple[int, int]]:
    """Map world coords to centred upper-left anchors on a raster domain.

    Shared by `SpatialAlongTrack` and `SpatialExplicitCoords`: each point
    goes through the inverse affine to a pixel, and the yielded anchor is
    the UL corner that centres the geometry's patch on it. Points outside
    the raster are skipped. Only the default ``"drop"`` boundary clamps
    anchors to keep the patch fully in-domain; the other modes preserve
    the raw (possibly negative / overflowing) anchor so ``"pad"`` reads
    boundless context and ``"raise"`` can detect the overflow instead of
    silently shifting the patch inward.
    """
    h, w = int(domain.shape[-2]), int(domain.shape[-1])
    size = getattr(geometry, "size", (1, 1))
    ph, pw = int(size[-2]), int(size[-1])
    boundary = getattr(geometry, "boundary", "drop")
    inv = ~domain.transform
    for x, y in points:
        col_f, row_f = inv * (float(x), float(y))
        r, c = int(np.floor(row_f)), int(np.floor(col_f))
        if not (0 <= r < h and 0 <= c < w):
            continue
        ar, ac = r - ph // 2, c - pw // 2
        if boundary == "drop":
            ar = min(max(h - ph, 0), max(0, ar))
            ac = min(max(w - pw, 0), max(0, ac))
        yield (ar, ac)


def _validate_polar_guard(policy: str) -> None:
    if policy not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"invalid polar_guard {policy!r}; expected 'warn', 'raise', or 'ignore'."
        )


def _domain_crs(domain: Any) -> Any:
    crs = getattr(domain, "crs", None)
    if crs is None:
        raise ValueError(
            "cannot reproject anchors: the domain exposes no CRS. Provide "
            "coordinates already in the domain's grid and leave crs=None."
        )
    return crs


@lru_cache(maxsize=128)
def _transformer(src: str, dst: str) -> Any:
    import pyproj

    return pyproj.Transformer.from_crs(src, dst, always_xy=True)


def _to_domain_crs(
    coords: Any, src_crs: Any, dst_crs: Any, polar_guard: str
) -> np.ndarray:
    """Reproject ``(N, 2)`` xy from ``src_crs`` to ``dst_crs``.

    A no-op (returns the coordinates unchanged) when the two CRSs compare
    equal, so ``crs`` equal to the domain CRS is bit-identical to
    ``crs=None``.
    """
    coords = np.asarray(coords, dtype=float)
    from georeader import compare_crs

    if compare_crs(str(src_crs), str(dst_crs)):
        return coords
    _polar_dateline_check(coords, src_crs, polar_guard)
    transformer = _transformer(str(src_crs), str(dst_crs))
    xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])


def _polar_dateline_check(coords: np.ndarray, src_crs: Any, policy: str) -> None:
    """Warn / raise when a geographic source CRS hits unreliable reprojection.

    Only meaningful for a geographic (lon/lat) source CRS: near the poles
    (``|lat| > 80``) and across the ±180° antimeridian the planar
    transform is untrustworthy.
    """
    if policy == "ignore":
        return
    import pyproj

    crs = pyproj.CRS.from_user_input(src_crs)
    if not crs.is_geographic:
        return
    lon, lat = coords[:, 0], coords[:, 1]
    problems = []
    if lat.size and np.any(np.abs(lat) > 80.0):
        problems.append("a latitude beyond ±80°")
    if lon.size > 1 and np.any(np.abs(np.diff(lon)) > 180.0):
        problems.append("a step across the ±180° antimeridian")
    if not problems:
        return
    message = (
        "reprojecting geographic coordinates with "
        + " and ".join(problems)
        + " is unreliable; set polar_guard='ignore' to silence or "
        "'raise' to fail."
    )
    if policy == "raise":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _track_coords(track: Any) -> np.ndarray:
    """Coerce a track-like object into an ``(N, 2)`` float coordinate array."""
    if hasattr(track, "geometry"):  # GeoDataFrame
        track = track.geometry
    if hasattr(track, "x") and hasattr(track, "y"):  # GeoSeries of points
        coords = np.column_stack([np.asarray(track.x), np.asarray(track.y)])
    elif hasattr(track, "coords"):  # shapely LineString
        coords = np.asarray(track.coords, dtype=float)[:, :2]
    else:
        coords = np.asarray(track, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError(
            f"track must be an ordered (N, 2) coordinate array, "
            f"got shape {coords.shape}."
        )
    return coords


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
