"""`SpatialGeometry` — shape + scale of the neighborhood the operator sees.

Each `SpatialGeometry` subclass knows how to translate an anchor into
backend-specific indices on each `Domain` it supports. The dispatch is
explicit ``isinstance`` rather than `functools.singledispatchmethod`
because the raster path matches a Protocol-like surface (``transform`` +
``shape`` + ``crs``) rather than a single concrete class — Protocol
nominal-typing is unreliable in `singledispatch` registries.

Five geometries:

- `SpatialRectangular`     — Raster + Grid
- `SpatialSphericalCap`    — Grid + Point
- `SpatialKNNGraph`        — Point + Vector
- `SpatialRadiusGraph`     — Point + Vector
- `SpatialPolygonIntersection` — Raster + Vector
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.domains import GridDomain, PointDomain, VectorDomain


BoundaryMode = Literal["drop", "pad", "shrink", "raise", "reflect"]
_VALID_BOUNDARY_MODES: tuple[str, ...] = (
    "drop",
    "pad",
    "shrink",
    "raise",
    "reflect",
)


def _is_raster_domain(domain: Any) -> bool:
    """A domain is "raster-shaped" if it carries `transform`, `shape`, `crs`."""
    return (
        hasattr(domain, "transform")
        and hasattr(domain, "shape")
        and hasattr(domain, "crs")
    )


class SpatialGeometry:
    """Base for spatial neighborhood definitions.

    Subclasses override `neighborhood` (anchor → backend-specific
    indices). Anchor placement is the sampler's job.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class SpatialRectangular(SpatialGeometry):
    """Axis-aligned box — the bread-and-butter raster / grid geometry.

    Args:
        size: For raster, ``(height, width)`` in pixels. For grid, one
            length per declared dim in the domain's coord order.
        boundary: How to treat anchors whose patch would overflow the
            domain edge. See `BoundaryMode` and ``docs/patcher/patching.md``.

            - ``"drop"`` (default): samplers only place anchors whose
              patch fits in-domain; the trailing residual is dropped, and
              a patch larger than the domain yields no anchors at all.
            - ``"pad"``: the sampler places the edge anchor (the first
              whose patch reaches the edge); out-of-bounds cells are
              filled with the reader's nodata, or ``pad_value``.
            - ``"reflect"``: as ``"pad"``, but the overflow is
              mirror-padded from the interior (numpy ``mode="reflect"``,
              repeated when the overflow exceeds the domain).
            - ``"shrink"``: as ``"pad"``, but the window is clipped to the
              domain on every side (a negative anchor included), so the
              patch is smaller at the edge; weights crop to match.
            - ``"raise"``: as ``"pad"``, but `SpatialPatcher.split`
              raises on the first overflowing window.

            Honoured on raster domains and on `GridDomain` slice dicts.
            ``"pad"`` / ``"reflect"`` are applied by the patcher itself —
            the overflowing window is clipped to the domain, read once,
            then padded up to the full geometry size with the chip's
            georeferencing kept exact — so they work for any `Field`.
            Merging is likewise boundary-agnostic: every dense
            aggregation crops chip data and weights to the in-domain part
            of the patch's indices.
        pad_value: Constant fill for ``boundary="pad"``. When ``None``
            (default) the out-of-bounds region is filled with the
            reader's nodata; set a number to force a specific constant.
            It must be representable in the field's dtype (checked at
            read time — ``-999`` into a ``uint16`` raster raises instead
            of wrapping). Ignored by every other boundary mode.
    """

    size: tuple[int, ...]
    boundary: BoundaryMode = "drop"
    pad_value: float | None = None

    def __post_init__(self) -> None:
        if self.boundary not in _VALID_BOUNDARY_MODES:
            raise ValueError(
                f"invalid boundary mode {self.boundary!r}; "
                f"expected one of {_VALID_BOUNDARY_MODES}"
            )
        if self.pad_value is not None and (
            isinstance(self.pad_value, bool)
            or not isinstance(self.pad_value, numbers.Real)
        ):
            raise TypeError(
                f"pad_value must be a real number or None, got {self.pad_value!r}"
            )

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        if _is_raster_domain(domain):
            from rasterio.windows import Window

            row_off, col_off = int(anchor[0]), int(anchor[1])
            ph, pw = int(self.size[-2]), int(self.size[-1])
            if self.boundary == "shrink":
                dh, dw = int(domain.shape[-2]), int(domain.shape[-1])
                row_off, ph = _shrink_axis(row_off, ph, dh)
                col_off, pw = _shrink_axis(col_off, pw, dw)
            return Window(col_off=col_off, row_off=row_off, width=pw, height=ph)
        if isinstance(domain, GridDomain):
            dims = list(domain.coords)
            if len(self.size) != len(dims):
                raise ValueError(
                    f"size must name every GridDomain dim: got size={tuple(self.size)} "
                    f"for dims {tuple(dims)}."
                )
            out = {}
            for d, sz in zip(dims, self.size, strict=True):
                start, length = int(anchor[d]), int(sz)
                if self.boundary == "shrink":
                    start, length = _shrink_axis(start, length, len(domain.coords[d]))
                out[d] = slice(start, start + length)
            return out
        raise NotImplementedError(
            f"SpatialRectangular doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialSphericalCap(SpatialGeometry):
    """Geodesic cap of radius ``radius_km`` — for lat/lon fields near the poles.

    On a `GridDomain` the latitude / longitude dims are found by name
    (``lat`` / ``latitude`` and ``lon`` / ``longitude``). The anchor is
    either the index dict a grid sampler yields (the cap is centred on
    the cell at those indices) or a ``(lat, lon)`` value pair. The
    neighborhood is the cap's bounding box as a ``{dim: slice}`` dict —
    every other dim kept whole — plus the boolean mask of cells inside
    the cap, which becomes the patch weights. On a `PointDomain` it is
    the indices of the points inside the cap.

    Args:
        radius_km: Cap radius in kilometres, used as great-circle distance
            from the anchor. Earth radius is fixed at 6371 km.
    """

    radius_km: float

    _EARTH_RADIUS_KM: ClassVar[float] = 6371.0

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        if isinstance(domain, GridDomain):
            return self._grid_neighborhood(domain, anchor)
        if isinstance(domain, PointDomain):
            # PointDomain coords are (x, y) = (lon, lat), matching the
            # GeoPandas / xvec adapter convention. The KNN/radius haversine
            # paths on PointDomain use anchor[0] = lon, anchor[1] = lat;
            # mirror that here so spherical-cap on the natural (lon, lat)
            # anchor doesn't quietly swap coordinates.
            lon_a, lat_a = float(anchor[0]), float(anchor[1])
            lat_pts = domain.coords[:, 1]
            lon_pts = domain.coords[:, 0]
            d = _haversine_km(lat_a, lon_a, lat_pts, lon_pts)
            return np.flatnonzero(d <= self.radius_km)
        raise NotImplementedError(
            f"SpatialSphericalCap doesn't support {type(domain).__name__} domains."
        )

    def _grid_neighborhood(self, domain: GridDomain, anchor: Any) -> _MaskedWindow:
        lat_dim = _find_dim(domain, ("lat", "latitude"))
        lon_dim = _find_dim(domain, ("lon", "longitude"))
        lat = np.asarray(domain.coords[lat_dim], dtype=float)
        lon = np.asarray(domain.coords[lon_dim], dtype=float)
        if isinstance(anchor, dict):
            lat_a, lon_a = lat[int(anchor[lat_dim])], lon[int(anchor[lon_dim])]
        else:
            lat_a, lon_a = float(anchor[0]), float(anchor[1])
        llat, llon = np.meshgrid(lat, lon, indexing="ij")
        inside = _haversine_km(lat_a, lon_a, llat, llon) <= self.radius_km
        rows, cols = (
            np.flatnonzero(inside.any(axis=1)),
            np.flatnonzero(inside.any(axis=0)),
        )
        if rows.size == 0:
            raise ValueError(
                f"SpatialSphericalCap of {self.radius_km} km around "
                f"({lat_a}, {lon_a}) contains no grid cell."
            )
        box = {
            lat_dim: slice(int(rows[0]), int(rows[-1]) + 1),
            lon_dim: slice(int(cols[0]), int(cols[-1]) + 1),
        }
        mask_2d = inside[box[lat_dim], box[lon_dim]]
        indexer = {
            d: box.get(d, slice(0, len(domain.coords[d]))) for d in domain.coords
        }
        # Lay the (lat, lon) mask out in the domain's dim order, broadcast
        # over every other dim, so it matches the selected chip's shape.
        spatial = [d for d in domain.coords if d in box]
        if spatial != [lat_dim, lon_dim]:
            mask_2d = mask_2d.T
        shape = [s.stop - s.start for s in indexer.values()]
        expand = [1 if d not in box else n for d, n in zip(indexer, shape, strict=True)]
        mask = np.broadcast_to(mask_2d.reshape(expand), shape)
        return _MaskedWindow(window=indexer, mask=mask)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _find_dim(domain: GridDomain, names: tuple[str, ...]) -> str:
    """The first of ``names`` that is a dim of ``domain``."""
    for name in names:
        if name in domain.coords:
            return name
    raise ValueError(
        f"SpatialSphericalCap needs one of the dims {names} on the GridDomain; "
        f"got {tuple(domain.coords)}."
    )


def _shrink_axis(start: int, size: int, length: int) -> tuple[int, int]:
    """Clip ``[start, start + size)`` to ``[0, length)`` → ``(start, size)``."""
    lo = min(max(start, 0), length)
    hi = min(max(start + size, lo), length)
    return lo, hi - lo


def _haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km on a unit Earth (R = 6371)."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2.0 * SpatialSphericalCap._EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@dataclass(eq=False)
class SpatialKNNGraph(SpatialGeometry):
    """Fixed-k nearest-neighbor neighborhood.

    Args:
        k: Number of neighbors to return per anchor (``>= 1``); capped at
            the number of features, so ``k > N`` returns all ``N``.
        metric: ``"euclidean"`` (planar; uses the domain's kdtree) or
            ``"haversine"`` (great-circle; requires lat/lon coords).

    An integer anchor (e.g. from `SpatialRandom`) is the index of a point,
    or of a vector feature whose centroid is used.
    """

    k: int
    metric: str = "euclidean"

    def __post_init__(self) -> None:
        if int(self.k) != self.k or self.k < 1:
            raise ValueError(f"k must be an integer >= 1, got {self.k!r}")

    def neighborhood(self, domain: Any, anchor: Any) -> np.ndarray:
        if isinstance(domain, PointDomain):
            anchor_xy = _to_xy(domain, anchor)
            if self.metric == "euclidean":
                # scipy pads k > N with the sentinel index N; cap k instead.
                k = min(int(self.k), len(domain.coords))
                _, idx = domain.kdtree.query(anchor_xy, k=k)
                return np.atleast_1d(idx).astype(int)
            if self.metric == "haversine":
                lats = domain.coords[:, 1]
                lons = domain.coords[:, 0]
                d = _haversine_km(anchor_xy[1], anchor_xy[0], lats, lons)
                return np.argsort(d)[: self.k]
            raise ValueError(f"unknown metric: {self.metric!r}")
        if isinstance(domain, VectorDomain):
            anchor_geom = _to_shapely_point(anchor, domain)
            centroids = domain.geometry.centroid
            d = centroids.distance(anchor_geom).values
            return np.argsort(d)[: self.k]
        raise NotImplementedError(
            f"SpatialKNNGraph doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialRadiusGraph(SpatialGeometry):
    """All neighbors within a fixed radius — variable patch size.

    Args:
        radius: Radius in the domain's coordinate units.
        metric: ``"euclidean"`` for planar coords, ``"haversine"`` for
            lat/lon (radius then interpreted as km).
    """

    radius: float
    metric: str = "euclidean"

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        if isinstance(domain, PointDomain):
            anchor_xy = _to_xy(domain, anchor)
            if self.metric == "euclidean":
                idx = domain.kdtree.query_ball_point(anchor_xy, r=self.radius)
                return np.asarray(idx, dtype=int)
            if self.metric == "haversine":
                lats = domain.coords[:, 1]
                lons = domain.coords[:, 0]
                d = _haversine_km(anchor_xy[1], anchor_xy[0], lats, lons)
                return np.flatnonzero(d <= self.radius)
            raise ValueError(f"unknown metric: {self.metric!r}")
        if isinstance(domain, VectorDomain):
            anchor_geom = _to_shapely_point(anchor, domain)
            buf = anchor_geom.buffer(self.radius)
            tree = domain.sindex
            hits = tree.query(buf, predicate="intersects")
            return np.asarray(hits, dtype=int)
        raise NotImplementedError(
            f"SpatialRadiusGraph doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialPolygonIntersection(SpatialGeometry):
    """Patch = pixels (or features) lying inside a given polygon.

    On a `RasterDomain`, ``neighborhood`` returns a ``MaskedWindow``: the
    bounding `rasterio.windows.Window` of the polygon's footprint, rounded
    outward to whole pixels and clipped to the domain, + the boolean mask
    of the window's pixels whose centre lies inside the polygon. A polygon
    that does not overlap the raster at all raises ``ValueError``. On a
    `VectorDomain`, it returns the indices of geometries that intersect.

    Args:
        polygons: Sequence (typically a ``geopandas.GeoSeries``) of
            polygons. The ``anchor`` passed to ``neighborhood`` indexes
            into this sequence.
    """

    polygons: Any

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        poly = self.polygons.iloc[int(anchor)]
        if _is_raster_domain(domain):
            return _polygon_masked_window(domain, poly, int(anchor))
        if isinstance(domain, VectorDomain):
            hits = domain.sindex.query(poly, predicate="intersects")
            return np.asarray(hits, dtype=int)
        raise NotImplementedError(
            "SpatialPolygonIntersection doesn't support "
            f"{type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return {"n_polygons": len(self.polygons)}


@dataclass(eq=False)
class _MaskedWindow:
    """A bounding window + an interior boolean mask.

    Returned by `SpatialPolygonIntersection.neighborhood` on `RasterDomain`
    (a rasterio `Window`) and by `SpatialSphericalCap.neighborhood` on a
    `GridDomain` (a ``{dim: slice}`` dict). ``Field.select`` reads the
    rectangular window; the mask is forwarded to the `Patch.weights` so
    downstream aggregation honours it.
    """

    window: Any
    mask: np.ndarray


def _polygon_masked_window(domain: Any, poly: Any, anchor: int) -> _MaskedWindow:
    """Pixel-aligned bounding window + interior mask of ``poly`` on a raster.

    ``from_bounds`` yields a fractional window for any polygon whose bounds
    are not on pixel edges, which raster readers cannot slice. The window is
    therefore rounded *outward* to whole pixels and clipped to the domain
    (pixels outside the raster do not exist, and the chip must fit the merge
    accumulator). The mask is rasterised on that exact window's grid, so
    ``mask[i, j]`` is True iff the centre of raster pixel
    ``(row_off + i, col_off + j)`` lies inside ``poly``.
    """
    from georeader.window_utils import round_outer_window
    from rasterio import features
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds, transform as window_transform

    height, width = (int(s) for s in domain.shape[-2:])
    rounded = round_outer_window(from_bounds(*poly.bounds, transform=domain.transform))
    try:
        clipped = rounded.intersection(
            Window(col_off=0, row_off=0, width=width, height=height)
        )
    except WindowError:
        clipped = None
    if clipped is None or clipped.width <= 0 or clipped.height <= 0:
        raise ValueError(
            f"SpatialPolygonIntersection: polygon {anchor} (bounds {poly.bounds}) "
            "does not overlap the raster domain."
        )
    window = Window(
        col_off=int(clipped.col_off),
        row_off=int(clipped.row_off),
        width=int(clipped.width),
        height=int(clipped.height),
    )
    mask = features.geometry_mask(
        [poly],
        out_shape=(window.height, window.width),
        transform=window_transform(window, domain.transform),
        invert=True,
    )
    return _MaskedWindow(window=window, mask=mask)


def _to_xy(domain: PointDomain, anchor: Any) -> np.ndarray:
    """Coerce an anchor into an ``(x, y)`` pair on a `PointDomain`."""
    if isinstance(anchor, int | np.integer):
        return domain.coords[int(anchor)]
    return np.asarray(anchor, dtype=float)


def _to_shapely_point(anchor: Any, domain: VectorDomain) -> Any:
    """Coerce an anchor into a `shapely.Point`.

    An integer anchor (what `SpatialRandom` yields on a `VectorDomain`) is
    a feature index and maps to that feature's centroid.
    """
    import shapely

    if isinstance(anchor, int | np.integer):
        return domain.geometry.iloc[int(anchor)].centroid
    if hasattr(anchor, "x") and hasattr(anchor, "y"):
        return anchor
    return shapely.Point(*anchor)
