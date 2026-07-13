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
    indices) and `extent` (domain → bounds on anchor placement).
    """

    forbid_in_yaml: ClassVar[bool] = False

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} doesn't support {type(domain).__name__} domains."
        )

    def extent(self, domain: Any) -> Any:
        """Return the placement-space the SpatialSampler should iterate over.

        For raster: ``(height, width)`` tuple.
        For grid: ``{dim_name: length}`` dict.
        For point/vector: the number of features (``int``).
        """
        if _is_raster_domain(domain):
            return tuple(domain.shape[-2:])
        if isinstance(domain, GridDomain):
            return {d: len(c) for d, c in domain.coords.items()}
        if isinstance(domain, PointDomain):
            return len(domain.coords)
        if isinstance(domain, VectorDomain):
            return len(domain.geometry)
        raise NotImplementedError(
            f"extent() doesn't support {type(domain).__name__} domains."
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
            domain edge. ``"drop"`` (default — current behavior) clips
            the sampler so overflowing anchors are never placed.
            ``"pad"`` emits edge anchors and fills out-of-bounds pixels
            with the reader's nodata (or ``pad_value`` when set).
            ``"reflect"`` emits edge anchors and mirror-pads the
            out-of-bounds region from the in-domain interior — the
            spectrally correct choice for overlap-add stitching with
            tapered windows (no DC dip at the scene boundary).
            ``"shrink"`` clips the returned Window so the patch becomes
            smaller at the edge. ``"raise"`` emits edge anchors and
            raises at split time. See `BoundaryMode`.

            Only honored on raster domains in v0.x; GridDomain treats
            every mode as ``"drop"`` until an xarray-pad story lands.

            ``"pad"`` and ``"reflect"`` are guaranteed by the patcher
            itself — the overflowing window is clipped to the domain,
            read once, then padded up to the full geometry size. This is
            field-independent: it works identically for `RasterField`,
            `RioXarrayField`, and any other `Field`, with the chip's
            georeferencing (transform origin) shifted to stay exact.
        pad_value: Constant fill for ``boundary="pad"``. When ``None``
            (default) the out-of-bounds region is filled with the
            reader's nodata; set a float to force a specific constant.
            Ignored by every other boundary mode.
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

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        if _is_raster_domain(domain):
            from rasterio.windows import Window

            row_off, col_off = int(anchor[0]), int(anchor[1])
            ph, pw = int(self.size[-2]), int(self.size[-1])
            if self.boundary == "shrink":
                dh, dw = int(domain.shape[-2]), int(domain.shape[-1])
                ph = min(ph, max(dh - row_off, 0))
                pw = min(pw, max(dw - col_off, 0))
            return Window(col_off=col_off, row_off=row_off, width=pw, height=ph)
        if isinstance(domain, GridDomain):
            dims = list(domain.coords)
            return {
                d: slice(int(anchor[d]), int(anchor[d]) + int(sz))
                for d, sz in zip(dims, self.size, strict=False)
            }
        raise NotImplementedError(
            f"SpatialRectangular doesn't support {type(domain).__name__} domains."
        )

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialSphericalCap(SpatialGeometry):
    """Geodesic cap of radius ``radius_km`` — for lat/lon fields near the poles.

    Args:
        radius_km: Cap radius in kilometres, used as great-circle distance
            from the anchor. Earth radius is fixed at 6371 km.
    """

    radius_km: float

    _EARTH_RADIUS_KM: ClassVar[float] = 6371.0

    def neighborhood(self, domain: Any, anchor: tuple[float, float]) -> np.ndarray:
        if isinstance(domain, GridDomain):
            # GridDomain stores named axes; anchor is (lat, lon) by convention.
            lat_a, lon_a = float(anchor[0]), float(anchor[1])
            lat = np.asarray(domain.coords["lat"])
            lon = np.asarray(domain.coords["lon"])
            llat, llon = np.meshgrid(lat, lon, indexing="ij")
            d = _haversine_km(lat_a, lon_a, llat, llon)
            return np.argwhere(d <= self.radius_km)
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

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


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
        k: Number of neighbors to return per anchor.
        metric: ``"euclidean"`` (planar; uses the domain's kdtree) or
            ``"haversine"`` (great-circle; requires lat/lon coords).
    """

    k: int
    metric: str = "euclidean"

    def neighborhood(self, domain: Any, anchor: Any) -> np.ndarray:
        if isinstance(domain, PointDomain):
            anchor_xy = _to_xy(domain, anchor)
            if self.metric == "euclidean":
                _, idx = domain.kdtree.query(anchor_xy, k=self.k)
                return np.atleast_1d(idx).astype(int)
            if self.metric == "haversine":
                lats = domain.coords[:, 1]
                lons = domain.coords[:, 0]
                d = _haversine_km(anchor_xy[1], anchor_xy[0], lats, lons)
                return np.argsort(d)[: self.k]
            raise ValueError(f"unknown metric: {self.metric!r}")
        if isinstance(domain, VectorDomain):
            anchor_geom = _to_shapely_point(anchor)
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
            anchor_geom = _to_shapely_point(anchor)
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
    bounding `rasterio.windows.Window` of the polygon's footprint + the
    boolean mask of pixels strictly inside it. On a `VectorDomain`, it
    returns the indices of geometries that intersect.

    Args:
        polygons: Sequence (typically a ``geopandas.GeoSeries``) of
            polygons. The ``anchor`` passed to ``neighborhood`` indexes
            into this sequence.
    """

    polygons: Any

    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        poly = self.polygons.iloc[int(anchor)]
        if _is_raster_domain(domain):
            from rasterio import features
            from rasterio.windows import from_bounds

            window = from_bounds(*poly.bounds, transform=domain.transform)
            mask = features.geometry_mask(
                [poly],
                out_shape=(int(window.height), int(window.width)),
                transform=domain.transform
                * domain.transform.translation(window.col_off, window.row_off),
                invert=True,
            )
            return _MaskedWindow(window=window, mask=mask)
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
    """A bounding rasterio Window + an interior boolean mask.

    Returned by `SpatialPolygonIntersection.neighborhood` on `RasterDomain`. The
    `RasterField.select` will read the rectangular window; the mask is
    forwarded to the `Patch.weights` so downstream aggregation honours it.
    """

    window: Any
    mask: np.ndarray


def _to_xy(domain: PointDomain, anchor: Any) -> np.ndarray:
    """Coerce an anchor into an ``(x, y)`` pair on a `PointDomain`."""
    if isinstance(anchor, int | np.integer):
        return domain.coords[int(anchor)]
    return np.asarray(anchor, dtype=float)


def _to_shapely_point(anchor: Any) -> Any:
    """Coerce an anchor into a `shapely.Point`."""
    import shapely

    if hasattr(anchor, "x") and hasattr(anchor, "y"):
        return anchor
    return shapely.Point(*anchor)
