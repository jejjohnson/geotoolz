"""`InMemoryGeoCatalog` — the in-RAM catalog backend.

Wraps a ``geopandas.GeoDataFrame`` whose row index is a
``pd.IntervalIndex`` over the time axis and whose ``geometry`` column
carries each file's footprint in a uniform target CRS. Queries leverage
geopandas's R-tree (built lazily on first access via ``gdf.sindex``)
and pandas's IntervalIndex for O(log n + k) lookups in both axes.

Good for catalogs up to ~10⁵ rows. Beyond that, switch to
`DuckDBGeoCatalog` (same Protocol, different store).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Any, Literal, get_args

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely

from geocatalog._src._timeutil import (
    naive_utc_datetimes,
    naive_utc_interval_index,
    to_naive_utc,
)
from geocatalog._src.base import CatalogRow
from geocatalog._src.geoslice import GeoSlice


_BACKEND_T = Literal["raster", "xarray", "vector"]
_INTERSECT_ENGINE_T = Literal["sjoin", "overlay"]
_GEOMETRY_TYPE_FAMILY = {
    "Point": "Point",
    "MultiPoint": "Point",
    "LineString": "LineString",
    "LinearRing": "LineString",
    "MultiLineString": "LineString",
    "Polygon": "Polygon",
    "MultiPolygon": "Polygon",
}


class InMemoryGeoCatalog:
    """A `geopandas.GeoDataFrame`-backed catalog.

    Args:
        gdf: The catalog rows. Required columns: ``geometry`` (footprint
            polygons in ``crs``) and either a row-level ``pd.IntervalIndex``
            (preferred — set the index via ``gdf.set_index(IntervalIndex)``)
            or ``start_time`` + ``end_time`` columns that this constructor
            will promote to the index. The ``geometry`` column's CRS is
            authoritative; ``gdf.crs`` must be set.
        backend: ``"raster"`` / ``"xarray"`` / ``"vector"``. Drives the
            dispatching choice in `geocatalog.load_*`.
    """

    backend: _BACKEND_T

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        *,
        backend: _BACKEND_T,
    ) -> None:
        if gdf.crs is None:
            raise ValueError("InMemoryGeoCatalog requires gdf.crs to be set; got None.")
        # Stored time contract: naive UTC (see `_timeutil`). Convert any
        # tz-aware / mixed time columns before they become the index so
        # naive and aware inputs never meet in a comparison.
        time_cols = {
            c: naive_utc_datetimes(gdf[c])
            for c in ("start_time", "end_time")
            if c in gdf.columns
        }
        if any(time_cols[c] is not gdf[c] for c in time_cols):
            gdf = gdf.assign(**time_cols)
        if not isinstance(gdf.index, pd.IntervalIndex):
            if "start_time" in gdf.columns and "end_time" in gdf.columns:
                idx = pd.IntervalIndex.from_arrays(
                    gdf["start_time"], gdf["end_time"], closed="both", name="datetime"
                )
                gdf = gdf.set_index(idx)
            else:
                raise ValueError(
                    "InMemoryGeoCatalog needs either an IntervalIndex row "
                    "index or 'start_time'+'end_time' columns; got "
                    f"index={type(gdf.index).__name__}, "
                    f"columns={list(gdf.columns)}."
                )
        normalized_index = naive_utc_interval_index(gdf.index)
        if normalized_index is not gdf.index:
            gdf = gdf.set_axis(normalized_index, axis=0)
        if gdf.index.closed != "both":
            # query() uses IntervalIndex.overlaps, whose endpoint semantics
            # follow ``closed``; GeoSlice intervals are always closed='both'.
            raise ValueError(
                "InMemoryGeoCatalog requires a closed='both' IntervalIndex; "
                f"got closed={gdf.index.closed!r}."
            )
        if backend not in get_args(_BACKEND_T):
            raise ValueError(
                f"InMemoryGeoCatalog backend must be one of "
                f"{list(get_args(_BACKEND_T))}; got {backend!r}."
            )
        self.gdf = gdf
        self.backend = backend

    @property
    def crs(self) -> pyproj.CRS:
        """The catalog CRS (``gdf.crs``) as a ``pyproj.CRS``."""
        return pyproj.CRS.from_user_input(self.gdf.crs)

    @cached_property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Union bbox over all rows.

        Returns:
            ``(xmin, ymin, xmax, ymax)`` in catalog-CRS units, or four
            NaNs if the catalog is empty.
        """
        if len(self.gdf) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return tuple(self.gdf.total_bounds.tolist())  # type: ignore[return-value]

    @cached_property
    def temporal_extent(self) -> pd.Interval | None:
        """Tightest interval spanning every row's time window.

        Returns:
            ``pd.Interval(min(start_times), max(end_times),
            closed='both')``, or ``None`` for an empty catalog
            (``pd.Interval`` cannot hold ``NaT`` endpoints).
        """
        if len(self.gdf) == 0:
            return None
        return pd.Interval(
            self.gdf.index.left.min(),
            self.gdf.index.right.max(),
            closed="both",
        )

    def __len__(self) -> int:
        """Number of rows in the catalog."""
        return len(self.gdf)

    def __repr__(self) -> str:
        return (
            f"InMemoryGeoCatalog(backend={self.backend!r}, "
            f"len={len(self)}, crs={self.gdf.crs!r})"
        )

    def query(
        self,
        slice_: GeoSlice | None = None,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        crs: Any | None = None,
        time: tuple[Any, Any] | pd.Interval | None = None,
    ) -> InMemoryGeoCatalog:
        """Spatial + temporal filter on the catalog.

        Either pass a `GeoSlice` (carries bounds/time/crs together) or
        pass the parts. Bounds in a non-catalog CRS are reprojected
        internally — the canonical anti-footgun path (§10.1 of the
        design plan).
        """
        if slice_ is not None and (
            bounds is not None or time is not None or crs is not None
        ):
            raise TypeError(
                "query: pass either slice_ or (bounds + crs + time), not both; "
                "a GeoSlice carries its own crs"
            )
        if slice_ is not None:
            q_bounds = slice_.bounds
            q_crs = slice_.crs
            q_interval = slice_.interval
        else:
            if bounds is None and time is None:
                # No filter — return a copy view.
                return InMemoryGeoCatalog(self.gdf.copy(), backend=self.backend)
            q_bounds = bounds
            q_crs = crs
            q_interval = _coerce_interval(time) if time is not None else None

        out = self.gdf
        if q_interval is not None:
            out = out[out.index.overlaps(q_interval)]
        if q_bounds is not None:
            envelopes = _query_envelopes(q_bounds, q_crs, self.gdf.crs)
            if len(envelopes) == 1:
                xmin, ymin, xmax, ymax = envelopes[0]
                out = out.cx[xmin:xmax, ymin:ymax]
            else:
                # Antimeridian split: same predicate as `.cx` (geometry
                # intersects the box), OR-ed over both halves.
                out = out[
                    out.intersects(
                        shapely.MultiPolygon([shapely.box(*env) for env in envelopes])
                    )
                ]
        return InMemoryGeoCatalog(out, backend=self.backend)

    def intersect(
        self,
        other: InMemoryGeoCatalog,
        *,
        spatial_only: bool = False,
        engine: _INTERSECT_ENGINE_T = "sjoin",
    ) -> InMemoryGeoCatalog:
        """Cross-catalog AND — rows whose footprints and times overlap.

        Args:
            other: Another catalog, possibly a different backend. The
                returned catalog has ``self``'s backend tag (it indexes
                the same kind of file as ``self``).
            spatial_only: If True, ignore the temporal axis — useful for
                pairing imagery with static labels.
            engine: Spatial join engine. ``"sjoin"`` uses the GeoPandas
                spatial index, while ``"overlay"`` preserves the legacy
                overlay implementation.
        """
        if other.gdf.crs != self.gdf.crs:
            right_gdf = other.gdf.to_crs(self.gdf.crs)
        else:
            right_gdf = other.gdf

        right_renamed = right_gdf.rename(
            columns={c: f"_right_{c}" for c in right_gdf.columns if c != "geometry"}
        )
        left = self.gdf.reset_index(names="_left_interval")
        right = right_renamed.reset_index(names="_right_interval")
        if engine == "overlay":
            joined = gpd.overlay(left, right, how="intersection", keep_geom_type=True)
        elif engine == "sjoin":
            joined = gpd.sjoin(left, right, how="inner", predicate="intersects")
            if not joined.empty:
                left_geometry = joined.geometry.reset_index(drop=True)
                right_geometry = right.geometry.iloc[joined["index_right"]].reset_index(
                    drop=True
                )
                # Mirror gpd.overlay's default ``make_valid=True``: GEOS will
                # raise on self-intersecting inputs to ``shapely.intersection``.
                # Pay the repair cost only on rows actually flagged invalid.
                left_array = _repair_invalid(left_geometry.to_numpy())
                right_array = _repair_invalid(right_geometry.to_numpy())
                clipped = gpd.GeoSeries(
                    _extract_same_family(
                        _symmetric_intersection(left_array, right_array),
                        left_geometry.to_numpy(),
                    ),
                    index=joined.index,
                    crs=self.gdf.crs,
                )
                keep_geom_mask = _keep_geom_type_mask(joined.geometry, clipped)
                joined = joined.loc[keep_geom_mask].copy()
                clipped = clipped.loc[keep_geom_mask]
                joined = joined.set_geometry(clipped)
                joined = joined.drop(columns=["index_right"], errors="ignore")
        else:
            raise ValueError(f"Unsupported intersect engine: {engine!r}")

        if joined.empty:
            return self._empty()

        if spatial_only:
            mint = joined["_left_interval"].apply(lambda i: i.left)
            maxt = joined["_left_interval"].apply(lambda i: i.right)
            keep_mask = pd.Series(True, index=joined.index)
        else:
            li = joined["_left_interval"]
            ri = joined["_right_interval"]
            mint = np.maximum(
                li.apply(lambda i: i.left).to_numpy(),
                ri.apply(lambda i: i.left).to_numpy(),
            )
            maxt = np.minimum(
                li.apply(lambda i: i.right).to_numpy(),
                ri.apply(lambda i: i.right).to_numpy(),
            )
            keep_mask = pd.Series(maxt >= mint, index=joined.index)
            joined = joined[keep_mask]
            mint = mint[keep_mask.to_numpy()]
            maxt = maxt[keep_mask.to_numpy()]

        if joined.empty:
            return self._empty()

        idx = pd.IntervalIndex.from_arrays(mint, maxt, closed="both", name="datetime")
        joined = joined.drop(
            columns=[
                "_left_interval",
                "_right_interval",
                # The right side's time columns duplicate the index.
                "_right_start_time",
                "_right_end_time",
            ],
            errors="ignore",
        ).set_index(idx)
        # Keep the time columns (if present) in step with the new,
        # clipped index rather than the left side's original times.
        if "start_time" in joined.columns:
            joined["start_time"] = idx.left
        if "end_time" in joined.columns:
            joined["end_time"] = idx.right
        return InMemoryGeoCatalog(joined, backend=self.backend)

    def _empty(self) -> InMemoryGeoCatalog:
        """Zero-row catalog with this catalog's columns, dtypes, CRS and tag."""
        return InMemoryGeoCatalog(self.gdf.iloc[:0].copy(), backend=self.backend)

    def union(self, other: InMemoryGeoCatalog) -> InMemoryGeoCatalog:
        """Cross-catalog OR — concatenate rows.

        ``self``'s CRS and backend tag win. If ``other`` is in a
        different CRS it's reprojected into ``self.crs`` first. The
        backend tags are *not* required to match: the caller is
        responsible for ensuring it makes sense to treat the merged
        rows uniformly (e.g. unioning two raster catalogs is fine;
        unioning raster + vector would lie about what the result
        indexes, but no exception is raised — the downstream loader
        will catch it via its own backend-tag check).
        """
        if other.gdf.crs != self.gdf.crs:
            right_gdf = other.gdf.to_crs(self.gdf.crs)
        else:
            right_gdf = other.gdf
        geom_name = str(self.gdf.geometry.name)
        if right_gdf.geometry.name != geom_name:
            # Otherwise concat puts other's footprints in a separate
            # column and the merged geometry is null for those rows.
            right_gdf = right_gdf.rename_geometry(geom_name, inplace=False)
        merged = gpd.GeoDataFrame(
            pd.concat([self.gdf, right_gdf], axis=0), crs=self.gdf.crs
        )
        return InMemoryGeoCatalog(merged, backend=self.backend)

    def iter_rows(self, *, batch_size: int = 1024) -> Iterator[CatalogRow]:
        """Yield each row as a backend-neutral `CatalogRow`.

        Streaming-friendly: rows are constructed lazily so a caller can
        short-circuit a large catalog. ``batch_size`` is accepted for
        Protocol parity with `DuckDBGeoCatalog` and is ignored here —
        in-memory iteration is row-at-a-time regardless.

        Args:
            batch_size: Accepted for Protocol parity, ignored.

        Yields:
            `CatalogRow` in catalog row order. ``filepath`` comes from
            the ``filepath`` column (or the row index as a fallback);
            ``extras`` carries every other non-geometry column.
        """
        del batch_size
        crs = pyproj.CRS.from_user_input(self.gdf.crs)
        # Mirror `DuckDBGeoCatalog.iter_rows`: drop the GeoParquet 1.1
        # `bbox` covering struct and any underscore-prefixed on-disk
        # schema column (e.g. `_backend`, `_schema_version`) so they
        # don't leak through `CatalogRow.extras` into downstream
        # consumers (STAC export, matchup, …).
        reserved = {"geometry", "filepath", "start_time", "end_time", "bbox"}
        extra_cols = [
            c for c in self.gdf.columns if c not in reserved and not c.startswith("_")
        ]
        # Use ``.array`` rather than ``.to_numpy(copy=False)`` so pandas
        # extension scalars (notably ``Timestamp``) survive iteration with
        # the same types DuckDB's ``Series.iloc[i]`` produces — going
        # through NumPy would silently coerce them to ``np.datetime64``
        # etc. and make ``CatalogRow.extras`` backend-dependent.
        geoms = self.gdf.geometry.array
        intervals = self.gdf.index.array
        paths = self.gdf["filepath"].array if "filepath" in self.gdf.columns else None
        extras_data = {c: self.gdf[c].array for c in extra_cols}
        extras_items = tuple(extras_data.items())
        n_rows = len(self.gdf)

        for i in range(n_rows):
            filepath = paths[i] if paths is not None else None
            if filepath is None:
                filepath = str(intervals[i])
            extras = {c: values[i] for c, values in extras_items}
            yield CatalogRow(
                filepath=str(filepath),
                geometry=geoms[i],
                interval=intervals[i],
                crs=crs,
                extras=extras,
            )

    def iter_slices(self, *, resolution: tuple[float, float]) -> Iterator[GeoSlice]:
        """Yield one `GeoSlice` per row, at the given target resolution.

        Lazy — each `GeoSlice` is constructed on demand so callers can
        short-circuit large catalogs. The slice's ``bounds`` come from
        the row's footprint polygon, its ``interval`` from the
        IntervalIndex entry, and ``crs`` from ``self.gdf.crs``.

        Args:
            resolution: ``(x_res, y_res)`` in CRS units, baked into
                every emitted slice. Drives the pixel shape the
                downstream loader will produce.

        Yields:
            `GeoSlice` instances in catalog row order. Point and line
            footprints with a zero-width axis get a one-pixel extent on
            that axis, centred on the footprint; rows with a missing or
            empty footprint are skipped with a single warning.
        """
        crs = self.crs
        yield from _footprint_slices(
            zip(self.gdf.geometry, self.gdf.index, strict=True), resolution, crs
        )

    def where(self, query: str) -> InMemoryGeoCatalog:
        """Filter by a non-geometric predicate — escape hatch via pandas ``.query()``.

        Useful when the catalog carries extra columns (sensor, cloud %,
        mission ID) and you want to filter on them without writing a
        boolean mask by hand. The geometry / time axes are unaffected;
        for those use `query`.

        Args:
            query: A pandas query string, e.g. ``"cloud_pct < 20 and
                sensor == 'S2A'"``. Column names available are whatever
                the underlying ``GeoDataFrame`` has.

        Returns:
            A new catalog with the matching rows; same backend tag,
            same CRS.

        Example:
            >>> imagery.where("mission == 'S2A' and cloud_pct < 20")
        """
        return InMemoryGeoCatalog(self.gdf.query(query), backend=self.backend)

    def get_config(self) -> dict[str, Any]:
        """JSON-serialisable summary — backend tag, row count, CRS.

        Returns:
            ``{"backend": str, "len": int, "crs": str}``. The CRS is
            stringified via ``str(self.gdf.crs)``; it round-trips
            through ``pyproj.CRS.from_user_input``.
        """
        return {
            "backend": self.backend,
            "len": len(self),
            "crs": str(self.gdf.crs),
        }


def _repair_invalid(geometries: np.ndarray) -> np.ndarray:
    """Apply ``shapely.make_valid`` only to invalid geometries.

    ``gpd.overlay`` defaults to ``make_valid=True``; the sjoin path must
    mirror that to avoid GEOS exceptions from ``shapely.intersection`` on
    self-intersecting polygons. Vectorised ``shapely.is_valid`` keeps the
    cost proportional to the number of actually-invalid rows.
    """
    invalid_mask = ~shapely.is_valid(geometries)
    if not invalid_mask.any():
        return geometries
    repaired = geometries.copy()
    repaired[invalid_mask] = shapely.make_valid(geometries[invalid_mask])
    return repaired


def _symmetric_intersection(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise ``shapely.intersection`` with a canonical operand order.

    GEOS intersection is not bit-symmetric under operand order for
    near-degenerate overlaps: a sliver a few ULPs wide can come back as
    a `Polygon` for ``intersection(a, b)`` and as empty for
    ``intersection(b, a)`` (gh #40). Ordering each pair canonically (by
    WKB bytes) makes ``intersect(a, b)`` and ``intersect(b, a)`` compute
    the same geometry for every row pair, so the result cardinality is
    symmetric by construction.
    """
    left_wkb = shapely.to_wkb(left)
    right_wkb = shapely.to_wkb(right)
    swap = np.fromiter(
        (lw > rw for lw, rw in zip(left_wkb, right_wkb, strict=True)),
        dtype=bool,
        count=len(left),
    )
    first = np.where(swap, right, left)
    second = np.where(swap, left, right)
    return shapely.intersection(first, second)


def _extract_same_family(clipped: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Reduce ``GeometryCollection`` clips to the left geometry's family.

    Two polygons that overlap in an area *and* touch along an edge
    intersect as ``GeometryCollection(Polygon, LineString)``. Mirror
    ``gpd.overlay(..., keep_geom_type=True)``: keep the parts in the
    left row's family (e.g. the polygon) and union them, rather than
    dropping the whole row. Collections with no matching part are left
    as they are and filtered out by `_keep_geom_type_mask`.
    """
    is_collection = (
        shapely.get_type_id(clipped) == shapely.GeometryType.GEOMETRYCOLLECTION
    )
    if not is_collection.any():
        return clipped
    out = clipped.copy()
    for i in np.flatnonzero(is_collection):
        family = _GEOMETRY_TYPE_FAMILY.get(left[i].geom_type)
        parts = [
            p
            for p in shapely.get_parts(clipped[i])
            if _GEOMETRY_TYPE_FAMILY.get(p.geom_type) == family
        ]
        if parts:
            out[i] = shapely.union_all(parts)
    return out


def _keep_geom_type_mask(
    left_geometry: gpd.GeoSeries, intersection_geometry: gpd.GeoSeries
) -> pd.Series:
    """Match `gpd.overlay(..., keep_geom_type=True)` after vectorized clipping.

    Known single/multi geometry pairs share a family. Unknown geometry
    types fall through unchanged and must match exactly.
    """
    left_family = left_geometry.geom_type.replace(_GEOMETRY_TYPE_FAMILY)
    intersection_family = intersection_geometry.geom_type.replace(_GEOMETRY_TYPE_FAMILY)
    return (
        intersection_geometry.notna()
        & ~intersection_geometry.is_empty
        & (left_family == intersection_family)
    )


def _coerce_interval(time: tuple[Any, Any] | pd.Interval) -> pd.Interval:
    """Normalise user-supplied time bounds into a ``closed='both'`` interval.

    Accepts either a ``(start, end)`` pair of ``pd.Timestamp``-likes or
    a `pd.Interval`. Half-open intervals are rebuilt as ``closed='both'``
    to match the catalog's IntervalIndex convention.
    Timestamp endpoints are returned in naive UTC, the catalog's stored
    time form, so tz-aware and naive bounds query identically.
    """
    if isinstance(time, pd.Interval):
        t0, t1 = time.left, time.right
    else:
        t0, t1 = time
    return pd.Interval(to_naive_utc(t0), to_naive_utc(t1), closed="both")


def _footprint_slices(
    rows: Iterable[tuple[Any, pd.Interval]],
    resolution: tuple[float, float],
    crs: pyproj.CRS,
) -> Iterator[GeoSlice]:
    """One `GeoSlice` per ``(footprint, interval)``; shared by both backends.

    ``align="off"`` because footprints are arbitrary shapes; their bbox
    extents are almost never integer multiples of an arbitrary target
    resolution. A zero-width axis (Point, or an axis-parallel line) is
    widened to one pixel centred on the footprint, since `GeoSlice`
    requires ``xmin < xmax`` and ``ymin < ymax``. Missing or empty
    footprints have no extent at all; they are skipped and counted, and
    one warning reports the count once iteration finishes.
    """
    x_res, y_res = resolution
    skipped = 0
    for geom, interval in rows:
        if geom is None or geom.is_empty:
            skipped += 1
            continue
        xmin, ymin, xmax, ymax = geom.bounds
        if not xmin < xmax:
            cx = (xmin + xmax) / 2
            xmin, xmax = cx - x_res / 2, cx + x_res / 2
        if not ymin < ymax:
            cy = (ymin + ymax) / 2
            ymin, ymax = cy - y_res / 2, cy + y_res / 2
        yield GeoSlice(
            bounds=(xmin, ymin, xmax, ymax),
            interval=interval,
            resolution=resolution,
            crs=crs,
            align="off",
        )
    if skipped:
        warnings.warn(
            f"iter_slices skipped {skipped} row(s) with a missing or empty footprint.",
            UserWarning,
            stacklevel=3,
        )


def _reproject_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: Any | None,
    dst_crs: Any,
) -> tuple[float, float, float, float]:
    """Reproject a bbox into the catalog CRS — no-op if they already match.

    Silently returning empty results when the user's AOI is in the wrong
    CRS is the §10.1 footgun this helper exists to avoid. ``src_crs=None``
    is treated as "already in catalog CRS" — preserves the bbox
    unchanged.
    """
    if src_crs is None:
        return bounds
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return bounds
    transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)
    return transformer.transform_bounds(*bounds)  # type: ignore[return-value]


def _query_envelopes(
    bounds: tuple[float, float, float, float],
    src_crs: Any,
    dst_crs: Any,
) -> list[tuple[float, float, float, float]]:
    """Query bbox(es) in the catalog CRS, split at the antimeridian if needed.

    ``transform_bounds`` into a geographic CRS reports an AOI that
    crosses ±180° as ``xmin > xmax`` (and raw geographic ``bounds=``
    may use the same convention). A single envelope built from those
    numbers is the *complement* of the AOI, so it is split into
    ``[xmin, 180]`` and ``[-180, xmax]`` instead. Inverted x in a
    projected CRS, or inverted y anywhere, is an error.

    Returns:
        One envelope, or two for an antimeridian-crossing AOI.

    Raises:
        ValueError: If the (reprojected) bounds are inverted and cannot
            be read as an antimeridian crossing.
    """
    xmin, ymin, xmax, ymax = (
        float(v) for v in _reproject_bounds(bounds, src_crs, dst_crs)
    )
    if ymin > ymax:
        raise ValueError(
            f"query bounds have ymin > ymax in the catalog CRS: "
            f"{(xmin, ymin, xmax, ymax)!r}"
        )
    if xmin <= xmax:
        return [(xmin, ymin, xmax, ymax)]
    if not pyproj.CRS.from_user_input(dst_crs).is_geographic:
        raise ValueError(
            f"query bounds have xmin > xmax in the projected catalog CRS: "
            f"{(xmin, ymin, xmax, ymax)!r}"
        )
    return [(xmin, ymin, 180.0, ymax), (-180.0, ymin, xmax, ymax)]
