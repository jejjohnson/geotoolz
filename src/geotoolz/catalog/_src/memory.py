"""`InMemoryGeoCatalog` — Phase 1 in-RAM catalog backend.

Wraps a ``geopandas.GeoDataFrame`` whose row index is a
``pd.IntervalIndex`` over the time axis and whose ``geometry`` column
carries each file's footprint in a uniform target CRS. Queries leverage
geopandas's R-tree (built lazily on first access via ``gdf.sindex``)
and pandas's IntervalIndex for O(log n + k) lookups in both axes.

Good for catalogs up to ~10⁵ rows. Beyond that, switch to the v0.2
DuckDB backend (same Protocol, different store).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj

from geotoolz.catalog._src.base import CatalogRow
from geotoolz.types import GeoSlice


_BACKEND_T = Literal["raster", "xarray", "vector"]


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
            dispatching choice in `geotoolz.catalog.load_*`.
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
        self.gdf = gdf
        self.backend = backend

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Union bbox over all rows.

        Returns:
            ``(xmin, ymin, xmax, ymax)`` in catalog-CRS units, or four
            NaNs if the catalog is empty.
        """
        if len(self.gdf) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return tuple(self.gdf.total_bounds.tolist())  # type: ignore[return-value]

    @property
    def temporal_extent(self) -> pd.Interval:
        """Tightest interval spanning every row's time window.

        Returns:
            ``pd.Interval(min(start_times), max(end_times),
            closed='both')``. Both endpoints are ``pd.NaT`` for an
            empty catalog.
        """
        if len(self.gdf) == 0:
            return pd.Interval(pd.NaT, pd.NaT, closed="both")
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
        if slice_ is not None and (bounds is not None or time is not None):
            raise TypeError("query: pass either slice_ or (bounds + time), not both")
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
            xmin, ymin, xmax, ymax = _reproject_bounds(q_bounds, q_crs, self.gdf.crs)
            out = out.cx[xmin:xmax, ymin:ymax]
        return InMemoryGeoCatalog(out, backend=self.backend)

    def intersect(
        self, other: InMemoryGeoCatalog, *, spatial_only: bool = False
    ) -> InMemoryGeoCatalog:
        """Cross-catalog AND — rows whose footprints and times overlap.

        Args:
            other: Another catalog, possibly a different backend. The
                returned catalog has ``self``'s backend tag (it indexes
                the same kind of file as ``self``).
            spatial_only: If True, ignore the temporal axis — useful for
                pairing imagery with static labels.
        """
        if other.gdf.crs != self.gdf.crs:
            right_gdf = other.gdf.to_crs(self.gdf.crs)
        else:
            right_gdf = other.gdf

        # Spatial intersect via gpd.overlay. Rename the right-side
        # attribute columns up-front so they survive the overlay without
        # collisions, *then* attach the interval columns under a sentinel
        # name `gpd.overlay` won't touch.
        right_renamed = right_gdf.rename(
            columns={c: f"_right_{c}" for c in right_gdf.columns if c != "geometry"}
        )
        left = self.gdf.reset_index(names="_left_interval")
        right = right_renamed.reset_index(names="_right_interval")
        overlay = gpd.overlay(left, right, how="intersection", keep_geom_type=True)
        if overlay.empty:
            return _empty_catalog(self.gdf.crs, self.backend)

        if spatial_only:
            mint = overlay["_left_interval"].apply(lambda i: i.left)
            maxt = overlay["_left_interval"].apply(lambda i: i.right)
            keep_mask = pd.Series(True, index=overlay.index)
        else:
            li = overlay["_left_interval"]
            ri = overlay["_right_interval"]
            mint = np.maximum(
                li.apply(lambda i: i.left).to_numpy(),
                ri.apply(lambda i: i.left).to_numpy(),
            )
            maxt = np.minimum(
                li.apply(lambda i: i.right).to_numpy(),
                ri.apply(lambda i: i.right).to_numpy(),
            )
            keep_mask = pd.Series(maxt >= mint, index=overlay.index)
            overlay = overlay[keep_mask]
            mint = mint[keep_mask.to_numpy()]
            maxt = maxt[keep_mask.to_numpy()]

        if overlay.empty:
            return _empty_catalog(self.gdf.crs, self.backend)

        idx = pd.IntervalIndex.from_arrays(mint, maxt, closed="both", name="datetime")
        overlay = overlay.drop(
            columns=["_left_interval", "_right_interval"], errors="ignore"
        ).set_index(idx)
        return InMemoryGeoCatalog(overlay, backend=self.backend)

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
        reserved = {"geometry", "filepath", "start_time", "end_time"}
        extra_cols = [c for c in self.gdf.columns if c not in reserved]
        for interval, row in zip(self.gdf.index, self.gdf.itertuples(), strict=True):
            row_dict = row._asdict()
            filepath = row_dict.get("filepath")
            if filepath is None:
                filepath = str(row_dict.get("Index", ""))
            extras = {c: row_dict[c] for c in extra_cols if c in row_dict}
            yield CatalogRow(
                filepath=str(filepath),
                geometry=row_dict["geometry"],
                interval=interval,
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
            `GeoSlice` instances in catalog row order.
        """
        crs = pyproj.CRS.from_user_input(self.gdf.crs)
        for interval, geom in zip(self.gdf.index, self.gdf.geometry, strict=True):
            yield GeoSlice(
                bounds=tuple(geom.bounds),  # type: ignore[arg-type]
                interval=interval,
                resolution=resolution,
                crs=crs,
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


def _empty_catalog(crs: Any, backend: _BACKEND_T) -> InMemoryGeoCatalog:
    """Build a zero-row `InMemoryGeoCatalog` with the right schema.

    Constructed on demand when `intersect` / `query` filters to nothing
    — the caller still needs a typed catalog (right CRS, right
    IntervalIndex schema, right backend tag) rather than a bare
    GeoDataFrame.
    """
    empty_gdf = gpd.GeoDataFrame(
        {"geometry": []},
        geometry="geometry",
        crs=crs,
        index=pd.IntervalIndex.from_arrays(
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype="datetime64[ns]"),
            closed="both",
            name="datetime",
        ),
    )
    return InMemoryGeoCatalog(empty_gdf, backend=backend)


def _coerce_interval(time: tuple[Any, Any] | pd.Interval) -> pd.Interval:
    """Normalise user-supplied time bounds into a ``closed='both'`` interval.

    Accepts either a ``(start, end)`` pair of ``pd.Timestamp``-likes or
    a `pd.Interval`. Half-open intervals are rebuilt as ``closed='both'``
    to match the catalog's IntervalIndex convention.
    """
    if isinstance(time, pd.Interval):
        if time.closed != "both":
            return pd.Interval(time.left, time.right, closed="both")
        return time
    t0, t1 = time
    return pd.Interval(pd.Timestamp(t0), pd.Timestamp(t1), closed="both")


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
