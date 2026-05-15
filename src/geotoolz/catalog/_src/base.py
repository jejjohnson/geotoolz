"""`GeoCatalog` Protocol — the catalog contract.

A *geocatalog* is a queryable spatiotemporal index over geospatial
files. Each row records the file's bbox, time interval, CRS, and path.
Given a query like "files overlapping AOI X between dates Y and Z," the
catalog returns the matching rows fast without opening any file.

This Protocol is what downstream callers depend on. Two backends
implement it (`InMemoryGeoCatalog` in v0.1, `DuckDBGeoCatalog` in v0.2);
new backends can join without touching consumers.

See ``research_journal_v2/notes/geotoolz/plans/geodatabase/geocatalog.md``
for the design report.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import pandas as pd


if TYPE_CHECKING:
    import geopandas as gpd
    import pyproj
    import shapely.geometry.base

    from geotoolz.types import GeoSlice


@dataclasses.dataclass(frozen=True)
class CatalogRow:
    """Backend-neutral view of a single catalog row.

    Yielded by `GeoCatalog.iter_rows`; consumed by loaders and the
    `geotoolz.patch` bridge. The dataclass exists so streaming code
    (DuckDB cursors, on-disk shards) can hand rows downstream without
    callers caring whether the row came from a `gpd.GeoDataFrame` or a
    SQL relation.

    Attributes:
        filepath: Path or URI of the indexed file. The loader resolves
            this with the configured reader class.
        geometry: Footprint polygon (or multipolygon) in ``crs`` units.
            Already decoded from WKB for DuckDB-backed catalogs.
        interval: Time window for the row, ``closed='both'``.
        crs: Coordinate reference of ``geometry``.
        extras: Backend-specific metadata (``layer``, ``data_vars``,
            ``time_var``, …). Empty dict for rows that carry no extras.
    """

    filepath: str
    geometry: shapely.geometry.base.BaseGeometry
    interval: pd.Interval
    crs: pyproj.CRS
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class GeoCatalog(Protocol):
    """A queryable spatiotemporal index over geospatial files.

    Implementations carry a backend-specific store (in-memory
    GeoDataFrame; DuckDB+GeoParquet; future others) but expose the same
    query / intersect / union / iter-slices surface. Downstream code
    (loaders, the patch bridge, training loops) targets this Protocol,
    not a concrete class, so swapping the backend never touches the
    consumer.

    Attributes:
        gdf: A `geopandas.GeoDataFrame` view of the catalog rows. For
            in-memory backends this is the canonical store; for
            out-of-core backends it's a materialised view of the rows
            most recently queried. Always non-None; may be empty. The
            geometry column is in CRS units; the row index is a
            ``pd.IntervalIndex`` (``closed='both'``) over the time axis.
        backend: One of ``"raster"``, ``"xarray"``, ``"vector"``.
            Drives the dispatching choice in the per-backend loaders
            (`load_raster`, `load_xarray`, `load_vector`).
    """

    gdf: gpd.GeoDataFrame
    backend: Literal["raster", "xarray", "vector"]

    def query(
        self,
        slice_: GeoSlice | None = None,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        crs: Any | None = None,
        time: tuple[Any, Any] | pd.Interval | None = None,
    ) -> GeoCatalog:
        """Spatial + temporal filter — return rows overlapping the query.

        Two call shapes are supported: pass a `GeoSlice` (carries bounds,
        time, and CRS together), or pass the parts. Bounds in a non-
        catalog CRS are reprojected internally so an AOI in the wrong
        CRS doesn't silently match zero rows (§10.1 of the design plan).

        Args:
            slice_: A `GeoSlice` whose bbox + interval drive the filter.
                Mutually exclusive with the keyword args.
            bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` units.
            crs: CRS of ``bounds``; defaults to the catalog CRS if
                ``None``.
            time: Either a ``(start, end)`` pair (`pd.Timestamp`-like)
                or a `pd.Interval`. ``None`` skips the temporal filter.

        Returns:
            A new catalog of the same backend tag containing only the
            matching rows. May be empty.

        Raises:
            TypeError: If both ``slice_`` and any of (``bounds``,
                ``time``) are passed.
        """
        ...

    def intersect(self, other: GeoCatalog, *, spatial_only: bool = False) -> GeoCatalog:
        """Cross-catalog AND — pair rows by spatial (+ temporal) overlap.

        For each pair of rows whose footprints intersect, the result
        contains one row with the clipped intersection footprint and the
        intersected time interval. Pairs with disjoint intervals are
        dropped unless ``spatial_only=True``.

        Args:
            other: The catalog to pair against. May use a different
                backend; reprojection to ``self``'s CRS happens
                internally if needed.
            spatial_only: If True, ignore the temporal axis. The right
                tool for pairing imagery with static labels (DEMs,
                land-cover maps) that have no meaningful time interval.

        Returns:
            A new catalog with ``self``'s backend tag — the result
            indexes the same kind of file as ``self``, just constrained
            to footprints where ``other`` agrees. May be empty.
        """
        ...

    def union(self, other: GeoCatalog) -> GeoCatalog:
        """Cross-catalog OR — concatenate rows.

        Useful when you want to treat two sensors as one virtual dataset
        (Landsat 7 + Landsat 8 for a long time series; S2A + S2B for
        better revisit). ``other`` is reprojected into ``self.crs`` if
        the CRSs differ; the result's backend tag is ``self``'s.
        Backend tags are *not* checked — unioning a raster catalog with
        a vector one will not raise, but the downstream loader you
        eventually call will reject the mismatch.

        Args:
            other: Catalog to concatenate. CRS is auto-reprojected.

        Returns:
            A new catalog containing every row of ``self`` followed by
            every (possibly reprojected) row of ``other``. No dedup is
            performed — call `query` afterwards if you want to filter.
        """
        ...

    def iter_rows(self, *, batch_size: int = 1024) -> Iterator[CatalogRow]:
        """Yield rows as backend-neutral `CatalogRow` instances.

        Consumed by loaders, the sampler, and the `geotoolz.patch`
        bridge. Backends should stream — for the DuckDB backend this
        fetches in batches of ``batch_size``; the in-memory backend
        ignores ``batch_size`` and iterates the underlying gdf.

        Args:
            batch_size: Rows per fetch for streaming backends. Default
                1024. In-memory backends ignore this.

        Yields:
            `CatalogRow` instances in catalog order. Order is stable for
            in-memory backends; out-of-core backends should document
            their ordering explicitly.
        """
        ...

    def iter_slices(self, *, resolution: tuple[float, float]) -> Iterator[GeoSlice]:
        """Yield one `GeoSlice` per catalog row at the given resolution.

        Each row's footprint becomes the slice's bbox; the row's
        IntervalIndex entry becomes its interval. The slices are the
        unit of work consumed by loaders and the `CatalogDomain` bridge
        — emit them lazily so callers can short-circuit.

        Args:
            resolution: Target ``(x_res, y_res)`` in CRS units, baked
                into every emitted slice. Drives the pixel shape a
                loader will produce.

        Yields:
            `GeoSlice` instances in row order. Order is stable for
            in-memory backends; out-of-core backends should document
            their ordering explicitly.
        """
        ...

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Union bbox over all rows.

        Returns:
            ``(xmin, ymin, xmax, ymax)`` in catalog-CRS units, or four
            NaNs if the catalog is empty.
        """
        ...

    @property
    def temporal_extent(self) -> pd.Interval:
        """Tightest interval that contains every row's time window.

        Returns:
            A ``pd.Interval(closed='both')`` from ``min(start_times)``
            to ``max(end_times)``. For an empty catalog, both endpoints
            are ``pd.NaT``.
        """
        ...

    def __len__(self) -> int:
        """Number of rows in the catalog."""
        ...

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary — for logging / reproducibility audits.

        The exact shape is backend-specific but always includes the
        backend tag, the row count, and the catalog CRS so two catalogs
        can be compared at a glance.
        """
        ...
