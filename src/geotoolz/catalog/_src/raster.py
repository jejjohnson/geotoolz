"""Raster catalog builder + loaders.

`build_raster_catalog` reads each file's bounds via ``rasterio`` (lazy
via `WarpedVRT`) and assembles them into a `InMemoryGeoCatalog`.
`load_raster` opens the matching rows for a `GeoSlice` and mosaics
them into a single `GeoTensor`.

This is the bedrock backend; xarray and vector builders mirror its
shape.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
from georeader.geotensor import GeoTensor
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT

from geotoolz.catalog._src.memory import InMemoryGeoCatalog
from geotoolz.types import GeoSlice


log = logging.getLogger(__name__)


_RasterMergeMethod = Literal["first", "last", "min", "max", "sum"]
_VALID_MERGE_METHODS: tuple[_RasterMergeMethod, ...] = (
    "first",
    "last",
    "min",
    "max",
    "sum",
)


def _parse_date(value: str, fmt: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` for the UTC day containing ``value``.

    `Timestamp.ceil('D')` is a no-op when the input is already at a day
    boundary, which would produce start > end for a date-only string
    like "20240115". Add a full day to the floor instead.
    """
    ts = pd.to_datetime(value, format=fmt)
    start = ts.floor("D")
    end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end


def _filepath_to_row(
    filepath: str | Path,
    *,
    filename_regex: re.Pattern[str] | None,
    date_format: str,
    target_crs: Any | None,
) -> dict[str, Any] | None:
    filepath = Path(filepath)
    with rasterio.open(filepath) as src:
        if target_crs is None:
            crs = src.crs
            bounds = src.bounds
        else:
            with WarpedVRT(src, crs=target_crs) as vrt:
                crs = vrt.crs
                bounds = vrt.bounds

    polygon = shapely.geometry.box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    if filename_regex is None:
        # No date parsing — files are treated as time-invariant. Use a
        # narrow sentinel interval that won't dominate IntervalIndex
        # logs the way Timestamp.min/max would (§7.2 of the design plan).
        start = pd.Timestamp("1900-01-01")
        end = pd.Timestamp("2100-01-01")
    else:
        match = filename_regex.search(filepath.name)
        if match is None:
            log.warning("Skipping %s: filename does not match regex", filepath)
            return None
        groups = match.groupdict()
        if "date" in groups:
            start, end = _parse_date(groups["date"], date_format)
        elif "start" in groups and "stop" in groups:
            start, _ = _parse_date(groups["start"], date_format)
            _, end = _parse_date(groups["stop"], date_format)
        else:
            raise ValueError(
                f"filename_regex must capture either 'date' or "
                f"'start'+'stop' named groups; got {list(groups.keys())}"
            )

    return {
        "filepath": str(filepath),
        "geometry": polygon,
        "start_time": start,
        "end_time": end,
        "crs": str(crs),
    }


def build_raster_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str | None = None,
    date_format: str = "%Y%m%d",
    target_crs: Any | None = None,
) -> InMemoryGeoCatalog:
    """Build an in-memory catalog from a collection of raster files.

    For each input file, opens it with `rasterio`, extracts its bounds
    (optionally projected through a lazy `WarpedVRT` if ``target_crs``
    is set so no pixels are read), and parses the time interval from
    the filename via ``filename_regex``. The result is a catalog
    queryable in milliseconds even over tens of thousands of files —
    `rasterio` builds the spatial R-tree lazily on first access.

    Files whose filenames don't match the regex are *skipped with a
    warning*, not raised — so a heterogeneous directory partially
    populated by one product still produces a usable catalog.

    Args:
        filepaths: Files to index. Anything `rasterio.open` accepts —
            local paths, S3 URLs via `/vsis3/`, HTTP URLs via
            `/vsicurl/`, etc.
        filename_regex: A regex with a named group capturing the date:
            either ``(?P<date>...)`` for single-date captures (treated
            as covering the full UTC day) or ``(?P<start>...)`` +
            ``(?P<stop>...)`` for explicit ranges. ``None`` treats every
            file as time-invariant and writes a sentinel interval
            (``1900-01-01`` to ``2100-01-01``) instead.
        date_format: ``strptime`` format for the named date groups.
            Default ``"%Y%m%d"``.
        target_crs: CRS to project each file's bounds into. ``None``
            keeps each file's native CRS, which is fine for a uniform
            archive but a footgun for multi-CRS ones: the catalog
            gdf-level CRS must be a single value, and set algebra will
            reproject but bounds extraction won't. For multi-CRS
            archives, set this explicitly.

    Returns:
        An `InMemoryGeoCatalog` with backend ``"raster"`` and one row
        per matching file. Columns: ``filepath``, ``geometry``,
        ``start_time``, ``end_time``, ``crs``.

    Raises:
        ValueError: If no files matched the regex (the catalog would
            be empty — usually a sign of a wrong regex or wrong
            ``filepaths`` list).
    """
    pattern = re.compile(filename_regex) if filename_regex is not None else None
    rows: list[dict[str, Any]] = []
    for fp in filepaths:
        row = _filepath_to_row(
            fp,
            filename_regex=pattern,
            date_format=date_format,
            target_crs=target_crs,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("build_raster_catalog: no files matched the regex")

    crs_value = target_crs if target_crs is not None else rows[0]["crs"]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_value)
    return InMemoryGeoCatalog(gdf, backend="raster")


def load_raster(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Any | None = None,
    merge_method: _RasterMergeMethod = "last",
    nodata: float | None = None,
) -> GeoTensor:
    """Read + mosaic the catalog rows matching ``slice_`` into one `GeoTensor`.

    Filters the catalog with ``catalog.query(slice_)``, opens every
    surviving row, and calls `rasterio.merge.merge` to produce a single
    array clipped to ``slice_.bounds`` at ``slice_.resolution``. The
    returned `GeoTensor` carries the right transform + CRS for
    downstream operators.

    Args:
        catalog: A raster-backend catalog.
        slice_: The window to read. Bounds may be in a different CRS
            than the catalog; the loader reprojects internally on the
            query.
        band_indexes: 1-indexed bands to keep, in the order requested.
            ``None`` means all bands in their native order.
        resampling: A ``rasterio.enums.Resampling`` enum value (e.g.
            ``Resampling.nearest`` for categorical labels). Defaults to
            bilinear.
        merge_method: How to combine overlapping pixels from multiple
            source files. One of ``"first"``, ``"last"``, ``"min"``,
            ``"max"``, ``"sum"`` — exactly what `rasterio.merge.merge`
            accepts. The original snippet's ``"count"`` is not a valid
            rasterio mode and is rejected (§7.7 of the design plan).
        nodata: Override the nodata value used in the mosaic. ``None``
            respects each source's declared ``_FillValue`` / ``nodata``.

    Returns:
        A `GeoTensor` of shape ``(bands, H, W)`` with ``transform`` and
        ``crs`` set from ``slice_``; ``fill_value_default`` is
        ``nodata`` if supplied, else 0.

    Raises:
        ValueError: If no catalog rows match the slice, or if
            ``merge_method`` is not one of the valid rasterio modes.
        TypeError: If the catalog's backend tag is not ``"raster"``.
    """
    if merge_method not in _VALID_MERGE_METHODS:
        raise ValueError(
            f"merge_method must be one of {_VALID_MERGE_METHODS}; got {merge_method!r}"
        )
    if catalog.backend != "raster":
        raise TypeError(
            f"load_raster requires a raster-backend catalog; got {catalog.backend!r}"
        )
    filtered = catalog.query(slice_)
    if len(filtered) == 0:
        raise ValueError("load_raster: no catalog rows match the slice")

    resampling = resampling or Resampling.bilinear
    sources = []
    handles = []
    try:
        for fp in filtered.gdf["filepath"].tolist():
            src = rasterio.open(fp)
            handles.append(src)
            sources.append(src)
        target_resolution = slice_.resolution
        merged, transform = rio_merge(
            sources,
            bounds=slice_.bounds,
            res=target_resolution,
            indexes=list(band_indexes) if band_indexes is not None else None,
            method=merge_method,
            nodata=nodata,
            resampling=resampling,
            dst_path=None,
        )
    finally:
        for h in handles:
            h.close()

    return GeoTensor(
        values=merged,
        transform=transform,
        crs=slice_.crs,
        fill_value_default=nodata if nodata is not None else 0,
    )


def load_raster_timeseries(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Any | None = None,
    nodata: float | None = None,
) -> GeoTensor:
    """Stack daily mosaics across the slice's interval into ``(time, b, h, w)``.

    For each distinct day with matching rows in ``slice_.interval``,
    runs `load_raster` for that day's sub-slice and stacks the results
    along a new leading time axis. Days without coverage are silently
    dropped — the time axis is the *observed* day count, not a dense
    calendar. The transform / CRS come from the last successful day's
    load (same for every day since the slice is fixed).

    Args:
        catalog: A raster-backend catalog with at least one row in
            ``slice_.interval``.
        slice_: Window to read for each day. The same ``bounds`` /
            ``resolution`` / ``crs`` apply to every time step.
        band_indexes: Forwarded to per-day `load_raster`; see that
            function for semantics.
        resampling: Forwarded to per-day `load_raster`.
        nodata: Forwarded to per-day `load_raster`.

    Returns:
        A `GeoTensor` of shape ``(T, bands, H, W)`` where ``T`` is the
        number of distinct days with coverage.

    Raises:
        ValueError: If no catalog rows match the slice, or if every
            matching day failed to produce a usable mosaic.
    """
    filtered = catalog.query(slice_)
    if len(filtered) == 0:
        raise ValueError("load_raster_timeseries: no catalog rows match the slice")

    days = sorted({i.left.floor("D") for i in filtered.gdf.index})
    stacks: list[np.ndarray] = []
    last: GeoTensor | None = None
    for day in days:
        day_interval = pd.Interval(
            day,
            day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
            closed="both",
        )
        day_slice = dataclasses.replace(slice_, interval=day_interval)
        try:
            day_tensor = load_raster(
                catalog,
                day_slice,
                band_indexes=band_indexes,
                resampling=resampling,
                nodata=nodata,
            )
        except ValueError:
            continue
        stacks.append(day_tensor.values)
        last = day_tensor

    if last is None:
        raise ValueError("load_raster_timeseries: no usable days in interval")
    stacked = np.stack(stacks, axis=0)
    return GeoTensor(
        values=stacked,
        transform=last.transform,
        crs=last.crs,
        fill_value_default=last.fill_value_default,
    )
