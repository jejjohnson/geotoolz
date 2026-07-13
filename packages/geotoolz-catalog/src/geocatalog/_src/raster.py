"""Raster catalog builder + loaders.

`build_raster_catalog` reads each file's bounds via ``rasterio`` (lazy
via `WarpedVRT`) and assembles them into a `InMemoryGeoCatalog`.
`load_raster` opens the matching rows for a `GeoSlice` and mosaics
them into a single `GeoTensor`.

This is the bedrock backend; xarray and vector builders mirror its
shape.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
from georeader.geotensor import GeoTensor
from loguru import logger as log
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT

from geocatalog._src.geoslice import GeoSlice
from geocatalog._src.io import _close_resolved_uri, _resolve_uri, _uri_name
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.retry import retry_transient_io


if TYPE_CHECKING:
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog


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
    retries: int = 3,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    resolved = _resolve_uri(filepath, storage_options=storage_options)
    try:
        with retry_transient_io(rasterio.open, resolved, retries=retries) as src:
            if target_crs is None:
                crs = src.crs
                bounds = src.bounds
            else:
                with WarpedVRT(src, crs=target_crs) as vrt:
                    crs = vrt.crs
                    bounds = vrt.bounds
    finally:
        _close_resolved_uri(resolved)

    polygon = shapely.geometry.box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    if filename_regex is None:
        # No date parsing — files are treated as time-invariant. Use a
        # narrow sentinel interval that won't dominate IntervalIndex
        # logs the way Timestamp.min/max would (§7.2 of the design plan).
        start = pd.Timestamp("1900-01-01")
        end = pd.Timestamp("2100-01-01")
    else:
        match = filename_regex.search(_uri_name(filepath))
        if match is None:
            log.warning("Skipping {}: filename does not match regex", filepath)
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


_Concurrency = Literal["sequential", "async"]


async def _filepath_to_row_async(
    filepath: str | Path,
    *,
    filename_regex: re.Pattern[str] | None,
    date_format: str,
    target_crs: Any | None,
    storage_options: dict[str, Any] | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Async wrapper around :func:`_filepath_to_row` for ``concurrency="async"``.

    Runs the sync extractor in a worker thread under a fan-out cap.
    For remote URIs each thread spends almost all its time blocked on
    network I/O, so a moderate ``Semaphore`` limit (default 8) gives
    a wall-clock win without saturating the rasterio thread pool.
    """
    async with semaphore:
        return await asyncio.to_thread(
            _filepath_to_row,
            filepath,
            filename_regex=filename_regex,
            date_format=date_format,
            target_crs=target_crs,
            storage_options=storage_options,
        )


async def _extract_rows_async(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: re.Pattern[str] | None,
    date_format: str,
    target_crs: Any | None,
    storage_options: dict[str, Any] | None,
    max_concurrent: int,
) -> list[dict[str, Any]]:
    """Gather row extraction across ``filepaths`` with a fan-out cap.

    Returns one row per input file in completion order. Rows whose
    filename doesn't match ``filename_regex`` come back as ``None`` and
    are filtered out here (matching the sequential path's semantics).
    Transient I/O failures *are not* swallowed — ``retry_transient_io``
    inside ``_filepath_to_row`` re-raises after exhausting its retry
    budget, and ``asyncio.gather`` propagates the first exception.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _filepath_to_row_async(
            fp,
            filename_regex=filename_regex,
            date_format=date_format,
            target_crs=target_crs,
            storage_options=storage_options,
            semaphore=semaphore,
        )
        for fp in filepaths
    ]
    results = await asyncio.gather(*tasks)
    return [row for row in results if row is not None]


def _run_coroutine_safely(coro: Any) -> Any:
    """Run ``coro`` to completion regardless of whether a loop is running.

    The catalog builder's public surface is sync (returns an
    ``InMemoryGeoCatalog``), so ``concurrency="async"`` needs a way to
    drive its coroutine even when the caller is already inside a
    running event loop — Jupyter, a FastAPI request handler,
    ``pytest-asyncio``, etc. Calling :func:`asyncio.run` from inside a
    running loop raises ``RuntimeError``, which would make the new
    mode unusable in interactive contexts.

    Strategy:

    1. If no loop is running on the calling thread, dispatch to
       :func:`asyncio.run` — the simple, default path.
    2. If a loop *is* running, spin up a worker thread with its own
       event loop and run the coroutine there via
       :meth:`asyncio.new_event_loop().run_until_complete`. The
       calling thread blocks on ``.join()`` so the sync return type is
       preserved.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop on this thread — safe to use asyncio.run.
        return asyncio.run(coro)

    # A loop is already running on this thread. Run the coroutine on a
    # helper thread with its own loop so we don't try to nest.
    import threading

    result_box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            result_box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]


def build_raster_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str | None = None,
    date_format: str = "%Y%m%d",
    target_crs: Any | None = None,
    backend: Literal["memory", "duckdb"] = "memory",
    out_path: str | Path | None = None,
    write_bbox: bool = True,
    sort_by: tuple[str, ...] | None = ("start_time", "geometry_hilbert"),
    partition_by: tuple[str, ...] | None = None,
    batch_size: int = 10_000,
    n_workers: int = 1,
    ordered: bool = False,
    storage_options: dict[str, Any] | None = None,
    concurrency: _Concurrency = "sequential",
    max_concurrent: int = 8,
) -> InMemoryGeoCatalog | DuckDBGeoCatalog:
    """Build a raster catalog — in-memory (default) or streamed to GeoParquet.

    For each input file, opens it with `rasterio`, extracts its bounds
    (optionally projected through a lazy `WarpedVRT` if ``target_crs``
    is set so no pixels are read), and parses the time interval from
    the filename via ``filename_regex``. The result is a catalog
    queryable in milliseconds even over tens of thousands of files —
    `rasterio` builds the spatial R-tree lazily on first access.

    Files whose filenames don't match the regex are *skipped with a
    warning*, not raised — so a heterogeneous directory partially
    populated by one product still produces a usable catalog.

    Two backends:

    - ``backend="memory"`` (default): collects rows into a
      `gpd.GeoDataFrame` and returns an `InMemoryGeoCatalog`. Peak RAM is
      ``O(n_rows)``. Good up to ~10⁵ files.
    - ``backend="duckdb"``: streams rows through a `pyarrow.parquet.ParquetWriter`
      directly to ``out_path``, then sorts via DuckDB ``(start_time,
      ST_Hilbert(ST_Centroid(geometry)))`` for row-group pruning. Peak RAM
      is ``O(batch_size)``. Scales to 10⁶+ files. Returns a
      `DuckDBGeoCatalog` opened on the freshly written artifact. Requires
      the ``[duckdb]`` extra.

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
            keeps each file's native CRS in the memory backend; for the
            duckdb backend ``None`` is upgraded to ``"EPSG:4326"`` —
            the canonical wire format the design prescribes for shared
            GeoParquet artifacts (§sharp-edges of the design plan).
        backend: ``"memory"`` for the existing in-RAM path,
            ``"duckdb"`` for the streamed GeoParquet path.
        out_path: Destination GeoParquet path. Required when
            ``backend="duckdb"``. Ignored when ``backend="memory"``.
        write_bbox: Emit the GeoParquet 1.1 per-row ``bbox`` covering
            struct (used for predicate pushdown). Default True. Only
            consulted when ``backend="duckdb"``.
        sort_by: Sort keys for the post-write DuckDB rewrite. Each
            plain column name passes through; the literal token
            ``"geometry_hilbert"`` expands to
            ``ST_Hilbert(ST_Centroid(geometry))``. ``None`` skips the
            rewrite and leaves rows in extraction order. Only consulted
            when ``backend="duckdb"``.
        partition_by: Optional Hive partition columns for directory
            output when ``backend="duckdb"``. ``"year"``, ``"month"``,
            and ``"day"`` are derived from ``start_time``.
        batch_size: Rows per Arrow record batch in the streaming
            writer. Default 10 000. Only consulted when
            ``backend="duckdb"``.
        n_workers: Process-pool size for per-file metadata extraction.
            ``1`` runs sequentially in-process. ``>1`` spawns a
            ``ProcessPoolExecutor`` (``spawn`` start method) feeding
            the single in-process writer.
        ordered: With ``backend="duckdb"`` and ``n_workers>1``, preserve
            input row order instead of completion order. Useful for
            reproducible artifacts when ``sort_by=None``. A slow input
            earlier in the queue stalls every subsequent yield and can
            temporarily reduce parallelism (workers may sit idle waiting
            on the next-in-line future). Prefer ``ordered=False`` for
            skewed workloads and sort post-hoc if you need a stable byte
            layout.
        concurrency: Extraction strategy for the ``backend="memory"``
            branch. ``"sequential"`` (default) extracts rows one at a
            time on the calling thread — the historical behaviour, no
            changes for existing callers. ``"async"`` fans extraction
            out via ``asyncio.gather`` + ``asyncio.to_thread`` so I/O
            on independent files overlaps; meaningful win when reading
            from a remote bucket (a 30-file build over WAN typically
            drops from O(n_files * RTT) to O((n_files / max_concurrent)
            * RTT)). Ignored for ``backend="duckdb"``, which already
            has ``n_workers`` for the same purpose.
        max_concurrent: Maximum in-flight file extractions when
            ``concurrency="async"``. Default 8 — a sweet spot between
            connection-reuse benefits and saturating the
            ``ThreadPoolExecutor`` that ``asyncio.to_thread`` shares
            across the process.

    Returns:
        ``InMemoryGeoCatalog`` for ``backend="memory"``, otherwise a
        ``DuckDBGeoCatalog`` opened on ``out_path``.

    Raises:
        ValueError: No files matched (or `out_path` missing in the
            duckdb branch).
    """
    if backend not in ("memory", "duckdb"):
        raise ValueError(
            f"build_raster_catalog: backend must be 'memory' or 'duckdb'; "
            f"got {backend!r}"
        )
    if backend == "duckdb":
        if out_path is None:
            raise ValueError("build_raster_catalog(backend='duckdb') requires out_path")
        return _build_raster_catalog_duckdb(
            filepaths,
            filename_regex=filename_regex,
            date_format=date_format,
            target_crs=target_crs,
            out_path=out_path,
            write_bbox=write_bbox,
            sort_by=sort_by,
            partition_by=partition_by,
            batch_size=batch_size,
            n_workers=n_workers,
            ordered=ordered,
            storage_options=storage_options,
        )

    if concurrency not in ("sequential", "async"):
        raise ValueError(
            f"build_raster_catalog: concurrency must be 'sequential' or 'async'; "
            f"got {concurrency!r}"
        )
    if max_concurrent < 1:
        raise ValueError(
            f"build_raster_catalog: max_concurrent must be >= 1; got {max_concurrent}"
        )

    pattern = re.compile(filename_regex) if filename_regex is not None else None
    rows: list[dict[str, Any]]
    if concurrency == "async":
        rows = _run_coroutine_safely(
            _extract_rows_async(
                filepaths,
                filename_regex=pattern,
                date_format=date_format,
                target_crs=target_crs,
                storage_options=storage_options,
                max_concurrent=max_concurrent,
            )
        )
    else:
        rows = []
        for fp in filepaths:
            row = _filepath_to_row(
                fp,
                filename_regex=pattern,
                date_format=date_format,
                target_crs=target_crs,
                storage_options=storage_options,
            )
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError("build_raster_catalog: no files matched the regex")

    crs_value = target_crs if target_crs is not None else rows[0]["crs"]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_value)
    return InMemoryGeoCatalog(gdf, backend="raster")


def _build_raster_catalog_duckdb(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str | None,
    date_format: str,
    target_crs: Any | None,
    out_path: str | Path,
    write_bbox: bool,
    sort_by: tuple[str, ...] | None,
    partition_by: tuple[str, ...] | None,
    batch_size: int,
    n_workers: int,
    ordered: bool,
    storage_options: dict[str, Any] | None,
) -> DuckDBGeoCatalog:
    """Streaming-write branch for `build_raster_catalog`.

    Canonicalises CRS to EPSG:4326 by default (design §sharp-edges line
    588), assembles a picklable extractor via `functools.partial`, and
    delegates to `stream_build_duckdb`.
    """
    from geocatalog._src.streaming import stream_build_duckdb

    if target_crs is None:
        target_crs = "EPSG:4326"
        log.info(
            "build_raster_catalog(backend='duckdb'): target_crs=None → "
            "canonicalising footprints to EPSG:4326 (design §sharp-edges)."
        )
    pattern = re.compile(filename_regex) if filename_regex is not None else None
    extract_fn = functools.partial(
        _filepath_to_row,
        filename_regex=pattern,
        date_format=date_format,
        target_crs=target_crs,
        storage_options=storage_options,
    )
    return stream_build_duckdb(
        filepaths,
        extract_fn,
        out_path=out_path,
        crs=target_crs,
        backend="raster",
        write_bbox=write_bbox,
        sort_by=sort_by,
        partition_by=partition_by,
        batch_size=batch_size,
        n_workers=n_workers,
        ordered=ordered,
    )


def load_raster(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Any | None = None,
    merge_method: _RasterMergeMethod = "last",
    nodata: float | None = None,
    retries: int = 3,
    max_open_workers: int = 8,
    storage_options: dict[str, Any] | None = None,
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
        retries: Number of retries for transient remote I/O failures.
            ``0`` disables retry/backoff.
        max_open_workers: Thread-pool width for the file-*open* phase.
            Remote (S3/HTTP) opens are latency-bound metadata round
            trips and rasterio releases the GIL during them, so opening
            concurrently cuts the open phase from ``N x latency`` to
            ``ceil(N / workers) x latency``. The effective width is
            ``min(max_open_workers, n_files)``; ``1`` (or ``0``) keeps
            the serial behaviour. The merge phase is unchanged.
        storage_options: Options forwarded to fsspec for cloud/HTTP URIs
            (e.g. ``{"anon": True}`` for public S3). ``None`` uses fsspec
            defaults / GDAL's native VSI handling for ``s3://``-style paths.

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
    filepaths = filtered.gdf["filepath"].tolist()

    def _open_one(fp: str) -> tuple[Any, Any]:
        resolved = _resolve_uri(fp, storage_options=storage_options)
        try:
            src = retry_transient_io(rasterio.open, resolved, retries=retries)
        except BaseException:
            _close_resolved_uri(resolved)
            raise
        return resolved, src

    handles = []
    resolved_handles = []
    try:
        n_workers = max(1, min(max_open_workers, len(filepaths)))
        if n_workers == 1:
            for fp in filepaths:
                resolved, src = _open_one(fp)
                resolved_handles.append(resolved)
                handles.append(src)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit in row order and consume in the same order so
                # order-sensitive merge methods ("first" / "last") see
                # the same source sequence as the serial path. Track
                # every successfully opened handle even when an earlier
                # file failed, so the `finally` below closes them all.
                futures = [executor.submit(_open_one, fp) for fp in filepaths]
                first_exc: BaseException | None = None
                for future in futures:
                    try:
                        resolved, src = future.result()
                    except BaseException as exc:
                        if first_exc is None:
                            first_exc = exc
                        continue
                    resolved_handles.append(resolved)
                    handles.append(src)
                if first_exc is not None:
                    raise first_exc
        sources = list(handles)
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
        for h in resolved_handles:
            _close_resolved_uri(h)

    return GeoTensor(
        values=merged,
        transform=transform,
        crs=slice_.crs,
        fill_value_default=nodata if nodata is not None else 0,
    )


async def aload_raster(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Any | None = None,
    merge_method: _RasterMergeMethod = "last",
    nodata: float | None = None,
    retries: int = 3,
    concurrency: int = 8,
    storage_options: dict[str, Any] | None = None,
) -> GeoTensor:
    """Async mirror of `load_raster` for event-loop consumers.

    Runs `load_raster` in a worker thread via `asyncio.to_thread` so the
    event loop stays responsive during rasterio I/O (which releases the
    GIL), with the file-open phase parallelised to ``concurrency``
    threads inside the worker. Same arguments and return value as
    `load_raster`; ``concurrency`` maps to ``max_open_workers``.
    """
    return await asyncio.to_thread(
        functools.partial(
            load_raster,
            catalog,
            slice_,
            band_indexes=band_indexes,
            resampling=resampling,
            merge_method=merge_method,
            nodata=nodata,
            retries=retries,
            max_open_workers=concurrency,
            storage_options=storage_options,
        )
    )


def load_raster_timeseries(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    band_indexes: Sequence[int] | None = None,
    resampling: Any | None = None,
    nodata: float | None = None,
    n_workers: int = 4,
    on_missing_day: Literal["skip", "raise"] = "skip",
    storage_options: dict[str, Any] | None = None,
) -> GeoTensor:
    """Stack daily mosaics across the slice's interval into ``(time, b, h, w)``.

    For each distinct day with matching rows in ``slice_.interval``,
    runs `load_raster` for that day's sub-slice and stacks the results
    along a new leading time axis sorted in chronological order. By
    default, days whose per-day load raises `ValueError` are dropped —
    the time axis is the *observed* day count, not a dense calendar.
    The transform / CRS come from the chronologically last successful
    day's load (same for every day since the slice is fixed).

    Args:
        catalog: A raster-backend catalog with at least one row in
            ``slice_.interval``.
        slice_: Window to read for each day. The same ``bounds`` /
            ``resolution`` / ``crs`` apply to every time step.
        band_indexes: Forwarded to per-day `load_raster`; see that
            function for semantics.
        resampling: Forwarded to per-day `load_raster`.
        nodata: Forwarded to per-day `load_raster`.
        n_workers: Thread-pool size for per-day `load_raster` calls.
            ``1`` preserves serial execution.
        on_missing_day: ``"skip"`` preserves the historical behavior of
            dropping days whose per-day load raises `ValueError`; ``"raise"``
            propagates the first such error.

    Returns:
        A `GeoTensor` of shape ``(T, bands, H, W)`` where ``T`` is the
        number of distinct days with coverage.

    Raises:
        ValueError: If no catalog rows match the slice, or if every
            matching day failed to produce a usable mosaic.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1; got {n_workers!r}")
    if on_missing_day not in ("skip", "raise"):
        raise ValueError(
            f"on_missing_day must be one of {{'skip', 'raise'}}; got {on_missing_day!r}"
        )

    filtered = catalog.query(slice_)
    if len(filtered) == 0:
        raise ValueError("load_raster_timeseries: no catalog rows match the slice")

    days = sorted({i.left.floor("D") for i in filtered.gdf.index})

    def load_day(day: pd.Timestamp) -> tuple[pd.Timestamp, GeoTensor] | None:
        day_interval = pd.Interval(
            day,
            day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
            closed="both",
        )
        day_slice = dataclasses.replace(slice_, interval=day_interval)
        try:
            return day, load_raster(
                catalog,
                day_slice,
                band_indexes=band_indexes,
                resampling=resampling,
                nodata=nodata,
                storage_options=storage_options,
            )
        except ValueError:
            if on_missing_day == "skip":
                return None
            raise

    results: list[tuple[pd.Timestamp, GeoTensor]] = []
    if n_workers == 1:
        for day in days:
            result = load_day(day)
            if result is not None:
                results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit in day order and consume results in the same order so
            # that, with on_missing_day="raise", the earliest failing day
            # raises deterministically regardless of completion order.
            futures = [executor.submit(load_day, day) for day in days]
            try:
                for future in futures:
                    result = future.result()
                    if result is not None:
                        results.append(result)
            except BaseException:
                # Cancel any pending work before propagating so large
                # intervals do not block on already-submitted tasks.
                for pending in futures:
                    pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    if not results:
        raise ValueError("load_raster_timeseries: no usable days in interval")

    ordered = [tensor for _, tensor in sorted(results, key=lambda item: item[0])]
    last = ordered[-1]
    stacked = np.stack([tensor.values for tensor in ordered], axis=0)
    return GeoTensor(
        values=stacked,
        transform=last.transform,
        crs=last.crs,
        fill_value_default=last.fill_value_default,
    )
