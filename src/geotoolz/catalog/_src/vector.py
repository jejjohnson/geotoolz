"""Vector catalog builder + loader — extras-gated via `[vector]`.

Each row's footprint is the vector file's ``total_bounds`` in the
target CRS; loaders rasterise the matching features into a label
`GeoTensor` for one of three ML tasks: ``"semantic_segmentation"``,
``"object_detection"``, ``"instance_segmentation"``.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from georeader.geotensor import GeoTensor
from rasterio.features import rasterize

from geotoolz.catalog._src.memory import InMemoryGeoCatalog
from geotoolz.types import GeoSlice


if TYPE_CHECKING:
    from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog


log = logging.getLogger(__name__)


_VectorTask = Literal[
    "semantic_segmentation",
    "object_detection",
    "instance_segmentation",
]


def _vector_row(
    filepath: str | Path,
    *,
    filename_regex: re.Pattern[str] | None,
    date_format: str,
    target_crs: Any | None,
    layer: str | int | None,
) -> tuple[dict[str, Any] | None, Any]:
    """Build one catalog row from a vector file.

    Opens the file once, reprojects to ``target_crs`` if supplied, and
    returns ``(row_dict, observed_crs)``. The observed CRS is the
    file's *native* CRS — `build_vector_catalog` uses the first
    observed CRS to anchor the catalog when no ``target_crs`` was
    provided, then reprojects every subsequent file to that anchor.
    """
    filepath = Path(filepath)
    gdf = (
        gpd.read_file(filepath, layer=layer)
        if layer is not None
        else gpd.read_file(filepath)
    )
    if gdf.empty:
        log.warning("Skipping empty vector file %s", filepath)
        return None, None
    observed_crs = gdf.crs
    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)

    if filename_regex is None:
        start = pd.Timestamp("1900-01-01")
        end = pd.Timestamp("2100-01-01")
    else:
        match = filename_regex.search(filepath.name)
        if match is None:
            log.warning("Skipping %s: filename does not match regex", filepath)
            return None, observed_crs
        groups = match.groupdict()
        if "date" in groups:
            ts = pd.to_datetime(groups["date"], format=date_format)
            start = ts.floor("D")
            end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        elif "start" in groups and "stop" in groups:
            t0 = pd.to_datetime(groups["start"], format=date_format)
            t1 = pd.to_datetime(groups["stop"], format=date_format)
            start = t0.floor("D")
            end = t1.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        else:
            raise ValueError(
                f"filename_regex must capture either 'date' or "
                f"'start'+'stop'; got {list(groups.keys())}"
            )

    return (
        {
            "filepath": str(filepath),
            "geometry": polygon,
            "start_time": start,
            "end_time": end,
            "layer": layer,
        },
        observed_crs,
    )


def _vector_row_for_stream(
    filepath: str | Path,
    *,
    filename_regex: re.Pattern[str] | None,
    date_format: str,
    target_crs: Any,
    layer: str | int | None,
) -> dict[str, Any] | None:
    """Picklable streaming extractor for the duckdb branch.

    Wraps `_vector_row` and discards the observed-CRS second return — the
    streaming branch always carries a fixed catalog CRS (EPSG:4326 by
    default), so first-file-wins CRS latching does not apply. Lives at
    module scope so `functools.partial` over it pickles cleanly across
    a ``ProcessPoolExecutor`` (spawn).
    """
    row, _ = _vector_row(
        filepath,
        filename_regex=filename_regex,
        date_format=date_format,
        target_crs=target_crs,
        layer=layer,
    )
    return row


def build_vector_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str | None = None,
    date_format: str = "%Y%m%d",
    target_crs: Any | None = None,
    layer: str | int | None = None,
    backend: Literal["memory", "duckdb"] = "memory",
    out_path: str | Path | None = None,
    write_bbox: bool = True,
    sort_by: tuple[str, ...] | None = ("start_time", "geometry_hilbert"),
    batch_size: int = 10_000,
    n_workers: int = 1,
) -> InMemoryGeoCatalog | DuckDBGeoCatalog:
    """Build a vector catalog — in-memory (default) or streamed to GeoParquet.

    For each input file, opens it with `geopandas.read_file`, optionally
    reprojects to ``target_crs``, and records the file's
    ``total_bounds`` polygon as the catalog footprint. Time is parsed
    from the filename via ``filename_regex``, mirroring
    `build_raster_catalog`. Empty files and non-matching filenames are
    skipped with a warning.

    Backends mirror `build_raster_catalog`:

    - ``backend="memory"`` (default): collects rows into a gdf and
      returns an `InMemoryGeoCatalog`. First-non-empty file's native CRS
      latches the catalog CRS when ``target_crs=None``.
    - ``backend="duckdb"``: streams rows to a GeoParquet at ``out_path``
      and returns a `DuckDBGeoCatalog`. CRS-latching is disabled — every
      file is reprojected to ``target_crs`` (EPSG:4326 by default).
      Requires the ``[duckdb]`` extra.

    Args:
        filepaths: Files to index. Anything `geopandas.read_file`
            accepts (Shapefile, GeoPackage, GeoJSON, FlatGeobuf, …).
        filename_regex: A regex with a ``(?P<date>...)`` group, or
            ``(?P<start>...)`` + ``(?P<stop>...)`` groups, used to parse
            the time interval out of each filename. ``None`` treats
            every file as time-invariant (sentinel interval).
        date_format: ``strptime`` format for the named date groups.
            Default ``"%Y%m%d"``.
        target_crs: CRS for footprint + storage. For ``backend="memory"``,
            ``None`` latches onto the first non-empty file's native CRS
            and reprojects every subsequent file to match. For
            ``backend="duckdb"``, ``None`` is upgraded to ``"EPSG:4326"``.
        layer: Layer name or index for multi-layer files (GeoPackage,
            GDB). ``None`` opens the file's default layer.
        backend: ``"memory"`` for the existing in-RAM path,
            ``"duckdb"`` for the streamed GeoParquet path.
        out_path: Destination GeoParquet path. Required when
            ``backend="duckdb"``.
        write_bbox: Emit the GeoParquet 1.1 covering ``bbox`` struct.
            Only consulted when ``backend="duckdb"``.
        sort_by: Sort keys for the post-write DuckDB rewrite; literal
            ``"geometry_hilbert"`` expands to
            ``ST_Hilbert(ST_Centroid(geometry))``. ``None`` skips the
            rewrite. Only consulted when ``backend="duckdb"``.
        batch_size: Rows per Arrow record batch. Default 10 000.
        n_workers: Process-pool size for per-file extraction. ``1``
            runs sequentially.

    Returns:
        `InMemoryGeoCatalog` for ``backend="memory"``, otherwise a
        `DuckDBGeoCatalog`.

    Raises:
        ValueError: If no files yielded a row, or ``out_path`` missing
            in the duckdb branch.
    """
    if backend not in ("memory", "duckdb"):
        raise ValueError(
            f"build_vector_catalog: backend must be 'memory' or 'duckdb'; "
            f"got {backend!r}"
        )
    if backend == "duckdb":
        if out_path is None:
            raise ValueError("build_vector_catalog(backend='duckdb') requires out_path")
        return _build_vector_catalog_duckdb(
            filepaths,
            filename_regex=filename_regex,
            date_format=date_format,
            target_crs=target_crs,
            layer=layer,
            out_path=out_path,
            write_bbox=write_bbox,
            sort_by=sort_by,
            batch_size=batch_size,
            n_workers=n_workers,
        )

    pattern = re.compile(filename_regex) if filename_regex is not None else None
    rows: list[dict[str, Any]] = []
    # `effective_crs` starts as the user-supplied `target_crs` (possibly
    # None). On the first non-empty file we latch onto its native CRS,
    # then pass that down to subsequent calls so every later file's
    # footprint is reprojected into the catalog's uniform CRS. This is
    # what keeps mixed-CRS archives honest.
    effective_crs = target_crs
    for fp in filepaths:
        row, observed_crs = _vector_row(
            fp,
            filename_regex=pattern,
            date_format=date_format,
            target_crs=effective_crs,
            layer=layer,
        )
        if row is None:
            continue
        rows.append(row)
        if effective_crs is None and observed_crs is not None:
            effective_crs = observed_crs
    if not rows:
        raise ValueError("build_vector_catalog: no files yielded a row")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=effective_crs)
    return InMemoryGeoCatalog(gdf, backend="vector")


def _build_vector_catalog_duckdb(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str | None,
    date_format: str,
    target_crs: Any | None,
    layer: str | int | None,
    out_path: str | Path,
    write_bbox: bool,
    sort_by: tuple[str, ...] | None,
    batch_size: int,
    n_workers: int,
) -> DuckDBGeoCatalog:
    """Streaming-write branch for `build_vector_catalog`.

    Forces EPSG:4326 when `target_crs=None` (no first-file CRS latching
    in the streaming branch — the writer needs a single fixed CRS up
    front).
    """
    from geotoolz.catalog._src.streaming import stream_build_duckdb

    if target_crs is None:
        target_crs = "EPSG:4326"
        log.info(
            "build_vector_catalog(backend='duckdb'): target_crs=None → "
            "canonicalising footprints to EPSG:4326 (design §sharp-edges)."
        )
    pattern = re.compile(filename_regex) if filename_regex is not None else None
    extract_fn = functools.partial(
        _vector_row_for_stream,
        filename_regex=pattern,
        date_format=date_format,
        target_crs=target_crs,
        layer=layer,
    )
    return stream_build_duckdb(
        filepaths,
        extract_fn,
        out_path=out_path,
        crs=target_crs,
        backend="vector",
        write_bbox=write_bbox,
        sort_by=sort_by,
        batch_size=batch_size,
        n_workers=n_workers,
    )


def load_vector(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    task: _VectorTask = "semantic_segmentation",
    label_field: str | None = None,
    burn_value: int | None = None,
    fill: int = 0,
) -> GeoTensor:
    """Rasterise the catalog's vector rows matching ``slice_`` into a `GeoTensor`.

    Filters the catalog with ``catalog.query(slice_)``, opens every
    matching vector file, clips its features to ``slice_.bounds`` (in
    ``slice_.crs``), and burns them into a single-channel int64 raster
    via `rasterio.features.rasterize`. The output's transform / CRS
    come straight from ``slice_``, so the result aligns pixel-for-pixel
    with whatever the matching imagery loader produces for the same
    slice.

    Args:
        catalog: A vector-backend catalog.
        slice_: Window to rasterise into. The result has
            ``shape == slice_.shape`` and ``crs``/``transform`` from
            ``slice_``.
        task: Which ML target shape to produce:

            - ``"semantic_segmentation"``: one channel of class IDs
              read from ``label_field``. With ``label_field=None`` each
              feature burns as ``burn_value`` (default 1).
            - ``"instance_segmentation"``: one integer per feature
              (1..N) — distinct values for distinct objects.
            - ``"object_detection"``: not implemented in v0.1; build
              the bbox tensor yourself from the filtered features.
        label_field: Column in the vector file to read class IDs from
            for ``"semantic_segmentation"``. Must be integer-valued.
            ``None`` falls back to ``burn_value``.
        burn_value: The value burnt in for ``"semantic_segmentation"``
            when ``label_field`` is ``None``. Default 1.
        fill: Background value written outside any feature. Default 0.

    Returns:
        A `GeoTensor` of shape ``(1, H, W)`` with ``dtype=int64``,
        ``transform`` and ``crs`` matching ``slice_``.

    Raises:
        TypeError: If the catalog's backend tag is not ``"vector"``.
        NotImplementedError: For ``task="object_detection"`` (v0.2+).
        ValueError: If no catalog rows match the slice.
    """
    if catalog.backend != "vector":
        raise TypeError(
            f"load_vector requires a vector-backend catalog; got {catalog.backend!r}"
        )
    if task == "object_detection":
        raise NotImplementedError(
            "load_vector(task='object_detection') is v0.2+; build the "
            "bounding-box tensor yourself from the matched features for now."
        )

    filtered = catalog.query(slice_)
    if len(filtered) == 0:
        raise ValueError("load_vector: no catalog rows match the slice")

    height, width = slice_.shape
    transform = slice_.transform
    features = []
    instance_id = 1
    # `layer` is always set by `_vector_row` for catalogs built here, but
    # externally constructed catalogs (or `from_geoparquet` of foreign
    # files) may not carry the column. Fall back to a same-length sequence
    # of `None`s so the zip below never crashes on a missing column.
    if "layer" in filtered.gdf.columns:
        layers: Sequence[Any] = filtered.gdf["layer"].tolist()
    else:
        layers = [None] * len(filtered.gdf)
    for fp, layer in zip(filtered.gdf["filepath"], layers, strict=True):
        sub = gpd.read_file(fp, layer=layer) if layer is not None else gpd.read_file(fp)
        if sub.crs != slice_.crs:
            sub = sub.to_crs(slice_.crs)
        # Clip to the slice bbox.
        xmin, ymin, xmax, ymax = slice_.bounds
        bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)
        sub = sub[sub.intersects(bbox)]
        for _, row in sub.iterrows():
            if task == "semantic_segmentation":
                value = (
                    int(row[label_field])
                    if label_field is not None
                    else (burn_value if burn_value is not None else 1)
                )
            else:  # instance_segmentation
                value = instance_id
                instance_id += 1
            features.append((row.geometry, value))

    if not features:
        raster = np.full((1, height, width), fill, dtype=np.int64)
    else:
        burned = rasterize(
            features,
            out_shape=(height, width),
            transform=transform,
            fill=fill,
            dtype=np.int64,
        )
        raster = burned[np.newaxis, ...]

    return GeoTensor(
        values=raster,
        transform=transform,
        crs=slice_.crs,
        fill_value_default=fill,
    )
