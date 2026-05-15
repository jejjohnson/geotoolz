"""Vector catalog builder + loader — extras-gated via `[vector]`.

Each row's footprint is the vector file's ``total_bounds`` in the
target CRS; loaders rasterise the matching features into a label
`GeoTensor` for one of three ML tasks: ``"semantic_segmentation"``,
``"object_detection"``, ``"instance_segmentation"``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from georeader.geotensor import GeoTensor
from rasterio.features import rasterize

from geotoolz.catalog._src.memory import InMemoryGeoCatalog
from geotoolz.types import GeoSlice


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
) -> dict[str, Any] | None:
    filepath = Path(filepath)
    gdf = (
        gpd.read_file(filepath, layer=layer)
        if layer is not None
        else gpd.read_file(filepath)
    )
    if gdf.empty:
        log.warning("Skipping empty vector file %s", filepath)
        return None
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
            return None
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

    return {
        "filepath": str(filepath),
        "geometry": polygon,
        "start_time": start,
        "end_time": end,
        "layer": layer,
    }


def build_vector_catalog(
    filepaths: Sequence[str | Path],
    *,
    filename_regex: str | None = None,
    date_format: str = "%Y%m%d",
    target_crs: Any | None = None,
    layer: str | int | None = None,
) -> InMemoryGeoCatalog:
    """Build an in-memory catalog from vector files (Shapefile / GeoPackage / GeoJSON).

    For each input file, opens it with `geopandas.read_file`, optionally
    reprojects to ``target_crs``, and records the file's
    ``total_bounds`` polygon as the catalog footprint. Time is parsed
    from the filename via ``filename_regex``, mirroring
    `build_raster_catalog`. Empty files and non-matching filenames are
    skipped with a warning.

    Args:
        filepaths: Files to index. Anything `geopandas.read_file`
            accepts (Shapefile, GeoPackage, GeoJSON, FlatGeobuf, …).
        filename_regex: A regex with a ``(?P<date>...)`` group, or
            ``(?P<start>...)`` + ``(?P<stop>...)`` groups, used to parse
            the time interval out of each filename. ``None`` treats
            every file as time-invariant (sentinel interval).
        date_format: ``strptime`` format for the named date groups.
            Default ``"%Y%m%d"``.
        target_crs: CRS for footprint + storage. ``None`` keeps the
            first observed file's CRS (mismatched files get
            reprojected).
        layer: Layer name or index for multi-layer files (GeoPackage,
            GDB). ``None`` opens the file's default layer.

    Returns:
        An `InMemoryGeoCatalog` with backend ``"vector"`` and one row
        per matching file. Columns: ``filepath``, ``geometry``,
        ``start_time``, ``end_time``, ``layer``.

    Raises:
        ValueError: If no files yielded a row (every file empty or
            unmatched).
    """
    pattern = re.compile(filename_regex) if filename_regex is not None else None
    rows: list[dict[str, Any]] = []
    crs_value = target_crs
    for fp in filepaths:
        row = _vector_row(
            fp,
            filename_regex=pattern,
            date_format=date_format,
            target_crs=target_crs,
            layer=layer,
        )
        if row is not None:
            rows.append(row)
            if crs_value is None:
                # Persist the first observed CRS so the gdf has a uniform one.
                opened = (
                    gpd.read_file(fp, layer=layer)
                    if layer is not None
                    else (gpd.read_file(fp))
                )
                crs_value = opened.crs
    if not rows:
        raise ValueError("build_vector_catalog: no files yielded a row")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_value)
    return InMemoryGeoCatalog(gdf, backend="vector")


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
    for fp, layer in zip(
        filtered.gdf["filepath"], filtered.gdf.get("layer"), strict=False
    ):
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
