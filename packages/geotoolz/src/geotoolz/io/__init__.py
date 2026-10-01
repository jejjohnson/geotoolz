"""Reader and writer operators for geospatial rasters."""

from __future__ import annotations

from geotoolz.io._src.array import (
    affine_from_geotransform,
    fill_value_from_attrs,
    read_indexes,
    select_indexes,
)
from geotoolz.io._src.errors import GeoToolzIOError
from geotoolz.io._src.operators import (
    LoadFromEE,
    LoadFromSTAC,
    ReadBounds,
    ReadCenterCoords,
    ReadHDF,
    ReadNetCDF,
    ReadPolygon,
    ReadReprojectLike,
    ReadTile,
    ReadToCRS,
    ReadWindow,
    SinkOperator,
    SourceOperator,
    WriteCOG,
    WriteGeoTIFF,
    WriteZarr,
)


__all__ = [
    "GeoToolzIOError",
    "LoadFromEE",
    "LoadFromSTAC",
    "ReadBounds",
    "ReadCenterCoords",
    "ReadHDF",
    "ReadNetCDF",
    "ReadPolygon",
    "ReadReprojectLike",
    "ReadTile",
    "ReadToCRS",
    "ReadWindow",
    "SinkOperator",
    "SourceOperator",
    "WriteCOG",
    "WriteGeoTIFF",
    "WriteZarr",
    "affine_from_geotransform",
    "fill_value_from_attrs",
    "read_indexes",
    "select_indexes",
]
