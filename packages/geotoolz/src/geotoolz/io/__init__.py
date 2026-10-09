"""Reader and writer operators for geospatial rasters."""

from __future__ import annotations

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
]
