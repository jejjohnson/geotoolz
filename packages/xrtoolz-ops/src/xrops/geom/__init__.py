"""Geometry — CRS, footprints, mosaics, vector/raster bridge, subsetting, regions.

Layer-1 ``Operator`` wrappers live in :mod:`xrops.geom.operators`.
"""

from __future__ import annotations

from xrops.geom._src.crs import (
    LocalFrame,
    assign_crs,
    assign_local_xy,
    calc_latlon,
    get_crs,
    get_dataset_resolution,
    local_frame,
    lonlat_to_xy,
    reproject,
    reproject_match,
    utm_crs_for,
    xy_to_lonlat,
)
from xrops.geom._src.footprint import (
    footprint,
    valid_footprint,
)
from xrops.geom._src.mosaic import (
    spatial_mosaic,
)
from xrops.geom._src.rasterize import (
    rasterize,
    rasterize_like,
)
from xrops.geom._src.regions import (
    REGIONS,
    RegionSpec,
    bbox_region,
    custom_region,
    load_region_file,
    polygon_from_geojson,
    region_from_dict,
    region_to_dict,
    resolve_region,
)
from xrops.geom._src.subset import (
    select_variables,
    subset_bbox,
    subset_time,
    subset_to_region,
    subset_where,
)
from xrops.geom._src.vectorize import (
    transform_polygon,
    vectorize,
)
from xrops.geom.operators import (
    AssignLocalXY,
    Footprint,
    Rasterize,
    ReprojectMatch,
    SelectVariables,
    SpatialMosaic,
    SubsetBBox,
    SubsetTime,
    SubsetToRegion,
    ValidFootprint,
    Vectorize,
)


__all__ = [
    "REGIONS",
    "AssignLocalXY",
    "Footprint",
    "LocalFrame",
    "Rasterize",
    "RegionSpec",
    "ReprojectMatch",
    "SelectVariables",
    "SpatialMosaic",
    "SubsetBBox",
    "SubsetTime",
    "SubsetToRegion",
    "ValidFootprint",
    "Vectorize",
    "assign_crs",
    "assign_local_xy",
    "bbox_region",
    "calc_latlon",
    "custom_region",
    "footprint",
    "get_crs",
    "get_dataset_resolution",
    "load_region_file",
    "local_frame",
    "lonlat_to_xy",
    "polygon_from_geojson",
    "rasterize",
    "rasterize_like",
    "region_from_dict",
    "region_to_dict",
    "reproject",
    "reproject_match",
    "resolve_region",
    "select_variables",
    "spatial_mosaic",
    "subset_bbox",
    "subset_time",
    "subset_to_region",
    "subset_where",
    "transform_polygon",
    "utm_crs_for",
    "valid_footprint",
    "vectorize",
    "xy_to_lonlat",
]
