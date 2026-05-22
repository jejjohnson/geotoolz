"""Public region registry helpers for :mod:`xrtoolz.geo`."""

from __future__ import annotations

from xrtoolz.geo._src.regions import (
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


__all__ = [
    "REGIONS",
    "RegionSpec",
    "bbox_region",
    "custom_region",
    "load_region_file",
    "polygon_from_geojson",
    "region_from_dict",
    "region_to_dict",
    "resolve_region",
]
