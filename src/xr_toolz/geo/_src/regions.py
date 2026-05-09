"""Named geographic region specifications backed by ``regionmask``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regionmask
from shapely.geometry import MultiPolygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass(frozen=True)
class RegionSpec:
    """Named region pairing ``regionmask`` geometry with viz metadata."""

    id: str
    display_name: str
    regions: regionmask.Regions
    projection: str | None = None


def bbox_region(
    *,
    id: str,
    name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> regionmask.Regions:
    """Construct a ``regionmask.Regions`` from a rectangular bounding box."""
    _validate_id(id)
    _validate_bounds(lat_min=lat_min, lat_max=lat_max)
    if lon_min <= lon_max:
        poly = box(lon_min, lat_min, lon_max, lat_max)
    else:
        poly = MultiPolygon(
            [
                box(lon_min, lat_min, 180.0, lat_max),
                box(-180.0, lat_min, lon_max, lat_max),
            ]
        )
    return regionmask.Regions([poly], names=[name], abbrevs=[id], name=id)


def custom_region(
    *,
    id: str,
    display_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    projection: str | None = "PlateCarree",
) -> RegionSpec:
    """Build a named ``RegionSpec`` from a rectangular bounding box."""
    _validate_id(id)
    _validate_bounds(lat_min=lat_min, lat_max=lat_max)
    return RegionSpec(
        id=id,
        display_name=display_name,
        regions=bbox_region(
            id=id,
            name=display_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        ),
        projection=projection,
    )


def polygon_from_geojson(
    data: dict[str, Any] | str | Path,
    *,
    name: str = "custom",
) -> regionmask.Regions:
    """Construct ``Regions`` from a GeoJSON Feature, Polygon, or MultiPolygon."""
    geojson = _load_geojson(data)
    geometry = _geometry_from_geojson(geojson)
    return regionmask.Regions([geometry], names=[name], abbrevs=[name], name=name)


def _validate_id(id: str) -> None:
    if not id:
        raise ValueError("region id must be non-empty.")


def _validate_bounds(*, lat_min: float, lat_max: float) -> None:
    if lat_min >= lat_max:
        raise ValueError("lat_min must be less than lat_max.")


REGIONS: dict[str, RegionSpec] = {
    "global": custom_region(
        id="global",
        display_name="Global",
        lat_min=-90.0,
        lat_max=90.0,
        lon_min=-180.0,
        lon_max=180.0,
        projection="Robinson",
    ),
    "north_atlantic": custom_region(
        id="north_atlantic",
        display_name="North Atlantic",
        lat_min=10.0,
        lat_max=65.0,
        lon_min=-80.0,
        lon_max=0.0,
    ),
    "gulf_stream": custom_region(
        id="gulf_stream",
        display_name="Gulf Stream",
        lat_min=30.0,
        lat_max=45.0,
        lon_min=-80.0,
        lon_max=-50.0,
    ),
    "kuroshio": custom_region(
        id="kuroshio",
        display_name="Kuroshio",
        lat_min=25.0,
        lat_max=45.0,
        lon_min=130.0,
        lon_max=180.0,
    ),
    "mediterranean": custom_region(
        id="mediterranean",
        display_name="Mediterranean",
        lat_min=30.0,
        lat_max=46.0,
        lon_min=-6.0,
        lon_max=36.0,
    ),
    "ibi": custom_region(
        id="ibi",
        display_name="Iberia-Biscay-Ireland (IBI)",
        # Bounds mirror the Mercator/CMEMS IBI domain extent from OB-1.3.
        lat_min=26.17,
        lat_max=56.08,
        lon_min=-19.08,
        lon_max=5.08,
    ),
}


def resolve_region(region: str | RegionSpec) -> RegionSpec:
    """Resolve a registered region id or pass through a ``RegionSpec``."""
    if isinstance(region, RegionSpec):
        return region
    try:
        return REGIONS[region]
    except KeyError as exc:
        raise KeyError(
            f"Unknown region {region!r}. Available regions: {sorted(REGIONS)}."
        ) from exc


def region_to_dict(region: RegionSpec | regionmask.Regions) -> dict[str, Any]:
    """Serialize a ``RegionSpec`` or raw ``Regions`` to a JSON-safe dict."""
    if isinstance(region, RegionSpec):
        spec = region
    else:
        name = region.name or "custom"
        spec = RegionSpec(id=name, display_name=name, regions=region)
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "projection": spec.projection,
        "regions": _regions_to_dict(spec.regions),
    }


def region_from_dict(data: dict[str, Any]) -> RegionSpec:
    """Deserialize a ``RegionSpec`` from native or bbox-only JSON data."""
    if "bounds" in data:
        bounds = data["bounds"]
        return custom_region(
            id=data["id"],
            display_name=data["display_name"],
            lat_min=bounds["lat_min"],
            lat_max=bounds["lat_max"],
            lon_min=bounds["lon_min"],
            lon_max=bounds["lon_max"],
            projection=data.get("projection"),
        )

    regions_data = data["regions"]
    regions = regionmask.Regions(
        [_geometry_from_geojson(geom) for geom in regions_data["geometries"]],
        numbers=regions_data.get("numbers"),
        names=regions_data["names"],
        abbrevs=regions_data["abbrevs"],
        name=regions_data.get("name", data["id"]),
    )
    return RegionSpec(
        id=data["id"],
        display_name=data["display_name"],
        regions=regions,
        projection=data.get("projection"),
    )


def load_region_file(path: str | Path) -> RegionSpec:
    """Load a JSON-encoded ``RegionSpec`` from disk."""
    with Path(path).open() as f:
        return region_from_dict(json.load(f))


def _load_geojson(data: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, Path):
        return _load_json_file(data)
    text = str(data)
    if text.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Could not parse GeoJSON string.") from exc
        # GeoJSON Features/Polygons are JSON objects; reject top-level
        # arrays so the downstream `.get("type")` doesn't AttributeError.
        if not isinstance(parsed, dict):
            raise ValueError(
                "GeoJSON must be a JSON object (Feature, Polygon, "
                f"MultiPolygon, etc.); got top-level {type(parsed).__name__}."
            )
        return parsed
    return _load_json_file(Path(text))


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"GeoJSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse GeoJSON file: {path}") from exc


def _geometry_from_geojson(data: dict[str, Any]) -> BaseGeometry:
    kind = data.get("type")
    if kind == "Feature":
        if "geometry" not in data:
            raise ValueError("GeoJSON Feature must include a 'geometry' field.")
        return shape(data["geometry"])
    if kind == "FeatureCollection":
        geometries = []
        for index, feature in enumerate(data["features"]):
            if "geometry" not in feature:
                raise ValueError(
                    "GeoJSON FeatureCollection feature "
                    f"at index {index} must include a 'geometry' field."
                )
            geometries.append(shape(feature["geometry"]))
        return unary_union(geometries)
    if kind not in {"Polygon", "MultiPolygon", "GeometryCollection"}:
        raise ValueError(
            f"Unsupported GeoJSON geometry type {kind!r}; expected Feature, "
            "FeatureCollection, Polygon, MultiPolygon, or GeometryCollection."
        )
    return shape(data)


def _regions_to_dict(regions: regionmask.Regions) -> dict[str, Any]:
    return {
        "name": regions.name,
        "numbers": list(regions.numbers),
        "names": list(regions.names),
        "abbrevs": list(regions.abbrevs),
        "geometries": [mapping(poly) for poly in regions.polygons],
    }


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
