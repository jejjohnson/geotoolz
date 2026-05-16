"""Carrier-aware mask operators."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.request import urlretrieve
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.geometry.base
from georeader import rasterize

from geotoolz.cloud._src.array import apply_mask
from geotoolz.core import Operator
from geotoolz.mask._src.array import (
    altitude_mask,
    buffer_mask,
    clean_mask,
    close_mask,
    combine_masks,
    dilate_mask,
    distance_mask,
    erode_mask,
    invert_mask,
    open_mask,
    remove_small_holes,
    remove_small_objects,
    slope_mask,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


_NATURAL_EARTH_URLS = {
    "land": "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
    "ocean": "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_ocean.zip",
    "countries": "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
}


class PolygonMask(Operator):
    """Rasterize a shapely geometry or GeoDataFrame into a boolean mask."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
        crs: str | None = None,
        all_touched: bool = False,
        inside: bool = True,
    ) -> None:
        self.geometry = geometry
        self.crs = crs
        self.all_touched = all_touched
        self.inside = inside

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mask = _rasterize_geometry_like(
            self.geometry,
            gt,
            crs=self.crs,
            all_touched=self.all_touched,
        )
        if not self.inside:
            mask = ~mask
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": _geometry_config(self.geometry),
            "crs": self.crs,
            "all_touched": self.all_touched,
            "inside": self.inside,
        }


class BBoxMask(PolygonMask):
    """Rasterize an axis-aligned bounding box into a boolean mask."""

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        crs: str | None = None,
        inside: bool = True,
    ) -> None:
        if len(bounds) != 4:
            raise ValueError("BBoxMask: `bounds` must contain exactly four values")
        self.bounds = tuple(bounds)
        super().__init__(
            geometry=shapely.geometry.box(*self.bounds), crs=crs, inside=inside
        )

    def get_config(self) -> dict[str, Any]:
        return {"bounds": self.bounds, "crs": self.crs, "inside": self.inside}


class DistanceMask(PolygonMask):
    """Mask pixels within or beyond a distance from a geometry."""

    def __init__(
        self,
        *,
        geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
        distance: float,
        inside: bool = True,
        crs: str | None = None,
        all_touched: bool = True,
    ) -> None:
        self.distance = distance
        super().__init__(
            geometry=geometry, crs=crs, all_touched=all_touched, inside=inside
        )

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        burned = _rasterize_geometry_like(
            self.geometry,
            gt,
            crs=self.crs,
            all_touched=self.all_touched,
        )
        mask = distance_mask(
            burned,
            self.distance,
            inside=self.inside,
            pixel_size=_pixel_size(gt),
        )
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": _geometry_config(self.geometry),
            "distance": self.distance,
            "inside": self.inside,
            "crs": self.crs,
            "all_touched": self.all_touched,
        }


class LandMask(PolygonMask):
    """Rasterize Natural Earth land polygons."""

    def __init__(self, *, source: str = "natural_earth_10m") -> None:
        self.source = source
        super().__init__(geometry=_load_natural_earth("land", source), crs="EPSG:4326")

    def get_config(self) -> dict[str, Any]:
        return {"source": self.source}


class OceanMask(PolygonMask):
    """Rasterize Natural Earth ocean polygons."""

    def __init__(self, *, source: str = "natural_earth_10m") -> None:
        self.source = source
        super().__init__(geometry=_load_natural_earth("ocean", source), crs="EPSG:4326")

    def get_config(self) -> dict[str, Any]:
        return {"source": self.source}


class CountryMask(PolygonMask):
    """Rasterize Natural Earth country polygons selected by ISO A3 code."""

    def __init__(
        self,
        *,
        iso_a3: str | Sequence[str],
        source: str = "natural_earth_10m",
    ) -> None:
        self.iso_a3 = (iso_a3,) if isinstance(iso_a3, str) else tuple(iso_a3)
        self.source = source
        countries = _load_natural_earth("countries", source)
        selected = countries[countries["ISO_A3"].isin(self.iso_a3)]
        if selected.empty:
            raise ValueError(f"CountryMask: no countries found for {list(self.iso_a3)}")
        super().__init__(geometry=selected, crs="EPSG:4326")

    def get_config(self) -> dict[str, Any]:
        iso_a3: str | list[str]
        iso_a3 = self.iso_a3[0] if len(self.iso_a3) == 1 else list(self.iso_a3)
        return {"iso_a3": iso_a3, "source": self.source}


class AltitudeMask(Operator):
    """Build a boolean mask from DEM elevation bounds."""

    def __init__(
        self,
        *,
        dem: GeoTensor,
        min_elev: float | None = None,
        max_elev: float | None = None,
    ) -> None:
        self.dem = dem
        self.min_elev = min_elev
        self.max_elev = max_elev

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mask = altitude_mask(
            np.asarray(self.dem), min_elev=self.min_elev, max_elev=self.max_elev
        )
        _check_spatial_match(mask, gt, "AltitudeMask")
        return gt.array_as_geotensor(mask, fill_value_default=False)

    forbid_in_yaml: ClassVar[bool] = True

    def get_config(self) -> dict[str, Any]:
        return {
            "dem": {"shape": list(self.dem.shape), "dtype": str(self.dem.dtype)},
            "min_elev": self.min_elev,
            "max_elev": self.max_elev,
        }


class SlopeMask(AltitudeMask):
    """Build a boolean mask from DEM slope bounds in degrees."""

    def __init__(
        self,
        *,
        dem: GeoTensor,
        min_slope_deg: float | None = None,
        max_slope_deg: float | None = None,
    ) -> None:
        self.dem = dem
        self.min_slope_deg = min_slope_deg
        self.max_slope_deg = max_slope_deg

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mask = slope_mask(
            np.asarray(self.dem),
            _pixel_size(self.dem),
            min_slope_deg=self.min_slope_deg,
            max_slope_deg=self.max_slope_deg,
        )
        _check_spatial_match(mask, gt, "SlopeMask")
        return gt.array_as_geotensor(mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "dem": {"shape": list(self.dem.shape), "dtype": str(self.dem.dtype)},
            "min_slope_deg": self.min_slope_deg,
            "max_slope_deg": self.max_slope_deg,
        }


class DilateMask(Operator):
    """Dilate a boolean mask over its trailing spatial axes."""

    def __init__(
        self, *, iterations: int = 1, structure: np.ndarray | None = None
    ) -> None:
        self.iterations = iterations
        self.structure = structure

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = dilate_mask(np.asarray(mask), self.iterations, self.structure)
        return _wrap_like(mask, out)

    def get_config(self) -> dict[str, Any]:
        return _morph_config(self.iterations, self.structure)


class ErodeMask(DilateMask):
    """Erode a boolean mask over its trailing spatial axes."""

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = erode_mask(np.asarray(mask), self.iterations, self.structure)
        return _wrap_like(mask, out)


class OpenMask(Operator):
    """Open a boolean mask over its trailing spatial axes."""

    def __init__(self, *, iterations: int = 1) -> None:
        self.iterations = iterations

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = open_mask(np.asarray(mask), self.iterations)
        return _wrap_like(mask, out)

    def get_config(self) -> dict[str, Any]:
        return {"iterations": self.iterations}


class CloseMask(OpenMask):
    """Close a boolean mask over its trailing spatial axes."""

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = close_mask(np.asarray(mask), self.iterations)
        return _wrap_like(mask, out)


class BufferMask(Operator):
    """Radially expand True pixels in a mask."""

    def __init__(self, *, radius: float, unit: str = "pixels") -> None:
        self.radius = radius
        self.unit = unit

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        pixel_size = (
            _pixel_size(mask) if self.unit in {"meter", "meters"} else (1.0, 1.0)
        )
        out = buffer_mask(
            np.asarray(mask), self.radius, unit=self.unit, pixel_size=pixel_size
        )
        return _wrap_like(mask, out)

    def get_config(self) -> dict[str, Any]:
        return {"radius": self.radius, "unit": self.unit}


class RemoveSmallObjects(Operator):
    """Remove connected True components smaller than ``min_size`` pixels."""

    def __init__(self, *, min_size: int) -> None:
        self.min_size = min_size

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = remove_small_objects(np.asarray(mask), self.min_size)
        return _wrap_like(mask, out)

    def get_config(self) -> dict[str, Any]:
        return {"min_size": self.min_size}


class RemoveSmallHoles(Operator):
    """Fill enclosed False components up to ``area_threshold`` pixels."""

    def __init__(self, *, area_threshold: int) -> None:
        self.area_threshold = area_threshold

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = remove_small_holes(np.asarray(mask), self.area_threshold)
        return _wrap_like(mask, out)

    def get_config(self) -> dict[str, Any]:
        return {"area_threshold": self.area_threshold}


class CleanMask(Operator):
    """Remove small objects, fill small holes, then close the mask."""

    def __init__(
        self,
        *,
        min_object_size: int = 25,
        max_hole_size: int = 25,
        close_iter: int = 1,
    ) -> None:
        self.min_object_size = min_object_size
        self.max_hole_size = max_hole_size
        self.close_iter = close_iter

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = clean_mask(
            np.asarray(mask),
            min_object_size=self.min_object_size,
            max_hole_size=self.max_hole_size,
            close_iter=self.close_iter,
        )
        return _wrap_like(mask, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_object_size": self.min_object_size,
            "max_hole_size": self.max_hole_size,
            "close_iter": self.close_iter,
        }


class CombineMasks(Operator):
    """Combine boolean masks with ``or``, ``and``, ``xor``, or unary ``not``."""

    def __init__(self, *, op: str = "or") -> None:
        self.op = op

    def _apply(self, masks: Sequence[GeoTensor | np.ndarray]) -> GeoTensor | np.ndarray:
        out = combine_masks([np.asarray(mask) for mask in masks], self.op)
        return _wrap_like(masks[0], out)

    def get_config(self) -> dict[str, Any]:
        return {"op": self.op}


class InvertMask(Operator):
    """Invert a boolean mask."""

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = invert_mask(np.asarray(mask))
        return _wrap_like(mask, out)


class ApplyMask(Operator):
    """Apply a boolean mask to a GeoTensor, filling True pixels."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        mask: Operator | np.ndarray | Any,
        fill: float = float("nan"),
        invert: bool = False,
    ) -> None:
        self.mask = mask
        self.fill = fill
        self.invert = invert

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mask_arr = np.asarray(
            self.mask(gt) if isinstance(self.mask, Operator) else self.mask
        )
        out = apply_mask(
            np.asarray(gt), mask_arr, fill_value=self.fill, invert=self.invert
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        if isinstance(self.mask, Operator):
            mask_config: Any = {
                "class": type(self.mask).__name__,
                "config": self.mask.get_config(),
            }
        else:
            arr = np.asarray(self.mask)
            mask_config = {
                "type": "ndarray",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        return {"mask": mask_config, "fill": self.fill, "invert": self.invert}


def _rasterize_geometry_like(
    geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
    gt: GeoTensor,
    *,
    crs: str | None,
    all_touched: bool,
) -> np.ndarray:
    if isinstance(geometry, gpd.GeoDataFrame):
        gdf = geometry.copy()
        if gdf.crs is None:
            if crs is None:
                raise ValueError("PolygonMask: GeoDataFrame CRS is missing")
            gdf = gdf.set_crs(crs)
        gdf["__mask__"] = 1
        burned = rasterize.rasterize_geopandas_like(
            gdf, gt, column="__mask__", all_touched=all_touched, return_only_data=True
        )
    else:
        burned = rasterize.rasterize_geometry_like(
            geometry,
            gt,
            value=1,
            dtype=np.uint8,
            crs_geometry=crs,
            fill=0,
            all_touched=all_touched,
            return_only_data=True,
        )
    return np.asarray(burned, dtype=bool)


@cache
def _load_natural_earth(kind: str, source: str) -> gpd.GeoDataFrame:
    """Load and cache Natural Earth vectors.

    Supports ``kind`` values ``"land"``, ``"ocean"``, and ``"countries"``.
    The special ``source="natural_earth_10m"`` downloads the corresponding
    Natural Earth 1:10m zip once into the geotoolz cache directory and reuses
    the extracted shapefile. Any other source is passed to
    ``geopandas.read_file``. The in-process ``@cache`` avoids repeated reads
    across multiple operator instances.
    """
    if source != "natural_earth_10m":
        return gpd.read_file(source)

    cache_dir = _natural_earth_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / Path(_NATURAL_EARTH_URLS[kind]).name
    extract_dir = cache_dir / zip_path.stem
    if not extract_dir.exists():
        if not zip_path.exists():
            try:
                urlretrieve(_NATURAL_EARTH_URLS[kind], zip_path)
            except OSError as exc:
                raise RuntimeError(
                    f"failed to download Natural Earth {kind!r} data from "
                    f"{_NATURAL_EARTH_URLS[kind]}"
                ) from exc
        with ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    shapefiles = list(extract_dir.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"Natural Earth {kind!r} archive did not contain a shapefile"
        )
    shp = shapefiles[0]
    return gpd.read_file(shp)


def _natural_earth_cache_dir() -> Path:
    """Return the on-disk Natural Earth cache directory.

    The location follows ``$XDG_CACHE_HOME/geotoolz/natural_earth`` when
    ``XDG_CACHE_HOME`` is set, otherwise ``~/.cache/geotoolz/natural_earth``.
    In restricted environments where the home directory cannot be resolved,
    it falls back to the system temporary directory.
    """
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root is not None:
        return Path(cache_root) / "geotoolz" / "natural_earth"
    try:
        return Path.home() / ".cache" / "geotoolz" / "natural_earth"
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "geotoolz" / "natural_earth"


def _wrap_like(mask: Any, out: np.ndarray) -> Any:
    if hasattr(mask, "array_as_geotensor"):
        return mask.array_as_geotensor(out, fill_value_default=False)
    return out


def _pixel_size(gt: Any) -> tuple[float, float]:
    transform = gt.transform
    return (abs(float(transform.e)), abs(float(transform.a)))


def _check_spatial_match(mask: np.ndarray, gt: GeoTensor, name: str) -> None:
    if mask.shape[-2:] != gt.shape[-2:]:
        raise ValueError(f"{name}: DEM spatial shape must match the input GeoTensor")


def _morph_config(iterations: int, structure: np.ndarray | None) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "structure": None if structure is None else np.asarray(structure).tolist(),
    }


def _geometry_config(
    geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
) -> dict[str, Any]:
    if isinstance(geometry, gpd.GeoDataFrame):
        return {
            "type": "GeoDataFrame",
            "length": len(geometry),
            "crs": None if geometry.crs is None else str(geometry.crs),
        }
    return {"type": type(geometry).__name__, "bounds": tuple(geometry.bounds)}
