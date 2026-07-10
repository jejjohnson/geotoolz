"""Carrier-aware mask operators."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.request import urlopen
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.geometry.base
from georeader import rasterize
from pipekit import Operator

from geotoolz._src.config import jsonable
from geotoolz._src.wrap import wrap_like
from geotoolz.cloud._src.array import apply_mask
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
    """Rasterize a shapely geometry or GeoDataFrame into a boolean mask.

    Geo-dependent: rasterization needs the carrier's transform/CRS, so
    the input must be a georeferenced ``GeoTensor``. The mask matches
    the carrier's spatial grid and is returned as a boolean GeoTensor
    with ``fill_value_default=False``.

    Args:
        geometry: Shapely geometry or ``GeoDataFrame`` to burn in.
        crs: CRS of ``geometry`` when it doesn't carry one itself
            (required for a CRS-less GeoDataFrame).
        all_touched: Burn every pixel touched by the geometry instead
            of only pixels whose center is inside.
        inside: If True (default) the mask is True inside the geometry;
            if False the complement is returned.

    Examples:
        >>> PolygonMask(geometry=box(1.0, 1.0, 3.0, 3.0))(scene)  # doctest: +SKIP
    """

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
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": _geometry_config(self.geometry),
            "crs": self.crs,
            "all_touched": self.all_touched,
            "inside": self.inside,
        }


class BBoxMask(PolygonMask):
    """Rasterize an axis-aligned bounding box into a boolean mask.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (see
    :class:`PolygonMask`).

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` box in ``crs`` coordinates.
        crs: CRS of ``bounds``; defaults to the carrier's CRS.
        inside: If True (default) the mask is True inside the box.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        crs: str | None = None,
        inside: bool = True,
    ) -> None:
        self.bounds = tuple(bounds)
        if len(self.bounds) != 4:
            raise ValueError("BBoxMask: `bounds` must contain exactly four values")
        super().__init__(
            geometry=shapely.geometry.box(*self.bounds), crs=crs, inside=inside
        )

    def get_config(self) -> dict[str, Any]:
        return {"bounds": self.bounds, "crs": self.crs, "inside": self.inside}


class DistanceMask(PolygonMask):
    """Mask pixels within or beyond a distance from a geometry.

    Rasterizes the geometry onto the carrier grid, then thresholds the
    Euclidean distance transform at ``distance`` (CRS units, using the
    carrier pixel size). Geo-dependent: requires a georeferenced
    ``GeoTensor`` input.

    Args:
        geometry: Shapely geometry or ``GeoDataFrame``.
        distance: Maximum distance from the geometry, in CRS units.
        inside: If True (default), True within ``distance`` of the
            geometry; if False, the complement.
        crs: CRS of ``geometry`` when it doesn't carry one itself.
        all_touched: Rasterization rule; see :class:`PolygonMask`.
            Default True so thin geometries are not lost.
    """

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
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": _geometry_config(self.geometry),
            "distance": self.distance,
            "inside": self.inside,
            "crs": self.crs,
            "all_touched": self.all_touched,
        }


class LandMask(PolygonMask):
    """Rasterize Natural Earth land polygons into a boolean mask.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (see
    :class:`PolygonMask`).

    Args:
        source: ``"natural_earth_10m"`` (default) downloads and caches
            the Natural Earth 1:10m land polygons; any other value is a
            local vector file path passed to ``geopandas.read_file``.
    """

    def __init__(self, *, source: str = "natural_earth_10m") -> None:
        self.source = source
        super().__init__(geometry=_load_natural_earth("land", source), crs="EPSG:4326")

    def get_config(self) -> dict[str, Any]:
        return {"source": self.source}


class OceanMask(PolygonMask):
    """Rasterize Natural Earth ocean polygons into a boolean mask.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (see
    :class:`PolygonMask`).

    Args:
        source: ``"natural_earth_10m"`` (default) downloads and caches
            the Natural Earth 1:10m ocean polygons; any other value is a
            local vector file path passed to ``geopandas.read_file``.
    """

    def __init__(self, *, source: str = "natural_earth_10m") -> None:
        self.source = source
        super().__init__(geometry=_load_natural_earth("ocean", source), crs="EPSG:4326")

    def get_config(self) -> dict[str, Any]:
        return {"source": self.source}


class CountryMask(PolygonMask):
    """Rasterize Natural Earth country polygons selected by ISO A3 code.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (see
    :class:`PolygonMask`).

    Args:
        iso_a3: One ISO A3 country code or a sequence of them (e.g.
            ``"ESP"`` or ``("ESP", "PRT")``).
        source: ``"natural_earth_10m"`` (default) downloads and caches
            the Natural Earth 1:10m admin-0 countries; any other value
            is a local vector file path with an ``ISO_A3`` column.

    Raises:
        ValueError: If none of the requested codes are found.
    """

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
    """Build a boolean mask from DEM elevation bounds.

    Marks True where the carrier DEM falls inside the requested elevation
    interval. Either bound may be ``None`` for an open-ended interval.
    Accepts a GeoTensor or plain ndarray carrier; when both the DEM and
    the carrier are georeferenced, their transform/CRS must match.

    Args:
        dem: Single-band ``GeoTensor`` whose spatial shape matches the
            input scene.
        min_elev: Inclusive lower bound (units of the DEM).
        max_elev: Inclusive upper bound.

    Examples:
        >>> AltitudeMask(dem=dem, min_elev=500.0)(scene)  # doctest: +SKIP
    """

    forbid_in_yaml: ClassVar[bool] = True

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        mask = altitude_mask(
            np.asarray(self.dem), min_elev=self.min_elev, max_elev=self.max_elev
        )
        _check_spatial_match(mask, gt, "AltitudeMask")
        _check_dem_grid_alignment(self.dem, gt, "AltitudeMask")
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "dem": {"shape": list(self.dem.shape), "dtype": str(self.dem.dtype)},
            "min_elev": self.min_elev,
            "max_elev": self.max_elev,
        }


class SlopeMask(Operator):
    """Build a boolean mask from DEM slope bounds in degrees.

    Slope is computed from the DEM with central differences scaled by
    the DEM pixel size; True where the slope (degrees) falls inside
    the requested interval. Geo-dependent on the DEM side: the DEM must
    be a georeferenced ``GeoTensor`` (its transform supplies the pixel
    size). The carrier may be a GeoTensor or plain ndarray; when both
    are georeferenced, their transform/CRS must match.

    Args:
        dem: Single-band ``GeoTensor`` whose spatial shape matches the
            input scene and whose pixel size is in the same units as
            the elevation values.
        min_slope_deg: Inclusive lower bound, degrees.
        max_slope_deg: Inclusive upper bound, degrees.

    Examples:
        >>> SlopeMask(dem=dem, max_slope_deg=10.0)(scene)  # doctest: +SKIP
    """

    forbid_in_yaml: ClassVar[bool] = True

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        mask = slope_mask(
            np.asarray(self.dem),
            _pixel_size(self.dem),
            min_slope_deg=self.min_slope_deg,
            max_slope_deg=self.max_slope_deg,
        )
        _check_spatial_match(mask, gt, "SlopeMask")
        _check_dem_grid_alignment(self.dem, gt, "SlopeMask")
        return wrap_like(gt, mask, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "dem": {"shape": list(self.dem.shape), "dtype": str(self.dem.dtype)},
            "min_slope_deg": self.min_slope_deg,
            "max_slope_deg": self.max_slope_deg,
        }


class DilateMask(Operator):
    """Dilate a boolean mask over its trailing spatial axes.

    Wraps :func:`scipy.ndimage.binary_dilation`. The structuring element
    defaults to a 3x3 box (8-connectivity). Stacks of masks of shape
    ``(C, H, W)`` are processed per channel. Accepts a GeoTensor or a
    plain ndarray and returns the same carrier kind.

    Args:
        iterations: Number of dilation passes. ``0`` is a no-op copy.
        structure: Optional 2-D structuring element.

    Examples:
        >>> DilateMask(iterations=2)(mask)  # doctest: +SKIP
    """

    def __init__(
        self, *, iterations: int = 1, structure: np.ndarray | None = None
    ) -> None:
        self.iterations = iterations
        self.structure = structure

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = dilate_mask(np.asarray(mask), self.iterations, self.structure)
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return _morph_config(self.iterations, self.structure)


class ErodeMask(Operator):
    """Erode a boolean mask over its trailing spatial axes.

    Wraps :func:`scipy.ndimage.binary_erosion`; see :class:`DilateMask`
    for the structuring-element and carrier conventions.

    Args:
        iterations: Number of erosion passes. ``0`` is a no-op copy.
        structure: Optional 2-D structuring element.

    Examples:
        >>> ErodeMask(iterations=1)(mask)  # doctest: +SKIP
    """

    def __init__(
        self, *, iterations: int = 1, structure: np.ndarray | None = None
    ) -> None:
        self.iterations = iterations
        self.structure = structure

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = erode_mask(np.asarray(mask), self.iterations, self.structure)
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return _morph_config(self.iterations, self.structure)


class OpenMask(Operator):
    """Open (erode then dilate) a boolean mask.

    Wraps :func:`scipy.ndimage.binary_opening`. Useful for removing
    isolated True pixels (salt) while preserving large True components.
    Accepts a GeoTensor or a plain ndarray and returns the same carrier
    kind.

    Args:
        iterations: Number of opening passes. ``0`` is a no-op copy.

    Examples:
        >>> OpenMask(iterations=1)(mask)  # doctest: +SKIP
    """

    def __init__(self, *, iterations: int = 1) -> None:
        self.iterations = iterations

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = open_mask(np.asarray(mask), self.iterations)
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {"iterations": self.iterations}


class CloseMask(Operator):
    """Close (dilate then erode) a boolean mask.

    Wraps :func:`scipy.ndimage.binary_closing`. Useful for filling
    pin-holes (pepper) inside otherwise solid True regions. Accepts a
    GeoTensor or a plain ndarray and returns the same carrier kind.

    Args:
        iterations: Number of closing passes. ``0`` is a no-op copy.

    Examples:
        >>> CloseMask(iterations=1)(mask)  # doctest: +SKIP
    """

    def __init__(self, *, iterations: int = 1) -> None:
        self.iterations = iterations

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = close_mask(np.asarray(mask), self.iterations)
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {"iterations": self.iterations}


class BufferMask(Operator):
    """Radially expand True pixels in a mask by a Euclidean buffer.

    Wraps :func:`geotoolz.mask.buffer_mask`. With ``unit="pixels"``
    (default) the op is metadata-free and accepts a GeoTensor or a
    plain ndarray, returning the same carrier kind. With
    ``unit="meters"`` the op is geo-dependent: the input must be a
    georeferenced ``GeoTensor`` whose transform supplies the pixel
    size, and ``radius`` is measured in CRS units.

    Args:
        radius: Buffer distance, in pixels or CRS units per ``unit``.
        unit: ``"pixels"`` (default), ``"meters"``, or ``"meter"``.

    Examples:
        >>> BufferMask(radius=2.0)(mask)  # doctest: +SKIP
    """

    def __init__(self, *, radius: float, unit: str = "pixels") -> None:
        self.radius = radius
        self.unit = unit

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        if self.unit in {"meter", "meters"}:
            if not hasattr(mask, "transform"):
                raise ValueError(
                    "BufferMask(unit='meters') requires a GeoTensor with a "
                    f"transform; got {type(mask).__name__}."
                )
            pixel_size = _pixel_size(mask)
        else:
            pixel_size = (1.0, 1.0)
        out = buffer_mask(
            np.asarray(mask), self.radius, unit=self.unit, pixel_size=pixel_size
        )
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {"radius": self.radius, "unit": self.unit}


class RemoveSmallObjects(Operator):
    """Remove connected True components smaller than ``min_size`` pixels.

    Wraps :func:`geotoolz.mask.remove_small_objects` (4-connectivity).
    Accepts a GeoTensor or a plain ndarray and returns the same carrier
    kind.

    Args:
        min_size: Minimum component area, in pixels, to keep.
    """

    def __init__(self, *, min_size: int) -> None:
        self.min_size = min_size

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = remove_small_objects(np.asarray(mask), self.min_size)
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {"min_size": self.min_size}


class RemoveSmallHoles(Operator):
    """Fill enclosed False components up to ``area_threshold`` pixels.

    Wraps :func:`geotoolz.mask.remove_small_holes`; holes touching the
    image border are never filled. Accepts a GeoTensor or a plain
    ndarray and returns the same carrier kind.

    Args:
        area_threshold: Maximum hole area, in pixels, to fill.
    """

    def __init__(self, *, area_threshold: int) -> None:
        self.area_threshold = area_threshold

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = remove_small_holes(np.asarray(mask), self.area_threshold)
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {"area_threshold": self.area_threshold}


class CleanMask(Operator):
    """Remove small objects, fill small holes, then close the mask.

    Convenience composition of :class:`RemoveSmallObjects`,
    :class:`RemoveSmallHoles`, and :class:`CloseMask`, in that order.
    Accepts a GeoTensor or a plain ndarray and returns the same carrier
    kind.

    Args:
        min_object_size: Components smaller than this are removed.
        max_hole_size: Enclosed holes up to this size are filled.
        close_iter: Binary-closing iterations applied last; ``0`` skips
            the closing step.
    """

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
        return wrap_like(mask, out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_object_size": self.min_object_size,
            "max_hole_size": self.max_hole_size,
            "close_iter": self.close_iter,
        }


class CombineMasks(Operator):
    """Combine boolean masks with ``or``, ``and``, ``xor``, or unary ``not``.

    Called with a *sequence* of masks (GeoTensors or plain ndarrays);
    the output takes its carrier kind — and, for GeoTensors, its
    metadata — from the first mask in the sequence.

    Args:
        op: ``"or"`` (default), ``"and"``, ``"xor"`` (n-ary), or the
            unary ``"not"`` which expects exactly one mask.
    """

    def __init__(self, *, op: str = "or") -> None:
        self.op = op

    def _apply(self, masks: Sequence[GeoTensor | np.ndarray]) -> GeoTensor | np.ndarray:
        out = combine_masks([np.asarray(mask) for mask in masks], self.op)
        return wrap_like(masks[0], out, fill_value_default=False)

    def get_config(self) -> dict[str, Any]:
        return {"op": self.op}


class InvertMask(Operator):
    """Invert a boolean mask element-wise.

    Accepts a GeoTensor or a plain ndarray and returns the same carrier
    kind. Takes no parameters.
    """

    def _apply(self, mask: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = invert_mask(np.asarray(mask))
        return wrap_like(mask, out, fill_value_default=False)


class ApplyMask(Operator):
    """Apply a boolean mask to the carrier, filling True pixels.

    Accepts a GeoTensor or a plain ndarray and returns the same
    carrier kind. Mirrors :class:`geotoolz.cloud.ApplyMask` but lives in the ``mask``
    namespace so geometry / morphology / algebra pipelines can compose
    without importing the cloud submodule. Delegates the actual masking
    to :func:`geotoolz.cloud._src.array.apply_mask` so the broadcasting
    + dtype-preservation rules stay in one place.

    Convention: the mask is True where pixels should be *masked out*.
    Geometry masks built with ``inside=True`` return True *inside* the
    polygon, so use ``invert=True`` to keep only the polygon interior
    (or build the geometry mask with ``inside=False``).

    Args:
        mask: Boolean array, ``GeoTensor``, or ``Operator`` producing
            one when called on the input.
        fill_value: Value substituted where the mask says "drop".
            Default ``np.nan``.
        invert: Flip the mask before applying it.

    Examples:
        >>> import geotoolz as gz, numpy as np
        >>> aoi = gz.mask.BBoxMask(bounds=(1.0, 1.0, 3.0, 3.0))
        >>> keep_aoi = gz.mask.ApplyMask(mask=aoi, invert=True)
        >>> out = keep_aoi(scene)  # doctest: +SKIP
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        mask: Operator | np.ndarray | Any,
        fill_value: float = float("nan"),
        invert: bool = False,
    ) -> None:
        self.mask = mask
        self.fill_value = fill_value
        self.invert = invert

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        mask_arr = np.asarray(
            self.mask(gt) if isinstance(self.mask, Operator) else self.mask
        )
        out = apply_mask(
            np.asarray(gt), mask_arr, fill_value=self.fill_value, invert=self.invert
        )
        return wrap_like(gt, out)

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
        return {
            "mask": mask_config,
            "fill_value": self.fill_value,
            "invert": self.invert,
        }


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
                raise ValueError(
                    "PolygonMask: GeoDataFrame CRS is missing; pass `crs=...` "
                    "or set `geometry.crs` before creating the operator"
                )
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
    across multiple operator instances in one Python session; the downloaded
    zip and extracted shapefile are cached separately on disk across sessions.
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
                _download_url(_NATURAL_EARTH_URLS[kind], zip_path)
            except OSError as exc:
                raise RuntimeError(
                    f"failed to download Natural Earth {kind!r} data from "
                    f"{_NATURAL_EARTH_URLS[kind]}; check network/firewall "
                    "access or pass a local vector file path as `source`"
                ) from exc
        with ZipFile(zip_path) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(
                    f"Natural Earth {kind!r} archive failed CRC validation at "
                    f"{bad_member!r}; remove the cached archive to re-download it"
                )
            _safe_extract_zip(zf, extract_dir)
    shapefiles = list(extract_dir.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"Natural Earth {kind!r} archive did not contain a top-level .shp "
            "file; remove the cached archive to re-download it or pass a "
            "valid local vector file path as `source`"
        )
    shp = shapefiles[0]
    return gpd.read_file(shp)


def _download_url(url: str, destination: Path, *, timeout: float = 60.0) -> None:
    with urlopen(url, timeout=timeout) as response, destination.open("wb") as dst:
        shutil.copyfileobj(response, dst)


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


def _safe_extract_zip(zf: ZipFile, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    for member in zf.infolist():
        target = (extract_root / member.filename).resolve()
        if not target.is_relative_to(extract_root):
            raise RuntimeError(
                f"refusing to extract unsafe zip member {member.filename!r}"
            )
    zf.extractall(extract_root)


def _pixel_size(gt: Any) -> tuple[float, float]:
    transform = gt.transform
    return (abs(float(transform.e)), abs(float(transform.a)))


def _check_spatial_match(mask: np.ndarray, gt: GeoTensor, name: str) -> None:
    if mask.shape[-2:] != gt.shape[-2:]:
        raise ValueError(f"{name}: DEM spatial shape must match the input GeoTensor")


def _check_dem_grid_alignment(dem: Any, gt: Any, name: str) -> None:
    """Reject DEMs whose georeferencing disagrees with the carrier.

    A DEM with the same raster shape but a different ``transform`` or
    ``crs`` would be silently misapplied to the carrier. When both
    inputs expose ``transform`` / ``crs`` (i.e. are GeoTensor-like), we
    require them to match.
    """
    if not (hasattr(dem, "transform") and hasattr(gt, "transform")):
        return
    if not (hasattr(dem, "crs") and hasattr(gt, "crs")):
        return
    if dem.transform != gt.transform or str(dem.crs) != str(gt.crs):
        raise ValueError(f"{name}: DEM grid (transform/crs) doesn't match carrier")


def _morph_config(iterations: int, structure: np.ndarray | None) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "structure": None if structure is None else np.asarray(structure).tolist(),
    }


def _geometry_config(
    geometry: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Return a JSON-safe summary of a geometry / GeoDataFrame.

    Shapely geometries are emitted as GeoJSON-ish ``__geo_interface__``
    dicts (with tuples cast to lists for strict JSON safety). GeoDataFrames
    are summarised by length + CRS to avoid embedding many features.
    """
    if isinstance(geometry, gpd.GeoDataFrame):
        return {
            "type": "GeoDataFrame",
            "length": len(geometry),
            "crs": None if geometry.crs is None else str(geometry.crs),
        }
    # `shapely.geometry.mapping` returns the GeoJSON-style dict for any
    # base geometry (including MultiPolygon / GeometryCollection). We
    # coerce nested tuples to lists for strict JSON serialisability.
    return jsonable(shapely.geometry.mapping(geometry))
