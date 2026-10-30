"""Layer-1 ``Operator`` wrappers around the :mod:`xrops.geom._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`pipekit.Operator`, so they compose with
:class:`~pipekit.Sequential`, the ``|`` pipe, and the functional
:class:`~pipekit.Graph` API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import regionmask
import xarray as xr
from pipekit import Signature

from xrops._src import signature as _sig
from xrops._src.operator import Operator
from xrops.geom._src import (
    crs as _crs,
    footprint as _footprint,
    mosaic as _mosaic,
    rasterize as _rasterize,
    regions as _regions,
    subset as _subset,
    vectorize as _vectorize,
)


class SubsetBBox(Operator):
    """Subset a dataset to a lon/lat bounding box.

    Args:
        lon_bnds: ``(min, max)`` longitude bounds.
        lat_bnds: ``(min, max)`` latitude bounds.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.

    Returns:
        The dataset restricted to the bounding box.
    """

    def __init__(
        self,
        lon_bnds: tuple[float, float],
        lat_bnds: tuple[float, float],
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.lon_bnds = tuple(lon_bnds)
        self.lat_bnds = tuple(lat_bnds)
        self.lon = lon
        self.lat = lat

    def _apply(self, ds):
        return _subset.subset_bbox(
            ds,
            lon_bnds=self.lon_bnds,
            lat_bnds=self.lat_bnds,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "lon_bnds": list(self.lon_bnds),
            "lat_bnds": list(self.lat_bnds),
            "lon": self.lon,
            "lat": self.lat,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _sig.replace_dims(input_signature, {self.lon: None, self.lat: None})


class SubsetToRegion(Operator):
    """Subset a dataset to a named or custom geographic region.

    Masks to the region and trims to its bounding box.

    Args:
        region: Region specification — a registry name, a
            :class:`~xrops.geom._src.regions.RegionSpec`, a
            ``regionmask.Regions``, or the dict form from ``get_config()``.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.
        validate: Whether to validate coordinates before masking.

    Returns:
        The dataset restricted to the region.
    """

    def __init__(
        self,
        region: str | _regions.RegionSpec | regionmask.Regions | dict[str, Any],
        *,
        lon: str = "lon",
        lat: str = "lat",
        validate: bool = True,
    ):
        # Accept the dict form emitted by ``get_config`` so the standard
        # ``cls(**op.get_config())`` round-trip used by ApplyToEach works
        # for custom regions, not just registry strings.
        if isinstance(region, dict):
            region = _regions.region_from_dict(region)
        self.region = region
        self.lon = lon
        self.lat = lat
        self.validate = validate

    def _apply(self, ds):
        return _subset.subset_to_region(
            ds,
            self.region,
            lon=self.lon,
            lat=self.lat,
            validate=self.validate,
        )

    def get_config(self) -> dict[str, Any]:
        region: str | dict[str, Any]
        if isinstance(self.region, str):
            region = self.region
        else:
            region = _regions.region_to_dict(self.region)
        return {
            "region": region,
            "lon": self.lon,
            "lat": self.lat,
            "validate": self.validate,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _sig.replace_dims(input_signature, {self.lon: None, self.lat: None})


class SubsetTime(Operator):
    """Subset a dataset to an inclusive ``[time_min, time_max]`` window.

    Args:
        time_min: Start of the time window (ISO string).
        time_max: End of the time window (ISO string).
        time: Name of the time coordinate.

    Returns:
        The dataset restricted to the time window.
    """

    def __init__(self, time_min: str, time_max: str, time: str = "time"):
        self.time_min = time_min
        self.time_max = time_max
        self.time = time

    def _apply(self, ds):
        return _subset.subset_time(
            ds, time_min=self.time_min, time_max=self.time_max, time=self.time
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "time_min": self.time_min,
            "time_max": self.time_max,
            "time": self.time,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _sig.replace_dims(input_signature, {self.time: None})


class SelectVariables(Operator):
    """Select a subset of data variables.

    Args:
        variables: A variable name or sequence of names to keep.

    Returns:
        The dataset containing only the selected variables.
    """

    def __init__(self, variables: str | Sequence[str]):
        self.variables = [variables] if isinstance(variables, str) else list(variables)

    def _apply(self, ds):
        return _subset.select_variables(ds, self.variables)

    def get_config(self) -> dict[str, Any]:
        return {"variables": list(self.variables)}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


class ReprojectMatch(Operator):
    """Reproject a dataset onto ``target``'s CRS, transform and shape.

    Wraps :func:`xrops.geom.reproject_match`. Use this instead of
    :class:`xrops.interpolate.RegridLike` when source and target live
    in different CRSs. Takes a single ``Dataset`` / ``DataArray``, so it
    is ``DataTree``-mappable — each leaf lands on the same target grid.

    Args:
        target: Raster whose CRS, affine transform and shape define the
            output grid.
        resampling: Name of the :class:`rasterio.enums.Resampling`
            member. Default ``"bilinear"``.

    Returns:
        The dataset on ``target``'s grid.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        target: xr.Dataset | xr.DataArray,
        *,
        resampling: str = "bilinear",
    ):
        self.target = target
        self.resampling = resampling

    def _apply(self, ds):
        return _crs.reproject_match(ds, self.target, resampling=self.resampling)

    def get_config(self) -> dict[str, Any]:
        kind = "Dataset" if isinstance(self.target, xr.Dataset) else "DataArray"
        return {"target": f"<{kind}>", "resampling": self.resampling}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        # rio.reproject_match returns the *target's* spatial dim names, so
        # when the two rasters disagree the source's names disappear rather
        # than surviving alongside the target's. Drop any source spatial dim
        # before inserting the target's; non-spatial dims pass through.
        y_dim, x_dim = self.target.rio.y_dim, self.target.rio.x_dim
        source_spatial = {"x", "y", "lat", "lon", "latitude", "longitude"}
        dims = {
            name: size
            for name, size in _sig.sizes(input_signature).items()
            if name not in source_spatial
        }
        dims[y_dim] = self.target.sizes[y_dim]
        dims[x_dim] = self.target.sizes[x_dim]
        return _sig.make(dims, dtype=input_signature.dtype)


class AssignLocalXY(Operator):
    """Add metric ``x``/``y`` coordinates relative to a :class:`LocalFrame`.

    Wraps :func:`xrops.geom.assign_local_xy` — the point / swath
    counterpart of :class:`ReprojectMatch`. The frame is stored as its
    JSON payload, so ``get_config()`` round-trips through the constructor.

    Args:
        frame: A :class:`xrops.geom.LocalFrame` or its ``to_dict()``
            payload (``{"crs", "origin_lon", "origin_lat"}``).
        lon: Longitude coordinate name.
        lat: Latitude coordinate name.
        x: Name of the output east coordinate.
        y: Name of the output north coordinate.

    Returns:
        The dataset with ``x``/``y`` coordinates in metres (2-D for grids
        and swaths, 1-D for along-track points) and ``attrs["local_frame"]``
        set.
    """

    def __init__(
        self,
        frame: _crs.LocalFrame | dict[str, Any],
        *,
        lon: str = "lon",
        lat: str = "lat",
        x: str = "x",
        y: str = "y",
    ):
        self.frame = (
            frame
            if isinstance(frame, _crs.LocalFrame)
            else _crs.LocalFrame.from_dict(frame)
        )
        self.lon = lon
        self.lat = lat
        self.x = x
        self.y = y

    def _apply(self, ds):
        return _crs.assign_local_xy(
            ds, self.frame, lon=self.lon, lat=self.lat, x=self.x, y=self.y
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "lon": self.lon,
            "lat": self.lat,
            "x": self.x,
            "y": self.y,
        }


class SpatialMosaic(Operator):
    """Merge a sequence of georeferenced tiles into a single cube.

    Wraps :func:`xrops.geom.spatial_mosaic`. Expects a **sequence** of
    DataArrays or Datasets as the carrier payload, so — unlike the
    single-Dataset operators in this module — it is *not*
    ``DataTree``-mappable.

    Args:
        method: Overlap resolution rule — ``"first"``, ``"last"``,
            ``"min"``, ``"max"`` or ``"sum"``.
        resampling: :class:`rasterio.enums.Resampling` member name used
            when harmonising inputs whose CRS differs from the first
            input's.
        nodata: Fill value for cells covered by no input.

    Returns:
        One cube spanning the union of the input extents.
    """

    def __init__(
        self,
        *,
        method: _mosaic.MosaicMethod = "first",
        resampling: str = "nearest",
        nodata: float | None = None,
    ):
        self.method = method
        self.resampling = resampling
        self.nodata = nodata

    def _apply(self, datasets):
        return _mosaic.spatial_mosaic(
            datasets,
            method=self.method,
            resampling=self.resampling,
            nodata=self.nodata,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "resampling": self.resampling,
            "nodata": self.nodata,
        }


class Vectorize(Operator):
    """Polygonize a label / binary mask into a ``GeoDataFrame``.

    Wraps :func:`xrops.geom.vectorize`. Returns a ``GeoDataFrame``, so
    it is *terminal* in ``DataTree`` mode: mapping it over a tree raises
    ``TypeError`` — call it on a single leaf instead.

    Args:
        min_area: Drop polygons smaller than this, in CRS units².
        connectivity: Pixel connectivity, ``4`` or ``8``.
        attr_name: Column holding each polygon's mask value.

    Returns:
        ``GeoDataFrame`` with one row per polygon and ``crs=mask.rio.crs``.
    """

    def __init__(
        self,
        *,
        min_area: float = 0.0,
        connectivity: int = 4,
        attr_name: str = "value",
    ):
        self.min_area = min_area
        self.connectivity = connectivity
        self.attr_name = attr_name

    def _apply(self, mask):
        return _vectorize.vectorize(
            mask,
            min_area=self.min_area,
            connectivity=self.connectivity,
            attr_name=self.attr_name,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "connectivity": self.connectivity,
            "attr_name": self.attr_name,
        }


class Rasterize(Operator):
    """Burn stored vector geometries onto the grid of the input raster.

    Wraps :func:`xrops.geom.rasterize_like` with the input as ``like``.
    Returns a ``DataArray``, so it *is* ``DataTree``-mappable — each leaf
    gets the geometries burned onto its own grid.

    Args:
        geometries: ``GeoDataFrame`` / ``GeoSeries``, a shapely geometry,
            or a sequence of them.
        column: ``GeoDataFrame`` column holding the burn values.
        fill: Value for pixels no geometry covers.
        value: Burn value when ``column`` is ``None``.
        all_touched: Burn every pixel a geometry touches.
        dtype: Output dtype.
        geometries_crs: CRS of bare shapely geometries.

    Returns:
        2-D DataArray on the input's grid with its CRS.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        geometries: Any,
        *,
        column: str | None = None,
        fill: float = 0.0,
        value: float = 1.0,
        all_touched: bool = False,
        dtype: str = "float32",
        geometries_crs: Any = None,
    ):
        self.geometries = geometries
        self.column = column
        self.fill = fill
        self.value = value
        self.all_touched = all_touched
        self.dtype = dtype
        self.geometries_crs = geometries_crs

    def _apply(self, like):
        return _rasterize.rasterize_like(
            self.geometries,
            like,
            column=self.column,
            fill=self.fill,
            value=self.value,
            all_touched=self.all_touched,
            dtype=self.dtype,
            geometries_crs=self.geometries_crs,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "geometries": f"<{type(self.geometries).__name__}>",
            "column": self.column,
            "fill": self.fill,
            "value": self.value,
            "all_touched": self.all_touched,
            "dtype": self.dtype,
            "geometries_crs": (
                None if self.geometries_crs is None else str(self.geometries_crs)
            ),
        }


class Footprint(Operator):
    """Bounding polygon of the input raster's extent.

    Wraps :func:`xrops.geom.footprint`. Returns a shapely ``Polygon``,
    so it is *terminal* in ``DataTree`` mode (mapping over a tree raises
    ``TypeError``).

    Args:
        crs: Output CRS; ``None`` keeps the raster's CRS.

    Returns:
        ``shapely.Polygon``.
    """

    def __init__(self, *, crs: str | None = None):
        self.crs = crs

    def _apply(self, ds):
        return _footprint.footprint(ds, crs=self.crs)

    def get_config(self) -> dict[str, Any]:
        return {"crs": self.crs}


class ValidFootprint(Operator):
    """Polygon enclosing the input raster's valid (non-nodata) pixels.

    Wraps :func:`xrops.geom.valid_footprint`. Returns a shapely
    geometry, so it is *terminal* in ``DataTree`` mode (mapping over a
    tree raises ``TypeError``).

    Args:
        crs: Output CRS; ``None`` keeps the raster's CRS.
        nodata: Invalid-pixel value; ``None`` uses ``rio.nodata``.
        method: ``"all"`` or ``"any"`` — how extra dims / variables combine.
        connectivity: Pixel connectivity, ``4`` or ``8``.

    Returns:
        ``shapely.Polygon`` or ``shapely.MultiPolygon``.
    """

    def __init__(
        self,
        *,
        crs: str | None = None,
        nodata: float | None = None,
        method: Literal["all", "any"] = "all",
        connectivity: int = 4,
    ):
        self.crs = crs
        self.nodata = nodata
        self.method = method
        self.connectivity = connectivity

    def _apply(self, ds):
        return _footprint.valid_footprint(
            ds,
            crs=self.crs,
            nodata=self.nodata,
            method=self.method,
            connectivity=self.connectivity,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "nodata": self.nodata,
            "method": self.method,
            "connectivity": self.connectivity,
        }


__all__ = [
    "AssignLocalXY",
    "Footprint",
    "Rasterize",
    "ReprojectMatch",
    "SelectVariables",
    "SpatialMosaic",
    "SubsetBBox",
    "SubsetTime",
    "SubsetToRegion",
    "ValidFootprint",
    "Vectorize",
]
