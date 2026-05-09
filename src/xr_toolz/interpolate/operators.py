"""Layer-1 ``Operator`` wrappers around :mod:`xr_toolz.interpolate._src`.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`xr_toolz.core.Operator`, so they compose with
:class:`~xr_toolz.core.Sequential`, the ``|`` pipe, and the functional
:class:`~xr_toolz.core.Graph` API.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from xr_toolz.core import Operator, Signature
from xr_toolz.interpolate._src import (
    binning as _binning,
    coord_remap as _coord_remap,
    downscale as _downscale,
    gap_fill as _gap_fill,
    grid_to_grid as _grid_to_grid,
    knn as _knn,
    mask_ops as _mask_ops,
    points_to_grid as _points_to_grid,
    resample as _resample,
    smooth as _smooth,
)


ResizeMode = Literal["reflect", "constant", "edge", "symmetric", "wrap"]


def _as_integer_factor(dim: str, factor: int | float) -> int:
    if isinstance(factor, bool) or int(factor) != factor:
        raise ValueError(
            f"refinement factor for {dim!r} must be an integer for the default "
            "interpolation method."
        )
    return int(factor)


# ---------- gap fill -------------------------------------------------------


class FillNaNSpatial(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_spatial`."""

    def __init__(self, method: str = "linear", lon: str = "lon", lat: str = "lat"):
        self.method = method
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _gap_fill.fillnan_spatial(
            da, method=self.method, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "lon": self.lon, "lat": self.lat}


class FillNaNTemporal(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_temporal`."""

    def __init__(
        self,
        method: str = "linear",
        time: str = "time",
        max_gap: Any = None,
    ):
        self.method = method
        self.time = time
        self.max_gap = max_gap

    def _apply(self, ds):
        return _gap_fill.fillnan_temporal(
            ds, method=self.method, time=self.time, max_gap=self.max_gap
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "time": self.time, "max_gap": self.max_gap}


class FillNaNClimatology(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_climatology`."""

    def __init__(
        self,
        *,
        time: str = "time",
        group: Literal["month", "dayofyear", "season"] = "month",
        residual: Literal["zero", "linear"] = "linear",
        min_count: int = 1,
    ):
        self.time = time
        self.group = group
        self.residual = residual
        self.min_count = min_count

    def _apply(self, da):
        return _gap_fill.fillnan_climatology(
            da,
            time=self.time,
            group=self.group,
            residual=self.residual,
            min_count=self.min_count,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "group": self.group,
            "residual": self.residual,
            "min_count": self.min_count,
        }


class FillNaNLaplacian(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_laplacian`."""

    def __init__(
        self,
        *,
        max_iter: int = 1000,
        tol: float = 1e-4,
        relaxation: float = 1.0,
        boundary: str = "reflect",
        lon: str = "lon",
        lat: str = "lat",
    ):
        # Validate eagerly so misconfigured operators fail at construction
        # time rather than deep inside _apply (mirrors Coarsen).
        _gap_fill._validate_laplacian_args(max_iter, tol, relaxation, boundary)
        self.max_iter = max_iter
        self.tol = tol
        self.relaxation = relaxation
        self.boundary = boundary
        self.lon = lon
        self.lat = lat

    def _apply(self, ds):
        def _fill(da):
            if {self.lat, self.lon} <= set(da.dims):
                return _gap_fill.fillnan_laplacian(
                    da,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    relaxation=self.relaxation,
                    boundary=self.boundary,
                    lon=self.lon,
                    lat=self.lat,
                )
            return da

        if isinstance(ds, xr.Dataset):
            return ds.map(_fill)
        return _fill(ds)

    def get_config(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "tol": self.tol,
            "relaxation": self.relaxation,
            "boundary": self.boundary,
            "lon": self.lon,
            "lat": self.lat,
        }


class FillNaNRBF(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_rbf`."""

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        neighbors: int | None = 32,
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.kernel = kernel
        self.neighbors = neighbors
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _gap_fill.fillnan_rbf(
            da,
            kernel=self.kernel,
            neighbors=self.neighbors,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "neighbors": self.neighbors,
            "lon": self.lon,
            "lat": self.lat,
        }


class FillNaNIDW(Operator):
    """Wrap :func:`xr_toolz.interpolate.fillnan_idw`."""

    def __init__(
        self,
        *,
        lon: str = "lon",
        lat: str = "lat",
        k: int = 8,
        power: float = 2.0,
        metric: _knn.Metric = "euclidean",
        max_distance: float | None = None,
        eps: float = 1e-12,
    ):
        _knn._validate_idw_args(k, power, metric, max_distance, eps)
        self.lon = lon
        self.lat = lat
        self.k = k
        self.power = power
        self.metric = metric
        self.max_distance = max_distance
        self.eps = eps

    def _apply(self, da):
        return _knn.fillnan_idw(
            da,
            lon=self.lon,
            lat=self.lat,
            k=self.k,
            power=self.power,
            metric=self.metric,
            max_distance=self.max_distance,
            eps=self.eps,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "lon": self.lon,
            "lat": self.lat,
            "k": self.k,
            "power": self.power,
            "metric": self.metric,
            "max_distance": self.max_distance,
            "eps": self.eps,
        }


# ---------- mask cleanup ----------------------------------------------------


def _footprint_config(footprint: _mask_ops.Footprint | None) -> Any:
    # ndarray footprints are encoded as a JSON-safe dict so cls(**get_config())
    # round-trips without losing the exact mask. The helpers in mask_ops
    # accept the dict form via ``_decode_footprint_config``.
    if isinstance(footprint, np.ndarray):
        return {"kind": "ndarray", "data": footprint.astype(bool).tolist()}
    return footprint


def _decode_footprint_config(footprint: Any) -> _mask_ops.Footprint | None:
    if isinstance(footprint, dict) and footprint.get("kind") == "ndarray":
        return np.asarray(footprint["data"], dtype=bool)
    return footprint


class MaskRemoveSmallHoles(Operator):
    """Operator wrapper for filling small unmasked holes in mask pipelines.

    Args:
        area: False regions smaller than this pixel count are filled.
        lon: Longitude dimension name.
        lat: Latitude dimension name.
    """

    def __init__(self, *, area: int = 4, lon: str = "lon", lat: str = "lat"):
        self.area = area
        self.lon = lon
        self.lat = lat

    def _apply(self, mask):
        return _mask_ops.remove_small_holes_2d(
            mask, area=self.area, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {"area": self.area, "lon": self.lon, "lat": self.lat}


class MaskRemoveSmallObjects(Operator):
    """Operator wrapper for dropping small masked specks in mask pipelines.

    Args:
        area: True regions smaller than this pixel count are dropped.
        lon: Longitude dimension name.
        lat: Latitude dimension name.
    """

    def __init__(self, *, area: int = 4, lon: str = "lon", lat: str = "lat"):
        self.area = area
        self.lon = lon
        self.lat = lat

    def _apply(self, mask):
        return _mask_ops.remove_small_objects_2d(
            mask, area=self.area, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {"area": self.area, "lon": self.lon, "lat": self.lat}


class MaskBinaryOpening(Operator):
    """Operator wrapper for binary opening inside ``Sequential`` pipelines.

    Args:
        footprint: Structuring element specification.
        lon: Longitude dimension name.
        lat: Latitude dimension name.
    """

    def __init__(
        self,
        *,
        footprint: _mask_ops.Footprint = 1,
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.footprint = _decode_footprint_config(footprint)
        self.lon = lon
        self.lat = lat

    def _apply(self, mask):
        return _mask_ops.binary_opening_2d(
            mask, footprint=self.footprint, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "footprint": _footprint_config(self.footprint),
            "lon": self.lon,
            "lat": self.lat,
        }


class MaskBinaryClosing(Operator):
    """Operator wrapper for binary closing inside ``Sequential`` pipelines.

    Args:
        footprint: Structuring element specification.
        lon: Longitude dimension name.
        lat: Latitude dimension name.
    """

    def __init__(
        self,
        *,
        footprint: _mask_ops.Footprint = 1,
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.footprint = _decode_footprint_config(footprint)
        self.lon = lon
        self.lat = lat

    def _apply(self, mask):
        return _mask_ops.binary_closing_2d(
            mask, footprint=self.footprint, lon=self.lon, lat=self.lat
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "footprint": _footprint_config(self.footprint),
            "lon": self.lon,
            "lat": self.lat,
        }


class CleanMask(Operator):
    """Convenience operator for the common mask-cleanup pipeline.

    The fixed order is remove small holes, remove small objects, binary
    closing, then binary opening; each step is opt-in through its keyword.

    Args:
        fill_holes_area: Optional hole area threshold.
        drop_objects_area: Optional object area threshold.
        closing_footprint: Optional binary-closing footprint.
        opening_footprint: Optional binary-opening footprint.
        lon: Longitude dimension name.
        lat: Latitude dimension name.
    """

    def __init__(
        self,
        *,
        fill_holes_area: int | None = 4,
        drop_objects_area: int | None = None,
        closing_footprint: _mask_ops.Footprint | None = None,
        opening_footprint: _mask_ops.Footprint | None = None,
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.fill_holes_area = fill_holes_area
        self.drop_objects_area = drop_objects_area
        self.closing_footprint = _decode_footprint_config(closing_footprint)
        self.opening_footprint = _decode_footprint_config(opening_footprint)
        self.lon = lon
        self.lat = lat

    def _apply(self, mask):
        return _mask_ops.clean_mask(
            mask,
            fill_holes_area=self.fill_holes_area,
            drop_objects_area=self.drop_objects_area,
            closing_footprint=self.closing_footprint,
            opening_footprint=self.opening_footprint,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "fill_holes_area": self.fill_holes_area,
            "drop_objects_area": self.drop_objects_area,
            "closing_footprint": _footprint_config(self.closing_footprint),
            "opening_footprint": _footprint_config(self.opening_footprint),
            "lon": self.lon,
            "lat": self.lat,
        }


# ---------- resample -------------------------------------------------------


class ResampleTime(Operator):
    """Wrap :func:`xr_toolz.interpolate.resample_time`."""

    def __init__(
        self,
        freq: str = "1D",
        method: str = "mean",
        time: str = "time",
        interp_method: Literal["linear", "nearest", "cubic"] = "linear",
    ):
        self.freq = freq
        self.method = method
        self.time = time
        self.interp_method = interp_method

    def _apply(self, ds):
        return _resample.resample_time(
            ds,
            freq=self.freq,
            method=self.method,
            time=self.time,
            interp_method=self.interp_method,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "freq": self.freq,
            "method": self.method,
            "time": self.time,
            "interp_method": self.interp_method,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature.replace_dims({self.time: None})


# ---------- grid-to-grid ---------------------------------------------------


class Coarsen(Operator):
    """Wrap :func:`xr_toolz.interpolate.coarsen`."""

    _VALID_BOUNDARY = ("exact", "trim", "pad")

    def __init__(
        self,
        factor: dict[str, int],
        method: str = "mean",
        boundary: str = "trim",
        conservative: bool = False,
        lat: str = "lat",
    ):
        if boundary not in self._VALID_BOUNDARY:
            raise ValueError(
                f"Coarsen boundary must be one of {self._VALID_BOUNDARY!r}, "
                f"got {boundary!r}."
            )
        if conservative and method != "mean":
            raise ValueError(
                f"conservative coarsen only supports method='mean', got {method!r}."
            )
        # Reuse the layer-0 validator so int-likes (np.int64) are accepted and
        # negative / zero / non-integer factors fail at construction time.
        self.factor = _grid_to_grid._validate_coarsen_factor(factor)
        self.method = method
        self.boundary = boundary
        self.conservative = conservative
        self.lat = lat

    def _apply(self, ds):
        if self.conservative:
            return _grid_to_grid.coarsen_conservative(
                ds, factor=self.factor, lat=self.lat, boundary=self.boundary
            )
        return _grid_to_grid.coarsen(
            ds, factor=self.factor, method=self.method, boundary=self.boundary
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": dict(self.factor),
            "method": self.method,
            "boundary": self.boundary,
            "conservative": self.conservative,
            "lat": self.lat,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        updates: dict[str, int | None] = {}
        for dim, factor in self.factor.items():
            size = input_signature.dims.get(dim)
            if size is None:
                updates[dim] = None
            elif self.boundary == "trim":
                updates[dim] = size // factor
            elif self.boundary == "exact" and size % factor:
                raise ValueError(
                    f"coarsen boundary='exact' requires {dim!r} size {size} "
                    f"to be divisible by factor {factor}."
                )
            else:
                updates[dim] = (size + factor - 1) // factor
        return input_signature.replace_dims(updates)


class Refine(Operator):
    """Wrap :func:`xr_toolz.interpolate.refine`.

    If ``order`` is set, dispatches to the scikit-image-backed 2-D resize
    path and requires ``factor`` to include both ``lat`` and ``lon``.

    Args:
        factor: Per-dimension refinement factors.
        method: Interpolation method for the default ``xr.interp`` path.
        order: Optional scikit-image spline order (0..5). When set, uses the
            2-D resize path.
        lat: Latitude-like dimension name for the 2-D resize path.
        lon: Longitude-like dimension name for the 2-D resize path.
        anti_aliasing: Anti-aliasing setting passed to scikit-image.
        mode: Boundary mode for scikit-image (``"reflect"``, ``"constant"``,
            ``"edge"``, ``"symmetric"``, or ``"wrap"``).
        cval: Fill value used when ``mode="constant"``.
    """

    def __init__(
        self,
        factor: dict[str, int | float],
        method: str = "linear",
        *,
        order: int | None = None,
        lat: str = "lat",
        lon: str = "lon",
        anti_aliasing: bool | None = None,
        mode: ResizeMode = "reflect",
        cval: float = 0.0,
    ):
        if order is not None:
            extra = set(factor) - {lat, lon}
            if extra:
                raise ValueError(
                    "Refine(order=...) only resizes the (lat, lon) plane; "
                    f"got extra factor dims {sorted(extra)!r}. Drop them or "
                    "use order=None."
                )
        self.factor = dict(factor)
        self.method = method
        self.order = order
        self.lat = lat
        self.lon = lon
        self.anti_aliasing = anti_aliasing
        self.mode = mode
        self.cval = cval

    def _apply(self, ds):
        if self.order is not None:
            if isinstance(ds, xr.Dataset):
                # refine_2d is DataArray-only; map over variables that have
                # both core dims and pass others through unchanged so we keep
                # the same Dataset/DataArray contract as the default path.
                def _resize_var(da: xr.DataArray) -> xr.DataArray:
                    if {self.lat, self.lon} <= set(da.dims):
                        return _grid_to_grid.refine_2d(
                            da,
                            factor=self.factor,
                            lat=self.lat,
                            lon=self.lon,
                            order=self.order,
                            anti_aliasing=self.anti_aliasing,
                            mode=self.mode,
                            cval=self.cval,
                        )
                    return da

                return ds.map(_resize_var)
            return _grid_to_grid.refine_2d(
                ds,
                factor=self.factor,
                lat=self.lat,
                lon=self.lon,
                order=self.order,
                anti_aliasing=self.anti_aliasing,
                mode=self.mode,
                cval=self.cval,
            )
        factor = {
            dim: _as_integer_factor(dim, value) for dim, value in self.factor.items()
        }
        return _grid_to_grid.refine(ds, factor=factor, method=self.method)

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": dict(self.factor),
            "method": self.method,
            "order": self.order,
            "lat": self.lat,
            "lon": self.lon,
            "anti_aliasing": self.anti_aliasing,
            "mode": self.mode,
            "cval": self.cval,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        # When order is set, only lat / lon dims are resized; pass other dims
        # through unchanged even if the user passed them in `factor` (we
        # rejected that case in __init__, so this is just a safety net).
        active_dims = (
            {self.lat, self.lon} if self.order is not None else set(self.factor)
        )
        updates: dict[str, int | None] = {}
        for dim, factor in self.factor.items():
            if dim not in active_dims:
                continue
            size = input_signature.dims.get(dim)
            if size is None:
                updates[dim] = None
            elif self.order is None or float(factor).is_integer():
                # Integer factors use refine()'s endpoint-preserving formula.
                updates[dim] = (size - 1) * _as_integer_factor(dim, factor) + 1
            else:
                updates[dim] = max(1, round(size * factor))
        return input_signature.replace_dims(updates)


class RegridLike(Operator):
    """Wrap :func:`xr_toolz.interpolate.regrid_like` — bilinear resample
    of the input onto another Dataset's coordinate grid along ``dims``.
    """

    def __init__(
        self,
        target: xr.Dataset | xr.DataArray,
        *,
        dims: tuple[str, ...] = ("lat", "lon"),
        method: str = "linear",
    ):
        self.target = target
        self.dims = tuple(dims)
        self.method = method

    def _apply(self, ds):
        return _grid_to_grid.regrid_like(
            ds, self.target, dims=self.dims, method=self.method
        )

    def get_config(self) -> dict[str, Any]:
        target_shape = {
            d: int(self.target.sizes[d]) for d in self.dims if d in self.target.sizes
        }
        return {
            "target_shape": target_shape,
            "dims": list(self.dims),
            "method": self.method,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        updates = {
            dim: int(self.target.sizes[dim])
            for dim in self.dims
            if dim in self.target.sizes
        }
        return input_signature.replace_dims(updates)


# ---------- binning --------------------------------------------------------


class Bin2D(Operator):
    """Wrap :func:`xr_toolz.interpolate.bin_2d`."""

    def __init__(
        self,
        grid: _binning.Grid,
        statistic: str = "mean",
        lon: str = "lon",
        lat: str = "lat",
    ):
        self.grid = grid
        self.statistic = statistic
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _binning.bin_2d(
            da,
            grid=self.grid,
            statistic=self.statistic,
            lon=self.lon,
            lat=self.lat,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "grid": "<Grid>",
            "statistic": self.statistic,
            "lon": self.lon,
            "lat": self.lat,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return Signature(
            {self.lat: len(self.grid.lat), self.lon: len(self.grid.lon)},
            dtype=input_signature.dtype,
        )


class Histogram2D(Operator):
    """Wrap :func:`xr_toolz.interpolate.histogram_2d`."""

    def __init__(self, grid: _binning.Grid, lon: str = "lon", lat: str = "lat"):
        self.grid = grid
        self.lon = lon
        self.lat = lat

    def _apply(self, da):
        return _binning.histogram_2d(da, grid=self.grid, lon=self.lon, lat=self.lat)

    def get_config(self) -> dict[str, Any]:
        return {"grid": "<Grid>", "lon": self.lon, "lat": self.lat}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return Signature(
            {self.lat: len(self.grid.lat), self.lon: len(self.grid.lon)},
            dtype=input_signature.dtype,
        )


# ---------- points → grid --------------------------------------------------


class PointsToGrid(Operator):
    """Wrap :func:`xr_toolz.interpolate.points_to_grid`.

    Expects a 3-tuple ``(lons, lats, values)`` as input.
    """

    def __init__(self, grid: _binning.Grid, statistic: str = "mean"):
        self.grid = grid
        self.statistic = statistic

    def _apply(self, payload) -> xr.DataArray:
        lons, lats, values = payload
        return _points_to_grid.points_to_grid(
            lons, lats, values, grid=self.grid, statistic=self.statistic
        )

    def get_config(self) -> dict[str, Any]:
        return {"grid": "<Grid>", "statistic": self.statistic}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return Signature(
            {"lat": len(self.grid.lat), "lon": len(self.grid.lon)},
            dtype=input_signature.dtype,
        )


class KDEToGrid(Operator):
    """Wrap :func:`xr_toolz.interpolate.kde_to_grid`.

    Expects ``(lons, lats)`` or ``(lons, lats, weights)`` as input.
    """

    def __init__(
        self,
        grid: _binning.Grid,
        *,
        bandwidth: float | str = "scott",
        kernel: str = "gaussian",
        metric: str = "euclidean",
        algorithm: str = "auto",
        output: str = "density",
        rtol: float = 1e-4,
    ):
        self.grid = grid
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.algorithm = algorithm
        self.output = output
        self.rtol = float(rtol)

    def _apply(self, payload) -> xr.DataArray:
        if len(payload) == 2:
            lons, lats = payload
            weights = None
        elif len(payload) == 3:
            lons, lats, weights = payload
        else:
            raise ValueError(
                "KDEToGrid expects (lons, lats) or (lons, lats, weights); "
                f"got payload of length {len(payload)}"
            )
        return _points_to_grid.kde_to_grid(
            lons,
            lats,
            self.grid,
            weights=weights,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            metric=self.metric,
            algorithm=self.algorithm,
            output=self.output,
            rtol=self.rtol,
        )

    def get_config(self) -> dict[str, Any]:
        # Coerce numpy scalar bandwidths so json.dumps(get_config()) works.
        bw = (
            self.bandwidth if isinstance(self.bandwidth, str) else float(self.bandwidth)
        )
        return {
            "grid": "<Grid>",
            "bandwidth": bw,
            "kernel": self.kernel,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "output": self.output,
            "rtol": self.rtol,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return Signature(
            {"lat": len(self.grid.lat), "lon": len(self.grid.lon)},
            dtype=input_signature.dtype,
        )


class IDWToGrid(Operator):
    """Wrap :func:`xr_toolz.interpolate.idw_to_grid`.

    Expects a 3-tuple ``(lons, lats, values)`` as input.
    """

    def __init__(
        self,
        grid: _binning.Grid,
        *,
        k: int = 8,
        power: float = 2.0,
        metric: _knn.Metric = "euclidean",
        max_distance: float | None = None,
        eps: float = 1e-12,
    ):
        _knn._validate_idw_args(k, power, metric, max_distance, eps)
        self.grid = grid
        self.k = k
        self.power = power
        self.metric = metric
        self.max_distance = max_distance
        self.eps = eps

    def _apply(self, payload):
        lons, lats, values = payload
        return _knn.idw_to_grid(
            lons,
            lats,
            values,
            self.grid,
            k=self.k,
            power=self.power,
            metric=self.metric,
            max_distance=self.max_distance,
            eps=self.eps,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "grid": "<Grid>",
            "k": self.k,
            "power": self.power,
            "metric": self.metric,
            "max_distance": self.max_distance,
            "eps": self.eps,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return Signature(
            {"lat": len(self.grid.lat), "lon": len(self.grid.lon)},
            dtype=input_signature.dtype,
        )


class IDWToPoints(Operator):
    """Wrap :func:`xr_toolz.interpolate.idw_to_points`.

    Expects source ``(lons, lats, values)`` as input.
    """

    def __init__(
        self,
        dst_lons: np.ndarray,
        dst_lats: np.ndarray,
        *,
        k: int = 8,
        power: float = 2.0,
        metric: _knn.Metric = "euclidean",
        max_distance: float | None = None,
        eps: float = 1e-12,
    ):
        _knn._validate_idw_args(k, power, metric, max_distance, eps)
        self.dst_lons = np.asarray(dst_lons)
        self.dst_lats = np.asarray(dst_lats)
        self.k = k
        self.power = power
        self.metric = metric
        self.max_distance = max_distance
        self.eps = eps

    def _apply(self, payload):
        lons, lats, values = payload
        return _knn.idw_to_points(
            lons,
            lats,
            values,
            self.dst_lons,
            self.dst_lats,
            k=self.k,
            power=self.power,
            metric=self.metric,
            max_distance=self.max_distance,
            eps=self.eps,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dst_lons": self.dst_lons.tolist(),
            "dst_lats": self.dst_lats.tolist(),
            "k": self.k,
            "power": self.power,
            "metric": self.metric,
            "max_distance": self.max_distance,
            "eps": self.eps,
        }

    def compute_output_signature(self, input_signature: Any) -> Signature:
        # Payload is a (lons, lats, values) tuple — pull dtype from values.
        if isinstance(input_signature, tuple) and len(input_signature) == 3:
            dtype = input_signature[2].dtype
        else:
            dtype = getattr(input_signature, "dtype", None)
        out_shape = np.broadcast_shapes(self.dst_lons.shape, self.dst_lats.shape)
        dims = (
            {"point": int(out_shape[0])}
            if len(out_shape) == 1
            else {f"dim_{i}": int(s) for i, s in enumerate(out_shape)}
        )
        return Signature(dims, dtype=dtype)


# ---------- smoothers ------------------------------------------------------


class MovingAverage(Operator):
    """Wrap :func:`xr_toolz.interpolate._src.smooth.moving_average`."""

    def __init__(
        self,
        dim: str,
        window: int,
        *,
        center: bool = True,
        min_periods: int | None = None,
    ):
        if not isinstance(window, int) or isinstance(window, bool):
            raise TypeError(f"window must be an int, got {type(window).__name__}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if min_periods is not None and (
            not isinstance(min_periods, int) or min_periods < 0
        ):
            raise ValueError(
                f"min_periods must be a non-negative int or None, got {min_periods!r}"
            )
        self.dim = dim
        self.window = window
        self.center = bool(center)
        self.min_periods = min_periods

    def _apply(self, ds):
        return _smooth.moving_average(
            ds,
            dim=self.dim,
            window=self.window,
            center=self.center,
            min_periods=self.min_periods,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "window": self.window,
            "center": self.center,
            "min_periods": self.min_periods,
        }


class GaussianSmooth(Operator):
    """Wrap :func:`xr_toolz.interpolate._src.smooth.gaussian_smooth`."""

    def __init__(self, dim: str, sigma: float, *, truncate: float = 4.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.dim = dim
        self.sigma = float(sigma)
        self.truncate = float(truncate)

    def _apply(self, ds):
        return _smooth.gaussian_smooth(
            ds, dim=self.dim, sigma=self.sigma, truncate=self.truncate
        )

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "sigma": self.sigma, "truncate": self.truncate}


class LowpassFilter(Operator):
    """Wrap :func:`xr_toolz.interpolate._src.smooth.lowpass_filter`.

    For ``btype`` in ``{"low", "high", "lowpass", "highpass"}`` ``cutoff``
    is a scalar in ``(0, 1)``. For ``btype`` in
    ``{"bandpass", "bandstop"}`` it is a length-2 sequence. Validation
    is delegated to the Tier A kernel.
    """

    def __init__(
        self,
        dim: str,
        cutoff: Any,
        *,
        order: int = 4,
        btype: str = "low",
    ):
        if not isinstance(order, int) or isinstance(order, bool):
            raise TypeError(f"order must be an int, got {type(order).__name__}")
        self.dim = dim
        if np.isscalar(cutoff):
            self.cutoff: Any = float(cutoff)
        else:
            pair = tuple(float(v) for v in cutoff)
            if len(pair) != 2:
                raise ValueError(f"cutoff sequence must have length 2, got {len(pair)}")
            self.cutoff = pair
        self.order = order
        self.btype = btype

    def _apply(self, ds):
        return _smooth.lowpass_filter(
            ds,
            dim=self.dim,
            cutoff=self.cutoff,
            order=self.order,
            btype=self.btype,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "cutoff": (
                list(self.cutoff) if isinstance(self.cutoff, tuple) else self.cutoff
            ),
            "order": self.order,
            "btype": self.btype,
        }


# ---------- coord remap ----------------------------------------------------


class RemapAxis(Operator):
    """Generic axis remapping (D12).

    Replaces the ``source_axis`` dimension in the input Dataset with a
    new dimension whose coordinate values are ``target_axis``. Every
    numeric variable that carries ``source_axis`` is interpolated onto
    the target axis.

    Parameters
    ----------
    source_axis
        Name of the existing dimension to remap.
    target_axis
        Target coordinate values. If an :class:`xr.DataArray`, its
        ``.name`` becomes the new dim name; otherwise the new dim name
        defaults to ``target_name`` or ``source_axis``.
    target_name
        Optional explicit new dim name.
    method
        ``"linear"`` or ``"nearest"``.
    """

    def __init__(
        self,
        source_axis: str,
        target_axis: xr.DataArray | np.ndarray | list,
        *,
        target_name: str | None = None,
        method: str = "linear",
    ):
        self.source_axis = source_axis
        if isinstance(target_axis, xr.DataArray):
            self._target_da = target_axis
            self._target_values: np.ndarray = np.asarray(
                target_axis.values, dtype=float
            )
            self._inferred_name = target_axis.name
        else:
            self._target_da = None
            self._target_values = np.asarray(target_axis, dtype=float)
            self._inferred_name = None
        self.target_name = target_name
        self.method = method

    def _apply(self, ds):
        target = self._target_da if self._target_da is not None else self._target_values
        return _coord_remap.remap_axis(
            ds,
            source_dim=self.source_axis,
            target_coords=target,
            target_name=self.target_name,
            method=self.method,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "source_axis": self.source_axis,
            "target_axis": self._target_values.tolist(),
            "target_name": self._resolve_target_name(),
            "method": self.method,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        target_name = self._resolve_target_name()
        dims = {}
        for name, size in input_signature.dims.items():
            if name == self.source_axis:
                dims[target_name] = len(self._target_values)
            else:
                dims[name] = size
        return Signature(dims, dtype=input_signature.dtype)

    def _resolve_target_name(self) -> str:
        if self.target_name is not None:
            return self.target_name
        if self._inferred_name is not None:
            return str(self._inferred_name)
        return self.source_axis


# Vertical presets — thin specializations that pin convention names. The
# user supplies the target coordinate values; the preset names the new
# dim and the source dim conventionally.


class _VerticalPreset(RemapAxis):
    """Common base for vertical-axis presets — pins ``target_name`` only."""

    _DEFAULT_TARGET_NAME: str = ""
    _DEFAULT_SOURCE: str = "depth"

    def __init__(
        self,
        target_axis: xr.DataArray | np.ndarray | list,
        *,
        source_axis: str | None = None,
        target_name: str | None = None,
        method: str = "linear",
    ):
        super().__init__(
            source_axis=source_axis or self._DEFAULT_SOURCE,
            target_axis=target_axis,
            target_name=target_name or self._DEFAULT_TARGET_NAME,
            method=method,
        )


class ToSigma(_VerticalPreset):
    """Remap a depth axis to terrain-following ``sigma`` values."""

    _DEFAULT_TARGET_NAME = "sigma"
    _DEFAULT_SOURCE = "depth"


class FromSigma(_VerticalPreset):
    """Remap a ``sigma`` axis back to a fixed depth grid."""

    _DEFAULT_TARGET_NAME = "depth"
    _DEFAULT_SOURCE = "sigma"


class ToIsopycnal(_VerticalPreset):
    """Remap a depth axis to potential-density (isopycnal) levels."""

    _DEFAULT_TARGET_NAME = "sigma_theta"
    _DEFAULT_SOURCE = "depth"


class ToPressureLevels(_VerticalPreset):
    """Remap a height/depth axis to standard pressure levels."""

    _DEFAULT_TARGET_NAME = "pressure"
    _DEFAULT_SOURCE = "level"


class ToHeight(_VerticalPreset):
    """Remap a pressure or hybrid axis to geometric height."""

    _DEFAULT_TARGET_NAME = "height"
    _DEFAULT_SOURCE = "level"


class ToPhase(Operator):
    """Fold a time axis onto a phase axis by binning + averaging.

    Phase is computed as ``((t - epoch) / period) mod 1`` and binned
    into ``n_bins`` evenly-spaced bins on ``[0, 1)``.

    Parameters
    ----------
    time_dim
        Name of the time dimension.
    period
        Length of one cycle, in the same units as the time coordinate.
    n_bins
        Number of phase bins.
    epoch
        Reference time at which phase = 0.
    """

    def __init__(
        self,
        time_dim: str,
        period: float,
        n_bins: int,
        *,
        epoch: float = 0.0,
    ):
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.time_dim = time_dim
        self.period = float(period)
        self.n_bins = int(n_bins)
        self.epoch = float(epoch)

    def _apply(self, ds):
        return _coord_remap.to_phase(
            ds,
            time_dim=self.time_dim,
            period=self.period,
            n_bins=self.n_bins,
            epoch=self.epoch,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "time_dim": self.time_dim,
            "period": self.period,
            "n_bins": self.n_bins,
            "epoch": self.epoch,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        dims = {}
        for name, size in input_signature.dims.items():
            if name == self.time_dim:
                dims["phase"] = self.n_bins
            else:
                dims[name] = size
        return Signature(dims, dtype=input_signature.dtype)


# ---------- learned resolution change --------------------------------------

# Re-export Downscale / Upscale from _src.downscale so all Layer-1 Operators
# are reachable from xr_toolz.interpolate.operators.
Downscale = _downscale.Downscale
Upscale = _downscale.Upscale


__all__ = [
    "Bin2D",
    "CleanMask",
    "Coarsen",
    "Downscale",
    "FillNaNClimatology",
    "FillNaNIDW",
    "FillNaNLaplacian",
    "FillNaNRBF",
    "FillNaNSpatial",
    "FillNaNTemporal",
    "FromSigma",
    "GaussianSmooth",
    "Histogram2D",
    "IDWToGrid",
    "IDWToPoints",
    "KDEToGrid",
    "LowpassFilter",
    "MaskBinaryClosing",
    "MaskBinaryOpening",
    "MaskRemoveSmallHoles",
    "MaskRemoveSmallObjects",
    "MovingAverage",
    "PointsToGrid",
    "Refine",
    "RegridLike",
    "RemapAxis",
    "ResampleTime",
    "ToHeight",
    "ToIsopycnal",
    "ToPhase",
    "ToPressureLevels",
    "ToSigma",
    "Upscale",
]
