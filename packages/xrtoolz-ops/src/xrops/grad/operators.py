"""`Gradient` and `Difference`: the xrops.grad functions as operators (OP-O7).

Dims are named, steps come from the coordinates, and the stencils are
geoops's (:func:`geoops.geom.difference`), so ``xrops.Gradient`` and
``geoops.geom.Gradient2D`` give one set of numbers for one field (with
``ellipsoid=`` matching geoops's geographic metric).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import xarray as xr

from xrops._src.operator import Operator
from xrops.grad._src.constants import EARTH_RADIUS
from xrops.grad._src.operators import Geometry, partial


NanPolicy = Literal["propagate", "adaptive"]


def _geom_kw(
    geometry: Geometry, lon: str, lat: str, radius: float, ellipsoid: str | None
) -> dict[str, Any]:
    if geometry != "spherical":
        if ellipsoid is not None:
            raise ValueError("ellipsoid= needs geometry='spherical'.")
        return {}
    return {"lon": lon, "lat": lat, "radius": radius, "ellipsoid": ellipsoid}


def _each(x: xr.DataArray | xr.Dataset, fn: Any) -> xr.DataArray | xr.Dataset:
    if isinstance(x, xr.Dataset):
        return x.map(fn, keep_attrs=True)
    return fn(x)


class Difference(Operator):
    """The ``derivative``-th partial derivative along one dim.

    Args:
        dim: The dim to differentiate along.
        derivative: The derivative order.
        geometry: ``"cartesian"`` (uniform spacing), ``"rectilinear"``
            (non-uniform 1-D coordinate) or ``"spherical"`` (lon/lat in
            degrees, derivatives per metre).
        accuracy: The accuracy order of the one-sided stencils.
        method: ``"central"``, ``"forward"`` or ``"backward"``.
        periodic: The dim wraps (a global longitude).
        nan_policy: ``"propagate"`` (a NaN costs a halo) or ``"adaptive"``
            (per point, the widest stencil on finite data).
        lon: Longitude coordinate name (spherical).
        lat: Latitude coordinate name (spherical).
        radius: Sphere radius in metres (spherical, the default metric).
        ellipsoid: A pyproj ellipsoid name for an ellipsoidal metric
            (spherical), as geoops's on a geographic CRS.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0.0, 3.0, 4)
        >>> da = xr.DataArray(x**2, dims="x", coords={"x": x})
        >>> Difference("x")(da).values
        array([1., 2., 4., 5.])
    """

    def __init__(
        self,
        dim: str,
        *,
        derivative: int = 1,
        geometry: Geometry = "cartesian",
        accuracy: int = 1,
        method: str = "central",
        periodic: bool = False,
        nan_policy: NanPolicy = "propagate",
        lon: str = "lon",
        lat: str = "lat",
        radius: float = EARTH_RADIUS,
        ellipsoid: str | None = None,
    ) -> None:
        self.dim = dim
        self.derivative = derivative
        self.geometry = geometry
        self.accuracy = accuracy
        self.method = method
        self.periodic = periodic
        self.nan_policy = nan_policy
        self.lon = lon
        self.lat = lat
        self.radius = radius
        self.ellipsoid = ellipsoid
        _geom_kw(geometry, lon, lat, radius, ellipsoid)

    def _apply(self, x: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        kw = _geom_kw(self.geometry, self.lon, self.lat, self.radius, self.ellipsoid)

        def one(da: xr.DataArray) -> xr.DataArray:
            return partial(
                da,
                self.dim,
                order=self.derivative,
                geometry=self.geometry,
                accuracy=self.accuracy,
                method=self.method,
                periodic=self.periodic,
                nan_policy=self.nan_policy,
                **kw,
            )

        return _each(x, one)


class Gradient(Operator):
    """The first partial derivatives along ``dims``, stacked on a ``component`` dim.

    The result has a new leading ``component`` dim labelled ``"d/d<dim>"``
    in ``dims`` order; a Dataset gets one per variable.

    Args:
        dims: The dims to differentiate along.
        geometry: As for `Difference`.
        accuracy: The accuracy order of the one-sided stencils.
        method: ``"central"``, ``"forward"`` or ``"backward"``.
        periodic: The dim, or dims, that wrap (a global longitude).
        nan_policy: ``"propagate"`` or ``"adaptive"``.
        lon: Longitude coordinate name (spherical).
        lat: Latitude coordinate name (spherical).
        radius: Sphere radius in metres (spherical).
        ellipsoid: A pyproj ellipsoid name for an ellipsoidal metric.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> y, x = np.arange(3.0), np.arange(4.0)
        >>> da = xr.DataArray(3.0 * x + 2.0 * y[:, None], dims=("y", "x"),
        ...                   coords={"y": y, "x": x})
        >>> g = Gradient(dims=("y", "x"))(da)
        >>> g.dims, g["component"].values.tolist()
        (('component', 'y', 'x'), ['d/dy', 'd/dx'])
        >>> float(g.sel(component="d/dx").mean()), float(g.sel(component="d/dy").mean())
        (3.0, 2.0)
    """

    def __init__(
        self,
        dims: Sequence[str],
        *,
        geometry: Geometry = "cartesian",
        accuracy: int = 1,
        method: str = "central",
        periodic: str | Sequence[str] = (),
        nan_policy: NanPolicy = "propagate",
        lon: str = "lon",
        lat: str = "lat",
        radius: float = EARTH_RADIUS,
        ellipsoid: str | None = None,
    ) -> None:
        if isinstance(dims, str) or not dims:
            raise ValueError("Gradient needs a sequence of one or more dims.")
        self.dims = tuple(dims)
        self.geometry = geometry
        self.accuracy = accuracy
        self.method = method
        self.periodic = (periodic,) if isinstance(periodic, str) else tuple(periodic)
        self.nan_policy = nan_policy
        self.lon = lon
        self.lat = lat
        self.radius = radius
        self.ellipsoid = ellipsoid
        _geom_kw(geometry, lon, lat, radius, ellipsoid)

    def _apply(self, x: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        kw = _geom_kw(self.geometry, self.lon, self.lat, self.radius, self.ellipsoid)

        def one(da: xr.DataArray) -> xr.DataArray:
            parts = [
                partial(
                    da,
                    dim,
                    geometry=self.geometry,
                    accuracy=self.accuracy,
                    method=self.method,
                    periodic=dim in self.periodic,
                    nan_policy=self.nan_policy,
                    **kw,
                ).rename(da.name)
                for dim in self.dims
            ]
            labels = [f"d/d{dim}" for dim in self.dims]
            out = xr.concat(parts, dim="component", coords="minimal", join="exact")
            return out.assign_coords(component=labels).assign_attrs(da.attrs)

        return _each(x, one)


__all__ = ["Difference", "Gradient"]
