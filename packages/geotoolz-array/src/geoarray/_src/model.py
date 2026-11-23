"""`to_geoswath`: a model field on a lat-lon grid, in sensor geometry.

The field and the grid are duck-typed, so geoarray never imports the model's
grid library (gridax). A field has ``data``, ``dims`` and ``stagger``; a grid
has ``ghost`` and the coordinates of its cell centres.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geoarray._src.attrs import FrozenAttrs
from geoarray._src.carriers import GeoSwath
from geoarray._src.coords import Coordinate, CurvilinearGrid


__all__ = ["to_geoswath"]


def to_geoswath(
    f: Any,
    grid: Any,
    *,
    crs: Any = "EPSG:4326",
    coords: tuple[Coordinate, ...] = (),
    fill_value: Any = None,
    attrs: Any = None,
) -> GeoSwath:
    """A cell-centred model field on a latitude-longitude grid as a `GeoSwath`.

    The grid's ``y`` and ``x`` become ``row`` and ``col``, the ghost ring is
    dropped, and every pixel gets the latitude and longitude of its cell
    centre, in degrees. The values are not touched.

    Args:
        f: A field with ``data``, named ``dims`` (``"y"`` and ``"x"`` among
            them) and a ``stagger`` that is empty at the cell centres, such
            as a `gridax.CField`.
        grid: The grid of the field: its ``ghost`` width, and
            ``coords("y", at="T")`` and ``coords("x", at="T")`` in radians on
            a sphere of radius ``R`` (gridax's spherical grids), or 2-D
            ``lat_T`` and ``lon_T`` in radians.
        crs: The CRS of the latitudes and longitudes.
        coords: Coordinates of the other dims of the result.
        fill_value: The nodata marker.
        attrs: Whole-carrier metadata.

    Returns:
        The interior of the field, ``(..., row, col)``; the other dims keep
        their order.

    Raises:
        ValueError: The field is not at the cell centres, has no ``y`` or
            ``x`` dim, or its ``(y, x)`` do not match the grid.
        TypeError: The grid has no latitudes and longitudes.

    Example:
        >>> import numpy as np
        >>> from types import SimpleNamespace
        >>> y, x = np.deg2rad([9.0, 10.0, 11.0, 12.0]), np.deg2rad([-1.0, 0.0, 1.0])
        >>> grid = SimpleNamespace(
        ...     ghost=1, R=6.371e6, coords=lambda axis, at: {"y": y, "x": x}[axis]
        ... )
        >>> h = SimpleNamespace(data=np.ones((4, 3)), dims=("y", "x"), stagger=())
        >>> swath = to_geoswath(h, grid)
        >>> swath.dims, np.round(swath.lat[:, 0], 6).tolist()
        (('row', 'col'), [10.0, 11.0])
    """
    stagger = tuple(f.stagger)
    if stagger:
        where = getattr(f, "loc", None) or dict(stagger)
        raise ValueError(
            f"the field sits at {where}, not at the cell centres: move it there"
            ' first, e.g. gridax.interpolate(f, grid, dst="T")'
        )
    dims = tuple(f.dims)
    missing = [dim for dim in ("y", "x") if dim not in dims]
    if missing:
        raise ValueError(f"the field has dims {dims}, without {missing}")
    lat, lon = _latlon(grid)
    ghost = int(grid.ghost)
    axes = tuple(getattr(grid, "axes", ("y", "x")))
    others = tuple(dim for dim in dims if dim not in ("y", "x"))
    data = f.data.transpose([dims.index(dim) for dim in (*others, "y", "x")])
    if tuple(data.shape[-2:]) != lat.shape:
        raise ValueError(
            f"the field is {tuple(data.shape[-2:])} along (y, x), not {lat.shape}"
        )
    inner = slice(ghost, -ghost) if ghost else slice(None)
    # the ring of every grid axis is dropped, also of a vertical one
    index = tuple(inner if dim in axes else slice(None) for dim in others)
    cells = CurvilinearGrid(lat[inner, inner], lon[inner, inner], crs)
    return GeoSwath.from_grid(
        data[(*index, inner, inner)],
        cells,
        dims=(*others, "row", "col"),
        coords=coords,
        fill_value=fill_value,
        attrs=FrozenAttrs(attrs or {}),
    )


def _latlon(grid: Any) -> tuple[np.ndarray, np.ndarray]:
    """The 2-D latitudes and longitudes of the cell centres, in degrees."""
    lat, lon = getattr(grid, "lat_T", None), getattr(grid, "lon_T", None)
    if lat is None or lon is None:
        if not hasattr(grid, "R") or not hasattr(grid, "coords"):
            raise TypeError(
                f"{type(grid).__name__} has no latitudes and longitudes: to_geoswath "
                "takes a spherical grid, with coords() in radians and a radius R"
            )
        y, x = (np.asarray(grid.coords(axis, at="T")) for axis in ("y", "x"))
        lat, lon = np.meshgrid(y, x, indexing="ij")
    return np.rad2deg(np.asarray(lat)), np.rad2deg(np.asarray(lon))
