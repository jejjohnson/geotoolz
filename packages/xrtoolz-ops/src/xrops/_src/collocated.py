"""Run a kernel on the collocated grid that a field's coordinates describe.

gridax and diagnostix take plain arrays that end in (y, x) and a grid. The grid
of a DataArray is in its coordinates, so `on_grid` builds it, puts the
horizontal dims last, and puts them back. On dask the kernel runs per chunk
with a halo of one cell, and nothing is computed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import gridax
import numpy as np
import xarray as xr

from xrops.grad._src import cartesian
from xrops.grad._src.constants import EARTH_RADIUS


type Geometry = Literal["cartesian", "rectilinear", "spherical"]
type Grid = gridax.CartesianAGrid2D | gridax.SphericalAGrid2D


def horizontal_grid(
    x: np.ndarray, y: np.ndarray, geometry: Geometry, *, periodic: bool = False
) -> Grid:
    """The collocated grid of two coordinates: metres, or degrees on a sphere."""
    if geometry == "rectilinear":
        return gridax.CartesianAGrid2D.from_coords(y=y, x=x)
    nx, ny = x.size, y.size
    dx, dy = (x[-1] - x[0]) / (nx - 1), (y[-1] - y[0]) / (ny - 1)
    if geometry == "cartesian":
        return gridax.CartesianAGrid2D.from_interior(
            nx, ny, nx * dx, ny * dy, x0=x[0], y0=y[0]
        )
    # from_interior's latitudes are half a step inside the range
    return gridax.SphericalAGrid2D.from_interior(
        nx,
        ny,
        (x[0], x[0] + nx * dx),
        (y[0] - 0.5 * dy, y[0] + (ny - 0.5) * dy),
        R=EARTH_RADIUS,
        periodic=("x",) if periodic else (),
    )


def on_grid(
    kernel: Callable[..., Any],
    *fields: xr.DataArray,
    dims: tuple[str, str],
    geometry: Geometry,
    wrap: bool = False,
) -> xr.DataArray:
    """``kernel(*arrays, grid)`` on the grid of two dims of the fields.

    Args:
        kernel: Takes arrays that end in (y, x) and the grid, and returns one
            such array. Its stencil reaches one cell.
        *fields: The fields, each on both ``dims``; other dims broadcast.
        dims: The (x, y) dims, (lon, lat) on a sphere.
        geometry: ``"spherical"`` (degrees), ``"cartesian"`` (evenly spaced) or
            ``"rectilinear"``.
        wrap: Whether a grid of the whole circle of longitudes is periodic.

    Returns:
        The result with the dims of the fields, in the order they come in, and
        without a name or attributes.

    Raises:
        ValueError: If a dim or its coordinate is missing, or the coordinate is
            not 1-D, strictly monotonic and, but for ``"rectilinear"``, evenly
            spaced.
    """
    xdim, ydim = dims
    x, y = _coordinates(fields, dims, geometry)
    periodic = wrap and whole_circle(x, geometry)

    def run(*arrays: Any) -> Any:
        if any(cartesian._is_dask(array) for array in arrays):
            return _per_chunk(kernel, arrays, x, y, geometry, periodic)
        return kernel(*arrays, horizontal_grid(x, y, geometry, periodic=periodic))

    out = xr.apply_ufunc(
        run,
        *fields,
        input_core_dims=[[ydim, xdim]] * len(fields),
        output_core_dims=[[ydim, xdim]],
        dask="allowed",
        keep_attrs=False,
        join="inner",
    )
    order = dict.fromkeys(dim for field in fields for dim in field.dims)
    return out.transpose(*order)


def over_grid(
    kernel: Callable[..., Any],
    *fields: xr.DataArray,
    dims: tuple[str, str],
    geometry: Geometry,
) -> xr.DataArray:
    """As `on_grid`, for a kernel that takes the two dims away.

    The grid of a whole circle of longitudes is periodic. On dask the two dims
    become one chunk, and nothing is computed.
    """
    xdim, ydim = dims
    x, y = _coordinates(fields, dims, geometry)
    grid = horizontal_grid(x, y, geometry, periodic=whole_circle(x, geometry))
    return xr.apply_ufunc(
        lambda *arrays: kernel(*arrays, grid),
        *fields,
        input_core_dims=[[ydim, xdim]] * len(fields),
        dask="parallelized",
        keep_attrs=False,
        output_dtypes=[np.result_type(*(f.dtype for f in fields), np.float32)],
        dask_gufunc_kwargs={"allow_rechunk": True},
        join="inner",
    )


def whole_circle(x: np.ndarray, geometry: Geometry) -> bool:
    """Whether evenly spaced longitudes go once around, the last not the first."""
    if geometry != "spherical":
        return False
    return bool(np.isclose(abs(x[-1] - x[0]) * x.size / (x.size - 1), 360.0))


def _coordinates(
    fields: tuple[xr.DataArray, ...], dims: tuple[str, str], geometry: Geometry
) -> tuple[np.ndarray, np.ndarray]:
    """The x and y of the fields, once their dims and spacing are checked."""
    if geometry not in ("cartesian", "rectilinear", "spherical"):
        raise ValueError(
            f"Unknown geometry {geometry!r}; expected one of "
            "'cartesian', 'rectilinear', 'spherical'."
        )
    for field in fields:
        _check_dims(field, dims, geometry)
    x, y = (_coordinate(fields[0][dim], geometry) for dim in dims)
    return x, y


def _check_dims(field: xr.DataArray, dims: tuple[str, str], geometry: Geometry) -> None:
    for dim in dims:
        if dim not in field.dims:
            raise ValueError(
                f"Dimension {dim!r} not present on DataArray with dims={field.dims}."
            )
        cartesian._require_coord(field, dim)
        cartesian._require_1d_coord(field[dim])
        if geometry != "rectilinear":
            cartesian._uniform_step(field[dim])


def _coordinate(coord: xr.DataArray, geometry: Geometry) -> np.ndarray:
    if geometry == "rectilinear":
        return cartesian._coord_to_float(coord.values, name=coord.name)
    # an even coordinate, without the rounding of its dtype
    start = float(coord[0]) if geometry == "spherical" else 0.0
    values = cartesian._coord_to_float(coord.values, name=coord.name)
    step = (values[-1] - values[0]) / (values.size - 1)
    return start + step * np.arange(values.size)


def _at_least_two(chunks: tuple[int, ...]) -> tuple[int, ...]:
    """The chunks of an axis, each of one cell joined to its neighbour."""
    out: list[int] = []
    for size in chunks:
        if out and (size < 2 or out[-1] < 2):
            out[-1] += size
        else:
            out.append(size)
    return tuple(out)


def _per_chunk(
    kernel: Callable[..., Any],
    arrays: tuple[Any, ...],
    x: np.ndarray,
    y: np.ndarray,
    geometry: Geometry,
    periodic: bool,
) -> Any:
    """``kernel`` per chunk, each with one cell of its neighbours and its own grid.

    A periodic axis becomes one chunk, which wraps by itself.
    """
    import dask.array as dsa

    dtype = np.result_type(*(array.dtype for array in arrays), np.float32)
    shape = np.broadcast_shapes(*(array.shape for array in arrays))
    arrays = tuple(
        dsa.broadcast_to(dsa.asarray(array), shape)
        if array.shape != shape
        else dsa.asarray(array)
        for array in arrays
    )
    _, arrays = dsa.core.unify_chunks(
        *(item for array in arrays for item in (array, range(len(shape))))
    )
    # the one-sided difference at an edge needs three points, two and a halo
    ndim = len(shape)
    wide = {
        axis: _at_least_two(arrays[0].chunks[axis]) for axis in (ndim - 2, ndim - 1)
    }
    if periodic:
        wide[ndim - 1] = (shape[-1],)
    arrays = tuple(array.rechunk(wide) for array in arrays)
    chunks = arrays[0].chunks
    # Each block finds its part of the coordinates in a block of their indices,
    # of length 1 along every other axis.
    index = [
        dsa.arange(shape[axis], chunks=chunks[axis]).reshape(
            [-1 if i == axis else 1 for i in range(ndim)]
        )
        for axis in (ndim - 2, ndim - 1)
    ]

    def block(*blocks: Any) -> Any:
        *values, rows, columns = blocks
        along_x, along_y = x[columns.ravel()], y[rows.ravel()]
        grid = horizontal_grid(along_x, along_y, geometry, periodic=periodic)
        return kernel(*values, grid)

    halo = [{ndim - 2: 1, ndim - 1: 1}] * len(arrays)
    halo += [{ndim - 2: 1}, {ndim - 1: 1}]
    return dsa.map_overlap(
        block,
        *arrays,
        *index,
        depth=halo,
        boundary=["none"] * len(halo),
        trim=True,
        align_arrays=False,
        dtype=dtype,
        meta=np.empty((0,) * ndim, dtype=dtype),
    )
