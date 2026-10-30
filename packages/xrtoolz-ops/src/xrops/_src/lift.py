"""`lift`: a geoops geometry operator applied per item over an xarray cube (OP-O4).

geoops geometry operators (reproject, resample, gradient, …) take their
geometry from each item's grid, so they need a `geoarray.GeoArray` per item
and refuse a ``DataArray``. ``xrops.lift(op, dim="time")`` runs ``op`` on each
item along ``dim``: the item becomes a ``GeoArray`` (`GeoArray.from_xarray`,
reading the CF ``spatial_ref``), goes through ``op``, and comes back with
`GeoArray.to_xarray`. On dask the items run inside `xarray.map_blocks`, one
task per ``dim`` chunk, and nothing is computed until the result is.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, ClassVar

import numpy as np
import xarray as xr
from pipekit import Operator as _PipekitOperator

from xrops._src.operator import Operator


def _is_dask(da: xr.DataArray) -> bool:
    return da.chunks is not None


def _one(op: _PipekitOperator, item: xr.DataArray) -> xr.DataArray:
    from geoarray import GeoArray

    return op(GeoArray.from_xarray(item)).to_xarray()


def _run(op: _PipekitOperator, block: xr.DataArray, dim: Hashable) -> xr.DataArray:
    items = [_one(op, block.isel({dim: i})) for i in range(block.sizes[dim])]
    return xr.concat(items, dim=dim, join="exact").assign_coords(
        {dim: block[dim]} if dim in block.coords else {}
    )


def _run_block(block: xr.DataArray, *, op: _PipekitOperator, dim: Hashable) -> Any:
    return _run(op, block, dim)


class Lift(Operator):
    """``op`` applied to each item of a cube along ``dim``; built by `lift`.

    Args:
        op: A geoops operator that takes and returns one ``GeoArray``.
        dim: The dim whose items are run one at a time (``"time"``).

    Raises:
        ValueError: A dask cube is chunked along a dim other than ``dim``
            (each item must be whole), or ``dim`` is not a dim of the cube.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "dask")

    def __init__(self, op: _PipekitOperator, *, dim: str) -> None:
        self.op = op
        self.dim = dim

    def _apply(self, cube: xr.DataArray) -> xr.DataArray:
        if not isinstance(cube, xr.DataArray):
            raise TypeError(
                "lift takes a DataArray with a CF spatial_ref; "
                f"got {type(cube).__name__}."
            )
        if self.dim not in cube.dims:
            raise ValueError(f"lift: {self.dim!r} is not a dim of {tuple(cube.dims)}.")
        if not _is_dask(cube):
            return _run(self.op, cube, self.dim)
        assert cube.chunks is not None
        split = [
            d
            for d, sizes in zip(cube.dims, cube.chunks, strict=True)
            if d != self.dim and len(sizes) > 1
        ]
        if split:
            whole = ", ".join(f"{d!r}: -1" for d in split)
            raise ValueError(
                f"lift runs each item along {self.dim!r} whole, but the cube is "
                f"chunked along {split}; rechunk with .chunk({{{whole}}})."
            )
        template = self._template(cube)
        return cube.map_blocks(
            _run_block, kwargs={"op": self.op, "dim": self.dim}, template=template
        )

    def _template(self, cube: xr.DataArray) -> xr.DataArray:
        """The lazy output's layout, from ``op`` on one zero-filled item.

        A geometry operator's output grid follows the item's grid, not its
        values, so a zero item with the same coordinates gives the output's
        dims, sizes, coordinates and dtype without computing the cube.
        """
        first = cube.isel({self.dim: 0})
        dummy = xr.DataArray(
            np.zeros(first.shape, dtype=first.dtype),
            dims=first.dims,
            coords=first.coords,
            attrs=first.attrs,
            name=first.name,
        )
        item = _one(self.op, dummy).drop_vars(self.dim, errors="ignore")
        axis = cube.get_axis_num(self.dim)
        assert cube.chunks is not None
        # dask before the expansion: a lazy broadcast of the one item, never
        # an array the size of the whole output
        expanded = item.chunk().expand_dims({self.dim: cube.sizes[self.dim]})
        if self.dim in cube.coords:
            expanded = expanded.assign_coords({self.dim: cube[self.dim]})
        return expanded.chunk({self.dim: cube.chunks[axis]})


def lift(op: _PipekitOperator, *, dim: str) -> Lift:
    """Apply a geoops geometry operator per item over an xarray cube.

    Args:
        op: A geoops operator on one ``GeoArray`` (``geoops.geom.Reproject``,
            ``Resample``, ``Gradient2D``, …).
        dim: The dim to iterate over; each item along it is one ``GeoArray``.

    Returns:
        An operator on a ``DataArray`` with a CF ``spatial_ref`` (geoarray's
        `to_xarray` layout): ``dim`` first, then ``op``'s output dims, with
        ``dim``'s coordinate carried through. Lazy on dask, one task per
        ``dim`` chunk; the cube must be whole along every other dim.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from geoarray import GeoArray
        >>> import geoops.geom as gg
        >>> items = [
        ...     GeoArray(np.full((4, 5), float(t)), (1.0, 0, 0, 0, -1.0, 4.0),
        ...              "EPSG:4326", dims=("y", "x")).to_xarray()
        ...     for t in range(3)
        ... ]
        >>> cube = xr.concat(items, dim="time").assign_coords(time=[0, 1, 2])
        >>> lift(gg.Gradient2D(), dim="time")(cube).dims
        ('time', 'component', 'y', 'x')
    """
    return Lift(op, dim=dim)


__all__ = ["Lift", "lift"]
