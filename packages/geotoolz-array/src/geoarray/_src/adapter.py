"""geoarray's named-array adapter, registered with pipekit at import.

Carrier-agnostic code (``pipekit_array``'s dim-level functions and
``KernelOp``, ``Sequential.summary``) reaches a `GeoArray` through this
adapter, never through the carrier importing pipekit-array's machinery:

- ``isel`` slices the data, each dim's coordinate and, on ``y`` / ``x``,
  moves the transform origin; an integer or a list on a spatial dim drops
  the georeference, so the result is a ``pipekit_array.DimArray``.
- ``rewrap`` keeps the georeference while the spatial dims (and the batch
  dims that index transform rows) survive, and returns a ``DimArray``
  otherwise (PK-P7): per-column statistics that consume ``y`` are not on a
  grid any more.
- ``check_compatible`` requires one exact grid and hands the carriers
  back; under ``jit`` the transform comparison runs at run time, as a
  ``jax.debug.callback`` (F1, #541).
- ``batch_dims`` tells ``KernelOp`` that a batched carrier has one
  geometry per item (F4).
- The fast paths ``combine``, ``stack`` and ``concat`` keep geoarray's own
  rules: NumPy promotes to JAX when mixed (C1), fills must agree, and
  coordinates and per-band attrs are joined.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

import numpy as np
from pipekit import Signature, get_adapter, register_adapter
from pipekit_array import DimArray, DimArrayAdapter

from geoarray._src.array import SPATIAL_DIMS, GeoArray, build
from geoarray._src.attrs import drop_band_keys
from geoarray._src.backend import backend_of
from geoarray._src.carriers import GeoPoints, GeoSwath, _build
from geoarray._src.errors import GridMismatchError
from geoarray._src.geometry import (
    _require_carriers,
    _require_same_grid,
    combine_carriers,
    concat_carriers,
    require_equal_leaves,
    stack,
)
from geoarray._src.types import Kind, register_kind_reader, register_lazy_kind_reader


__all__ = ["GeoArrayAdapter", "GeoPointsAdapter", "GeoSwathAdapter", "to_dimarray"]

_DIMARRAY = DimArrayAdapter()


def to_dimarray(g: Any) -> DimArray:
    """A carrier's data, dims and labels as a ``pipekit_array.DimArray``; no grid."""
    labels = {c.dim: c.labels for c in g.coords if c.labels is not None}
    return DimArray(g.data, g.dims, labels)


def _fill_for_dtype(fill: Any, dtype: Any) -> Any:
    """``fill`` when data of ``dtype`` can hold it; ``None`` otherwise."""
    if fill is None:
        return None
    kind = np.dtype(dtype).kind
    if kind in "fc":
        return fill
    if kind == "b":
        return fill if fill in (0, 1) and isinstance(fill, bool | int) else None
    if isinstance(fill, float) and not fill.is_integer():
        return None
    info = np.iinfo(dtype)
    return fill if info.min <= fill <= info.max else None


class GeoArrayAdapter:
    """The `pipekit.NamedArrayAdapter` for `GeoArray`.

    Example:
        >>> import numpy as np
        >>> import pipekit_array as pa
        >>> from geoarray import GeoArray
        >>> t = (20, 0, 0, 0, -20, 0)
        >>> g = GeoArray(np.ones((3, 4, 4)), t, 32613, dims="time y x")
        >>> mean = pa.reduce(g, "mean", "time")
        >>> type(mean).__name__, mean.dims
        ('GeoArray', ('y', 'x'))
        >>> type(pa.reduce(g, "mean", "y")).__name__
        'DimArray'
    """

    def dims(self, x: GeoArray) -> tuple[str, ...]:
        """The stored dims."""
        return x.dims

    def data(self, x: GeoArray) -> Any:
        """The raw NumPy or JAX array."""
        return x.data

    def labels(self, x: GeoArray, dim: str) -> tuple[Any, ...] | None:
        """The dim's coordinate labels; ``None`` for ``y``, ``x`` and bare dims."""
        return None if dim in SPATIAL_DIMS else x.labels(dim)

    def batch_dims(self, x: GeoArray) -> tuple[str, ...]:
        """The dims with one transform per item (F4)."""
        return x.batch_dims

    def isel(self, x: GeoArray, indexers: dict[str, Any]) -> Any:
        """Select by position; dropping or picking spatial indices loses the grid."""
        spatial_ok = all(
            isinstance(k, slice) and (k.step is None or k.step > 0)
            for d, k in indexers.items()
            if d in SPATIAL_DIMS
        )
        if spatial_ok:
            return x.isel(indexers)
        geo = {d: k for d, k in indexers.items() if d not in SPATIAL_DIMS}
        rest = {d: k for d, k in indexers.items() if d in SPATIAL_DIMS}
        partial = x.isel(geo) if geo else x
        return _DIMARRAY.isel(to_dimarray(partial), rest)

    def rewrap(self, x: GeoArray, data: Any, dims: tuple[str, ...]) -> Any:
        """A `GeoArray` on ``x``'s grid while it still describes the data.

        That is when ``dims`` end in ``y x`` with ``x``'s spatial shape and
        start with ``x``'s batch dims and shape. Coordinates and per-band
        attrs are kept for dims that survive with their size; a fill the new
        dtype cannot hold is dropped. Otherwise a ``DimArray`` with the
        surviving labels.
        """
        dims = tuple(dims)
        shape = tuple(data.shape)
        n_batch = len(x.batch_shape)
        on_grid = (
            dims[-2:] == SPATIAL_DIMS
            and shape[-2:] == x.shape[-2:]
            and dims[:n_batch] == x.batch_dims
            and shape[:n_batch] == x.batch_shape
        )
        kept = tuple(
            c for c in x.coords if c.dim in dims and shape[dims.index(c.dim)] == c.size
        )
        if not on_grid:
            labels = {c.dim: c.labels for c in kept if c.labels is not None}
            return DimArray(data, dims, labels)
        attrs = x.attrs
        if "band" not in dims or shape[dims.index("band")] != x.count:
            attrs = drop_band_keys(attrs)
        transform = x.transform
        if backend_of(data) != backend_of(transform):
            # a JAX kernel on NumPy carriers: keep data and transform together
            transform = (
                x.to_jax().transform
                if backend_of(data) == "jax"
                else (np.asarray(transform, dtype=np.float64))
            )
        fill = _fill_for_dtype(x.fill_value, data.dtype)
        return build(data, transform, x.crs, fill, attrs, dims, kept)

    def check_compatible(self, *xs: GeoArray) -> tuple[GeoArray, ...]:
        """The carriers, checked to share one exact grid; use what it returns.

        Raises:
            TypeError: One of them is not a `GeoArray`.
            GridMismatchError: CRS, spatial shape, batch shape or transform
                differ (a traced transform: at run time).
        """
        gs = _require_carriers(xs, "check_compatible")
        return tuple(_require_same_grid(gs, "check_compatible"))

    def signature(self, x: GeoArray) -> Signature:
        """Dims with their sizes, and the dtype's name."""
        return Signature(
            dims=tuple(zip(x.dims, x.shape, strict=True)), dtype=str(x.dtype)
        )

    # -- fast paths ------------------------------------------------------------

    def combine(self, fn: Callable[..., Any], *xs: GeoArray) -> GeoArray:
        """``geoarray.combine``: one grid, one fill, inputs in the first's order."""
        return combine_carriers(fn, *xs)

    def stack(self, xs: Sequence[GeoArray], dim: str) -> GeoArray:
        """``geoarray.stack`` with a named dim."""
        return stack(xs, dim)

    def concat(self, xs: Sequence[GeoArray], dim: str) -> GeoArray:
        """``geoarray.concat``: coordinates and per-band attrs joined."""
        return concat_carriers(xs, dim)

    def map_over(self, x: GeoArray, fn: Callable[[Any], Any], dim: str) -> Any:
        """``fn`` on each item along ``dim``, stacked along ``dim`` first.

        Each item is ``x`` without ``dim``: its own transform row when
        ``dim`` is a batch dim (patches, per-frame grids), its own stamp
        when ``dim`` is ``time``. On JAX this is one ``jax.vmap`` over the
        data and the matching leaves, so 64 patches with 64 transforms are
        one trace (GA-G15); on NumPy, a loop.

        Raises:
            ValueError: ``dim`` is spatial or not a dim, or ``fn`` returns
                something other than one named carrier.
        """
        if dim not in x.dims or dim in SPATIAL_DIMS:
            raise ValueError(f"map_over: {dim!r} is not a non-spatial dim of {x.dims}")
        if x.backend == "jax":
            return _map_over_jax(x, fn, dim)
        n = x.sizes[dim]
        items = [fn(x.isel({dim: i})) for i in range(n)]
        return _restack(x, items, dim)


def _restack(x: GeoArray, items: list[Any], dim: str) -> Any:
    """Per-item results of `map_over` (NumPy) stacked along ``dim``, first."""
    first = items[0]
    if not all(type(item) is type(first) for item in items):
        raise ValueError("map_over: fn returned carriers of different types")
    batched = dim in x.batch_dims
    if isinstance(first, GeoArray):
        xp = first.xp
        data = xp.stack([item.data for item in items])
        if batched:
            transform = xp.stack([xp.asarray(item.transform) for item in items])
        else:
            _require_same_grid(list(items), "map_over")
            transform = first.transform
        coords = (
            tuple(c for c in x.coords if c.dim == dim and c.size == len(items))
            + first.coords
        )
        return build(
            data,
            transform,
            first.crs,
            first.fill_value,
            first.attrs,
            (dim, *first.dims),
            coords,
        )
    adapter = get_adapter(first)
    if adapter is None:
        raise ValueError(
            f"map_over: fn returns {type(first).__name__}, not a named carrier"
        )
    from pipekit_array import stack as stack_named

    out = stack_named(items, dim)
    labels = x.labels(dim)
    if isinstance(out, DimArray) and labels is not None:
        return DimArray(out.data, out.dims, {**out.labels, dim: labels})
    return out


def _map_over_jax(x: GeoArray, fn: Callable[[Any], Any], dim: str) -> Any:
    """`map_over` as one ``jax.vmap`` over the data and the leaves along ``dim``.

    The data is mapped along ``dim``; the transform too when ``dim`` is a
    batch dim; the time stamps when ``dim`` is ``time`` (the item then has
    no time dim, so they are dropped). Unmapped leaves come back broadcast,
    so the result keeps their first row.
    """
    import jax

    from geoarray._src.array import _STAMPS, _flatten, _unflatten

    axis = x.dims.index(dim)
    batched = dim in x.batch_dims
    leaves, (crs, fill, attrs, dims, coords) = _flatten(x)
    over_time = dim == "time"
    if over_time:
        leaves = leaves[:2]
    in_axes = (axis, axis if batched else None, *(None,) * (len(leaves) - 2))
    # the item drops dim's coordinate; over time that is the stamps' slot
    item_coords = tuple(
        c
        for c in coords
        if (c is _STAMPS and not over_time) or (c is not _STAMPS and c.dim != dim)
    )
    item_aux = (crs, fill, attrs, tuple(d for d in dims if d != dim), item_coords)
    captured: dict[str, Any] = {}

    def per_item(*item_leaves: Any) -> Any:
        out = fn(_unflatten(item_aux, item_leaves))
        captured["out"] = out
        return jax.tree_util.tree_leaves(out)

    flat = jax.vmap(per_item, in_axes=in_axes)(*leaves)
    template = captured["out"]
    own = tuple(c for c in x.coords if c.dim == dim)
    if isinstance(template, GeoArray):
        _, (o_crs, o_fill, o_attrs, o_dims, o_coords) = _flatten(template)
        data, transform, *stamps = flat
        out = _unflatten(
            (o_crs, o_fill, o_attrs, (dim, *o_dims), o_coords),
            (data, transform if batched else transform[0], *(s[0] for s in stamps)),
        )
        if own:
            out = build(
                out.data,
                out.transform,
                out.crs,
                out.fill_value,
                out.attrs,
                out.dims,
                own + out.coords,
            )
        return out
    if isinstance(template, DimArray):
        (data,) = flat
        labels = dict(template.labels)
        if own and own[0].labels is not None:
            labels[dim] = own[0].labels
        return DimArray(data, (dim, *template.dims), labels)
    raise ValueError(
        f"map_over: fn returns {type(template).__name__}; on JAX it must return a "
        "GeoArray or a DimArray"
    )


class _GeolocatedAdapter:
    """The adapter of a carrier located by per-sample leaves (swath, points).

    Subclasses name the carrier, its geometry leaves and how a rewrap
    carries them.
    """

    carrier: type
    leaves: tuple[str, ...]

    def dims(self, x: Any) -> tuple[str, ...]:
        """The stored dims."""
        return x.dims

    def data(self, x: Any) -> Any:
        """The raw NumPy or JAX array."""
        return x.data

    def labels(self, x: Any, dim: str) -> tuple[Any, ...] | None:
        """The dim's coordinate labels; ``None`` for the spatial dims."""
        return None if dim in x.spatial_dims else x.labels(dim)

    def batch_dims(self, x: Any) -> tuple[str, ...]:
        """The dims the geometry leaves vary along (F4)."""
        return x.batch_dims

    def signature(self, x: Any) -> Signature:
        """Dims with their sizes, and the dtype's name."""
        return Signature(
            dims=tuple(zip(x.dims, x.shape, strict=True)), dtype=str(x.dtype)
        )

    def _keeps_geometry(self, k: Any) -> bool:
        return isinstance(k, slice) or (
            self.carrier is GeoPoints and not isinstance(k, int | np.integer)
        )

    def isel(self, x: Any, indexers: dict[str, Any]) -> Any:
        """Select by position; an index that breaks the geometry gives a DimArray."""
        if all(
            self._keeps_geometry(k) for d, k in indexers.items() if d in x.spatial_dims
        ):
            return x.isel(indexers)
        geo = {d: k for d, k in indexers.items() if d not in x.spatial_dims}
        rest = {d: k for d, k in indexers.items() if d in x.spatial_dims}
        return _DIMARRAY.isel(to_dimarray(x.isel(geo) if geo else x), rest)

    def check_compatible(self, *xs: Any) -> tuple[Any, ...]:
        """The carriers, checked to share one geolocation exactly.

        Raises:
            TypeError: One of them is another carrier type.
            GridMismatchError: CRS or a geometry leaf differs (traced: at run
                time).
        """
        first = xs[0]
        for i, x in enumerate(xs[1:], start=1):
            if not isinstance(x, self.carrier):
                raise TypeError(
                    f"{self.carrier.__name__} carriers combine with their own kind, "
                    f"got {type(x).__name__}"
                )
            if x.crs != first.crs:
                raise GridMismatchError(f"carrier {i} has CRS {x.crs}, not {first.crs}")
            for name in self.leaves:
                a, b = getattr(first, name), getattr(x, name)
                if (a is None) != (b is None):
                    raise GridMismatchError(f"carrier {i}: {name} is set on one only")
                if a is not None:
                    require_equal_leaves(
                        a, b, f"carrier {i} has a different {name} than carrier 0"
                    )
        return xs

    def rewrap(self, x: Any, data: Any, dims: tuple[str, ...]) -> Any:
        """The carrier while its spatial and batch dims survive; else a DimArray."""
        dims = tuple(dims)
        shape = tuple(data.shape)
        spatial = x.spatial_dims
        n = len(spatial)
        batch = x.batch_dims
        on_geometry = (
            dims[len(dims) - n :] == spatial
            and shape[len(shape) - n :] == x.shape[x.ndim - n :]
            and dims[: len(batch)] == batch
            and shape[: len(batch)] == x.shape[: len(batch)]
        )
        kept = tuple(
            c for c in x.coords if c.dim in dims and shape[dims.index(c.dim)] == c.size
        )
        if not on_geometry:
            labels = {c.dim: c.labels for c in kept if c.labels is not None}
            return DimArray(data, dims, labels)
        fields = {name: getattr(x, name) for name in self._static_fields()}
        fields.update({name: getattr(x, name) for name in self.leaves})
        return _build(
            self.carrier,
            data=data,
            dims=dims,
            coords=kept,
            fill_value=_fill_for_dtype(x.fill_value, data.dtype),
            **fields,
        )

    def _static_fields(self) -> tuple[str, ...]:
        return ("crs", "attrs")


class GeoSwathAdapter(_GeolocatedAdapter):
    """The `pipekit.NamedArrayAdapter` for `GeoSwath`: lat/lon must match exactly.

    A reduction over ``row`` or ``col`` (per-column statistics) is a
    ``DimArray``.
    """

    carrier = GeoSwath
    leaves = ("lat", "lon")


class GeoPointsAdapter(_GeolocatedAdapter):
    """The `pipekit.NamedArrayAdapter` for `GeoPoints`: xy, z, time, valid match.

    A reduction over ``obs`` is a ``DimArray``.
    """

    carrier = GeoPoints
    leaves = ("xy", "z", "time", "valid")

    def _static_fields(self) -> tuple[str, ...]:
        return ("crs", "attrs", "vertical")


register_adapter(GeoArray, GeoArrayAdapter())
register_adapter(GeoSwath, GeoSwathAdapter())
register_adapter(GeoPoints, GeoPointsAdapter())


# the kinds the readable types read (geoarray.types); a DataArray's reader
# loads with the converters, the first time one is read (geotoolz-array[xarray])
def _load_xarray_kind_reader() -> None:
    import_module("geoarray._src.xarray")


register_lazy_kind_reader("xarray.", _load_xarray_kind_reader)
register_kind_reader(GeoArray, lambda g: Kind("grid", ("y", "x")))
register_kind_reader(GeoSwath, lambda s: Kind("swath", ("row", "col")))
register_kind_reader(
    GeoPoints,
    lambda p: Kind(
        "points",
        ("obs",),
        frozenset(k for k in ("z", "time", "valid") if getattr(p, k) is not None),
    ),
)
