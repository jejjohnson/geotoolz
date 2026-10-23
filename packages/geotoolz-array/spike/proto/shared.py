"""Shared dim-level functions over the adapters (pipekit-array; GA-Q3).

Each is written once and behaves the same on a NumPy or JAX GeoArray, a
DimArray and a (dask-backed) DataArray. Fast paths: ``map_over`` uses
``jax.vmap`` over data and transform rows on JAX carriers, and xarray
reductions stay native so dask stays lazy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from proto.adapters import XarrayAdapter, adapter_for, backend_name
from proto.carrier import GeoArray, _build


def _named(x: Any, what: str) -> Any:
    a = adapter_for(x)
    if a is None:
        raise TypeError(f"{what} takes named carriers, got {type(x).__name__}")
    return a


def _xp(data: Any) -> Any:
    if backend_name(data) == "jax":
        import jax.numpy as jnp

        return jnp
    return np


def combine(f: Callable[..., Any], *xs: Any) -> Any:
    """f of several carriers on one grid; never aligns (GA-G19)."""
    a = _named(xs[0], "combine")
    xs = a.check_compatible(*xs)
    if isinstance(a, XarrayAdapter):
        import xarray as xr

        return xr.apply_ufunc(
            f, *xs, dask="parallelized", join="exact", output_dtypes=[xs[0].dtype]
        )
    out = f(*(a.data(x) for x in xs))
    return a.rewrap(xs[0], out, a.dims(xs[0]), xs[0].labels)


def stack(xs: list[Any], dim: str) -> Any:
    """A new leading dim from carriers on one grid; never aligns (GA-G19)."""
    a = _named(xs[0], "stack")
    xs = list(a.check_compatible(*xs))
    if isinstance(a, XarrayAdapter):
        import xarray as xr

        return xr.concat(
            xs, dim=dim, join="exact", coords="minimal", compat="identical"
        )
    xp = _xp(a.data(xs[0]))
    data = xp.stack([a.data(x) for x in xs])
    return a.rewrap(xs[0], data, (dim, *a.dims(xs[0])), xs[0].labels)


def reduce(x: Any, fn: str, dim: str) -> Any:
    """Reduce over one named dim with ``mean``, ``max``, ``min`` or ``sum``."""
    a = _named(x, "reduce")
    if isinstance(a, XarrayAdapter):
        return getattr(x, fn)(dim=dim)
    dims = a.dims(x)
    xp = _xp(a.data(x))
    out = getattr(xp, fn)(a.data(x), axis=dims.index(dim))
    kept = tuple(d for d in dims if d != dim)
    return a.rewrap(x, out, kept, tuple((d, l) for d, l in x.labels if d in kept))


def isel(x: Any, dim: str, index: int) -> Any:
    """Position lookup on a non-spatial dim; drops the dim."""
    a = _named(x, "isel")
    if isinstance(a, XarrayAdapter):
        return x.isel({dim: index})
    dims = a.dims(x)
    axis = dims.index(dim)
    key = (slice(None),) * axis + (index,)
    kept = tuple(d for d in dims if d != dim)
    labels = tuple((d, l) for d, l in x.labels if d in kept)
    if isinstance(x, GeoArray) and dim in x.batch_dims:
        return _build(
            x.data[key], x.transform[index], kept, x.crs, x.fill_value, labels
        )
    return a.rewrap(x, a.data(x)[key], kept, labels)


def map_over(f: Callable[[Any], Any], x: Any, dim: str) -> Any:
    """f on each item along ``dim``; results stacked along ``dim`` again.

    On a JAX GeoArray this is one ``jax.vmap`` over the data and, when
    ``dim`` indexes transform rows, the transform rows too: one trace for
    every item, whatever its transform (GA-G15).
    """
    a = _named(x, "map_over")
    if isinstance(a, XarrayAdapter):
        import xarray as xr

        items = [f(x.isel({dim: i})) for i in range(x.sizes[dim])]
        return xr.concat(items, dim=dim, join="exact")
    if isinstance(x, GeoArray) and x.backend == "jax":
        return _map_over_jax(f, x, dim)
    items = [f(isel(x, dim, i)) for i in range(a.data(x).shape[a.dims(x).index(dim)])]
    return _restack(items, dim)


def _map_over_jax(f: Callable[[Any], Any], x: GeoArray, dim: str) -> Any:
    import jax

    if x.dims[0] != dim:
        raise ValueError(
            f"map_over on JAX maps the leading dim; {dim!r} is not {x.dims[0]!r}"
        )
    per_row = dim in x.batch_dims
    kept = x.dims[1:]
    labels = tuple((d, l) for d, l in x.labels if d in kept)

    def one(data: Any, transform: Any) -> Any:
        return f(_build(data, transform, kept, x.crs, x.fill_value, labels))

    out = jax.vmap(one, in_axes=(0, 0 if per_row else None))(x.data, x.transform)
    if isinstance(out, GeoArray):
        transform = out.transform if per_row else out.transform[0]
        return _build(
            out.data, transform, (dim, *out.dims), out.crs, out.fill_value, out.labels
        )
    return out


def _restack(items: list[Any], dim: str) -> Any:
    first = items[0]
    if isinstance(first, GeoArray):
        xp = _xp(first.data)
        data = xp.stack([g.data for g in items])
        rows = [np.asarray(g.transform) for g in items]
        same = all(np.array_equal(r, rows[0]) for r in rows)
        transform = first.transform if same else xp.stack([g.transform for g in items])
        return _build(
            data,
            transform,
            (dim, *first.dims),
            first.crs,
            first.fill_value,
            first.labels,
        )
    a = _named(first, "map_over")
    xp = _xp(a.data(first))
    data = xp.stack([a.data(g) for g in items])
    return a.rewrap(first, data, (dim, *a.dims(first)), first.labels)
