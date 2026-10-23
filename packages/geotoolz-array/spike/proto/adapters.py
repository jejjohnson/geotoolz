"""The adapter registry and three adapters (pipekit#99, #100, #104; geoarray #517).

An adapter is how carrier-agnostic code (KernelOp, the shared functions)
reaches a carrier without the carrier importing pipekit. Lookup walks the
carrier type's MRO.

One change from ``pipekit_kernels.md``: ``check_compatible`` returns the
carriers instead of only raising. Under ``jit`` a transform is a tracer, so
an exact grid check cannot be a Python ``if``; the adapter attaches a
runtime check (``equinox.error_if``) to the data and hands the guarded
carriers back, and the caller must use what it returns.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from geoarray._src.backend import is_tracer
from geoarray._src.errors import GridMismatchError
from proto.carrier import SPATIAL, GeoArray
from proto.dimarray import DimArray


class BackendError(TypeError):
    """Carriers on different backends, or an adapter that is not installed."""


class Adapter(Protocol):
    """What carrier-agnostic code needs from a carrier family."""

    kind: str

    def dims(self, x: Any) -> tuple[str, ...]: ...
    def data(self, x: Any) -> Any: ...
    def labels(self, x: Any, dim: str) -> tuple[Any, ...] | None: ...
    def rewrap(self, template: Any, data: Any, dims: tuple[str, ...], labels: Any) -> Any: ...
    def check_compatible(self, *xs: Any) -> tuple[Any, ...]: ...


_REGISTRY: dict[type, Adapter] = {}


def register_adapter(cls: type, adapter: Adapter) -> None:
    """Register an adapter for a carrier class (and its subclasses)."""
    _REGISTRY[cls] = adapter


def unregister_adapter(cls: type) -> None:
    """Remove a registration (tests use this to mimic a missing extra)."""
    _REGISTRY.pop(cls, None)


def adapter_for(x: Any) -> Adapter | None:
    """The adapter for x's type, or None for a raw array.

    Raises:
        BackendError: x is an xarray object and no xarray adapter is
            registered (the ``pipekit-array[xarray]`` extra is missing).
    """
    for cls in type(x).__mro__:
        if cls in _REGISTRY:
            return _REGISTRY[cls]
    if type(x).__module__.startswith("xarray."):
        raise BackendError(
            f"{type(x).__name__} needs the xarray adapter: "
            "pip install 'pipekit-array[xarray]'"
        )
    return None


def backend_name(x: Any) -> str:
    """``numpy``, ``jax`` or ``dask`` for a raw array."""
    mod = type(x).__module__
    if mod.startswith(("jax", "jaxlib")) or is_tracer(x):
        return "jax"
    if mod.startswith("dask"):
        return "dask"
    return "numpy"


def merged_labels(xs: Any, dims: tuple[str, ...]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Labels for ``dims``, taken from the first input that has them."""
    out: dict[str, tuple[Any, ...]] = {}
    for x in xs:
        ad = adapter_for(x)
        if ad is None:
            continue
        for d in dims:
            if d not in out:
                lab = ad.labels(x, d)
                if lab is not None:
                    out[d] = lab
    return tuple((d, out[d]) for d in dims if d in out)


class DimArrayAdapter:
    """The reference adapter: dims, sizes and labels must agree."""

    kind = "none"

    def dims(self, x: DimArray) -> tuple[str, ...]:
        return x.dims

    def data(self, x: DimArray) -> Any:
        return x.data

    def labels(self, x: DimArray, dim: str) -> tuple[Any, ...] | None:
        return x.labels_of(dim)

    def rewrap(self, template: Any, data: Any, dims: tuple[str, ...], labels: Any) -> DimArray:
        return DimArray(data, dims, labels)

    def check_compatible(self, *xs: Any) -> tuple[Any, ...]:
        sizes: dict[str, int] = {}
        for x in xs:
            for d, n in zip(x.dims, x.shape, strict=True):
                if sizes.setdefault(d, n) != n:
                    raise ValueError(f"dim {d!r} has sizes {sizes[d]} and {n}")
        return xs


class GeoArrayAdapter:
    """geoarray's adapter: grids must be exactly equal, on every backend."""

    kind = "grid"

    def dims(self, x: GeoArray) -> tuple[str, ...]:
        return x.dims

    def data(self, x: GeoArray) -> Any:
        return x.data

    def labels(self, x: GeoArray, dim: str) -> tuple[Any, ...] | None:
        return x.labels_of(dim)

    def rewrap(self, template: GeoArray, data: Any, dims: tuple[str, ...], labels: Any) -> Any:
        """A GeoArray while the spatial dims and the transform's batch dims survive.

        Otherwise a DimArray: the georeference no longer describes the data.
        """
        batch = template.batch_dims
        if dims[-2:] == SPATIAL and dims[: len(batch)] == batch:
            from proto.carrier import _build

            return _build(data, template.transform, dims, template.crs, template.fill_value, labels)
        return DimArray(data, dims, labels)

    def check_compatible(self, *xs: GeoArray) -> tuple[GeoArray, ...]:
        first = xs[0]
        out = [first]
        for i, g in enumerate(xs[1:], start=1):
            if g.crs != first.crs:
                raise GridMismatchError(f"carrier {i}: CRS {g.crs} is not {first.crs}")
            if g.shape[-2:] != first.shape[-2:] or g.transform.shape != first.transform.shape:
                raise GridMismatchError(
                    f"carrier {i}: grid {g.shape[-2:]} / {g.transform.shape} is not "
                    f"{first.shape[-2:]} / {first.transform.shape}"
                )
            if is_tracer(g.transform) or is_tracer(first.transform):
                out.append(_guard_equal(g, first, i))
                continue
            if not np.array_equal(np.asarray(g.transform), np.asarray(first.transform)):
                raise GridMismatchError(
                    f"carrier {i}: transform {np.asarray(g.transform).tolist()} is not "
                    f"{np.asarray(first.transform).tolist()}"
                )
            out.append(g)
        return tuple(out)


def _guard_equal(g: GeoArray, first: GeoArray, i: int) -> GeoArray:
    """g with a runtime check that its transform equals first's (under jit)."""
    import equinox as eqx
    import jax.numpy as jnp

    differ = jnp.any(g.transform != first.transform)
    data = eqx.error_if(g.data, differ, f"GridMismatchError: carrier {i} transform differs")
    return g.with_data(data)


class XarrayAdapter:
    """pipekit-array's ``[xarray]`` adapter: coordinates must be exactly equal."""

    kind = "xarray"

    def dims(self, x: Any) -> tuple[str, ...]:
        return tuple(x.dims)

    def data(self, x: Any) -> Any:
        return x.data

    def labels(self, x: Any, dim: str) -> tuple[Any, ...] | None:
        if dim in x.coords and x.coords[dim].ndim == 1 and dim not in SPATIAL:
            return tuple(x.coords[dim].values.tolist())
        return None

    def rewrap(self, template: Any, data: Any, dims: tuple[str, ...], labels: Any) -> Any:
        import xarray as xr

        coords = {k: v for k, v in template.coords.items() if set(v.dims) <= set(dims)}
        return xr.DataArray(data, dims=dims, coords=coords, attrs=template.attrs)

    def check_compatible(self, *xs: Any) -> tuple[Any, ...]:
        import xarray as xr

        try:
            xr.align(*xs, join="exact", copy=False)
        except ValueError as exc:
            raise GridMismatchError(f"xarray inputs are not aligned: {exc}") from exc
        return xs


register_adapter(DimArray, DimArrayAdapter())
register_adapter(GeoArray, GeoArrayAdapter())


def register_xarray() -> None:
    """What installing pipekit-array[xarray] does: register the DataArray adapter."""
    import xarray as xr

    register_adapter(xr.DataArray, XarrayAdapter())


try:
    register_xarray()
except ImportError:  # pragma: no cover
    pass
