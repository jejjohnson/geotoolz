"""A GeoArray with stored dims: the shape geoarray 0.3.0 takes (#517, #519).

Prototype only. It differs from today's carrier in one way that matters:
``dims`` is a static field, not ``("time", "band", "y", "x")[-ndim:]``.

Pytree split (design: geoarray.md, "The carrier"):

    data       leaf     (..., y, x)
    transform  leaf     (*batch, 6) float64, one row per leading batch item
    dims       static   names of every axis of data
    labels     static   ((dim, labels), ...) for labelled non-spatial dims
    crs        static
    fill_value static

The transform's batch dims are the first ``transform.ndim - 1`` entries of
``dims``: a stack of 64 patches is ``dims=("patch", "band", "y", "x")`` with a
``(64, 6)`` transform.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pyproj

from geoarray._src.backend import backend_of, is_tracer
from geoarray._src.grid import as_crs, as_transform


SPATIAL: tuple[str, str] = ("y", "x")


@dataclasses.dataclass(frozen=True, eq=False)
class GeoArray:
    """A NumPy or JAX array on a pixel grid, with its dims stored."""

    data: Any
    transform: Any
    dims: tuple[str, ...]
    crs: pyproj.CRS | None
    fill_value: float | int | bool | None = None
    labels: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def __post_init__(self) -> None:
        dims = tuple(self.dims)
        if dims[-2:] != SPATIAL:
            raise ValueError(f"a GeoArray's dims end in ('y', 'x'), got {dims}")
        if len(set(dims)) != len(dims):
            raise ValueError(f"repeated dim names in {dims}")
        if len(dims) != len(self.data.shape):
            raise ValueError(f"dims {dims} for data of shape {self.data.shape}")
        transform = self.transform
        if not is_tracer(transform):
            transform = _as_transform_array(transform, backend_of(self.data))
        batch = tuple(transform.shape[:-1])
        if tuple(self.data.shape[: len(batch)]) != batch:
            raise ValueError(f"transform batch {batch} vs data {self.data.shape}")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "crs", as_crs(self.crs))
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def shape(self) -> tuple[int, ...]:
        """The data's shape."""
        return tuple(self.data.shape)

    @property
    def backend(self) -> str:
        """``"numpy"`` or ``"jax"``."""
        return backend_of(self.data)

    @property
    def batch_dims(self) -> tuple[str, ...]:
        """The dims that index transform rows (``()`` for one scene)."""
        return self.dims[: self.transform.ndim - 1]

    def labels_of(self, dim: str) -> tuple[Any, ...] | None:
        """Static labels of one dim, or None."""
        return dict(self.labels).get(dim)

    def with_data(self, data: Any, dims: tuple[str, ...] | None = None) -> GeoArray:
        """The same georeference around new data (no validation, trace-safe)."""
        dims = self.dims if dims is None else dims
        kept = tuple((d, lab) for d, lab in self.labels if d in dims)
        return _build(data, self.transform, dims, self.crs, self.fill_value, kept)

    def to_jax(self) -> GeoArray:
        """Data and transform on JAX (x64 required for the transform)."""
        import jax.numpy as jnp

        return _build(
            jnp.asarray(self.data),
            jnp.asarray(np.asarray(self.transform), dtype=jnp.float64),
            self.dims,
            self.crs,
            self.fill_value,
            self.labels,
        )

    def to_numpy(self) -> GeoArray:
        """Data and transform on the host."""
        return _build(
            np.asarray(self.data),
            np.asarray(self.transform, dtype=np.float64),
            self.dims,
            self.crs,
            self.fill_value,
            self.labels,
        )

    def __repr__(self) -> str:
        return f"GeoArray[{' '.join(self.dims)}]{self.shape} {self.backend}"


def _as_transform_array(transform: Any, backend: str) -> Any:
    """A ``(*batch, 6)`` float64 array on the data's backend."""
    if backend == "jax":
        import jax.numpy as jnp

        arr = jnp.asarray(transform, dtype=jnp.float64)
    else:
        arr = np.asarray(transform, dtype=np.float64)
        if arr.shape == (6,) or arr.ndim == 0:
            arr = np.asarray(as_transform(transform), dtype=np.float64)
    if arr.shape[-1] != 6:
        raise ValueError(f"a transform ends in 6 numbers, got {arr.shape}")
    return arr


def _build(
    data: Any,
    transform: Any,
    dims: tuple[str, ...],
    crs: Any,
    fill_value: Any,
    labels: tuple[tuple[str, tuple[Any, ...]], ...],
) -> GeoArray:
    """A carrier without validation, as pytree unflattening needs."""
    g = object.__new__(GeoArray)
    for name, value in (
        ("data", data),
        ("transform", transform),
        ("dims", tuple(dims)),
        ("crs", crs),
        ("fill_value", fill_value),
        ("labels", tuple(labels)),
    ):
        object.__setattr__(g, name, value)
    return g


def _register() -> None:
    try:
        import jax
    except ImportError:  # pragma: no cover
        return
    from jax.tree_util import GetAttrKey

    def flatten(g: GeoArray) -> tuple[Any, Any]:
        return (g.data, g.transform), (g.dims, g.crs, g.fill_value, g.labels)

    def flatten_with_keys(g: GeoArray) -> tuple[Any, Any]:
        leaves, aux = flatten(g)
        return ((GetAttrKey("data"), leaves[0]), (GetAttrKey("transform"), leaves[1])), aux

    def unflatten(aux: Any, leaves: Any) -> GeoArray:
        dims, crs, fill, labels = aux
        return _build(leaves[0], leaves[1], dims, crs, fill, labels)

    jax.tree_util.register_pytree_with_keys(
        GeoArray, flatten_with_keys, unflatten, flatten_func=flatten
    )


_register()
