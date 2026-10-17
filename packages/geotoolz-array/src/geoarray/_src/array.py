"""`GeoArray`: a NumPy or JAX array with its affine transform, CRS, fill and attrs.

Composition, not subclassing: JAX operations always return a plain
``jax.Array``, so a subclass would lose its metadata on the first ``jnp``
call. A GeoArray *holds* its array and is registered as a JAX pytree:

    leaves  data       (..., H, W)  NumPy or JAX
            transform  (6,) float64 (a, b, c, d, e, f)
    static  crs, fill_value, attrs

Changing a leaf never recompiles a jitted function; changing a static field
does. So a new transform (another scene, another patch) reuses the compiled
code, and gradients reach the georeferencing.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, cast

import numpy as np
import pyproj

from geoarray._src.attrs import FrozenAttrs
from geoarray._src.backend import (
    Backend,
    SupportsArray,
    backend_of,
    import_jax,
    is_jax_array,
    is_tracer,
    namespace,
    require_x64,
)
from geoarray._src.errors import GeoArrayTraceError, X64Error
from geoarray._src.grid import Grid, Transform, as_crs, as_transform


__all__ = ["DIMS", "GeoArray"]

#: Names of the axes of a 4-D carrier; a carrier with ``ndim`` axes uses
#: ``DIMS[-ndim:]``, so the spatial axes are always the last two.
DIMS: tuple[str, str, str, str] = ("time", "band", "y", "x")


class _Inherit:
    """Sentinel: keep the carrier's own fill value."""

    def __repr__(self) -> str:
        return "INHERIT"


INHERIT = _Inherit()

_FillValue = float | int | bool | None


def _normalize_fill(fill: Any) -> _FillValue:
    """A hashable Python scalar fill; every NaN becomes the one `math.nan`."""
    if fill is None:
        return None
    if isinstance(fill, np.generic):
        fill = fill.item()
    if isinstance(fill, bool | int | float):
        if isinstance(fill, float) and math.isnan(fill):
            return math.nan
        return fill
    raise TypeError(f"fill_value must be a number, a bool or None, got {fill!r}")


def _coerce_transform(transform: Any, backend: Backend) -> Any:
    """The transform as a ``(6,)`` float64 array of the data's backend."""
    if backend == "numpy":
        if is_tracer(transform):
            raise GeoArrayTraceError(
                "NumPy data cannot take a traced JAX transform: build the GeoArray "
                "from JAX data (jnp.asarray(data)) inside jit / grad / vmap"
            )
        if is_jax_array(transform):
            transform = np.asarray(transform)
        if (
            isinstance(transform, np.ndarray)
            and transform.ndim >= 1
            and transform.shape[-1] == 6
        ):
            return transform.astype(np.float64, copy=False)
        return np.asarray(as_transform(transform), dtype=np.float64)

    import jax.numpy as jnp

    if is_jax_array(transform):
        if transform.ndim < 1 or transform.shape[-1] != 6:
            raise ValueError(
                f"the transform has shape (6,) (or (*batch, 6)), got {transform.shape}"
            )
        if transform.dtype != jnp.float64:
            require_x64()
            raise X64Error(
                f"the transform must be float64, got {transform.dtype}: build it "
                "after jax.config.update('jax_enable_x64', True)"
            )
        return transform
    require_x64()
    if (
        isinstance(transform, np.ndarray)
        and transform.ndim >= 1
        and transform.shape[-1] == 6
    ):
        return jnp.asarray(transform, dtype=jnp.float64)
    return jnp.asarray(as_transform(transform), dtype=jnp.float64)


def _crs_label(crs: pyproj.CRS | None) -> str:
    """A short label for repr: ``EPSG:32613``, or the CRS name."""
    if crs is None:
        return "None"
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0]}:{authority[1]}"
    return repr(crs.name)


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class GeoArray[A: SupportsArray]:
    """A georeferenced array whose data is a NumPy or a JAX array.

    The last two axes are spatial (H, W); a carrier is 2-D to 4-D with dims
    ``("time", "band", "y", "x")[-ndim:]``, so the band axis is always −3.

    Args:
        data: ``(..., H, W)`` array, ``numpy.ndarray`` or ``jax.Array``.
        transform: The affine transform (a, b, c, d, e, f): an
            ``affine.Affine``, six numbers, or a ``(6,)`` array. Stored as a
            ``(6,)`` float64 array on the data's backend.
        crs: Anything ``pyproj.CRS.from_user_input`` accepts, or ``None``.
        fill_value: The nodata marker (NaN allowed); ``None`` means none.
        attrs: Metadata, frozen into a hashable `FrozenAttrs`.

    Raises:
        TypeError: ``data`` is neither a NumPy nor a JAX array.
        ValueError: ``data`` is not 2-D to 4-D, or ``transform`` / ``crs``
            cannot be parsed.
        X64Error: JAX data without ``jax_enable_x64``.

    Example:
        >>> import numpy as np
        >>> scene = GeoArray(
        ...     np.zeros((2, 600, 600), dtype=np.float32),
        ...     (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
        ...     "EPSG:32613",
        ...     fill_value=np.nan,
        ...     attrs={"band_names": ["B11", "B12"]},
        ... )
        >>> scene.dims, scene.res
        (('band', 'y', 'x'), (20.0, 20.0))
        >>> scene.attrs["band_names"]
        ('B11', 'B12')
    """

    data: A
    transform: Any
    crs: pyproj.CRS | None
    fill_value: _FillValue = None
    attrs: FrozenAttrs = dataclasses.field(default_factory=FrozenAttrs)

    def __post_init__(self) -> None:
        backend = backend_of(self.data)
        transform = _coerce_transform(self.transform, backend)
        batch = tuple(transform.shape[:-1])
        data_shape = tuple(self.data.shape)
        if data_shape[: len(batch)] != batch:
            raise ValueError(
                f"a batched transform {tuple(transform.shape)} needs data with the "
                f"same leading batch shape {batch}, got data {data_shape}"
            )
        if not 2 <= len(data_shape) - len(batch) <= 4:
            raise ValueError(
                f"a GeoArray is 2-D to 4-D with dims {DIMS}[-ndim:], got shape "
                f"{data_shape}" + (f" with batch shape {batch}" if batch else "")
            )
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "crs", as_crs(self.crs))
        object.__setattr__(self, "fill_value", _normalize_fill(self.fill_value))
        if not isinstance(self.attrs, FrozenAttrs):
            object.__setattr__(self, "attrs", FrozenAttrs(self.attrs or {}))

    # -- what it is ---------------------------------------------------------

    @property
    def backend(self) -> Backend:
        """``"numpy"`` or ``"jax"``."""
        return backend_of(self.data)

    @property
    def xp(self) -> Any:
        """The array module of the data: ``numpy`` or ``jax.numpy``."""
        return namespace(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        """The data's shape, ``(..., H, W)``."""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """2, 3 or 4."""
        return len(self.data.shape)

    @property
    def dtype(self) -> Any:
        """The data's dtype."""
        return self.data.dtype

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch axes: ``()`` for one scene, ``(P,)`` for P patches.

        A batched carrier (one transform per element, ``(*batch, 6)``) is what
        ``jax.vmap`` builds and returns; index or ``vmap`` it to get scenes.
        """
        return tuple(self.transform.shape[:-1])

    @property
    def dims(self) -> tuple[str, ...]:
        """Axis names, ``("time", "band", "y", "x")[-n:]``, after any batch axes."""
        n_batch = len(self.batch_shape)
        return ("batch",) * n_batch + DIMS[-(self.ndim - n_batch) :]

    @property
    def height(self) -> int:
        """H, the number of rows."""
        return int(self.data.shape[-2])

    @property
    def width(self) -> int:
        """W, the number of columns."""
        return int(self.data.shape[-1])

    @property
    def count(self) -> int:
        """The number of bands: ``shape[-3]``, or 1 for a 2-D carrier."""
        return int(self.data.shape[-3]) if self.ndim - len(self.batch_shape) >= 3 else 1

    # -- where it is --------------------------------------------------------

    def _concrete_transform(self, what: str) -> Transform:
        if self.batch_shape:
            raise ValueError(
                f"{what} is per scene, but this carrier is batched {self.batch_shape}: "
                "index it, or read it inside jax.vmap"
            )
        if is_tracer(self.transform):
            raise GeoArrayTraceError(
                f"{what} needs a concrete transform, but this GeoArray is traced "
                "(inside jit / grad / vmap): read g.transform there, or compute "
                f"{what} outside the transform"
            )
        return as_transform(np.asarray(self.transform))

    @property
    def grid(self) -> Grid:
        """The pixel grid ``(transform, crs, (H, W))`` as a hashable `Grid`.

        Raises:
            GeoArrayTraceError: Inside ``jit`` / ``grad`` / ``vmap``.
        """
        return Grid(
            self._concrete_transform("grid"), self.crs, (self.height, self.width)
        )

    @property
    def res(self) -> tuple[float, float]:
        """Pixel size ``(x, y)`` in CRS units (see `Grid.res`)."""
        return self.grid.res

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` (see `Grid.bounds`)."""
        return self.grid.bounds

    # -- new carriers from this one ----------------------------------------

    def replace(self, **changes: Any) -> GeoArray:
        """A copy with fields replaced (``data``, ``transform``, ``crs``, …).

        Example:
            >>> import numpy as np
            >>> g = GeoArray(np.zeros((4, 4)), (1, 0, 0, 0, -1, 0), None)
            >>> g.replace(fill_value=0).fill_value
            0
        """
        return dataclasses.replace(self, **changes)

    def with_data(
        self,
        data: Any,
        *,
        transform: Any = None,
        fill_value: Any = INHERIT,
        attrs: Any = None,
    ) -> GeoArray:
        """A carrier with new data on this carrier's grid (or ``transform``).

        Args:
            data: The new ``(..., H, W)`` array.
            transform: A new transform, for data on a new grid. When omitted,
                ``data`` must keep this carrier's spatial shape.
            fill_value: The new fill; omitted keeps this carrier's.
            attrs: New attrs; omitted keeps this carrier's.

        Raises:
            ValueError: The spatial shape changed and no ``transform`` was
                given.

        Example:
            >>> import numpy as np
            >>> g = GeoArray(np.ones((2, 4, 4)), (1, 0, 0, 0, -1, 0), None)
            >>> g.with_data(np.zeros((1, 4, 4), dtype=bool), fill_value=False).shape
            (1, 4, 4)
        """
        if transform is None:
            if tuple(data.shape[-2:]) != tuple(self.data.shape[-2:]):
                raise ValueError(
                    f"spatial shape changed from {tuple(self.data.shape[-2:])} to "
                    f"{tuple(data.shape[-2:])}: pass transform= for data on a new grid"
                )
            transform = self.transform
        return GeoArray(
            data,
            transform,
            self.crs,
            self.fill_value if fill_value is INHERIT else fill_value,
            self.attrs if attrs is None else attrs,
        )

    def with_attrs(self, **items: Any) -> GeoArray:
        """A copy with attrs added or replaced (see `FrozenAttrs.set`)."""
        return dataclasses.replace(self, attrs=self.attrs.set(**items))

    def drop_attrs(self, *keys: str) -> GeoArray:
        """A copy without the given attrs keys (see `FrozenAttrs.drop`)."""
        return dataclasses.replace(self, attrs=self.attrs.drop(*keys))

    def astype(self, dtype: Any) -> GeoArray:
        """A copy with the data cast to ``dtype``; geometry unchanged."""
        return dataclasses.replace(self, data=self.data.astype(dtype))

    # -- backends -------------------------------------------------------------

    def to_jax(self, device: Any = None) -> GeoArray:
        """This carrier with data and transform on a JAX device.

        Args:
            device: A ``jax.Device``; ``None`` is JAX's default device.

        Raises:
            ImportError: JAX is not installed (``geotoolz-array[jax]``).
            X64Error: ``jax_enable_x64`` is off; the transform is float64.

        Example:
            >>> import jax
            >>> import numpy as np
            >>> jax.config.update("jax_enable_x64", True)
            >>> data = np.zeros((4, 4), np.float32)
            >>> g = GeoArray(data, (1, 0, 0, 0, -1, 0), None).to_jax()
            >>> g.backend, str(g.dtype), str(g.transform.dtype)
            ('jax', 'float32', 'float64')
        """
        jax = import_jax()
        require_x64()
        if self.backend == "jax" and device is None:
            return self
        data = jax.device_put(self.data, device)
        transform = jax.device_put(self.transform, device)
        return GeoArray(data, transform, self.crs, self.fill_value, self.attrs)

    def to_numpy(self) -> GeoArray:
        """This carrier with data and transform copied to NumPy on the host.

        Raises:
            GeoArrayTraceError: Inside ``jit`` / ``grad`` / ``vmap``.
        """
        if self.backend == "numpy":
            return self
        if is_tracer(self.data) or is_tracer(self.transform):
            raise GeoArrayTraceError(
                "to_numpy needs concrete data, but this GeoArray is traced "
                "(inside jit / grad / vmap): call it outside the transform"
            )
        return GeoArray(
            np.asarray(self.data),
            np.asarray(self.transform, dtype=np.float64),
            self.crs,
            self.fill_value,
            self.attrs,
        )

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """The data as a NumPy array, so ``np.asarray(g)`` works."""
        if copy:
            return np.array(self.data, dtype=dtype, copy=True)
        return np.asarray(self.data, dtype=dtype)

    # -- comparison and display ----------------------------------------------

    def identical(self, other: object) -> bool:
        """Equal data (NaN-aware), transform, CRS, fill value and attrs.

        Both carriers must be concrete; the backends may differ.

        Example:
            >>> import numpy as np
            >>> g = GeoArray(np.zeros((2, 2)), (1, 0, 0, 0, -1, 0), "EPSG:4326")
            >>> g.identical(g.with_data(np.zeros((2, 2))))
            True
        """
        if not isinstance(other, GeoArray):
            return False
        mine = self.to_numpy()
        theirs = cast("GeoArray[Any]", other).to_numpy()
        fill_equal = mine.fill_value is theirs.fill_value or (
            mine.fill_value == theirs.fill_value
            and type(mine.fill_value) is type(theirs.fill_value)
        )
        return (
            mine.data.shape == theirs.data.shape
            and mine.data.dtype == theirs.data.dtype
            and bool(
                np.array_equal(
                    mine.data, theirs.data, equal_nan=mine.data.dtype.kind in "fc"
                )
            )
            and bool(np.array_equal(mine.transform, theirs.transform))
            and mine.crs == theirs.crs
            and fill_equal
            and mine.attrs == theirs.attrs
        )

    def __repr__(self) -> str:
        head = f"{self.backend} {self.dtype} {self.shape}"
        if is_tracer(self.data) or is_tracer(self.transform):
            where = "traced"
        else:
            grid = self.grid
            where = f"res={grid.res}, bounds={grid.bounds}"
        keys = tuple(self.attrs)
        return (
            f"GeoArray({head}, crs={_crs_label(self.crs)}, {where}, "
            f"fill_value={self.fill_value!r}, attrs={keys})"
        )


# -- JAX pytree ---------------------------------------------------------------


def _flatten(g: GeoArray) -> tuple[tuple[Any, Any], tuple[Any, Any, Any]]:
    return (g.data, g.transform), (g.crs, g.fill_value, g.attrs)


def _unflatten(aux: tuple[Any, Any, Any], leaves: Any) -> GeoArray:
    # Never validate: JAX unflattens with placeholder leaves (None, object(),
    # ShapeDtypeStruct) while tracing, and a shape check would fail on them.
    g = object.__new__(GeoArray)
    data, transform = leaves
    crs, fill_value, attrs = aux
    object.__setattr__(g, "data", data)
    object.__setattr__(g, "transform", transform)
    object.__setattr__(g, "crs", crs)
    object.__setattr__(g, "fill_value", fill_value)
    object.__setattr__(g, "attrs", attrs)
    return g


def _register_pytree() -> bool:
    """Register `GeoArray` with JAX when JAX is installed; never fail without it."""
    try:
        jax = import_jax()
    except ImportError:  # pragma: no cover - exercised by the base-install job
        return False
    from jax.tree_util import GetAttrKey

    def flatten_with_keys(g: GeoArray) -> tuple[Any, Any]:
        leaves, aux = _flatten(g)
        keyed = ((GetAttrKey("data"), leaves[0]), (GetAttrKey("transform"), leaves[1]))
        return keyed, aux

    jax.tree_util.register_pytree_with_keys(
        GeoArray, flatten_with_keys, _unflatten, flatten_func=_flatten
    )
    return True


PYTREE_REGISTERED = _register_pytree()
