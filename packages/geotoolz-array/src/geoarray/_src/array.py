"""`GeoArray`: a NumPy or JAX array with its dims, transform, CRS, fill and attrs.

Composition, not subclassing: JAX operations always return a plain
``jax.Array``, so a subclass would lose its metadata on the first ``jnp``
call. A GeoArray *holds* its array and is registered as a JAX pytree:

    leaves  data       (..., y, x)  NumPy or JAX
            transform  (*batch, 6) float64 (a, b, c, d, e, f)
    static  crs, fill_value, attrs, dims, coords

Changing a leaf never recompiles a jitted function; changing a static field
does. So a new transform (another scene, another patch) reuses the compiled
code, and gradients reach the georeferencing.

The dims are stored, one name per axis, ending in ``("y", "x")``; the first
``transform.ndim - 1`` of them index the transform's rows (the batch dims:
one transform per patch or per frame). A carrier built without ``dims``
reads them from its rank, ``DIMS[-ndim:]``, with a ``DeprecationWarning``
(removed in 0.4.0).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, cast

import numpy as np
import pyproj
from pipekit_array.dimarray import vmapped_dims

from geoarray._src.attrs import FrozenAttrs, drop_band_keys, select_bands
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
from geoarray._src.coords import Coordinate, PixelGrid, TimeAxis
from geoarray._src.errors import GeoArrayTraceError, LabelError, X64Error
from geoarray._src.grid import (
    Grid,
    PixelWindow,
    Transform,
    as_crs,
    as_transform,
    shift_scale_transform,
)
from geoarray._src.options import warn_host_transfer


__all__ = ["DIMS", "SPATIAL_DIMS", "GeoArray"]

#: The dims a carrier built without ``dims`` reads from its rank:
#: ``DIMS[-ndim:]`` (deprecated in 0.3.0, removed in 0.4.0; pass ``dims=``).
DIMS: tuple[str, str, str, str] = ("time", "band", "y", "x")

#: The spatial dims every `GeoArray` ends in.
SPATIAL_DIMS: tuple[str, str] = ("y", "x")

_INFER_WARNING = (
    "GeoArray dims read from the data's rank ({dims}) are deprecated and "
    "removed in geotoolz-array 0.4.0: pass dims={dims!r}"
)


def infer_dims(ndim: int, n_batch: int) -> tuple[str, ...]:
    """The 0.2 layout: ``n_batch`` batch dims, then ``DIMS[-k:]`` for the rest.

    Raises:
        ValueError: The non-batch part is not 2-D to 4-D.
    """
    element = ndim - n_batch
    if not 2 <= element <= 4:
        raise ValueError(
            f"a GeoArray built without dims is 2-D to 4-D with dims {DIMS}[-ndim:], "
            f"got {ndim} axes"
            + (f" with {n_batch} batch axes" if n_batch else "")
            + "; pass dims= for any other layout"
        )
    batch = tuple("batch" if k == 0 else f"batch_{k}" for k in range(n_batch))
    return batch + DIMS[-element:]


def _check_dims(
    dims: Any,
    shape: tuple[int, ...],
    n_batch: int,
    spatial: tuple[str, ...] = SPATIAL_DIMS,
    carrier: str = "GeoArray",
) -> tuple[str, ...]:
    """``dims`` as a validated tuple for data of ``shape``, ending in ``spatial``."""
    if isinstance(dims, str):
        dims = tuple(dims.split())
    out = tuple(dims)
    if not all(isinstance(d, str) and d for d in out):
        raise TypeError(f"dims are non-empty strings, got {dims!r}")
    if len(set(out)) != len(out):
        raise ValueError(f"dims repeat a name: {out}")
    if len(out) != len(shape):
        raise ValueError(
            f"dims {out} name {len(out)} axes, but the data has shape {shape}"
        )
    if out[len(out) - len(spatial) :] != spatial:
        raise ValueError(f"a {carrier}'s dims end in {spatial}, got {out}")
    if n_batch > len(out) - len(spatial):
        raise ValueError(
            f"a georeference with {n_batch} leading axes needs {n_batch} batch dims"
        )
    return out


def _check_coords(
    coords: Any,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    spatial: tuple[str, ...] = SPATIAL_DIMS,
) -> tuple[Coordinate, ...]:
    """``coords`` as a validated tuple: one per non-spatial dim at most, sized."""
    out = tuple(coords or ())
    seen: set[str] = set()
    for c in out:
        if not isinstance(c, Coordinate):
            raise TypeError(
                f"coords holds geoarray.coords axes (SpectralAxis, TimeAxis, …), "
                f"got {type(c).__name__}"
            )
        if c.dim not in dims or c.dim in spatial:
            raise ValueError(
                f"{type(c).__name__} describes dim {c.dim!r}, which is not a "
                f"non-spatial dim of {dims}"
            )
        if c.dim in seen:
            raise ValueError(f"two coordinates for dim {c.dim!r}")
        seen.add(c.dim)
        size = shape[dims.index(c.dim)]
        if c.size != size:
            raise ValueError(
                f"{type(c).__name__} has {c.size} entries, but dim {c.dim!r} has "
                f"size {size}"
            )
    return out


def _coord_for(coords: tuple[Coordinate, ...], dim: str) -> Coordinate | None:
    for c in coords:
        if c.dim == dim:
            return c
    return None


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


def _check_fill_fits(fill: _FillValue, dtype: Any) -> None:
    """Raise unless ``dtype`` data can hold ``fill`` exactly.

    A fill the dtype cannot hold (``-9999.0`` or ``0.5`` in ``uint16``) would
    be written as some other value, which `valid_mask` then reads as data.
    Float and complex data hold any fill, NaN included.
    """
    if fill is None:
        return
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        fits = fill in (0, 1)
    elif dtype.kind in "iu":
        info = np.iinfo(dtype.str)
        fits = (
            not (isinstance(fill, float) and not fill.is_integer())
            and info.min <= fill <= info.max
        )
    else:
        return
    if not fits:
        raise ValueError(
            f"fill_value={fill!r} cannot be held by {dtype} data, so filled "
            f"pixels would read as data: pass a fill {dtype} can hold, or "
            "fill_value=None"
        )


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


def _expand_key(key: Any, ndim: int) -> tuple[Any, ...]:
    """A full-length index tuple: Ellipsis expanded, missing axes as ``:``."""
    if not isinstance(key, tuple):
        key = (key,)
    if any(k is None or isinstance(k, bool | np.bool_) for k in key):
        raise IndexError("GeoArray indexing cannot add axes; use expand_dims")
    ellipses = [i for i, k in enumerate(key) if k is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("an index can only have one ellipsis")
    if ellipses:
        i = ellipses[0]
        fill = ndim - (len(key) - 1)
        key = key[:i] + (slice(None),) * fill + key[i + 1 :]
    if len(key) > ndim:
        raise IndexError(f"too many indices for a {ndim}-D GeoArray")
    return key + (slice(None),) * (ndim - len(key))


def _require_concrete_spatial(spatial: tuple[Any, ...]) -> None:
    """Raise `GeoArrayTraceError` for slice bounds that are JAX tracers."""
    for s in spatial:
        bounds = (s.start, s.stop, s.step) if isinstance(s, slice) else (s,)
        if any(is_tracer(v) for v in bounds):
            raise GeoArrayTraceError(
                "spatial slices need concrete bounds (the output shape and "
                "transform depend on them); use geoarray.crop for traced offsets"
            )


def _drops_axis(k: Any) -> bool:
    """Whether index ``k`` removes its axis (an integer does)."""
    return isinstance(k, int | np.integer)


def _is_advanced(k: Any) -> bool:
    return not isinstance(k, slice | int | np.integer)


def _crs_label(crs: pyproj.CRS | None) -> str:
    """A short label for repr: ``EPSG:32613``, or the CRS name."""
    if crs is None:
        return "None"
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0]}:{authority[1]}"
    return repr(crs.name)


def _keeps_coord(c: Coordinate, dims: tuple[str, ...], shape: tuple[int, ...]) -> bool:
    return c.dim in dims and shape[dims.index(c.dim)] == c.size


def _drops_band_alone(dims: tuple[str, ...], dropped: set[str]) -> bool:
    """Dropping ``band`` but keeping ``time`` leaves (T, H, W), which 0.2 code
    (the geoops kernels, ``axis=-3``) reads as (C, H, W)."""
    return "band" in dropped and "time" in dims and "time" not in dropped


_BAND_ALONE = (
    "dropping the band dim of a (time, band, y, x) carrier would leave "
    "(time, y, x), which code reading the band axis as -3 takes for "
    "(band, y, x); keep a size-1 band ([k] instead of k) until the geoops "
    "kernels take dim= (W3)"
)


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class GeoArray[A: SupportsArray]:
    """A georeferenced array whose data is a NumPy or a JAX array.

    Every axis has a name (``dims``); the last two are ``("y", "x")``. The
    first ``transform.ndim - 1`` dims are batch dims, one transform row per
    item (patches, or frames with their own grids). Non-spatial dims may
    have a coordinate in ``coords`` (`geoarray.coords`: band names and
    wavelengths, level values, time stamps).

    Args:
        data: ``(..., y, x)`` array, ``numpy.ndarray`` or ``jax.Array``.
        transform: The affine transform (a, b, c, d, e, f): an
            ``affine.Affine``, six numbers, or a ``(6,)`` (or ``(*batch, 6)``)
            array. Stored as float64 on the data's backend.
        crs: Anything ``pyproj.CRS.from_user_input`` accepts, or ``None``.
        fill_value: The nodata marker (NaN allowed); ``None`` means none.
        attrs: Whole-carrier metadata, frozen into a hashable `FrozenAttrs`.
        dims: One name per axis, ending in ``("y", "x")``, as a tuple or a
            space-separated string (``"band y x"``). Omitted, they are read
            from the rank (``DIMS[-ndim:]``) with a ``DeprecationWarning``.
        coords: Coordinates of non-spatial dims, at most one per dim, each
            with one entry per index of its dim.

    Raises:
        TypeError: ``data`` is neither a NumPy nor a JAX array, or a
            ``coords`` entry is not a coordinate.
        ValueError: ``dims`` do not fit the data (wrong count, repeated,
            not ending in ``y x``), a coordinate has the wrong size, or
            ``transform`` / ``crs`` cannot be parsed.
        X64Error: JAX data without ``jax_enable_x64``.

    Example:
        >>> import numpy as np
        >>> from geoarray.coords import SpectralAxis
        >>> swir = SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0))
        >>> scene = GeoArray(
        ...     np.zeros((2, 600, 600), dtype=np.float32),
        ...     (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
        ...     "EPSG:32613",
        ...     fill_value=np.nan,
        ...     dims=("band", "y", "x"),
        ...     coords=(swir,),
        ... )
        >>> scene.dims, scene.res
        (('band', 'y', 'x'), (20.0, 20.0))
        >>> scene.sel(band="B12").dims
        ('y', 'x')
    """

    data: A
    transform: Any
    crs: pyproj.CRS | None
    fill_value: _FillValue = None
    attrs: FrozenAttrs = dataclasses.field(default_factory=FrozenAttrs)
    # None reads the dims from the rank (deprecated); stored as a tuple
    dims: tuple[str, ...] = dataclasses.field(default=cast(Any, None), kw_only=True)
    coords: tuple[Coordinate, ...] = dataclasses.field(default=(), kw_only=True)

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
        if self.dims is None:
            if not 2 <= len(data_shape) - len(batch) <= 4:
                raise ValueError(
                    f"a GeoArray is 2-D to 4-D with dims {DIMS}[-ndim:], got shape "
                    f"{data_shape}"
                    + (f" with batch shape {batch}" if batch else "")
                    + "; pass dims= for any other layout"
                )
            dims = infer_dims(len(data_shape), len(batch))
            warnings.warn(
                _INFER_WARNING.format(dims=dims), DeprecationWarning, stacklevel=3
            )
        else:
            dims = _check_dims(self.dims, data_shape, len(batch))
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", _check_coords(self.coords, dims, data_shape))
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "crs", as_crs(self.crs))
        object.__setattr__(self, "fill_value", _normalize_fill(self.fill_value))
        _check_fill_fits(self.fill_value, self.data.dtype)
        if not isinstance(self.attrs, FrozenAttrs):
            object.__setattr__(self, "attrs", FrozenAttrs(self.attrs or {}))

    @classmethod
    def from_coords(
        cls,
        data: Any,
        *coords: Coordinate | PixelGrid | str,
        transform: Any,
        fill_value: Any = None,
        attrs: Any = None,
    ) -> GeoArray:
        """A carrier from its coordinates, one per dim in axis order.

        Each entry describes the next axis or axes: a 1-D axis
        (`SpectralAxis`, `TimeAxis`, …) one dim, a dim name (``"patch"``) a
        dim without a coordinate, and the last entry a `PixelGrid`, which
        gives ``y x``, the CRS and the expected spatial shape. A
        point-convention grid's transform is shifted to the area convention
        carriers store (`PixelGrid.area_transform`).

        Raises:
            TypeError: The last entry is not a `PixelGrid`.
            ValueError: The data does not fit the coordinates.

        Example:
            >>> import numpy as np
            >>> from geoarray.coords import PixelGrid, SpectralAxis, TimeAxis
            >>> dates = TimeAxis(stamps=["2026-06-01", "2026-06-11"])
            >>> swir = SpectralAxis(names=("B11", "B12"))
            >>> grid = PixelGrid(shape=(4, 4), crs="EPSG:32613")
            >>> t = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
            >>> data = np.zeros((2, 2, 4, 4))
            >>> g = GeoArray.from_coords(data, dates, swir, grid, transform=t)
            >>> g.dims
            ('time', 'band', 'y', 'x')
        """
        if not coords or not isinstance(coords[-1], PixelGrid):
            raise TypeError("from_coords ends with a PixelGrid for the y x dims")
        grid = coords[-1]
        leading = coords[:-1]
        dims = tuple(c if isinstance(c, str) else c.dim for c in leading) + SPATIAL_DIMS
        if tuple(data.shape[-2:]) != grid.shape:
            raise ValueError(
                f"the PixelGrid has shape {grid.shape}, but the data's spatial "
                f"shape is {tuple(data.shape[-2:])}"
            )
        backend = backend_of(data)
        t = _coerce_transform(transform, backend)
        return cls(
            data,
            grid.area_transform(t),
            grid.crs,
            fill_value,
            attrs,
            dims=dims,
            coords=tuple(c for c in leading if not isinstance(c, str)),
        )

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
        """The number of axes (``len(dims)``)."""
        return len(self.data.shape)

    @property
    def dtype(self) -> Any:
        """The data's dtype."""
        return self.data.dtype

    @property
    def sizes(self) -> dict[str, int]:
        """Each dim's size: ``{"band": 2, "y": 600, "x": 600}``."""
        return dict(zip(self.dims, self.shape, strict=True))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch axes: ``()`` for one scene, ``(P,)`` for P patches.

        A batched carrier has one transform per item, ``(*batch, 6)``;
        `batch_dims` names those axes.
        """
        return tuple(self.transform.shape[:-1])

    @property
    def batch_dims(self) -> tuple[str, ...]:
        """The dims that index transform rows: ``()``, or ``("patch",)``."""
        return self.dims[: len(self.batch_shape)]

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
        """The number of bands: the size of the ``band`` dim, or 1 without one."""
        if "band" not in self.dims:
            return 1
        return int(self.data.shape[self.dims.index("band")])

    def coord(self, dim: str) -> Coordinate | None:
        """The coordinate of ``dim``, or ``None``.

        Raises:
            KeyError: ``dim`` is not one of this carrier's dims.
        """
        if dim not in self.dims:
            raise KeyError(f"{dim!r} is not a dim of {self.dims}")
        return _coord_for(self.coords, dim)

    def labels(self, dim: str) -> tuple[Any, ...] | None:
        """The exact labels of ``dim`` (band names, level values), or ``None``."""
        c = self.coord(dim)
        return None if c is None else c.labels

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
    def pixel_grid(self) -> PixelGrid:
        """The ``y x`` coordinate: shape and CRS, area convention."""
        return PixelGrid((self.height, self.width), self.crs)

    @property
    def res(self) -> tuple[float, float]:
        """Pixel size ``(x, y)`` in CRS units (see `Grid.res`)."""
        return self.grid.res

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` (see `Grid.bounds`)."""
        return self.grid.bounds

    # -- new carriers from this one ----------------------------------------

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuild through the constructor, so a NaN fill is the one
        # ``math.nan`` again and attrs are re-frozen in the new process.
        return (
            _restore,
            (
                self.data,
                self.transform,
                self.crs,
                self.fill_value,
                self.attrs,
                self.dims,
                self.coords,
            ),
        )

    def replace(self, **changes: Any) -> GeoArray:
        """A copy with fields replaced (``data``, ``transform``, ``dims``, …).

        Example:
            >>> import numpy as np
            >>> g = GeoArray(np.zeros((4, 4)), (1, 0, 0, 0, -1, 0), None, dims="y x")
            >>> g.replace(fill_value=0).fill_value
            0
        """
        return dataclasses.replace(self, **changes)

    def _dims_for(self, data: Any, transform: Any) -> tuple[str, ...]:
        """Dims for new data when none are given: ours, or a guess with a warning."""
        ndim = len(data.shape)
        if ndim == self.ndim:
            return self.dims
        n_batch = max(len(getattr(transform, "shape", (6,))) - 1, 0)
        if ndim < self.ndim and n_batch == len(self.batch_shape) == 0:
            # fewer axes: the leading dims went (a reduction keeping y x)
            return self.dims[self.ndim - ndim :]
        dims = infer_dims(ndim, n_batch)
        warnings.warn(
            f"with_data could not tell the new dims from {self.dims}; reading "
            f"them from the rank as {dims} is deprecated: pass dims=",
            DeprecationWarning,
            stacklevel=4,
        )
        return dims

    def with_data(
        self,
        data: Any,
        *,
        transform: Any = None,
        fill_value: Any = INHERIT,
        attrs: Any = None,
        dims: Any = None,
        coords: Any = INHERIT,
    ) -> GeoArray:
        """A carrier with new data on this carrier's grid (or ``transform``).

        Args:
            data: The new ``(..., H, W)`` array.
            transform: A new transform, for data on a new grid. When omitted,
                ``data`` must keep this carrier's spatial shape.
            fill_value: The new fill; omitted keeps this carrier's.
            attrs: New attrs; omitted keeps this carrier's, without the
                per-band keys when the band count changed.
            dims: The new data's dims; omitted keeps this carrier's when the
                rank is unchanged, and drops leading dims when it fell.
            coords: New coordinates; omitted keeps each of this carrier's
                whose dim survives with the same size.

        Raises:
            ValueError: The spatial shape changed and no ``transform`` was
                given.

        Example:
            >>> import numpy as np
            >>> t = (1, 0, 0, 0, -1, 0)
            >>> g = GeoArray(np.ones((2, 4, 4)), t, None, dims="band y x")
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
        new_dims = self._dims_for(data, transform) if dims is None else dims
        if isinstance(new_dims, str):
            new_dims = tuple(new_dims.split())
        shape = tuple(data.shape)
        if coords is INHERIT:
            coords = tuple(
                c
                for c in self.coords
                if len(new_dims) == len(shape) and _keeps_coord(c, new_dims, shape)
            )
        if attrs is None:
            attrs = self.attrs
            band = (
                shape[new_dims.index("band")]
                if "band" in new_dims and len(new_dims) == len(shape)
                else 1
            )
            if band != self.count:
                attrs = drop_band_keys(attrs)
        return GeoArray(
            data,
            transform,
            self.crs,
            self.fill_value if fill_value is INHERIT else fill_value,
            attrs,
            dims=new_dims,
            coords=coords,
        )

    def with_attrs(self, **items: Any) -> GeoArray:
        """A copy with attrs added or replaced (see `FrozenAttrs.set`)."""
        return dataclasses.replace(self, attrs=self.attrs.set(**items))

    def drop_attrs(self, *keys: str) -> GeoArray:
        """A copy without the given attrs keys (see `FrozenAttrs.drop`)."""
        return dataclasses.replace(self, attrs=self.attrs.drop(*keys))

    def astype(self, dtype: Any, *, fill_value: Any = INHERIT) -> GeoArray:
        """A copy with the data cast to ``dtype``; geometry unchanged.

        Args:
            dtype: The new dtype.
            fill_value: The new fill; omitted keeps this carrier's, which
                ``dtype`` must be able to hold (a NaN fill cannot be cast to
                integers, for example).

        Raises:
            ValueError: ``dtype`` cannot hold the fill.

        Example:
            >>> import numpy as np
            >>> t = (1, 0, 0, 0, -1, 0)
            >>> g = GeoArray(np.ones((4, 4)), t, None, np.nan, dims="y x")
            >>> g.astype(np.uint8, fill_value=255).fill_value
            255
        """
        fill = self.fill_value if fill_value is INHERIT else fill_value
        return dataclasses.replace(self, data=self.data.astype(dtype), fill_value=fill)

    def map(self, f: Any) -> GeoArray:
        """Apply ``f`` to the data and keep the geometry.

        ``f`` may change the leading axes (a band index collapses bands)
        but not the spatial shape. Works inside ``jit``.

        Raises:
            ValueError: ``f`` changed the spatial shape.

        Example:
            >>> import numpy as np
            >>> t = (1, 0, 0, 0, -1, 0)
            >>> g = GeoArray(np.full((2, 2), 9.0), t, None, dims="y x")
            >>> g.map(np.sqrt).data[0, 0]
            np.float64(3.0)
        """
        out = f(self.data)
        if tuple(out.shape[-2:]) != tuple(self.data.shape[-2:]):
            raise ValueError(
                f"map must keep the spatial shape {tuple(self.data.shape[-2:])}, "
                f"got {tuple(out.shape[-2:])}; for a new grid use "
                "with_data(..., transform=)"
            )
        return self.with_data(out)

    # -- indexing --------------------------------------------------------------

    def __getitem__(self, key: Any) -> GeoArray:
        """Index like NumPy; spatial slices move the transform.

        The last two (spatial) axes take slices with positive steps only;
        the leading axes take anything NumPy indexing allows, with at most
        one list / array index. An integer drops its dim and that dim's
        coordinate; other indices slice the coordinate with the data, and
        indexing the band dim selects the per-band attrs too.

        Example:
            >>> import numpy as np
            >>> g = GeoArray(
            ...     np.zeros((3, 8, 8)), (20, 0, 600000, 0, -20, 3550000), 32613,
            ...     attrs={"band_names": ["B04", "B08", "B11"]}, dims="band y x",
            ... )
            >>> sub = g[1:, 2:6, ::2]
            >>> sub.shape, sub.res, sub.attrs["band_names"]
            ((2, 4, 4), (40.0, 20.0), ('B08', 'B11'))
        """
        return self._index(key, guard_band=True)

    def _index(self, key: Any, *, guard_band: bool) -> GeoArray:
        """``g[key]``; ``guard_band`` refuses dropping ``band`` alone (positional)."""
        ndim = self.ndim
        key = _expand_key(key, ndim)
        _require_concrete_spatial(key[-2:])
        rows, cols = key[-2], key[-1]
        if not (isinstance(rows, slice) and isinstance(cols, slice)):
            raise IndexError(
                "the spatial axes take slices only (an integer or list would drop "
                "georeferencing); use g.data for plain arrays"
            )
        r0, _, row_step = rows.indices(self.height)
        c0, _, col_step = cols.indices(self.width)
        if row_step <= 0 or col_step <= 0:
            raise IndexError("spatial slices need positive steps")
        n_batch = len(self.batch_shape)
        leading = key[:-2]
        if sum(_is_advanced(k) for k in leading) > 1:
            raise IndexError("at most one list / array index on the leading axes")
        dims = self.dims
        dropped = {d for d, k in zip(dims, key, strict=True) if _drops_axis(k)}
        if guard_band and _drops_band_alone(dims, dropped):
            raise IndexError(
                "an integer band index on a (T, C, H, W) carrier would leave "
                "(T, H, W), which reads as (C, H, W); index with [k] to keep a "
                "(T, 1, H, W) band axis"
            )
        transform = self.transform[key[:n_batch]] if n_batch else self.transform
        attrs = self.attrs
        coords = []
        for d, k in zip(dims[:-2], leading, strict=True):
            c = _coord_for(self.coords, d)
            full = isinstance(k, slice) and k == slice(None)
            if d == "band" and not full:
                attrs = select_bands(attrs, k, self.count)
            if c is None or d in dropped:
                continue
            if not full and is_tracer(k):
                raise GeoArrayTraceError(
                    f"an index on {d!r} must be concrete: its coordinate is static, "
                    "so the selection must be known when tracing; index g.data "
                    "instead, or select outside jit"
                )
            coords.append(c if full else c.isel(k))
        new_dims = tuple(d for d in dims if d not in dropped)
        return GeoArray(
            self.data[key],
            shift_scale_transform(transform, r0, c0, row_step, col_step),
            self.crs,
            self.fill_value,
            attrs,
            dims=new_dims,
            coords=tuple(coords),
        )

    def isel(self, selection: dict[str, Any] | None = None, **by_dim: Any) -> GeoArray:
        """Index by dim name: ``g.isel(band=[0, 2], y=slice(0, 100))``.

        The rules of ``g[...]`` apply: ``y`` and ``x`` take positive-step
        slices and move the origin; other dims take ints, slices or
        positions, and slice their coordinate. By name, an integer may drop
        ``band`` from ``(time, band, y, x)``: the stored dims say what is
        left.

        Raises:
            KeyError: A name is not one of this carrier's dims.

        Example:
            >>> import numpy as np
            >>> t = (20, 0, 0, 0, -20, 0)
            >>> g = GeoArray(np.zeros((3, 8, 8)), t, 32613, dims="band y x")
            >>> g.isel(band=0, x=slice(0, 4)).shape
            (8, 4)
        """
        wanted = {**(selection or {}), **by_dim}
        dims = self.dims
        unknown = set(wanted) - set(dims)
        if unknown:
            raise KeyError(f"cannot select {sorted(unknown)}; dims are {dims}")
        # by name, the result's dims say what is left: no 0.2 band guard
        key = tuple(wanted.get(name, slice(None)) for name in dims)
        return self._index(key, guard_band=False)

    def sel(self, selection: dict[str, Any] | None = None, **by_dim: Any) -> GeoArray:
        """Select by exact label: ``g.sel(band="B12")``, ``g.sel(band=["B8", "B4"])``.

        One label drops its dim; a list keeps it. Labels come from the dim's
        coordinate (`labels`).

        Raises:
            LabelError: The dim has no labels or is spatial, a label is
                missing, or the value is a range (a ``slice``): selection by
                value lives in xarray (``to_xarray()``).
            KeyError: A name is not one of this carrier's dims.

        Example:
            >>> import numpy as np
            >>> from geoarray.coords import SpectralAxis
            >>> bands = SpectralAxis(names=("B4", "B8"))
            >>> g = GeoArray(np.zeros((2, 4, 4)), (20, 0, 0, 0, -20, 0), 32613,
            ...              dims="band y x", coords=(bands,))
            >>> g.sel(band="B8").dims
            ('y', 'x')
        """
        wanted = {**(selection or {}), **by_dim}
        indexers: dict[str, Any] = {}
        for dim, value in wanted.items():
            labels = self.labels(dim) if dim not in SPATIAL_DIMS else None
            if labels is None:
                raise LabelError(
                    f"sel: dim {dim!r} has no labels; use isel, or select by value "
                    "in xarray (g.to_xarray().sel(...))"
                )
            if isinstance(value, slice):
                raise LabelError(
                    f"sel: {dim}={value} is a range; GeoArray.sel takes exact labels "
                    "only: select by value in xarray (g.to_xarray().sel(...))"
                )
            position = {label: i for i, label in enumerate(labels)}
            many = isinstance(value, list) or getattr(value, "ndim", 0) == 1
            picked = (
                list(value.tolist() if hasattr(value, "tolist") else value)
                if many
                else [value]
            )
            for label in picked:
                if label not in position:
                    raise LabelError(
                        f"sel: {label!r} is not a label of {dim!r}: {labels}"
                    )
            indexers[dim] = [position[v] for v in picked] if many else position[value]
        return self.isel(indexers)

    def shift_origin(self, delta: Any) -> GeoArray:
        """Move the grid by ``delta = (Δx, Δy)`` in CRS units: c + Δx, f + Δy.

        The pixels do not move; their georeferencing does. Differentiable,
        so ``jax.grad`` can find a sub-pixel co-registration shift.

        Example:
            >>> import numpy as np
            >>> t = (20, 0, 600000, 0, -20, 3550000)
            >>> g = GeoArray(np.zeros((2, 2)), t, 32613, dims="y x")
            >>> g.shift_origin((7.3, 0.0)).transform[2]
            np.float64(600007.3)
        """
        xp = namespace(self.transform)
        dx, dy = (
            xp.asarray(v, dtype=self.transform.dtype) for v in (delta[0], delta[1])
        )
        t = self.transform
        columns = (
            t[..., 0],
            t[..., 1],
            t[..., 2] + dx,
            t[..., 3],
            t[..., 4],
            t[..., 5] + dy,
        )
        return dataclasses.replace(self, transform=xp.stack(columns, axis=-1))

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> GeoArray:
        """Drop size-1 leading axes (all of them when ``axis`` is ``None``).

        The band axis of a ``(T, 1, H, W)`` stack stays unless time goes
        too: ``(T, H, W)`` would read as ``(C, H, W)``.

        Raises:
            ValueError: ``axis`` names a spatial or batch axis.
        """
        n_batch = len(self.batch_shape)
        leading = range(n_batch, self.ndim - 2)
        dims = self.dims
        if axis is None:
            axes = tuple(i for i in leading if self.data.shape[i] == 1)
            names = {dims[i] for i in axes}
            if _drops_band_alone(dims, names):
                axes = tuple(i for i in axes if dims[i] != "band")
        else:
            requested = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(i + self.ndim if i < 0 else i for i in requested)
            if any(i not in leading for i in axes):
                raise ValueError(f"squeeze: {axis} is not a leading, non-batch axis")
            if _drops_band_alone(dims, {dims[i] for i in axes}):
                raise ValueError(
                    "squeeze: dropping the band axis of (T, 1, H, W) would leave "
                    "(T, H, W), which reads as (C, H, W)"
                )
        kept = tuple(d for i, d in enumerate(dims) if i not in axes)
        return self.with_data(self.xp.squeeze(self.data, axis=axes), dims=kept)

    def expand_dims(self, axis: int, dim: str | None = None) -> GeoArray:
        """Insert a size-1 leading axis at ``axis``, named ``dim``.

        Args:
            axis: Where the new axis goes, before the spatial axes.
            dim: Its name. Omitted, it is the name the 0.2 layout gives that
                position (``DIMS[-ndim:]``), with a ``DeprecationWarning``.

        Raises:
            ValueError: ``axis`` is not before the spatial axes, or the name
                is taken.
        """
        ndim = self.ndim + 1
        position = axis + ndim if axis < 0 else axis
        n_batch = len(self.batch_shape)
        if not n_batch <= position < ndim - 2:
            raise ValueError(
                f"expand_dims: axis {axis} must come before the spatial axes"
            )
        if dim is None:
            if ndim - n_batch > 4:
                raise ValueError(
                    f"expand_dims: a {ndim}-D result needs a name: pass dim="
                )
            dim = infer_dims(ndim, n_batch)[position]
            warnings.warn(
                f"expand_dims without dim= names the new axis {dim!r} from the "
                "rank; this is deprecated: pass dim=",
                DeprecationWarning,
                stacklevel=2,
            )
        if dim in self.dims:
            raise ValueError(f"expand_dims: {dim!r} is already a dim of {self.dims}")
        dims = (*self.dims[:position], dim, *self.dims[position:])
        return self.with_data(self.xp.expand_dims(self.data, axis=position), dims=dims)

    # -- nodata ----------------------------------------------------------------

    def valid_mask(self, *, keep_time: bool = False) -> Any:
        """Which pixels hold data: ``(H, W)`` booleans, True where valid.

        A pixel is valid when every entry there (over the non-batch,
        non-spatial dims) is finite and differs from the fill value (a NaN
        fill matches NaN). ``keep_time=True`` keeps a ``time`` dim,
        ``(T, H, W)``. Batch axes are kept. Works inside ``jit``.

        v_ij = ∧ₖ [isfinite(x_kij) ∧ x_kij ≠ φ]

        Example:
            >>> import numpy as np
            >>> data = np.array([[[1.0, np.nan]], [[2.0, 3.0]]])
            >>> g = GeoArray(data, (1, 0, 0, 0, -1, 0), None, np.nan, dims="band y x")
            >>> g.valid_mask().tolist()
            [[True, False]]
        """
        xp = self.xp
        data = self.data
        floating = data.dtype.kind in "fc"
        invalid = ~xp.isfinite(data) if floating else xp.zeros(data.shape, dtype=bool)
        fill = self.fill_value
        if fill is not None and not (isinstance(fill, float) and math.isnan(fill)):
            invalid = invalid | (data == fill)
        n_batch = len(self.batch_shape)
        axes = tuple(
            i
            for i in range(n_batch, self.ndim - 2)
            if not (keep_time and self.dims[i] == "time")
        )
        return ~invalid.any(axis=axes) if axes else ~invalid

    # -- RasterSource ----------------------------------------------------------

    def read_window(self, window: PixelWindow, *, boundless: bool = True) -> GeoArray:
        """The pixels of ``window``: `geoarray.read_window` as a method.

        With `load`, this makes a carrier a `RasterSource`, so code written
        for lazy readers takes a loaded carrier too.

        Example:
            >>> import numpy as np
            >>> from geoarray import PixelWindow
            >>> t = (20, 0, 0, 0, -20, 0)
            >>> g = GeoArray(np.ones((4, 4)), t, 32613, 0.0, dims="y x")
            >>> window = PixelWindow(col_off=2, row_off=2, width=4, height=4)
            >>> g.read_window(window).shape
            (4, 4)
        """
        from geoarray._src.geometry import read_window

        return read_window(self, window, boundless=boundless)

    def load(self) -> GeoArray:
        """This carrier: its pixels are already loaded (`RasterSource`)."""
        return self

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
            >>> g = GeoArray(data, (1, 0, 0, 0, -1, 0), None, dims="y x").to_jax()
            >>> g.backend, str(g.dtype), str(g.transform.dtype)
            ('jax', 'float32', 'float64')
        """
        jax = import_jax()
        require_x64()
        if self.backend == "jax" and device is None:
            return self
        data = jax.device_put(self.data, device)
        transform = jax.device_put(self.transform, device)
        coords = tuple(
            TimeAxis.from_leaf(jax.device_put(c.as_leaf(), device))
            if isinstance(c, TimeAxis)
            else c
            for c in self.coords
        )
        return GeoArray(
            data,
            transform,
            self.crs,
            self.fill_value,
            self.attrs,
            dims=self.dims,
            coords=coords,
        )

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
        coords = tuple(
            c.to_host() if isinstance(c, TimeAxis) else c for c in self.coords
        )
        return GeoArray(
            np.asarray(self.data),
            np.asarray(self.transform, dtype=np.float64),
            self.crs,
            self.fill_value,
            self.attrs,
            dims=self.dims,
            coords=coords,
        )

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """The data as a NumPy array, so ``np.asarray(g)`` works.

        NumPy data is returned as it is (no copy unless ``copy=True`` or a
        cast). JAX data is copied to the host, eagerly only: inside ``jit``,
        ``grad`` or ``vmap`` the data are tracers, and NumPy code cannot run
        on them. With ``set_options(warn_on_transfer=True)`` each host copy
        warns (`HostTransferWarning`).

        Raises:
            GeoArrayTraceError: The data are traced.
            ValueError: ``copy=False`` for JAX data or a cast, which need a
                copy.
        """
        if self.backend == "numpy":
            if copy:
                return np.array(self.data, dtype=dtype, copy=True)
            if copy is False and dtype is not None and np.dtype(dtype) != self.dtype:
                raise ValueError(
                    f"casting {self.dtype} data to {np.dtype(dtype)} needs a copy, "
                    "but copy=False was passed"
                )
            return np.asarray(self.data, dtype=dtype)
        if is_tracer(self.data):
            raise GeoArrayTraceError(
                "NumPy code received a GeoArray's data inside jit / grad / vmap, "
                "where the data are tracers; use a JAX function "
                "(geoops.functional) or call the NumPy code outside the transform"
            )
        if copy is False:
            raise ValueError(
                "a JAX GeoArray cannot become a NumPy array without a copy"
            )
        warn_host_transfer("np.asarray(GeoArray)", stacklevel=2)
        if copy:
            # np.asarray of a CPU JAX array can be a read-only view; a copy
            # the caller asked for must be writable.
            return np.array(self.data, dtype=dtype, copy=True)
        return np.asarray(self.data, dtype=dtype)

    # -- comparison and display ----------------------------------------------

    def identical(self, other: object) -> bool:
        """Equal data (NaN-aware), transform, dims, coords, CRS, fill and attrs.

        Both carriers must be concrete; the backends may differ.

        Example:
            >>> import numpy as np
            >>> t = (1, 0, 0, 0, -1, 0)
            >>> g = GeoArray(np.zeros((2, 2)), t, "EPSG:4326", dims="y x")
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
            and mine.dims == theirs.dims
            and mine.coords == theirs.coords
            and mine.crs == theirs.crs
            and fill_equal
            and mine.attrs == theirs.attrs
        )

    def __repr__(self) -> str:
        head = f"{self.backend} {self.dtype} {self.shape}"
        if is_tracer(self.data) or is_tracer(self.transform) or self.batch_shape:
            where = "traced" if not self.batch_shape else f"batch={self.batch_shape}"
        else:
            grid = self.grid
            where = f"res={grid.res}, bounds={grid.bounds}"
        keys = tuple(self.attrs)
        return (
            f"GeoArray({head}, crs={_crs_label(self.crs)}, {where}, "
            f"fill_value={self.fill_value!r}, attrs={keys}, dims={self.dims})"
        )


def _restore(
    data: Any,
    transform: Any,
    crs: Any,
    fill_value: Any,
    attrs: Any,
    dims: tuple[str, ...],
    coords: tuple[Coordinate, ...],
) -> GeoArray:
    """Unpickle through the constructor (``GeoArray.__reduce__``)."""
    return GeoArray(data, transform, crs, fill_value, attrs, dims=dims, coords=coords)


def build(
    data: Any,
    transform: Any,
    crs: Any,
    fill_value: Any,
    attrs: FrozenAttrs,
    dims: tuple[str, ...],
    coords: tuple[Coordinate, ...],
) -> GeoArray:
    """A carrier from already-valid fields, without validation.

    For pytree unflattening and the adapter's rewraps, which run under a
    trace and on the hot path of every kernel call.
    """
    g = object.__new__(GeoArray)
    object.__setattr__(g, "data", data)
    object.__setattr__(g, "transform", transform)
    object.__setattr__(g, "crs", crs)
    object.__setattr__(g, "fill_value", fill_value)
    object.__setattr__(g, "attrs", attrs)
    object.__setattr__(g, "dims", dims)
    object.__setattr__(g, "coords", coords)
    return g


# -- JAX pytree ---------------------------------------------------------------


class _StampSlot:
    """Where a `TimeAxis` sits in the static coords; its stamps are a leaf."""

    def __repr__(self) -> str:
        return "STAMPS"


_STAMPS = _StampSlot()


def _flatten(g: GeoArray) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    # leaves: data, transform, then TimeAxis stamps (int64 ns) if any;
    # static: everything else, with the TimeAxis replaced by a slot
    stamps: tuple[Any, ...] = ()
    coords = g.coords
    if any(isinstance(c, TimeAxis) for c in coords):
        stamps = tuple(c.as_leaf() for c in coords if isinstance(c, TimeAxis))
        coords = tuple(_STAMPS if isinstance(c, TimeAxis) else c for c in coords)
    aux = (g.crs, g.fill_value, g.attrs, g.dims, coords)
    return (g.data, g.transform, *stamps), aux


def _unflatten(aux: tuple[Any, ...], leaves: Any) -> GeoArray:
    # Never validate: JAX unflattens with placeholder leaves (None, object(),
    # ShapeDtypeStruct) while tracing, and a shape check would fail on them.
    # A bare jax.vmap maps leaves but not the static dims: match them.
    data, transform, *stamps = leaves
    crs, fill_value, attrs, dims, coords = aux
    if stamps:
        fills = iter(stamps)
        coords = tuple(
            TimeAxis.from_leaf(next(fills)) if c is _STAMPS else c for c in coords
        )
    matched = vmapped_dims(dims, getattr(data, "shape", None))
    if matched != dims:
        coords = tuple(c for c in coords if c.dim in matched)
    return build(data, transform, crs, fill_value, attrs, matched, coords)


def _register_pytree() -> bool:
    """Register `GeoArray` with JAX when JAX is installed; never fail without it."""
    try:
        jax = import_jax()
    except ImportError:  # pragma: no cover - exercised by the base-install job
        return False
    from jax.tree_util import GetAttrKey

    def flatten_with_keys(g: GeoArray) -> tuple[Any, Any]:
        leaves, aux = _flatten(g)
        names = ("data", "transform", "time_stamps")
        keyed = tuple(
            (GetAttrKey(name), leaf) for name, leaf in zip(names, leaves, strict=False)
        )
        return keyed, aux

    jax.tree_util.register_pytree_with_keys(
        GeoArray, flatten_with_keys, _unflatten, flatten_func=_flatten
    )
    return True


PYTREE_REGISTERED = _register_pytree()
