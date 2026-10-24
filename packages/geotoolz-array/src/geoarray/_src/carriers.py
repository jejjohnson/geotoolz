"""`GeoSwath` and `GeoPoints`: carriers whose georeference is not an affine grid.

A new carrier class exists only when the georeferencing changes; everything
else is a named dim.

| Carrier | Georeference leaves | Spatial dims (last) |
| --- | --- | --- |
| `GeoArray` | transform ``(…, 6)`` | ``y x`` |
| `GeoSwath` | lat, lon ``(…, row, col)`` | ``row col`` |
| `GeoPoints` | xy ``(obs, 2)``; optional z, time, valid ``(obs,)`` | ``obs`` |

Leaves change per call without recompiling (a new granule, a new day of
observations); dims, coordinates, CRS, fill, attrs and the `Vertical` of z
are static. ``valid`` marks padding, so point sets can be padded to a fixed
N under ``jit`` (`pad_points`, GA-G16).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar, cast

import numpy as np
import pyproj
from pipekit_array.dimarray import vmapped_dims

from geoarray._src.array import (
    _check_coords,
    _check_dims,
    _check_fill_fits,
    _coord_for,
    _crs_label,
    _normalize_fill,
    _subscript,
)
from geoarray._src.attrs import FrozenAttrs
from geoarray._src.backend import (
    backend_of,
    import_jax,
    is_tracer,
    namespace,
    require_x64,
    xarray_module,
)
from geoarray._src.coords import Coordinate, CurvilinearGrid, Vertical
from geoarray._src.errors import GeoArrayTraceError, LabelError
from geoarray._src.grid import as_crs


__all__ = ["GeoPoints", "GeoSwath", "pad_points"]

_WGS84 = "EPSG:4326"


def _float_leaf(value: Any, backend: str, name: str) -> Any:
    """A float64 leaf on ``backend`` (tracers kept as they are)."""
    if is_tracer(value):
        return value
    if backend == "jax":
        require_x64()
        import jax.numpy as jnp

        return jnp.asarray(value, dtype=jnp.float64)
    return np.asarray(value, dtype=np.float64)


def _time_leaf(value: Any, backend: str) -> Any:
    """Per-sample times: ``datetime64[ns]`` on NumPy, ``int64`` ns on JAX."""
    if is_tracer(value):
        return value
    if backend == "jax":
        require_x64()
        import jax.numpy as jnp

        if isinstance(value, np.ndarray) and value.dtype.kind == "M":
            value = value.astype("datetime64[ns]").view(np.int64)
        return jnp.asarray(value, dtype=jnp.int64)
    if type(value).__module__.startswith("jax"):
        return np.asarray(value, dtype=np.int64).view("datetime64[ns]")
    arr = np.asarray(value)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64).view("datetime64[ns]")
    return arr.astype("datetime64[ns]")


def _as_int_leaf(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.kind == "M":
        return value.view(np.int64)
    return value


def _to_jax_leaf(value: Any, device: Any) -> Any:
    jax = import_jax()
    return jax.device_put(_as_int_leaf(value), device)


def _host(value: Any, what: str) -> np.ndarray:
    if is_tracer(value):
        raise GeoArrayTraceError(
            f"{what} needs concrete values, but this carrier is traced (inside "
            "jit / grad / vmap): call it outside the transform"
        )
    return np.asarray(value)


class _Named:
    """What every carrier with stored dims and coordinates shares."""

    data: Any
    dims: tuple[str, ...]
    coords: tuple[Coordinate, ...]
    crs: pyproj.CRS | None
    fill_value: Any
    attrs: FrozenAttrs
    spatial_dims: ClassVar[tuple[str, ...]]

    @property
    def backend(self) -> str:
        """``"numpy"`` or ``"jax"``."""
        return backend_of(self.data)

    @property
    def xp(self) -> Any:
        """``numpy`` or ``jax.numpy``."""
        return namespace(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        """The data's shape."""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return len(self.data.shape)

    @property
    def dtype(self) -> Any:
        """The data's dtype."""
        return self.data.dtype

    @property
    def sizes(self) -> dict[str, int]:
        """Each dim's size."""
        return dict(zip(self.dims, self.shape, strict=True))

    def coord(self, dim: str) -> Coordinate | None:
        """The coordinate of ``dim``, or ``None``.

        Raises:
            KeyError: ``dim`` is not one of this carrier's dims.
        """
        if dim not in self.dims:
            raise KeyError(f"{dim!r} is not a dim of {self.dims}")
        return _coord_for(self.coords, dim)

    def labels(self, dim: str) -> tuple[Any, ...] | None:
        """The exact labels of ``dim``, or ``None``."""
        c = self.coord(dim)
        return None if c is None else c.labels

    def _leading_index(
        self, wanted: dict[str, Any]
    ) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[Coordinate, ...]]:
        """Index tuple, kept dims and sliced coordinates for a selection."""
        unknown = set(wanted) - set(self.dims)
        if unknown:
            raise KeyError(f"cannot select {sorted(unknown)}; dims are {self.dims}")
        key = tuple(wanted.get(d, slice(None)) for d in self.dims)
        dims, coords = [], []
        for d, k in zip(self.dims, key, strict=True):
            if isinstance(k, int | np.integer):
                continue
            dims.append(d)
            c = _coord_for(self.coords, d)
            if c is not None:
                full = isinstance(k, slice) and k == slice(None)
                coords.append(c if full else c.isel(k))
        return key, tuple(dims), tuple(coords)

    def sel(self, selection: dict[str, Any] | None = None, **by_dim: Any) -> Any:
        """Select by exact label (one drops the dim, a list keeps it).

        Raises:
            LabelError: The dim has no labels, a label is missing, or the
                value is a range (select by value in xarray).
        """
        wanted = {**(selection or {}), **by_dim}
        indexers: dict[str, Any] = {}
        for dim, value in wanted.items():
            labels = None if dim in self.spatial_dims else self.labels(dim)
            if labels is None:
                raise LabelError(
                    f"sel: dim {dim!r} has no labels; use isel, or select by value "
                    "in xarray (to_xarray().sel(...))"
                )
            if isinstance(value, slice):
                raise LabelError(
                    f"sel: {dim}={value} is a range; sel takes exact labels only"
                )
            position = {label: i for i, label in enumerate(labels)}
            many = isinstance(value, list) or getattr(value, "ndim", 0) == 1
            picked = list(value) if many else [value]
            for label in picked:
                if label not in position:
                    raise LabelError(f"sel: {label!r} is not a label of {dim!r}")
            indexers[dim] = [position[v] for v in picked] if many else position[value]
        return cast(Any, self).isel(indexers)

    def _identical_static(self, other: _Named) -> bool:
        fill_equal = self.fill_value is other.fill_value or (
            self.fill_value == other.fill_value
            and type(self.fill_value) is type(other.fill_value)
        )
        return (
            self.dims == other.dims
            and self.coords == other.coords
            and self.crs == other.crs
            and fill_equal
            and self.attrs == other.attrs
        )


def _same_array(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "M":
        a, b = a.view(np.int64), np.asarray(b).astype("datetime64[ns]").view(np.int64)
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    )


def _init_named(
    obj: Any, n_geo_batch: int, spatial: tuple[str, ...], carrier: str
) -> None:
    """Validate and normalise the fields every carrier shares."""
    shape = tuple(obj.data.shape)
    if obj.dims is None:
        raise TypeError(f"{carrier} needs dims=, e.g. {(*('band',), *spatial)}")
    dims = _check_dims(obj.dims, shape, n_geo_batch, spatial, carrier)
    object.__setattr__(obj, "dims", dims)
    object.__setattr__(obj, "coords", _check_coords(obj.coords, dims, shape, spatial))
    object.__setattr__(obj, "crs", as_crs(obj.crs))
    object.__setattr__(obj, "fill_value", _normalize_fill(obj.fill_value))
    _check_fill_fits(obj.fill_value, obj.data.dtype)
    if not isinstance(obj.attrs, FrozenAttrs):
        object.__setattr__(obj, "attrs", FrozenAttrs(obj.attrs or {}))


def _build(cls: type, **fields: Any) -> Any:
    """A carrier from already-valid fields, without validation."""
    obj = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    return obj


# -- GeoSwath -------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class GeoSwath(_Named):
    """Data in sensor geometry: per-pixel lat/lon on the ``row col`` dims.

    Rows run along track, columns across (EMIT, AVIRIS, TROPOMI L1B/L2).
    A model grid with fixed 2-D lat/lon (ORCA) is a swath too
    (`GeoSwath.from_grid`).

    Args:
        data: ``(..., row, col)``, NumPy or JAX.
        lat: ``(*batch, row, col)`` latitudes; ``batch`` are the leading dims
            of ``dims`` that index per-item geolocation (``time`` for a
            series of granules), usually none.
        lon: Longitudes, the shape of ``lat``.
        dims: One name per axis, ending in ``("row", "col")``.
        coords: Coordinates of non-spatial dims (`SpectralAxis`, …).
        crs: The CRS of lat/lon; WGS 84 by default.
        fill_value: The nodata marker.
        attrs: Whole-carrier metadata.

    Raises:
        ValueError: The dims do not fit, or lat/lon do not match the data's
            ``(row, col)`` or each other.

    Example:
        >>> import numpy as np
        >>> from geoarray.coords import SpectralAxis
        >>> lat, lon = np.meshgrid(np.linspace(31, 32, 4), np.linspace(-104, -103, 5),
        ...                        indexing="ij")
        >>> swath = GeoSwath(np.zeros((2, 4, 5)), lat, lon, dims="band row col",
        ...                  coords=(SpectralAxis(("b1", "b2")),))
        >>> swath.isel(row=slice(0, 2)).lat.shape
        (2, 5)
    """

    data: Any
    lat: Any
    lon: Any
    dims: tuple[str, ...] = dataclasses.field(default=cast(Any, None), kw_only=True)
    coords: tuple[Coordinate, ...] = dataclasses.field(default=(), kw_only=True)
    crs: pyproj.CRS | None = dataclasses.field(default=cast(Any, _WGS84), kw_only=True)
    fill_value: Any = dataclasses.field(default=None, kw_only=True)
    attrs: FrozenAttrs = dataclasses.field(default_factory=FrozenAttrs, kw_only=True)

    spatial_dims: ClassVar[tuple[str, ...]] = ("row", "col")

    def __class_getitem__(cls, item: Any) -> Any:
        """``GeoSwath["band row col"]``: the carrier annotated with a dims pattern."""
        return _subscript(cls, item)

    def __post_init__(self) -> None:
        backend = backend_of(self.data)
        lat = _float_leaf(self.lat, backend, "lat")
        lon = _float_leaf(self.lon, backend, "lon")
        if tuple(lat.shape) != tuple(lon.shape) or lat.ndim < 2:
            raise ValueError(
                f"lat and lon are (…, row, col) of one shape, got {tuple(lat.shape)} "
                f"and {tuple(lon.shape)}"
            )
        shape = tuple(self.data.shape)
        n_batch = lat.ndim - 2
        if shape[-2:] != tuple(lat.shape[-2:]) or shape[:n_batch] != tuple(
            lat.shape[:n_batch]
        ):
            raise ValueError(
                f"lat/lon {tuple(lat.shape)} do not match data {shape}: they have "
                "its (row, col) and its leading batch dims"
            )
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        _init_named(self, n_batch, self.spatial_dims, "GeoSwath")

    @classmethod
    def from_grid(
        cls,
        data: Any,
        grid: CurvilinearGrid,
        *,
        dims: Any,
        coords: tuple[Coordinate, ...] = (),
        fill_value: Any = None,
        attrs: Any = None,
    ) -> GeoSwath:
        """A swath on a fixed curvilinear model grid (ORCA): lat/lon from ``grid``."""
        return cls(
            data,
            grid.lat,
            grid.lon,
            dims=dims,
            coords=coords,
            crs=grid.crs,
            fill_value=fill_value,
            attrs=FrozenAttrs(attrs or {}),
        )

    @property
    def batch_dims(self) -> tuple[str, ...]:
        """The leading dims lat/lon vary along (``()`` for one granule)."""
        return self.dims[: self.lat.ndim - 2]

    def isel(self, selection: dict[str, Any] | None = None, **by_dim: Any) -> GeoSwath:
        """Select by position; lat/lon are sliced with ``row``, ``col`` and batch dims.

        Raises:
            IndexError: An integer or a list on ``row`` / ``col`` (the result
                would not be a swath; index ``data`` for a profile).
        """
        wanted = {**(selection or {}), **by_dim}
        for d in self.spatial_dims:
            if d in wanted and not isinstance(wanted[d], slice):
                raise IndexError(
                    f"GeoSwath.isel: {d} takes a slice (an integer or a list would "
                    "not leave a swath); use the adapter or .data for profiles"
                )
        key, dims, coords = self._leading_index(wanted)
        n_batch = self.lat.ndim - 2
        geo_key = key[:n_batch] + key[-2:]
        return GeoSwath(
            self.data[key],
            self.lat[geo_key],
            self.lon[geo_key],
            dims=dims,
            coords=coords,
            crs=self.crs,
            fill_value=self.fill_value,
            attrs=self.attrs,
        )

    def with_data(
        self, data: Any, *, dims: Any = None, fill_value: Any = ..., attrs: Any = None
    ) -> GeoSwath:
        """New data on this swath's geolocation (same ``row``, ``col``)."""
        new_dims = self.dims if dims is None else dims
        if isinstance(new_dims, str):
            new_dims = tuple(new_dims.split())
        shape = tuple(data.shape)
        coords = tuple(
            c
            for c in self.coords
            if c.dim in new_dims and shape[new_dims.index(c.dim)] == c.size
        )
        return GeoSwath(
            data,
            self.lat,
            self.lon,
            dims=new_dims,
            coords=coords,
            crs=self.crs,
            fill_value=self.fill_value if fill_value is ... else fill_value,
            attrs=self.attrs if attrs is None else attrs,
        )

    def to_jax(self, device: Any = None) -> GeoSwath:
        """Data and lat/lon on a JAX device (needs ``jax_enable_x64``)."""
        jax = import_jax()
        require_x64()
        return dataclasses.replace(
            self,
            data=jax.device_put(self.data, device),
            lat=jax.device_put(self.lat, device),
            lon=jax.device_put(self.lon, device),
        )

    def to_numpy(self) -> GeoSwath:
        """Data and lat/lon on the host."""
        return dataclasses.replace(
            self,
            data=_host(self.data, "to_numpy"),
            lat=_host(self.lat, "to_numpy"),
            lon=_host(self.lon, "to_numpy"),
        )

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """The data as a NumPy array (a host copy for JAX data)."""
        out = _host(self.data, "np.asarray(GeoSwath)")
        return np.array(out, dtype=dtype, copy=True) if copy else np.asarray(out, dtype)

    def to_xarray(self) -> Any:
        """This carrier as a ``DataArray`` with 2-D lat/lon (``[xarray]``)."""
        return xarray_module().swath_to_xarray(self)

    @classmethod
    def from_xarray(cls, obj: Any) -> GeoSwath:
        """A ``DataArray`` with 2-D lat/lon as a `GeoSwath` (see `to_xarray`)."""
        return xarray_module().swath_from_xarray(obj)

    def identical(self, other: object) -> bool:
        """Equal data, lat/lon (NaN-aware), dims, coords, CRS, fill and attrs."""
        if not isinstance(other, GeoSwath):
            return False
        return (
            _same_array(self.data, other.data)
            and _same_array(self.lat, other.lat)
            and _same_array(self.lon, other.lon)
            and self._identical_static(other)
        )

    def __repr__(self) -> str:
        return (
            f"GeoSwath({self.backend} {self.dtype} {self.shape}, "
            f"crs={_crs_label(self.crs)}, fill_value={self.fill_value!r}, "
            f"attrs={tuple(self.attrs)}, dims={self.dims})"
        )


def _swath_flatten(s: GeoSwath) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (s.data, s.lat, s.lon), (s.dims, s.coords, s.crs, s.fill_value, s.attrs)


def _swath_unflatten(aux: tuple[Any, ...], leaves: Any) -> GeoSwath:
    data, lat, lon = leaves
    dims, coords, crs, fill, attrs = aux
    matched = vmapped_dims(dims, getattr(data, "shape", None))
    if matched != dims:
        coords = tuple(c for c in coords if c.dim in matched)
    return _build(
        GeoSwath,
        data=data,
        lat=lat,
        lon=lon,
        dims=matched,
        coords=coords,
        crs=crs,
        fill_value=fill,
        attrs=attrs,
    )


# -- GeoPoints ------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class GeoPoints(_Named):
    """Scattered samples on the ``obs`` dim: positions, optional z, time, valid.

    - **xy** — ``(obs, 2)`` positions in ``crs`` (lon, lat for WGS 84).
    - **z** — ``(obs,)`` vertical positions, described by ``vertical``
      (height, depth or pressure: never a third column of xy).
    - **time** — ``(obs,)`` per-sample times, for moving platforms
      (along-track altimetry, Argo). Shared times are a ``time`` dim with a
      `TimeAxis` instead; both together raise.
    - **valid** — ``(obs,)`` booleans, False on padding (`pad_points`).

    Args:
        data: ``(..., obs)`` values, NumPy or JAX.
        xy: ``(obs, 2)`` positions.
        z: Optional ``(obs,)`` vertical positions; needs ``vertical``.
        vertical: What ``z`` measures (`geoarray.coords.Vertical`).
        time: Optional ``(obs,)`` times (``datetime64`` or ``int64`` ns).
        valid: Optional ``(obs,)`` booleans; omitted means all valid.
        dims: One name per axis, ending in ``"obs"``.
        coords: Coordinates of non-spatial dims (`VariableAxis`, …).
        crs: The CRS of xy; WGS 84 by default.
        fill_value: The nodata marker.
        attrs: Whole-carrier metadata.

    Example:
        >>> import numpy as np
        >>> from geoarray.coords import VariableAxis, Vertical
        >>> depth = Vertical(kind="depth", units="m", positive="down")
        >>> cycle = GeoPoints(
        ...     np.zeros((2, 3)), xy=np.zeros((3, 2)), z=[5.0, 10.0, 20.0],
        ...     vertical=depth, dims="variable obs",
        ...     coords=(VariableAxis(("temp", "psal"), ("degC", "1")),),
        ... )
        >>> cycle.isel(obs=[2]).z.tolist()
        [20.0]
    """

    data: Any
    xy: Any
    z: Any = dataclasses.field(default=None, kw_only=True)
    vertical: Vertical | None = dataclasses.field(default=None, kw_only=True)
    time: Any = dataclasses.field(default=None, kw_only=True)
    valid: Any = dataclasses.field(default=None, kw_only=True)
    dims: tuple[str, ...] = dataclasses.field(default=cast(Any, None), kw_only=True)
    coords: tuple[Coordinate, ...] = dataclasses.field(default=(), kw_only=True)
    crs: pyproj.CRS | None = dataclasses.field(default=cast(Any, _WGS84), kw_only=True)
    fill_value: Any = dataclasses.field(default=None, kw_only=True)
    attrs: FrozenAttrs = dataclasses.field(default_factory=FrozenAttrs, kw_only=True)

    spatial_dims: ClassVar[tuple[str, ...]] = ("obs",)

    def __class_getitem__(cls, item: Any) -> Any:
        """``GeoPoints["variable obs"]``: the carrier annotated with a dims pattern."""
        return _subscript(cls, item)

    def __post_init__(self) -> None:
        backend = backend_of(self.data)
        n = int(self.data.shape[-1]) if self.data.ndim else 0
        xy = _float_leaf(self.xy, backend, "xy")
        if tuple(xy.shape) != (n, 2):
            raise ValueError(f"xy is (obs, 2) = ({n}, 2), got {tuple(xy.shape)}")
        object.__setattr__(self, "xy", xy)
        if self.z is not None:
            if self.vertical is None:
                raise ValueError(
                    "z needs vertical= (height, depth or pressure differ in units and "
                    "direction)"
                )
            z = _float_leaf(self.z, backend, "z")
            if tuple(z.shape) != (n,):
                raise ValueError(f"z is (obs,) = ({n},), got {tuple(z.shape)}")
            object.__setattr__(self, "z", z)
        elif self.vertical is not None:
            raise ValueError("vertical= describes z, but no z was given")
        if self.time is not None:
            time = _time_leaf(self.time, backend)
            if tuple(time.shape) != (n,):
                raise ValueError(f"time is (obs,) = ({n},), got {tuple(time.shape)}")
            object.__setattr__(self, "time", time)
        if self.valid is not None:
            valid = self.valid
            if not is_tracer(valid):
                valid = namespace(self.data).asarray(valid, dtype=bool)
            if tuple(valid.shape) != (n,):
                raise ValueError(f"valid is (obs,) = ({n},), got {tuple(valid.shape)}")
            object.__setattr__(self, "valid", valid)
        _init_named(self, 0, self.spatial_dims, "GeoPoints")
        if self.time is not None and "time" in self.dims:
            raise ValueError(
                "a time dim and a per-sample time leaf together: shared times are "
                "a time dim (TimeAxis), one time per sample is the time leaf"
            )

    @property
    def size(self) -> int:
        """N, the number of samples."""
        return int(self.data.shape[-1])

    @property
    def batch_dims(self) -> tuple[str, ...]:
        """``()``: the leaves of a point set do not vary along other dims."""
        return ()

    def _leaves(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in ("xy", "z", "time", "valid")
            if getattr(self, k) is not None
        }

    def isel(self, selection: dict[str, Any] | None = None, **by_dim: Any) -> GeoPoints:
        """Select by position; ``obs`` slices xy, z, time and valid together.

        Raises:
            IndexError: An integer on ``obs`` (the result would not be a point
                set; take ``[k]`` to keep one sample).
        """
        wanted = {**(selection or {}), **by_dim}
        k = wanted.get("obs", slice(None))
        if isinstance(k, int | np.integer):
            raise IndexError(
                "GeoPoints.isel: an integer on obs would not leave a point set; "
                "take obs=[k]"
            )
        if is_tracer(k):
            raise GeoArrayTraceError("an obs index must be concrete")
        key, dims, coords = self._leading_index(wanted)
        leaves = {name: leaf[k] for name, leaf in self._leaves().items()}
        return GeoPoints(
            self.data[key],
            leaves.pop("xy"),
            **leaves,
            vertical=self.vertical,
            dims=dims,
            coords=coords,
            crs=self.crs,
            fill_value=self.fill_value,
            attrs=self.attrs,
        )

    def with_data(
        self, data: Any, *, dims: Any = None, fill_value: Any = ..., attrs: Any = None
    ) -> GeoPoints:
        """New values at these samples (same ``obs``)."""
        new_dims = self.dims if dims is None else dims
        if isinstance(new_dims, str):
            new_dims = tuple(new_dims.split())
        shape = tuple(data.shape)
        coords = tuple(
            c
            for c in self.coords
            if c.dim in new_dims and shape[new_dims.index(c.dim)] == c.size
        )
        leaves = self._leaves()
        return GeoPoints(
            data,
            leaves.pop("xy"),
            **leaves,
            vertical=self.vertical,
            dims=new_dims,
            coords=coords,
            crs=self.crs,
            fill_value=self.fill_value if fill_value is ... else fill_value,
            attrs=self.attrs if attrs is None else attrs,
        )

    def valid_mask(self) -> Any:
        """``(obs,)`` True where a sample holds data: valid, finite, not the fill."""
        xp = self.xp
        data = self.data
        bad = (
            ~xp.isfinite(data)
            if data.dtype.kind in "fc"
            else xp.zeros(data.shape, bool)
        )
        fill = self.fill_value
        if fill is not None and not (isinstance(fill, float) and math.isnan(fill)):
            bad = bad | (data == fill)
        axes = tuple(range(self.ndim - 1))
        ok = ~bad.any(axis=axes) if axes else ~bad
        return ok if self.valid is None else ok & self.valid

    def to_jax(self, device: Any = None) -> GeoPoints:
        """Data and leaves on a JAX device; times as ``int64`` ns."""
        jax = import_jax()
        require_x64()
        moved = {k: _to_jax_leaf(v, device) for k, v in self._leaves().items()}
        return dataclasses.replace(
            self, data=jax.device_put(self.data, device), **moved
        )

    def to_numpy(self) -> GeoPoints:
        """Data and leaves on the host; times as ``datetime64[ns]``."""
        moved = {k: _host(v, "to_numpy") for k, v in self._leaves().items()}
        if "time" in moved and moved["time"].dtype.kind != "M":
            moved["time"] = moved["time"].astype(np.int64).view("datetime64[ns]")
        return dataclasses.replace(self, data=_host(self.data, "to_numpy"), **moved)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """The values as a NumPy array (a host copy for JAX data)."""
        out = _host(self.data, "np.asarray(GeoPoints)")
        return np.array(out, dtype=dtype, copy=True) if copy else np.asarray(out, dtype)

    def to_xarray(self) -> Any:
        """This carrier as a CF DSG ``Dataset`` on ``obs`` (``[xarray]``)."""
        return xarray_module().points_to_xarray(self)

    @classmethod
    def from_xarray(cls, obj: Any) -> GeoPoints:
        """A CF DSG ``Dataset`` on ``obs`` as a `GeoPoints` (see `to_xarray`)."""
        return xarray_module().points_from_xarray(obj)

    def identical(self, other: object) -> bool:
        """Equal values and leaves (NaN-aware), vertical and static fields."""
        if not isinstance(other, GeoPoints):
            return False
        mine, theirs = self.to_numpy(), other.to_numpy()
        return (
            _same_array(mine.data, theirs.data)
            and all(
                _same_array(getattr(mine, k), getattr(theirs, k))
                for k in ("xy", "z", "time", "valid")
            )
            and self.vertical == other.vertical
            and self._identical_static(other)
        )

    def __repr__(self) -> str:
        extra = [k for k in ("z", "time", "valid") if getattr(self, k) is not None]
        return (
            f"GeoPoints({self.backend} {self.dtype} {self.shape}, "
            f"crs={_crs_label(self.crs)}, leaves=xy{''.join(', ' + k for k in extra)}, "
            f"fill_value={self.fill_value!r}, dims={self.dims})"
        )


_POINT_LEAVES = ("xy", "z", "time", "valid")


def _points_flatten(p: GeoPoints) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    present = tuple(k for k in _POINT_LEAVES if getattr(p, k) is not None)
    leaves = tuple(_as_int_leaf(getattr(p, k)) for k in present)
    aux = (present, p.vertical, p.dims, p.coords, p.crs, p.fill_value, p.attrs)
    return (p.data, *leaves), aux


def _points_unflatten(aux: tuple[Any, ...], leaves: Any) -> GeoPoints:
    present, vertical, dims, coords, crs, fill, attrs = aux
    data, *rest = leaves
    fields = dict.fromkeys(_POINT_LEAVES)
    fields.update(zip(present, rest, strict=True))
    matched = vmapped_dims(dims, getattr(data, "shape", None))
    if matched != dims:
        coords = tuple(c for c in coords if c.dim in matched)
    return _build(
        GeoPoints,
        data=data,
        vertical=vertical,
        dims=matched,
        coords=coords,
        crs=crs,
        fill_value=fill,
        attrs=attrs,
        **fields,
    )


def pad_points(points: GeoPoints, n: int) -> GeoPoints:
    """``points`` padded to ``n`` samples, with ``valid=False`` on the pad.

    Pad samples sit at the first sample's position (finite, so gradients
    through a sampling operator stay finite), carry the fill value (NaN for
    float data without one) and time / z of the first sample. A cost that
    masks with ``valid`` is then unchanged by the pad length (GA-G16).

    Raises:
        ValueError: ``n`` is smaller than the number of samples, or there
            are none to copy a position from.

    Example:
        >>> import numpy as np
        >>> p = GeoPoints(np.ones(3), xy=np.zeros((3, 2)), dims="obs")
        >>> padded = pad_points(p, 5)
        >>> padded.size, padded.valid.tolist()
        (5, [True, True, True, False, False])
    """
    have = points.size
    if n < have:
        raise ValueError(f"pad_points: n={n} is smaller than the {have} samples")
    if have == 0:
        raise ValueError("pad_points needs at least one sample to place the pad")
    xp = points.xp
    extra = n - have
    fill = points.fill_value
    if fill is None:
        if points.dtype.kind not in "fc":
            raise ValueError(
                "pad_points: integer values need a fill_value for the padded samples"
            )
        fill = math.nan
    lead = points.shape[:-1]
    data = xp.concatenate(
        [points.data, xp.full((*lead, extra), fill, points.dtype)], -1
    )

    def repeat_first(leaf: Any) -> Any:
        lx = namespace(leaf) if not isinstance(leaf, np.ndarray) else np
        reps = lx.repeat(leaf[:1], extra, axis=0)
        return lx.concatenate([leaf, reps], axis=0)

    valid = points.valid if points.valid is not None else xp.ones(have, dtype=bool)
    leaves = {
        "xy": repeat_first(points.xy),
        "z": None if points.z is None else repeat_first(points.z),
        "time": None if points.time is None else repeat_first(points.time),
        "valid": xp.concatenate([valid, xp.zeros(extra, dtype=bool)]),
    }
    return GeoPoints(
        data,
        leaves.pop("xy"),
        **leaves,
        vertical=points.vertical,
        dims=points.dims,
        coords=points.coords,
        crs=points.crs,
        fill_value=fill,
        attrs=points.attrs,
    )


def _register_pytrees() -> bool:
    try:
        jax = import_jax()
    except ImportError:  # pragma: no cover - the base-install job
        return False
    from jax.tree_util import GetAttrKey

    def swath_keys(s: GeoSwath) -> tuple[Any, Any]:
        leaves, aux = _swath_flatten(s)
        names = ("data", "lat", "lon")
        return tuple(
            (GetAttrKey(k), v) for k, v in zip(names, leaves, strict=True)
        ), aux

    def points_keys(p: GeoPoints) -> tuple[Any, Any]:
        leaves, aux = _points_flatten(p)
        names = ("data", *aux[0])
        return tuple(
            (GetAttrKey(k), v) for k, v in zip(names, leaves, strict=True)
        ), aux

    jax.tree_util.register_pytree_with_keys(
        GeoSwath, swath_keys, _swath_unflatten, flatten_func=_swath_flatten
    )
    jax.tree_util.register_pytree_with_keys(
        GeoPoints, points_keys, _points_unflatten, flatten_func=_points_flatten
    )
    return True


PYTREES_REGISTERED = _register_pytrees()
