"""HDF5 / NetCDF / HDF4 reads on the pool, and CF-convention decoding.

Most EO products ship as NetCDF-4 (GOES ABI, MTG FCI, TROPOMI, VIIRS,
Sentinel-3) — HDF5 files following the CF conventions — and the rest as
NetCDF-3 or HDF4 / HDF-EOS. This module reads all of them from any
location the pool accepts:

- HDF5 / NetCDF-4 through h5py over `geocloud.files.open` — ranged reads,
  so a remote file is never copied whole (`open_hdf5`);
- NetCDF-3 and HDF4, whose libraries need a real file, through a
  `geocloud.cache` copy (in place when the file is already local).

On top sit the CF decoders the product readers share (`PackedVariable`,
`attr`, `scalar`, `unpacked`, `time_attr`, `grid_variables`) and two
`geoarray.GeoArray` readers (`read_hdf`, `read_netcdf`).

h5py, netCDF4 and pyhdf are imported at use time; a missing one raises an
``ImportError`` naming the geotoolz-cloud extra (``[hdf5]``, ``[netcdf]``,
``[hdf4]``).
"""

from __future__ import annotations

import importlib
import io
import os
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any, cast

import numpy as np
from affine import Affine
from geoarray import DIMS, GeoArray
from geoarray.coords import TimeAxis, Vertical, VerticalAxis

from geocloud._src.extras import missing_extra
from geocloud._src.store import Location, local_path


__all__ = [
    "HDF4_SIGNATURE",
    "HDF5_SIGNATURE",
    "HdfSource",
    "PackedVariable",
    "affine_from_geotransform",
    "attr",
    "fill_value_from_attrs",
    "grid_variables",
    "open_hdf5",
    "read_hdf",
    "read_indexes",
    "read_netcdf",
    "scalar",
    "select_indexes",
    "time_attr",
    "unpacked",
]

#: A URI, a local path, or an open binary file object.
HdfSource = Location | IO[bytes]

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
"""The first eight bytes of an HDF5 (and NetCDF-4) file."""
HDF4_SIGNATURE = b"\x0e\x03\x13\x01"
"""The first four bytes of an HDF4 / HDF-EOS file."""


def _require(module: str, extra: str, feature: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        error = missing_extra(feature, extra)
        error.name = module  # which backend is missing, for callers' hints
        raise error from exc


def _plain(value: Any) -> Any:
    """An attribute value as plain Python (arrays → lists, bytes → ``str``)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return (
            [_plain(v) for v in value.tolist()] if value.ndim else _plain(value.item())
        )
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


# --- opening ---------------------------------------------------------------------


class _RangedFile(io.RawIOBase):
    """A standard read-only file over an obstore reader.

    obstore's ``ReadableFile.read`` returns its own buffer type, which
    h5py's file-object driver rejects; ``readinto`` copies into the caller's
    buffer instead, so h5py (and anything expecting a plain binary file)
    reads it.
    """

    def __init__(self, handle: Any) -> None:
        super().__init__()
        self._handle = handle

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = memoryview(self._handle.read(len(buffer)))
        view = memoryview(buffer).cast("B")
        view[: data.nbytes] = data.cast("B")
        return data.nbytes

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


@contextmanager
def open_hdf5(
    source: HdfSource, *, storage_options: Mapping[str, Any] | None = None
) -> Iterator[Any]:
    """An open ``h5py.File`` on ``source``, read-only.

    A local path is opened directly; a remote URI through
    `geocloud.files.open`, so h5py fetches only the byte ranges it needs
    (the header, then the chunks it reads); an open binary file object is
    used as is.

    Args:
        source: A URI in any form the pool accepts, a local path, or an
            open binary file.
        storage_options: Forwarded to `geocloud.store.get_obstore`.

    Yields:
        The ``h5py.File``; it (and the remote handle) close on exit.

    Raises:
        ImportError: h5py is not installed (the ``[hdf5]`` extra).

    Examples:
        >>> import tempfile, pathlib, h5py
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "a.h5"
        >>> with h5py.File(path, "w") as f:
        ...     f["x"] = [1, 2, 3]
        >>> with open_hdf5(path) as f:
        ...     f["x"][1]
        np.int64(2)
    """
    h5py = _require("h5py", "hdf5", "geocloud.hdf.open_hdf5")
    if hasattr(source, "read"):
        with h5py.File(source, "r") as f:
            yield f
        return
    local = local_path(source)  # type: ignore[arg-type]
    if local is not None:
        with h5py.File(local, "r") as f:
            yield f
        return
    from geocloud._src.files import open as open_object

    handle = _RangedFile(
        open_object(source, "rb", storage_options=storage_options)  # type: ignore[arg-type]
    )
    try:
        with h5py.File(handle, "r") as f:
            yield f
    finally:
        handle.close()


def _signature(source: Location, storage_options: Mapping[str, Any] | None) -> bytes:
    from geocloud._src.files import open as open_object

    local = local_path(source)
    if local is not None:
        with local.open("rb") as fh:
            return fh.read(8)
    handle = cast(Any, open_object(source, "rb", storage_options=storage_options))
    try:
        return bytes(handle.read(8))
    finally:
        handle.close()


def _localize(source: Location, storage_options: Mapping[str, Any] | None) -> str:
    from geocloud._src.cache import localize

    return str(localize(source, storage_options=storage_options))


# --- CF decoding -----------------------------------------------------------------


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """An HDF5 attribute as a plain Python scalar / ``str`` (arrays kept).

    Examples:
        >>> class V:
        ...     attrs = {"units": np.bytes_(b"K"), "scale": np.array([0.5])}
        >>> attr(V, "units"), attr(V, "scale"), attr(V, "nope", 1)
        ('K', 0.5, 1)
    """
    if name not in obj.attrs:
        return default
    value = obj.attrs[name]
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value
    if isinstance(value, bytes | np.bytes_):
        value = value.decode("utf-8")
    elif isinstance(value, np.generic):
        value = value.item()
    return value


def time_attr(obj: Any, name: str) -> datetime | None:
    """An ISO-8601 attribute (``time_coverage_start``, …) as a ``datetime``.

    Returns ``None`` when the attribute is absent or not a timestamp.

    Examples:
        >>> class V:
        ...     attrs = {"t": b"2024-06-01T12:00:00"}
        >>> time_attr(V, "t")
        datetime.datetime(2024, 6, 1, 12, 0)
    """
    value = attr(obj, name)
    try:
        return None if value is None else datetime.fromisoformat(str(value))
    except ValueError:
        return None


def scalar(f: Any, name: str, *, fill: float | None = None) -> float:
    """A scalar variable as ``float``; ``NaN`` when absent or filled.

    The fill is the variable's own ``_FillValue``, or ``fill`` when the
    product documents one the file does not declare.
    """
    if name not in f:
        return float("nan")
    var = f[name]
    value = float(np.asarray(var[()]).reshape(-1)[0])
    declared = attr(var, "_FillValue")
    fills = {v for v in (declared, fill) if v is not None}
    return float("nan") if value in fills else value


def unpacked(f: Any, name: str) -> np.ndarray:
    """A packed variable (typically a coordinate) decoded to ``float64``."""
    var = f[name]
    raw = np.asarray(var[()], dtype=np.float64)
    scale = float(attr(var, "scale_factor", 1.0))
    return raw * scale + float(attr(var, "add_offset", 0.0))


@dataclass(frozen=True)
class PackedVariable:
    """How to decode one CF-packed variable.

    Attributes:
        name: Variable name in the file.
        storage: The integer / float dtype pixels are *interpreted* as
            (the unsigned twin of the stored dtype when ``_Unsigned``).
        fill: ``_FillValue`` in ``storage`` terms, or ``None``.
        scale: ``scale_factor`` (``None`` when unpacked).
        offset: ``add_offset`` (``None`` when unpacked).
        units: CF ``units`` (``""`` when absent).
        long_name: CF ``long_name`` (``""`` when absent).

    Examples:
        >>> v = PackedVariable("t", np.dtype("u2"), 65535, 0.01, 200.0, "K", "")
        >>> v.decode(np.array([0, 100, 65535], dtype="u2")).tolist()
        [200.0, 201.0, nan]
    """

    name: str
    storage: np.dtype
    fill: int | float | None
    scale: float | None
    offset: float | None
    units: str
    long_name: str

    @classmethod
    def from_file(cls, f: Any, name: str) -> PackedVariable:
        """Read the packing attributes of ``name`` from an open file."""
        var = f[name]
        storage = np.dtype(var.dtype)
        if attr(var, "_Unsigned") == "true" and storage.kind == "i":
            storage = np.dtype(f"u{storage.itemsize}")
        fill = None
        if "_FillValue" in var.attrs:
            raw_fill = np.asarray(var.attrs["_FillValue"]).reshape(-1)[:1]
            fill = raw_fill.astype(var.dtype).view(storage).item()
        scale = attr(var, "scale_factor")
        offset = attr(var, "add_offset")
        return cls(
            name=name,
            storage=storage,
            fill=fill,
            scale=None if scale is None else float(scale),
            offset=None if offset is None else float(offset),
            units=str(attr(var, "units", "")),
            long_name=str(attr(var, "long_name", "")),
        )

    @property
    def is_packed(self) -> bool:
        """Whether decoding changes values (scale / offset applied)."""
        return self.scale is not None or self.offset is not None

    @property
    def is_categorical(self) -> bool:
        """An unpacked integer variable: masks, flags, class codes."""
        return not self.is_packed and self.storage.kind in "iub"

    def raw(self, values: np.ndarray) -> np.ndarray:
        """Stored values reinterpreted in ``storage`` (``_Unsigned`` view)."""
        values = np.asarray(values)
        if values.dtype == self.storage:
            return values
        return np.ascontiguousarray(values).view(self.storage)

    def decode(self, values: np.ndarray) -> np.ndarray:
        """Physical values as ``float32``; fill pixels become ``NaN``."""
        raw = self.raw(values)
        out = raw.astype(np.float32)
        if self.scale is not None:
            out *= np.float32(self.scale)
        if self.offset is not None:
            out += np.float32(self.offset)
        if self.fill is not None:
            out[raw == self.fill] = np.nan
        return out


def grid_variables(f: Any, shape: tuple[int, int]) -> tuple[str, ...]:
    """Names of the file's top-level variables of shape ``shape``."""
    # Groups carry no ``shape``; datasets do.
    return tuple(
        name
        for name, var in f.items()
        if tuple(getattr(var, "shape", ()) or ()) == shape
    )


# --- band selection, fills, georeferencing ----------------------------------------


def select_indexes(values: Any, indexes: list[int] | None) -> np.ndarray:
    """Select 1-based band ``indexes`` along the leading axis of ``values``.

    Args:
        values: Array-like ``(C, ...)`` stack, or a 2-D single-layer map.
        indexes: 1-based band numbers (rasterio convention), or ``None``
            for every band. A 2-D map accepts only ``[1]`` (a no-op).

    Returns:
        The selected bands (``values`` itself when ``indexes`` is ``None``).

    Raises:
        ValueError: An index is not positive, or ``values`` has no band
            axis and ``indexes`` is not ``[1]``.

    Examples:
        >>> select_indexes(np.arange(12).reshape(3, 2, 2), [3, 1])[:, 0, 0]
        array([8, 0])
    """
    array = np.asanyarray(values)
    if indexes is None:
        return array
    zero_based = [index - 1 for index in indexes]
    if any(index < 0 for index in zero_based):
        raise ValueError("indexes are 1-based and must be positive.")
    if array.ndim < 3:
        # A non-band dataset is a single-layer raster: indexes=[1] selects
        # the only layer and is a no-op; anything else is invalid.
        if zero_based != [0]:
            raise ValueError("indexes require a dataset with a leading band axis.")
        return array
    return np.take(array, zero_based, axis=0)


def read_indexes(source: Any, indexes: list[int] | None) -> np.ndarray:
    """Read 1-based band ``indexes`` from a lazily sliced dataset.

    Same contract as `select_indexes`, but ``source`` is sliced (an
    h5py-style hyperslab read) instead of materialised first, so requesting
    one band reads only that band. h5py only fancy-indexes the leading axis
    with increasing indices, so the read is sorted and then reordered.

    Args:
        source: Dataset supporting ``source[...]`` and
            ``source[list_of_ints]`` (e.g. ``h5py.Dataset``).
        indexes: 1-based band numbers, or ``None`` for every band.

    Returns:
        The selected bands.

    Raises:
        ValueError: See `select_indexes`.

    Examples:
        >>> read_indexes(np.arange(12).reshape(3, 2, 2), [3, 1])[:, 0, 0]
        array([8, 0])
    """
    if indexes is None:
        return np.asanyarray(source[...])
    zero_based = [index - 1 for index in indexes]
    if any(index < 0 for index in zero_based):
        raise ValueError("indexes are 1-based and must be positive.")
    ndim = getattr(source, "ndim", None)
    if ndim is None:
        ndim = np.asarray(source.shape).size
    if ndim < 3:
        if zero_based != [0]:
            raise ValueError("indexes require a dataset with a leading band axis.")
        return np.asanyarray(source[...])
    order = np.argsort(zero_based)
    sorted_indexes = [zero_based[i] for i in order]
    sliced = np.asanyarray(source[sorted_indexes])
    if list(order) == list(range(len(order))):
        return sliced
    return sliced[np.argsort(order)]


def fill_value_from_attrs(attrs: Mapping[str, Any]) -> Any:
    """The scalar fill value declared in ``attrs``, or ``None`` when none is.

    Looks up ``_FillValue``, ``missing_value``, ``fill_value`` and
    ``nodata`` in that order. Size-1 arrays / lists (how netCDF4 / xarray
    write HDF5 attributes) are unwrapped; multi-element values are skipped,
    since a carrier's ``fill_value`` must be a scalar.

    Examples:
        >>> fill_value_from_attrs({"_FillValue": [-9999]})
        -9999
        >>> fill_value_from_attrs({}) is None
        True
    """
    for name in ("_FillValue", "missing_value", "fill_value", "nodata"):
        if name not in attrs:
            continue
        value = attrs[name]
        if isinstance(value, list | tuple | np.ndarray):
            flat = np.ravel(np.asarray(value))
            if flat.size != 1:
                continue
            value = flat[0].item()
        return value
    return None


def affine_from_geotransform(geotransform: Any) -> Affine:
    """Parse a GDAL ``GeoTransform`` attribute into an ``Affine``.

    Args:
        geotransform: Six whitespace-separated numbers
            ``"x0 dx rx y0 ry dy"`` (GDAL order), or ``None``.

    Returns:
        The affine, or the identity when ``geotransform`` is ``None`` or
        does not hold exactly six numbers.

    Examples:
        >>> affine_from_geotransform("100 10 0 200 0 -10")
        Affine(10.0, 0.0, 100.0,
               0.0, -10.0, 200.0)
    """
    if geotransform is None:
        return Affine.identity()
    parts = [float(part) for part in str(geotransform).split()]
    if len(parts) != 6:
        return Affine.identity()
    return Affine.from_gdal(*parts)


def _geoarray(
    values: Any,
    *,
    crs: Any = None,
    fill_value: Any = 0,
    attrs: dict[str, Any] | None = None,
    transform: Affine | None = None,
    dims: tuple[str, ...] | None = None,
    coords: tuple[Any, ...] = (),
) -> GeoArray:
    values = np.asarray(values)
    return GeoArray(
        values,
        Affine.identity() if transform is None else transform,
        crs,
        fill_value=_representable_fill(fill_value, values.dtype),
        attrs=attrs or {},
        # HDF datasets carry no dim names: the 0.2 layout, explicitly
        dims=DIMS[-values.ndim :] if dims is None else dims,
        coords=coords,
    )


_TIME_NAMES = ("time", "t")
_VERTICAL_NAMES = (
    "depth",
    "deptht",
    "lev",
    "level",
    "plev",
    "pressure",
    "height",
    "altitude",
    "z",
)
_PRESSURE_UNITS = ("pa", "hpa", "kpa", "mbar", "millibar", "bar", "atm")


def _cf_dims(holder: Any, var: Any, indexes: list[int] | None) -> tuple[Any, ...]:
    """A NetCDF variable's dims and coordinates, named the geoarray way.

    The last two dims are ``y x``. A leading dim whose CF coordinate
    variable (or name) marks it as time becomes ``time`` with a `TimeAxis`;
    one marked vertical (``axis="Z"``, ``positive``, a vertical
    ``standard_name``, pressure units, or a name like ``depth`` / ``lev`` /
    ``plev``) becomes ``level`` with a `VerticalAxis`. Any other dim keeps
    its NetCDF name. ``indexes`` (1-based, on the first dim) select the
    first dim's coordinate values too.

    Returns:
        ``(dims, coords)``.
    """
    names = [str(name) for name in var.dimensions]
    lead = names[:-2]
    found = [_netcdf_lookup(holder, name) for name in lead]
    attrs = [_plain(dict(v.__dict__)) if v is not None else {} for v in found]
    kinds = [_cf_kind(name, a) for name, a in zip(lead, attrs, strict=True)]
    dims = _unique_dims(lead, kinds)
    coords: list[Any] = []
    for position, (name, coord_var, attr, kind, dim) in enumerate(
        zip(lead, found, attrs, kinds, dims, strict=True)
    ):
        named = (kind, dim) in (("time", "time"), ("vertical", "level"))
        if not named or coord_var is None or len(coord_var.dimensions) != 1:
            continue  # kept its NetCDF name, or no 1-D coordinate to read
        values = np.asarray(coord_var[:])
        if position == 0 and indexes is not None:
            values = values[[index - 1 for index in indexes]]
        if dim == "level":
            values_list = tuple(float(v) for v in values.tolist())
            coords.append(VerticalAxis(values_list, _vertical(name, attr)))
        else:  # time
            stamps = _cf_times(values, attr)
            if stamps is not None:
                coords.append(TimeAxis(stamps))
    return (*dims, "y", "x"), tuple(coords)


def _unique_dims(lead: list[str], kinds: list[str]) -> list[str]:
    """Leading dims named ``time`` / ``level`` by kind, without a repeat.

    A dim keeps its NetCDF name when renaming it would repeat another dim's
    name (two vertical dims, ``("depth", "level", y, x)``), or clash with
    ``y`` / ``x``; renaming stops once the names are unique.

    Example:
        >>> _unique_dims(["depth", "level"], ["vertical", "vertical"])
        ['depth', 'level']
        >>> _unique_dims(["time", "depth"], ["time", "vertical"])
        ['time', 'level']
    """
    dims = [
        {"time": "time", "vertical": "level"}.get(k, n)
        for n, k in zip(lead, kinds, strict=True)
    ]
    while True:
        taken = [*dims, "y", "x"]
        clash = [i for i, d in enumerate(dims) if d != lead[i] and taken.count(d) > 1]
        if not clash:
            break
        for i in clash:
            dims[i] = lead[i]
    if len(set(dims)) != len(dims) or {"y", "x"} & set(dims):
        return list(lead)  # the NetCDF names themselves repeat or clash: leave them
    return dims


def _cf_kind(name: str, attrs: Mapping[str, Any]) -> str:
    axis = str(attrs.get("axis", "")).upper()
    standard = str(attrs.get("standard_name", "")).lower()
    units = str(attrs.get("units", "")).lower()
    lower = name.lower()
    if axis == "T" or standard == "time" or lower in _TIME_NAMES:
        return "time"
    vertical_standard = standard in ("depth", "height", "altitude", "air_pressure")
    if (
        axis == "Z"
        or "positive" in attrs
        or vertical_standard
        or units in _PRESSURE_UNITS
        or lower in _VERTICAL_NAMES
    ):
        return "vertical"
    return "other"


def _vertical(name: str, attrs: Mapping[str, Any]) -> Vertical:
    """What a CF vertical coordinate measures (`geoarray.coords.Vertical`)."""
    standard = str(attrs.get("standard_name", "")).lower()
    units = str(attrs.get("units", "1"))
    lower = name.lower()
    if (
        units.lower() in _PRESSURE_UNITS
        or standard == "air_pressure"
        or lower in ("plev", "pressure")
    ):
        kind = "pressure"
    elif standard == "depth" or lower.startswith("depth"):
        kind = "depth"
    elif standard in ("height", "altitude") or lower in ("height", "altitude", "z"):
        kind = "height"
    else:
        kind = "level"
    positive = str(attrs.get("positive", "")).lower()
    if positive not in ("up", "down"):
        positive = "down" if kind in ("depth", "pressure") else "up"
    return Vertical(kind=kind, units=units, positive=positive)


def _cf_times(values: np.ndarray, attrs: Mapping[str, Any]) -> Any | None:
    """CF ``units: "<unit> since <date>"`` times as ``datetime64``, or ``None``."""
    units = attrs.get("units")
    if values.dtype.kind == "M":
        return values
    if not isinstance(units, str) or " since " not in units:
        return None
    try:
        import netCDF4  # the [netcdf] extra, already open in read_netcdf

        dates = netCDF4.num2date(
            values,
            units,
            calendar=str(attrs.get("calendar", "standard")),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    except Exception:  # a non-standard calendar, or unparsable units
        return None
    return np.asarray([np.datetime64(d, "ns") for d in np.ravel(np.asarray(dates))])


def _representable_fill(fill: Any, dtype: np.dtype) -> Any:
    """``fill`` when ``dtype`` data can hold it, else ``None``.

    Files declare fills their dtype cannot hold (``-9999.0`` on ``uint16``,
    a float fill on integers); no pixel can equal such a fill, so the read
    has no nodata and its carrier none (a `GeoArray` refuses the fill).
    """
    if fill is None or dtype.kind not in "biu":
        return fill
    if isinstance(fill, bool | np.bool_):
        fill = int(fill)
    if not isinstance(fill, int | float | np.integer | np.floating):
        return None
    value = float(fill)
    if not (np.isfinite(value) and value.is_integer()):
        return None
    low, high = (
        (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
    )
    return fill if low <= value <= high else None


# --- GeoArray readers --------------------------------------------------------------


def read_hdf(
    source: Location,
    dataset: str,
    *,
    indexes: list[int] | None = None,
    metadata_groups: list[str] | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> GeoArray:
    """Read one dataset of an HDF5 or HDF4 / HDF-EOS file as a `geoarray.GeoArray`.

    The format is told by the file signature. HDF5 is read through h5py
    with ranged reads (`open_hdf5`), only the requested bands off disk;
    HDF4 through pyhdf on a local copy (`geocloud.cache.localize`). The
    result has an identity transform. A swath product carries its
    georeferencing in latitude / longitude datasets on the same pixel grid:
    read each with its own ``read_hdf`` call (attrs are frozen and
    hashable, so they never hold 2-D arrays).

    Args:
        source: A URI in any form the pool accepts, or a local path.
        dataset: Dataset path inside the file.
        indexes: 1-based band indexes along the leading dataset axis.
        metadata_groups: HDF5 group paths whose attributes are copied into
            ``attrs["metadata"]``.
        storage_options: Forwarded to `geocloud.store.get_obstore`.

    Returns:
        The dataset ``(..., H, W)``, its attributes under ``attrs["attrs"]``
        and its declared fill (else ``None``) as the carrier's ``fill_value``
        (``None`` when the dataset's dtype cannot hold the declared fill).

    Raises:
        ImportError: h5py (``[hdf5]``) or pyhdf (``[hdf4]``) is missing.
        ValueError: The file is neither HDF5 nor HDF4, ``indexes`` do
            not fit the dataset, or the dataset is not 2-D to 4-D.
        KeyError: No such dataset.
    """
    signature = _signature(source, storage_options)
    if signature.startswith(HDF5_SIGNATURE):
        return _read_hdf5(source, dataset, indexes, metadata_groups, storage_options)
    if signature.startswith(HDF4_SIGNATURE):
        return _read_hdf4(source, dataset, indexes, storage_options)
    raise ValueError(f"{os.fspath(source)!r} is neither an HDF5 nor an HDF4 file.")


def _read_hdf5(
    source: Location,
    dataset: str,
    indexes: list[int] | None,
    metadata_groups: list[str] | None,
    storage_options: Mapping[str, Any] | None,
) -> GeoArray:
    _require("h5py", "hdf5", "geocloud.hdf.read_hdf (HDF5)")
    with open_hdf5(source, storage_options=storage_options) as f:
        var = f[dataset]
        attrs = _plain(dict(var.attrs))
        values = read_indexes(var, indexes)
        out_attrs: dict[str, Any] = {"attrs": attrs}
        if metadata_groups is not None:
            out_attrs["metadata"] = {
                group: _plain(dict(f[group].attrs)) for group in metadata_groups
            }
    return _geoarray(values, fill_value=fill_value_from_attrs(attrs), attrs=out_attrs)


def _read_hdf4(
    source: Location,
    dataset: str,
    indexes: list[int] | None,
    storage_options: Mapping[str, Any] | None,
) -> GeoArray:
    sd = _require("pyhdf.SD", "hdf4", "geocloud.hdf.read_hdf (HDF4)")
    hdf = sd.SD(_localize(source, storage_options), sd.SDC.READ)
    try:
        var = hdf.select(dataset)
        attrs = _plain(dict(var.attributes()))
        values = select_indexes(var.get(), indexes)
        out_attrs: dict[str, Any] = {"attrs": attrs}
    finally:
        hdf.end()
    return _geoarray(values, fill_value=fill_value_from_attrs(attrs), attrs=out_attrs)


def _netcdf_group(root: Any, group: str | None) -> Any:
    current = root
    for part in (group or "").strip("/").split("/"):
        if part:
            current = current.groups[part]
    return current


def _netcdf_crs(
    group: Any, variable: Any, use_cf_grid_mapping: bool
) -> tuple[Any, Any]:
    """``(crs, grid_mapping variable)`` from the CF ``grid_mapping`` attribute."""
    name = getattr(variable, "grid_mapping", None)
    # CF scope: the nearest ancestor group, or a path; the extended form
    # ("crs: x y") names the mapping before its colon.
    mapping = (
        _netcdf_lookup(group, name.split(":")[0].strip())
        if isinstance(name, str) and name.strip()
        else None
    )
    if not use_cf_grid_mapping or mapping is None:
        return None, mapping
    try:
        from pyproj import CRS

        return CRS.from_cf(_plain(dict(mapping.__dict__))), mapping
    except (KeyError, RuntimeError, ValueError):
        return None, mapping


def _netcdf_transform(group: Any, variable: Any, mapping: Any, crs: Any) -> Affine:
    """The variable's affine: GDAL's ``GeoTransform``, else its CF coordinates.

    GDAL's netCDF driver writes ``GeoTransform`` on the ``grid_mapping``
    variable; the data variable is checked as a fallback for other writers.
    Without it, the transform comes from evenly spaced 1-D coordinate
    variables of the last two dimensions (`_coordinate`), scaled to the
    CRS's metres when their ``units`` are kilometres; when those are
    missing, irregular, 2-D (curvilinear) or in units that cannot be
    converted, the identity is returned with a warning.
    """
    for holder in (mapping, variable):
        if holder is not None and getattr(holder, "GeoTransform", None) is not None:
            return affine_from_geotransform(holder.GeoTransform)
    if len(variable.dimensions) < 2:
        return Affine.identity()
    y_dim, x_dim = variable.dimensions[-2:]
    reason = _coordinate_affine(group, variable, (y_dim, x_dim), crs)
    if isinstance(reason, Affine):
        return reason
    warnings.warn(
        f"read_netcdf: no GeoTransform on {variable.name!r} and {reason}; "
        "the result has an identity transform.",
        UserWarning,
        stacklevel=3,
    )
    return Affine.identity()


def _coordinate_affine(
    group: Any, variable: Any, dims: tuple[str, str], crs: Any
) -> Affine | str:
    """The transform from the coordinates of ``dims``, or why there is none."""
    declared = [
        _netcdf_lookup(group, name)
        for name in str(getattr(variable, "coordinates", "")).split()
    ]
    coords = [_coordinate(group, declared, dim) for dim in dims]
    missing = [d for d, c in zip(dims, coords, strict=True) if c is None]
    if missing:
        # 2-D auxiliaries only matter when no regular 1-D pair exists.
        if any(
            c is not None and len(c.dimensions) == 2 and set(c.dimensions) == set(dims)
            for c in declared
        ):
            return (
                "its coordinates are 2-D (curvilinear); read the latitude / "
                "longitude variables alongside it (read_netcdf on each)"
            )
        return f"no 1-D coordinate variable for {', '.join(map(repr, missing))}"
    axes = []
    for dim, coord in zip(dims, coords, strict=True):
        factor = _unit_factor(coord, crs)
        if factor is None:
            return (
                f"the {dim!r} coordinate's units {coord.units!r} cannot be "
                "converted to the CRS's"
            )
        axis = _regular_axis(group, coord, factor)
        if axis is None:
            return f"the {dim!r} coordinate is not evenly spaced (or its cells are not)"
        axes.append(axis)
    (dy, y0), (dx, x0) = axes
    return Affine(dx, 0.0, x0, 0.0, dy, y0)


def _netcdf_lookup(group: Any, name: str) -> Any:
    """A variable by CF name: a path (absolute or relative), else the nearest
    group from ``group`` up to the root that defines it."""
    if "/" in name:
        current = group
        if name.startswith("/"):
            while current.parent is not None:
                current = current.parent
        *parts, leaf = [p for p in name.split("/") if p]
        for part in parts:
            current = current.parent if part == ".." else current.groups.get(part)
            if current is None:
                return None
        return current.variables.get(leaf)
    current = group
    while current is not None:
        if name in current.variables:
            return current.variables[name]
        current = current.parent
    return None


def _coordinate(group: Any, declared: list[Any], dim: str) -> Any:
    """The 1-D coordinate variable over ``dim``.

    A horizontal 1-D variable over ``dim`` listed in the data variable's
    ``coordinates`` wins (another auxiliary, such as a height, never
    does); else the CF coordinate variable named after the dimension
    (searched up the group tree), unless it is a bare index with no
    ``units``.
    """
    for coord in declared:
        if (
            coord is not None
            and tuple(coord.dimensions) == (dim,)
            and _is_horizontal(coord)
        ):
            return coord
    coord = _netcdf_lookup(group, dim)
    if (
        coord is not None
        and tuple(coord.dimensions) == (dim,)
        and getattr(coord, "units", None) is not None
    ):
        return coord
    return None


_HORIZONTAL_NAMES = {
    "projection_x_coordinate",
    "projection_y_coordinate",
    "longitude",
    "latitude",
    "grid_longitude",
    "grid_latitude",
}

#: Length units a projected coordinate may carry, in metres.
_METRES = {
    "m": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "us_survey_foot": 1200 / 3937,
    "us survey foot": 1200 / 3937,
    "ftus": 1200 / 3937,
}


def _is_horizontal(coord: Any) -> bool:
    """Whether a CF variable is an X / Y (or longitude / latitude) axis.

    Judged by ``axis``, ``standard_name`` or units (a length or degrees),
    as CF identifies coordinate types.
    """
    if str(getattr(coord, "axis", "")).upper() in ("X", "Y"):
        return True
    if str(getattr(coord, "standard_name", "")) in _HORIZONTAL_NAMES:
        return True
    units = str(getattr(coord, "units", "")).strip().lower()
    return units in _METRES or units.startswith("degree")


def _unit_factor(coord: Any, crs: Any) -> float | None:
    """Multiplier from ``coord``'s units into the CRS's axis units.

    ``1`` for geographic coordinates, a missing CRS or missing units;
    ``None`` for a projected grid whose length unit is unknown.
    """
    if crs is None or not crs.is_projected:
        return 1.0
    units = str(getattr(coord, "units", "")).strip().lower()
    if not units:
        return 1.0
    metres = _METRES.get(units)
    if metres is None:
        return None
    axis_metres = float(getattr(crs.axis_info[0], "unit_conversion_factor", 1.0))
    return metres / axis_metres


def _stored_eps(coord: Any) -> float:
    """Machine epsilon of how ``coord`` is stored (float32 rounds harder)."""
    dtype = np.dtype(getattr(coord, "dtype", np.float64))
    return float(np.finfo(dtype).eps) if dtype.kind == "f" else 0.0


def _is_regular(values: np.ndarray, step: float, eps: float) -> bool:
    """Steps within ``1e-6`` of a step plus the stored values' rounding."""
    atol = 1e-6 * abs(step) + 4.0 * eps * float(np.abs(values).max())
    return bool(np.allclose(np.diff(values), step, rtol=0.0, atol=atol))


def _regular_axis(group: Any, coord: Any, factor: float) -> tuple[float, float] | None:
    """``(step, origin edge)`` of an evenly spaced axis, or ``None``.

    With CF cell ``bounds`` the origin is the first cell's leading edge
    (the cells must be contiguous and as wide as the step); without, the
    values are pixel centres and the origin is ``c[0] - step / 2``. ``step``
    keeps its sign, so a south-up grid gets a positive ``e``.
    """
    values = np.ma.filled(np.ma.asarray(coord[:], dtype=np.float64), np.nan) * factor
    if values.size < 2 or not np.isfinite(values).all():
        return None
    eps = _stored_eps(coord)
    step = float((values[-1] - values[0]) / (values.size - 1))
    if step == 0 or not _is_regular(values, step, eps):
        return None
    name = getattr(coord, "bounds", None)
    cells = _netcdf_lookup(group, name) if isinstance(name, str) else None
    if cells is None:
        return step, float(values[0]) - step / 2
    edges = np.ma.filled(np.ma.asarray(cells[:], dtype=np.float64), np.nan) * factor
    if edges.shape != (values.size, 2) or not np.isfinite(edges).all():
        return None
    lead = edges.min(axis=1) if step > 0 else edges.max(axis=1)
    widths = np.abs(edges[:, 1] - edges[:, 0])
    if not (
        _is_regular(lead, step, eps)
        and np.allclose(
            widths,
            abs(step),
            rtol=0.0,
            atol=1e-6 * abs(step) + 4.0 * eps * float(np.abs(edges).max()),
        )
    ):
        return None
    return step, float(lead[0])


def read_netcdf(
    source: Location,
    variable: str,
    *,
    group: str | None = None,
    indexes: list[int] | None = None,
    decode_cf: bool = True,
    use_cf_grid_mapping: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> GeoArray:
    """Read one variable of a NetCDF-3 / NetCDF-4 file as a `geoarray.GeoArray`.

    netCDF4 opens a local copy (`geocloud.cache.localize`; a local path is
    used in place). With ``decode_cf`` the CF mask / ``scale_factor`` /
    ``add_offset`` are applied and masked pixels become the fill (``NaN``
    for decoded floats). The CRS comes from the variable's CF
    ``grid_mapping``; the transform from GDAL's ``GeoTransform`` attribute,
    or else from evenly spaced 1-D coordinate variables of the last two
    dimensions (CF ``x`` / ``y`` or ``lon`` / ``lat`` pixel centres). When
    neither is there the transform is the identity and a `UserWarning`
    says why.

    Args:
        source: A URI in any form the pool accepts, or a local path.
        variable: Variable name within ``group``.
        group: Slash-separated NetCDF group path.
        indexes: 1-based band indexes along the leading variable axis.
        decode_cf: Apply CF masking and scale / offset.
        use_cf_grid_mapping: Recover the CRS from ``grid_mapping``.
        storage_options: Forwarded to `geocloud.store.get_obstore`.

    Returns:
        The variable ``(..., H, W)``, its attributes under ``attrs["attrs"]``;
        a declared fill its dtype cannot hold becomes ``fill_value=None``.
        Dims are named from the variable's CF dims: the last two are
        ``y x``, a time dim is ``time`` (with a `TimeAxis` for CF ``since``
        units), a vertical one is ``level`` (with a `VerticalAxis`: CF
        ``depth`` is ``Vertical(kind="depth", positive="down")``), and any
        other keeps its NetCDF name.

    Raises:
        ImportError: netCDF4 is missing (the ``[netcdf]`` extra).
        KeyError: No such group or variable.
        ValueError: ``indexes`` do not fit the variable, or it is not 2-D
            to 4-D.
    """
    netcdf4 = _require("netCDF4", "netcdf", "geocloud.hdf.read_netcdf")
    with netcdf4.Dataset(_localize(source, storage_options), "r") as root:
        holder = _netcdf_group(root, group)
        var = holder.variables[variable]
        var.set_auto_maskandscale(decode_cf)
        attrs = _plain(dict(var.__dict__))
        values = select_indexes(var[:], indexes)
        fill_value = fill_value_from_attrs(attrs)
        if np.ma.isMaskedArray(values):
            # Decoded (scaled) floats are NaN-filled, so NaN is the
            # sentinel the returned tensor actually carries.
            if values.dtype.kind == "f":
                fill_value = np.nan
            elif fill_value is None:
                # Masked against netCDF's default fill: that value marks them.
                fill_value = np.asarray(values.fill_value).item()
            values = values.filled(fill_value)
        crs, mapping = _netcdf_crs(holder, var, use_cf_grid_mapping)
        dims, coords = _cf_dims(holder, var, indexes)
        return _geoarray(
            values,
            crs=crs,
            fill_value=fill_value,
            attrs={"attrs": attrs},
            transform=_netcdf_transform(holder, var, mapping, crs),
            dims=dims,
            coords=coords,
        )
