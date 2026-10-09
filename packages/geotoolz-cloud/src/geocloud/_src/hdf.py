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
GeoTensor readers (`read_hdf`, `read_netcdf`).

h5py, netCDF4 and pyhdf are imported at use time; a missing one raises an
``ImportError`` naming the geotoolz-cloud extra (``[hdf5]``, ``[netcdf]``,
``[hdf4]``).
"""

from __future__ import annotations

import importlib
import io
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any, cast

import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor

from geocloud._src.extras import missing_extra
from geocloud._src.store import Location, local_path


__all__ = [
    "HDF4_SIGNATURE",
    "HDF5_SIGNATURE",
    "PackedVariable",
    "Source",
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
Source = Location | IO[bytes]

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
    source: Source, *, storage_options: Mapping[str, Any] | None = None
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
    """The scalar fill value declared in ``attrs`` (default ``0``).

    Looks up ``_FillValue``, ``missing_value``, ``fill_value`` and
    ``nodata`` in that order. Size-1 arrays / lists (how netCDF4 / xarray
    write HDF5 attributes) are unwrapped; multi-element values are skipped,
    since a ``GeoTensor`` fill must be a scalar.

    Examples:
        >>> fill_value_from_attrs({"_FillValue": [-9999]})
        -9999
        >>> fill_value_from_attrs({})
        0
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
    return 0


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


def _geotensor(
    values: Any,
    *,
    crs: Any = None,
    fill_value: Any = 0,
    attrs: dict[str, Any] | None = None,
    transform: Affine | None = None,
) -> GeoTensor:
    return GeoTensor(
        np.asarray(values),
        transform=Affine.identity() if transform is None else transform,
        crs=crs,
        fill_value_default=fill_value,
        attrs=attrs,
    )


# --- GeoTensor readers -------------------------------------------------------------


def read_hdf(
    source: Location,
    dataset: str,
    *,
    indexes: list[int] | None = None,
    geolocation: tuple[str, str] | None = None,
    metadata_groups: list[str] | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> GeoTensor:
    """Read one dataset of an HDF5 or HDF4 / HDF-EOS file as a ``GeoTensor``.

    The format is told by the file signature. HDF5 is read through h5py
    with ranged reads (`open_hdf5`), only the requested bands off disk;
    HDF4 through pyhdf on a local copy (`geocloud.cache.localize`). The
    result has an identity transform: swath products carry their
    georeferencing in ``geolocation`` arrays instead.

    Args:
        source: A URI in any form the pool accepts, or a local path.
        dataset: Dataset path inside the file.
        indexes: 1-based band indexes along the leading dataset axis.
        geolocation: ``(latitude_dataset, longitude_dataset)`` to load into
            ``attrs["geolocation"]``.
        metadata_groups: HDF5 group paths whose attributes are copied into
            ``attrs["metadata"]``.
        storage_options: Forwarded to `geocloud.store.get_obstore`.

    Returns:
        The dataset, its attributes under ``attrs["attrs"]`` and its
        declared fill as the tensor's fill value.

    Raises:
        ImportError: h5py (``[hdf5]``) or pyhdf (``[hdf4]``) is missing.
        ValueError: The file is neither HDF5 nor HDF4, or ``indexes`` do
            not fit the dataset.
        KeyError: No such dataset.
    """
    signature = _signature(source, storage_options)
    if signature.startswith(HDF5_SIGNATURE):
        return _read_hdf5(
            source, dataset, indexes, geolocation, metadata_groups, storage_options
        )
    if signature.startswith(HDF4_SIGNATURE):
        return _read_hdf4(source, dataset, indexes, geolocation, storage_options)
    raise ValueError(f"{os.fspath(source)!r} is neither an HDF5 nor an HDF4 file.")


def _read_hdf5(
    source: Location,
    dataset: str,
    indexes: list[int] | None,
    geolocation: tuple[str, str] | None,
    metadata_groups: list[str] | None,
    storage_options: Mapping[str, Any] | None,
) -> GeoTensor:
    _require("h5py", "hdf5", "geocloud.hdf.read_hdf (HDF5)")
    with open_hdf5(source, storage_options=storage_options) as f:
        var = f[dataset]
        attrs = _plain(dict(var.attrs))
        values = read_indexes(var, indexes)
        out_attrs: dict[str, Any] = {"attrs": attrs}
        if geolocation is not None:
            lat, lon = geolocation
            out_attrs["geolocation"] = {
                "latitude": np.asarray(f[lat][...]),
                "longitude": np.asarray(f[lon][...]),
            }
        if metadata_groups is not None:
            out_attrs["metadata"] = {
                group: _plain(dict(f[group].attrs)) for group in metadata_groups
            }
    return _geotensor(values, fill_value=fill_value_from_attrs(attrs), attrs=out_attrs)


def _read_hdf4(
    source: Location,
    dataset: str,
    indexes: list[int] | None,
    geolocation: tuple[str, str] | None,
    storage_options: Mapping[str, Any] | None,
) -> GeoTensor:
    sd = _require("pyhdf.SD", "hdf4", "geocloud.hdf.read_hdf (HDF4)")
    hdf = sd.SD(_localize(source, storage_options), sd.SDC.READ)
    try:
        var = hdf.select(dataset)
        attrs = _plain(dict(var.attributes()))
        values = select_indexes(var.get(), indexes)
        out_attrs: dict[str, Any] = {"attrs": attrs}
        if geolocation is not None:
            lat, lon = geolocation
            out_attrs["geolocation"] = {
                "latitude": np.asarray(hdf.select(lat).get()),
                "longitude": np.asarray(hdf.select(lon).get()),
            }
    finally:
        hdf.end()
    return _geotensor(values, fill_value=fill_value_from_attrs(attrs), attrs=out_attrs)


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
    mapping = group.variables.get(name) if isinstance(name, str) else None
    if not use_cf_grid_mapping or mapping is None:
        return None, mapping
    try:
        from pyproj import CRS

        return CRS.from_cf(_plain(dict(mapping.__dict__))), mapping
    except (KeyError, RuntimeError, ValueError):
        return None, mapping


def _netcdf_transform(variable: Any, mapping: Any) -> Affine:
    """The GDAL ``GeoTransform`` attribute, on the grid mapping or the variable.

    GDAL's netCDF driver writes it on the ``grid_mapping`` variable; the
    data variable is checked as a fallback for other writers.
    """
    for holder in (mapping, variable):
        if holder is not None and getattr(holder, "GeoTransform", None) is not None:
            return affine_from_geotransform(holder.GeoTransform)
    return Affine.identity()


def read_netcdf(
    source: Location,
    variable: str,
    *,
    group: str | None = None,
    indexes: list[int] | None = None,
    decode_cf: bool = True,
    use_cf_grid_mapping: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> GeoTensor:
    """Read one variable of a NetCDF-3 / NetCDF-4 file as a ``GeoTensor``.

    netCDF4 opens a local copy (`geocloud.cache.localize`; a local path is
    used in place). With ``decode_cf`` the CF mask / ``scale_factor`` /
    ``add_offset`` are applied and masked pixels become the fill (``NaN``
    for decoded floats). The CRS comes from the variable's CF
    ``grid_mapping``, the transform from GDAL's ``GeoTransform`` attribute.

    Args:
        source: A URI in any form the pool accepts, or a local path.
        variable: Variable name within ``group``.
        group: Slash-separated NetCDF group path.
        indexes: 1-based band indexes along the leading variable axis.
        decode_cf: Apply CF masking and scale / offset.
        use_cf_grid_mapping: Recover the CRS from ``grid_mapping``.
        storage_options: Forwarded to `geocloud.store.get_obstore`.

    Returns:
        The variable, its attributes under ``attrs["attrs"]``.

    Raises:
        ImportError: netCDF4 is missing (the ``[netcdf]`` extra).
        KeyError: No such group or variable.
        ValueError: ``indexes`` do not fit the variable.
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
            values = values.filled(fill_value)
        crs, mapping = _netcdf_crs(holder, var, use_cf_grid_mapping)
        return _geotensor(
            values,
            crs=crs,
            fill_value=fill_value,
            attrs={"attrs": attrs},
            transform=_netcdf_transform(var, mapping),
        )
