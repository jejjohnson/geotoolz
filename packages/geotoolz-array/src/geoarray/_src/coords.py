"""Standard coordinates: what a named dim of a carrier means.

A carrier stores one name per axis (``("time", "band", "y", "x")``) and, for
any non-spatial dim, a coordinate saying what its entries are. Each
coordinate is a frozen, hashable dataclass, so it can be static pytree data
(changing it recompiles; keeping it reuses the compiled program):

| Coordinate | Dims | Holds |
| --- | --- | --- |
| `SpectralAxis` | ``band`` | band names; centre wavelengths and FWHM (nm) |
| `VariableAxis` | ``variable`` | names and units of physical quantities |
| `VerticalAxis` | ``level`` | level values and their `Vertical` |
| `TimeAxis` | ``time`` | ``datetime64[ns]`` stamps |
| `PixelGrid` | ``y x`` | shape, CRS, pixel convention (the transform is a leaf) |
| `SwathGrid` | ``row col`` | shape, CRS (lat/lon are leaves) |
| `CurvilinearGrid` | ``row col`` | fixed lat/lon of a model grid, CRS |
| `PointAxis` | ``obs`` | N, CRS (xy, z, time are leaves) |

``band`` and ``variable`` are separate dims (decided 21 Oct 2026): spectral
channels share units and have wavelengths, so a normalized difference or a
matched filter makes sense over them; physical variables carry different
units, and an op mixing channels must refuse them.

The 1-D axes (`SpectralAxis`, `VariableAxis`, `VerticalAxis`, `TimeAxis`)
live in a carrier's ``coords``; they slice with their dim (`Coordinate.isel`)
and give `labels` for selection by exact label. The grid coordinates
describe the spatial dims for `GeoArray.from_coords` and the readable types;
the georeference itself (transform, lat/lon, xy) stays a leaf on the carrier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
import pyproj

from geoarray._src.grid import apply_transform, as_crs, shift_scale_transform


__all__ = [
    "Coordinate",
    "CurvilinearGrid",
    "PixelGrid",
    "PointAxis",
    "SpectralAxis",
    "SwathGrid",
    "TimeAxis",
    "VariableAxis",
    "Vertical",
    "VerticalAxis",
]


@runtime_checkable
class Coordinate(Protocol):
    """What a 1-D coordinate of a carrier dim provides.

    ``dim`` names the dim it describes, ``size`` its length; ``labels`` are
    the exact labels `sel` looks up (or ``None``); ``isel`` slices it with
    its dim (a slice, or a list / 1-D array of positions; an integer drops
    the dim, so the carrier drops the coordinate instead).
    """

    @property
    def dim(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def labels(self) -> tuple[Any, ...] | None: ...

    def isel(self, index: Any) -> Coordinate: ...


def _positions(index: Any, size: int) -> list[int]:
    """Concrete positions an index picks from ``range(size)``."""
    picked = np.arange(size)[index]
    return np.atleast_1d(picked).tolist()


def _take(values: tuple[Any, ...] | None, positions: list[int]) -> Any:
    return None if values is None else tuple(values[i] for i in positions)


def _floats(values: Any, name: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    out = tuple(float(v) for v in np.asarray(values, dtype=np.float64).ravel())
    return tuple(math.nan if math.isnan(v) else v for v in out)


def _require_size(name: str, values: tuple[Any, ...] | None, size: int) -> None:
    if values is not None and len(values) != size:
        raise ValueError(f"{name} has {len(values)} entries for {size} names")


@dataclasses.dataclass(frozen=True)
class SpectralAxis:
    """Spectral channels on the ``band`` dim: names, wavelengths, FWHM.

    Args:
        names: One name per band (``"B11"``), unique.
        wavelength_nm: Centre wavelength of each band, in nm.
        fwhm_nm: Full width at half maximum of each band, in nm.

    Raises:
        ValueError: Names repeat, or a wavelength / FWHM list has another
            length.

    Example:
        >>> swir = SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0))
        >>> swir.dim, swir.size, swir.isel([1]).names
        ('band', 2, ('B12',))
    """

    names: tuple[str, ...]
    wavelength_nm: tuple[float, ...] | None = None
    fwhm_nm: tuple[float, ...] | None = None

    dim: ClassVar[str] = "band"

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.names)
        if len(set(names)) != len(names):
            raise ValueError(f"SpectralAxis names repeat: {names}")
        object.__setattr__(self, "names", names)
        for field in ("wavelength_nm", "fwhm_nm"):
            values = _floats(getattr(self, field), field)
            _require_size(field, values, len(names))
            object.__setattr__(self, field, values)

    @property
    def size(self) -> int:
        """The number of bands."""
        return len(self.names)

    @property
    def labels(self) -> tuple[str, ...]:
        """The band names."""
        return self.names

    def isel(self, index: Any) -> SpectralAxis:
        """The bands at ``index`` (a slice, or a list / array of positions)."""
        p = _positions(index, self.size)
        return SpectralAxis(
            _take(self.names, p), _take(self.wavelength_nm, p), _take(self.fwhm_nm, p)
        )


@dataclasses.dataclass(frozen=True)
class VariableAxis:
    """Physical quantities on the ``variable`` dim: names and units.

    Args:
        names: One name per variable (``"t2m"``), unique.
        units: The unit of each variable (``"K"``, ``"hPa"``).

    Example:
        >>> met = VariableAxis(names=("t2m", "psfc"), units=("K", "hPa"))
        >>> met.isel(slice(1, None)).units
        ('hPa',)
    """

    names: tuple[str, ...]
    units: tuple[str, ...] | None = None

    dim: ClassVar[str] = "variable"

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.names)
        if len(set(names)) != len(names):
            raise ValueError(f"VariableAxis names repeat: {names}")
        units = None if self.units is None else tuple(str(u) for u in self.units)
        _require_size("units", units, len(names))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)

    @property
    def size(self) -> int:
        """The number of variables."""
        return len(self.names)

    @property
    def labels(self) -> tuple[str, ...]:
        """The variable names."""
        return self.names

    def isel(self, index: Any) -> VariableAxis:
        """The variables at ``index``."""
        p = _positions(index, self.size)
        return VariableAxis(_take(self.names, p), _take(self.units, p))


_VERTICAL_KINDS = ("height", "depth", "pressure", "level")


@dataclasses.dataclass(frozen=True)
class Vertical:
    """What a vertical coordinate measures, in CF terms.

    Args:
        kind: ``"height"``, ``"depth"``, ``"pressure"`` or ``"level"``.
        units: ``"m"``, ``"hPa"``, ``"1"``, …
        positive: CF ``positive``: ``"up"`` or ``"down"``.
        datum: The reference surface (``"EGM2008"``), or ``None``.

    Example:
        >>> Vertical(kind="depth", units="m", positive="down").positive
        'down'
    """

    kind: str
    units: str
    positive: str
    datum: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VERTICAL_KINDS:
            raise ValueError(
                f"Vertical kind is one of {_VERTICAL_KINDS}, got {self.kind!r}"
            )
        if self.positive not in ("up", "down"):
            raise ValueError(
                f"Vertical positive is 'up' or 'down', got {self.positive!r}"
            )


@dataclasses.dataclass(frozen=True)
class VerticalAxis:
    """Shared vertical levels on the ``level`` dim.

    A ``level`` dim is for samples on one vertical grid (TROPOMI's retrieval
    layers, EN4 depths); a point with its own vertical position has a ``z``
    leaf on `GeoPoints` instead.

    Args:
        values: One value per level, in ``vertical.units``.
        vertical: What the values measure.

    Example:
        >>> depth = Vertical(kind="depth", units="m", positive="down")
        >>> VerticalAxis(values=(5.0, 15.0, 25.0), vertical=depth).isel([0, 2]).values
        (5.0, 25.0)
    """

    values: tuple[float, ...]
    vertical: Vertical

    dim: ClassVar[str] = "level"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _floats(self.values, "values") or ())

    @property
    def size(self) -> int:
        """The number of levels."""
        return len(self.values)

    @property
    def labels(self) -> tuple[float, ...]:
        """The level values."""
        return self.values

    def isel(self, index: Any) -> VerticalAxis:
        """The levels at ``index``."""
        return VerticalAxis(
            _take(self.values, _positions(index, self.size)), self.vertical
        )


def _array_digest(*arrays: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _frozen(values: Any, dtype: Any) -> np.ndarray:
    out = np.array(values, dtype=dtype)
    out.setflags(write=False)
    return out


@dataclasses.dataclass(frozen=True, eq=False)
class TimeAxis:
    """Shared acquisition times on the ``time`` dim.

    Args:
        stamps: One time per entry: ``datetime64`` values, ISO strings, or
            anything ``np.asarray(..., dtype="datetime64[ns]")`` takes.

    Hashes and compares by value. A moving platform (along-track altimetry,
    Argo) has one time per sample instead: a ``time`` leaf on `GeoPoints`.

    Example:
        >>> dates = TimeAxis(stamps=["2026-06-01", "2026-06-11"])
        >>> dates.size, str(dates.labels[1])
        (2, '2026-06-11T00:00:00.000000000')
    """

    stamps: Any

    dim: ClassVar[str] = "time"

    def __post_init__(self) -> None:
        stamps = _frozen(self.stamps, "datetime64[ns]")
        if stamps.ndim != 1:
            raise ValueError(f"TimeAxis stamps are 1-D, got shape {stamps.shape}")
        object.__setattr__(self, "stamps", stamps)

    @property
    def size(self) -> int:
        """The number of times."""
        return int(self.stamps.shape[0])

    @property
    def labels(self) -> tuple[np.datetime64, ...]:
        """The stamps, as ``numpy.datetime64[ns]``."""
        return tuple(self.stamps)

    def isel(self, index: Any) -> TimeAxis:
        """The times at ``index``."""
        return TimeAxis(
            self.stamps[np.asarray(_positions(index, self.size), dtype=int)]
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TimeAxis) and bool(
            np.array_equal(self.stamps, other.stamps)
        )

    def __hash__(self) -> int:
        return hash(("TimeAxis", _array_digest(self.stamps)))

    def __repr__(self) -> str:
        if self.size == 0:
            return "TimeAxis(size=0)"
        return f"TimeAxis(size={self.size}, {self.stamps[0]} … {self.stamps[-1]})"


def _shape2(shape: Sequence[int], name: str) -> tuple[int, int]:
    out = tuple(int(n) for n in shape)
    if len(out) != 2 or min(out) < 0:
        raise ValueError(f"{name} shape is two non-negative sizes, got {shape}")
    return out[0], out[1]


@dataclasses.dataclass(frozen=True)
class PixelGrid:
    """An affine pixel grid on the ``y x`` dims: shape, CRS, pixel convention.

    The transform itself is a leaf on the carrier, so a new scene of the same
    shape and CRS reuses the compiled program.

    Args:
        shape: ``(H, W)``.
        crs: Anything ``pyproj.CRS.from_user_input`` takes, or ``None``.
        convention: ``"area"`` when the transform maps pixel edges (GDAL,
            rasterio), ``"point"`` when it maps pixel centres. Carriers store
            ``"area"``; `GeoArray.from_coords` shifts a ``"point"`` transform
            by half a pixel.

    Example:
        >>> grid = PixelGrid(shape=(600, 600), crs="EPSG:32613")
        >>> grid.dims, grid.crs.to_epsg()
        (('y', 'x'), 32613)
    """

    shape: tuple[int, int]
    crs: pyproj.CRS | None
    convention: str = "area"

    dims: ClassVar[tuple[str, str]] = ("y", "x")

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape2(self.shape, "PixelGrid"))
        object.__setattr__(self, "crs", as_crs(self.crs))
        if self.convention not in ("area", "point"):
            raise ValueError(
                f"PixelGrid convention is 'area' or 'point', got {self.convention!r}"
            )

    def area_transform(self, transform: Any) -> Any:
        """``transform`` in the area convention carriers store.

        A point-convention transform maps pixel (j, i) to its centre, so the
        edge transform is T shifted by half a pixel: c′ = c − (a + b)/2,
        f′ = f − (d + e)/2. An area transform is returned as it is.
        """
        if self.convention == "area":
            return transform
        return shift_scale_transform(transform, -0.5, -0.5)

    def centres(self, transform: Any) -> tuple[Any, Any]:
        """Map coordinates ``(x, y)`` of every pixel centre, each ``(H, W)``.

        x_ij = a·(j + ½) + b·(i + ½) + c, y_ij = d·(j + ½) + e·(i + ½) + f for
        an area transform; without the ½ for a point transform.
        """
        offset = 0.5 if self.convention == "area" else 0.0
        h, w = self.shape
        jj, ii = np.meshgrid(np.arange(w) + offset, np.arange(h) + offset)
        return apply_transform(transform, jj, ii)


@dataclasses.dataclass(frozen=True)
class SwathGrid:
    """A swath in sensor geometry on the ``row col`` dims: shape and CRS.

    Rows run along track, columns across; per-pixel lat/lon are leaves on
    the carrier.

    Example:
        >>> SwathGrid(shape=(1280, 1242), crs="EPSG:4326").dims
        ('row', 'col')
    """

    shape: tuple[int, int]
    crs: pyproj.CRS | None

    dims: ClassVar[tuple[str, str]] = ("row", "col")

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape2(self.shape, "SwathGrid"))
        object.__setattr__(self, "crs", as_crs(self.crs))


@dataclasses.dataclass(frozen=True, eq=False)
class CurvilinearGrid:
    """A model grid with fixed 2-D lat/lon (ORCA) on the ``row col`` dims.

    Unlike a swath, the lat/lon never change between calls, so they are
    static; hashes and compares by value.

    Args:
        lat: ``(row, col)`` latitudes.
        lon: ``(row, col)`` longitudes.
        crs: The CRS of lat/lon (``"EPSG:4326"``).

    Example:
        >>> import numpy as np
        >>> lat, lon = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
        >>> CurvilinearGrid(lat, lon, "EPSG:4326").shape
        (3, 4)
    """

    lat: Any
    lon: Any
    crs: pyproj.CRS | None

    dims: ClassVar[tuple[str, str]] = ("row", "col")

    def __post_init__(self) -> None:
        lat = _frozen(self.lat, np.float64)
        lon = _frozen(self.lon, np.float64)
        if lat.ndim != 2 or lat.shape != lon.shape:
            raise ValueError(
                f"CurvilinearGrid lat/lon are 2-D of one shape, got {lat.shape} "
                f"and {lon.shape}"
            )
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "crs", as_crs(self.crs))

    @property
    def shape(self) -> tuple[int, int]:
        """``(row, col)``."""
        return _shape2(self.lat.shape, "CurvilinearGrid")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CurvilinearGrid)
            and self.crs == other.crs
            and bool(np.array_equal(self.lat, other.lat, equal_nan=True))
            and bool(np.array_equal(self.lon, other.lon, equal_nan=True))
        )

    def __hash__(self) -> int:
        return hash(("CurvilinearGrid", self.crs, _array_digest(self.lat, self.lon)))


@dataclasses.dataclass(frozen=True)
class PointAxis:
    """Scattered samples on the ``obs`` dim: N and the CRS of their xy.

    Example:
        >>> PointAxis(size=40, crs="EPSG:4326").dims
        ('obs',)
    """

    size: int
    crs: pyproj.CRS | None

    dims: ClassVar[tuple[str]] = ("obs",)

    def __post_init__(self) -> None:
        if int(self.size) < 0:
            raise ValueError(f"PointAxis size is non-negative, got {self.size}")
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "crs", as_crs(self.crs))


def _joined(parts: Sequence[Any], field: str) -> Any:
    values = [getattr(p, field) for p in parts]
    if any(v is None for v in values):
        return None
    return tuple(item for v in values for item in v)


def concat_coords(parts: Sequence[Coordinate | None]) -> Coordinate | None:
    """One coordinate for carriers joined along its dim, or ``None``.

    Every part must have a coordinate of one type; fields some part lacks
    (a wavelength list) are dropped. ``None`` when a part has none.

    Raises:
        ValueError: The parts are of different types, `VerticalAxis` parts
            measure different things, or the joined names repeat.
    """
    if not parts or any(p is None for p in parts):
        return None
    kinds = {type(p) for p in parts}
    if len(kinds) > 1:
        names = sorted(k.__name__ for k in kinds)
        raise ValueError(f"cannot join coordinates of different types: {names}")
    first = parts[0]
    if isinstance(first, SpectralAxis):
        return SpectralAxis(
            _joined(parts, "names"),
            _joined(parts, "wavelength_nm"),
            _joined(parts, "fwhm_nm"),
        )
    if isinstance(first, VariableAxis):
        return VariableAxis(_joined(parts, "names"), _joined(parts, "units"))
    if isinstance(first, VerticalAxis):
        verticals = {cast(VerticalAxis, p).vertical for p in parts}
        if len(verticals) > 1:
            raise ValueError(f"cannot join levels of different verticals: {verticals}")
        return VerticalAxis(_joined(parts, "values"), first.vertical)
    if isinstance(first, TimeAxis):
        return TimeAxis(np.concatenate([cast(TimeAxis, p).stamps for p in parts]))
    return None
