"""GOES-R ABI L1b radiance reader.

One ABI L1b file (``OR_ABI-L1b-Rad{F,C,M1,M2}-M6Cnn_Gnn_s..._e..._c....nc``)
holds one channel of one scan on the geostationary fixed grid: the ``x`` /
``y`` coordinates are scan angles in radians, so multiplying by the
perspective-point height gives a clean ``+proj=geos`` affine grid. The
reader recovers that grid from the file's own projection variable, reads
pixel windows lazily through HDF5 chunked I/O, and calibrates with the
coefficients stored in the same file (``kappa0`` for reflectance, the
``planck_*`` set for brightness temperature).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Literal, get_args

import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor
from rasterio.crs import CRS
from rasterio.windows import Window

from geoproducts._src.base import ProductReader, Track
from geoproducts._src.extras import missing_extra
from geoproducts.goes import constants


__all__ = ["Calibration", "QualityReader", "Reader"]

Calibration = Literal["counts", "radiance", "reflectance", "brightness_temperature"]
Source = str | os.PathLike[str] | IO[bytes]

# Scalar calibration variables use -999 as their ``_FillValue``.
_SCALAR_FILL = -999.0


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise missing_extra("geoproducts.goes", "goes") from exc
    return h5py


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """An HDF5 attribute as a plain Python scalar / ``str``."""
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


def _scalar(f: Any, name: str) -> float:
    """A scalar calibration variable, ``NaN`` when absent or filled."""
    if name not in f:
        return float("nan")
    value = float(np.asarray(f[name][()]).reshape(-1)[0])
    return float("nan") if value == _SCALAR_FILL else value


def _decoded_coordinate(f: Any, name: str) -> np.ndarray:
    """A packed ``x`` / ``y`` scan-angle coordinate, in radians (float64)."""
    var = f[name]
    raw = np.asarray(var[()], dtype=np.float64)
    return raw * float(_attr(var, "scale_factor", 1.0)) + float(
        _attr(var, "add_offset", 0.0)
    )


def _edges(centres: np.ndarray, height_m: float) -> tuple[float, float]:
    """Outer pixel edges (metres) of an evenly spaced scan-angle axis."""
    half = (centres[-1] - centres[0]) / (len(centres) - 1) / 2.0
    return float((centres[0] - half) * height_m), float((centres[-1] + half) * height_m)


def _parse_time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


@dataclass(frozen=True)
class _Header:
    """Everything the reader needs from one ABI L1b file, read once."""

    height: int
    width: int
    transform: Affine
    crs: CRS
    band_id: int
    wavelength_um: float
    rad_scale: float
    rad_offset: float
    rad_fill: int
    rad_units: str
    kappa0: float
    planck_fk1: float
    planck_fk2: float
    planck_bc1: float
    planck_bc2: float
    platform: str
    scene: str
    dataset_name: str
    start_time: datetime | None
    end_time: datetime | None
    satellite_lon_deg: float
    satellite_height_m: float

    @classmethod
    def from_file(cls, f: Any) -> _Header:
        if "Rad" not in f or "goes_imager_projection" not in f:
            raise ValueError(
                "not a GOES-R ABI L1b radiance file: expected the 'Rad' and "
                "'goes_imager_projection' variables."
            )
        rad = f["Rad"]
        proj = f["goes_imager_projection"]
        height_m = float(_attr(proj, "perspective_point_height"))
        left, right = _edges(_decoded_coordinate(f, "x"), height_m)
        top, bottom = _edges(_decoded_coordinate(f, "y"), height_m)
        n_rows, n_cols = (int(n) for n in rad.shape)
        transform = Affine(
            (right - left) / n_cols, 0.0, left, 0.0, (bottom - top) / n_rows, top
        )
        crs = CRS.from_proj4(
            f"+proj=geos +lon_0={float(_attr(proj, 'longitude_of_projection_origin'))}"
            f" +h={height_m} +a={float(_attr(proj, 'semi_major_axis'))}"
            f" +b={float(_attr(proj, 'semi_minor_axis'))}"
            f" +sweep={_attr(proj, 'sweep_angle_axis', 'x')} +units=m +no_defs"
        )
        # ``Rad`` is stored as int16 flagged ``_Unsigned``: the fill is
        # read through the same unsigned view as the pixel values.
        rad_fill = int(np.asarray(rad.attrs["_FillValue"]).astype(np.uint16).item())
        sat_height = _scalar(f, "nominal_satellite_height")
        if "nominal_satellite_height" in f and _attr(
            f["nominal_satellite_height"], "units"
        ) in {"km", "kilometers"}:
            sat_height *= 1000.0
        return cls(
            height=n_rows,
            width=n_cols,
            transform=transform,
            crs=crs,
            band_id=int(np.asarray(f["band_id"][()]).reshape(-1)[0]),
            wavelength_um=round(
                float(np.asarray(f["band_wavelength"][()]).reshape(-1)[0]), 6
            ),
            rad_scale=float(_attr(rad, "scale_factor", 1.0)),
            rad_offset=float(_attr(rad, "add_offset", 0.0)),
            rad_fill=rad_fill,
            rad_units=str(_attr(rad, "units", "")),
            kappa0=_scalar(f, "kappa0"),
            planck_fk1=_scalar(f, "planck_fk1"),
            planck_fk2=_scalar(f, "planck_fk2"),
            planck_bc1=_scalar(f, "planck_bc1"),
            planck_bc2=_scalar(f, "planck_bc2"),
            platform=str(_attr(f, "platform_ID", "")),
            scene=str(_attr(f, "scene_id", "")),
            dataset_name=str(_attr(f, "dataset_name", "")),
            start_time=_parse_time(_attr(f, "time_coverage_start")),
            end_time=_parse_time(_attr(f, "time_coverage_end")),
            satellite_lon_deg=_scalar(f, "nominal_satellite_subpoint_lon"),
            satellite_height_m=sat_height,
        )


class _ABIFile(ProductReader):
    """Shared grid, metadata and windowed HDF5 I/O for one ABI L1b file."""

    _variable: str
    _source: Source
    _header: _Header

    def _init_source(self, source: Source) -> None:
        self._source = source
        with self._open() as f:
            self._header = _Header.from_file(f)

    @contextmanager
    def _open(self) -> Iterator[Any]:
        # Re-opened per read: no handle outlives a call, so path-backed
        # readers pickle (process pools) and never leak file descriptors.
        with _h5py().File(self._source, "r") as f:
            yield f

    # -- georeferencing ---------------------------------------------------

    @property
    def _crs(self) -> CRS:
        return self._header.crs

    @property
    def _transform(self) -> Affine:
        return self._header.transform

    @property
    def _shape(self) -> tuple[int, ...]:
        return (1, self._header.height, self._header.width)

    @property
    def _track(self) -> Track:
        return "A"

    # -- file metadata ----------------------------------------------------

    @property
    def path(self) -> Path | None:
        """The file path, or ``None`` for a file-like source."""
        if isinstance(self._source, str | os.PathLike):
            return Path(self._source)
        return None

    @property
    def channel(self) -> str:
        """ABI channel name (``"C01"`` … ``"C16"``)."""
        return f"C{self._header.band_id:02d}"

    @property
    def wavelength_um(self) -> float:
        """Band central wavelength (µm)."""
        return self._header.wavelength_um

    @property
    def platform(self) -> str:
        """Platform ID (``"G16"`` … ``"G19"``)."""
        return self._header.platform

    @property
    def scene(self) -> str:
        """Scan sector: ``"Full Disk"``, ``"CONUS"`` or ``"Mesoscale"``."""
        return self._header.scene

    @property
    def start_time(self) -> datetime | None:
        """Scan start time (UTC)."""
        return self._header.start_time

    @property
    def end_time(self) -> datetime | None:
        """Scan end time (UTC)."""
        return self._header.end_time

    @property
    def satellite_lon_deg(self) -> float:
        """Nominal sub-satellite longitude (degrees east)."""
        return self._header.satellite_lon_deg

    @property
    def satellite_height_m(self) -> float:
        """Nominal satellite height above the equator (metres)."""
        return self._header.satellite_height_m

    # -- windowed I/O -----------------------------------------------------

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _read_window(self, window: Window) -> np.ndarray:
        row0, col0 = int(window.row_off), int(window.col_off)
        n_rows, n_cols = int(window.height), int(window.width)
        out = np.full((1, n_rows, n_cols), self._fill_value, dtype=self._dtype)
        r_start, r_stop = max(row0, 0), min(row0 + n_rows, self._header.height)
        c_start, c_stop = max(col0, 0), min(col0 + n_cols, self._header.width)
        if r_start >= r_stop or c_start >= c_stop:
            return out
        with self._open() as f:
            raw = f[self._variable][r_start:r_stop, c_start:c_stop]
        out[0, r_start - row0 : r_stop - row0, c_start - col0 : c_stop - col0] = (
            self._decode(np.asarray(raw))
        )
        return out


class Reader(_ABIFile):
    """GOES-R ABI L1b radiance reader (one channel, one scan).

    The grid is the file's ``+proj=geos`` fixed grid; ``load`` /
    ``read_from_window`` / ``read_from_bounds`` return a ``(1, H, W)``
    ``GeoTensor`` whose band is named after the channel (``"C13"``).
    Off-Earth-disk and missing pixels hold the fill: ``NaN`` for every
    calibrated output, the file's own count fill for ``"counts"``.

    Calibration uses the coefficients carried by the file itself:

    - ``"counts"`` — the raw unsigned detector counts (``uint16``).
    - ``"radiance"`` — ``counts * scale_factor + add_offset`` in the
      file's units (``W m-2 sr-1 um-1`` for C01-C06,
      ``mW m-2 sr-1 (cm-1)-1`` for C07-C16).
    - ``"reflectance"`` — ``kappa0 * radiance``, the ABI reflectance
      factor (C01-C06 only; not divided by the cosine of the solar zenith).
    - ``"brightness_temperature"`` — ``(fk2 / ln(fk1 / L + 1) - bc1) / bc2``
      in Kelvin (C07-C16 only).

    Args:
        source: Path to an L1b ``.nc`` file, or a binary file-like object
            (e.g. ``fsspec.open("s3://noaa-goes19/...", anon=True).open()``).
            A file-like source is not safe to read from several threads.
        calibration: Output quantity; see above. Default ``"radiance"``.

    Raises:
        ImportError: ``h5py`` is not installed (the ``[goes]`` extra).
        ValueError: ``source`` is not an ABI L1b radiance file, or
            ``calibration`` is unknown or does not apply to the channel.

    Examples:
        Brightness temperature of one CONUS C13 scan::

            reader = goes.Reader(path, calibration="brightness_temperature")
            reader.channel, reader.shape  # ('C13', (1, 3000, 5000))
            bt = reader.read_from_window(Window(0, 0, 512, 512))  # (1, 512, 512) K
    """

    _variable = "Rad"

    def __init__(
        self, source: Source, *, calibration: Calibration = "radiance"
    ) -> None:
        if calibration not in get_args(Calibration):
            raise ValueError(
                f"calibration must be one of {get_args(Calibration)}; "
                f"got {calibration!r}."
            )
        self._init_source(source)
        self.calibration = calibration
        header = self._header
        if calibration == "reflectance" and not np.isfinite(header.kappa0):
            raise ValueError(
                f"reflectance applies to the reflective channels "
                f"{constants.REFLECTIVE_CHANNELS[0]}-{constants.REFLECTIVE_CHANNELS[-1]};"
                f" {self.channel} has no kappa0."
            )
        planck = (
            header.planck_fk1,
            header.planck_fk2,
            header.planck_bc1,
            header.planck_bc2,
        )
        if calibration == "brightness_temperature" and not np.all(np.isfinite(planck)):
            raise ValueError(
                f"brightness_temperature applies to the emissive channels "
                f"{constants.EMISSIVE_CHANNELS[0]}-{constants.EMISSIVE_CHANNELS[-1]};"
                f" {self.channel} has no Planck coefficients."
            )

    def __repr__(self) -> str:
        name = self._header.dataset_name or type(self._source).__name__
        return (
            f"goes.Reader({name!r}, channel={self.channel!r}, "
            f"calibration={self.calibration!r}, shape={self.shape})"
        )

    @property
    def quality(self) -> QualityReader:
        """The file's per-pixel data-quality flags (``DQF``) on the same grid."""
        return QualityReader._sharing(self)

    @property
    def units(self) -> str:
        """Units of the calibrated output."""
        return {
            "counts": "1",
            "radiance": self._header.rad_units,
            "reflectance": "1",
            "brightness_temperature": "K",
        }[self.calibration]

    @property
    def calibration_coefficients(self) -> dict[str, float]:
        """The file's scaling and calibration coefficients (``NaN`` = n/a)."""
        h = self._header
        return {
            "scale_factor": h.rad_scale,
            "add_offset": h.rad_offset,
            "kappa0": h.kappa0,
            "planck_fk1": h.planck_fk1,
            "planck_fk2": h.planck_fk2,
            "planck_bc1": h.planck_bc1,
            "planck_bc2": h.planck_bc2,
        }

    @property
    def _bands(self) -> tuple[str, ...]:
        return (self.channel,)

    @property
    def _dtype(self) -> Any:
        return np.dtype(np.uint16 if self.calibration == "counts" else np.float32)

    @property
    def _fill_value(self) -> float | int:
        return self._header.rad_fill if self.calibration == "counts" else np.nan

    def read_from_window(self, window: Window, boundless: bool = True) -> GeoTensor:
        """Read a pixel window as a ``(1, h, w)`` ``GeoTensor``.

        Besides ``band_names``, the output's ``attrs`` carry
        ``wavelengths`` (nm), ``units`` and ``calibration``.
        """
        out = super().read_from_window(window, boundless=boundless)
        out.attrs = {
            **(out.attrs or {}),
            "wavelengths": (self.wavelength_um * 1000.0,),
            "units": self.units,
            "calibration": self.calibration,
        }
        return out

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        counts = raw.astype(np.int16, copy=False).view(np.uint16)
        if self.calibration == "counts":
            return counts
        h = self._header
        missing = counts == h.rad_fill
        radiance = counts.astype(np.float32) * np.float32(h.rad_scale) + np.float32(
            h.rad_offset
        )
        radiance[missing] = np.nan
        if self.calibration == "radiance":
            return radiance
        if self.calibration == "reflectance":
            return radiance * np.float32(h.kappa0)
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = np.where(radiance > 0, radiance, np.nan)
            bt = (
                h.planck_fk2 / np.log(h.planck_fk1 / positive + 1.0) - h.planck_bc1
            ) / h.planck_bc2
        return bt.astype(np.float32)


class QualityReader(_ABIFile):
    """The ``DQF`` data-quality flags of an ABI L1b file, as ``uint8``.

    Values follow :data:`geoproducts.goes.constants.DQF_FLAGS` (``0`` good,
    ``1`` conditionally usable, ``2`` out of range, ``3`` no value, ``4``
    focal-plane temperature exceeded); ``255`` marks pixels with no flag
    and reads outside the file's extent. Usually reached through
    ``Reader.quality``.

    Args:
        source: Path to an L1b ``.nc`` file, or a binary file-like object.

    Raises:
        ImportError: ``h5py`` is not installed (the ``[goes]`` extra).
        ValueError: ``source`` is not an ABI L1b radiance file.
    """

    _variable = "DQF"

    def __init__(self, source: Source) -> None:
        self._init_source(source)

    @classmethod
    def _sharing(cls, reader: _ABIFile) -> QualityReader:
        """A quality reader on ``reader``'s source, reusing its parsed header."""
        out = cls.__new__(cls)
        out._source = reader._source
        out._header = reader._header
        return out

    def __repr__(self) -> str:
        name = self._header.dataset_name or type(self._source).__name__
        return f"goes.QualityReader({name!r}, channel={self.channel!r})"

    @property
    def _bands(self) -> tuple[str, ...]:
        return ("DQF",)

    @property
    def _dtype(self) -> Any:
        return np.dtype(np.uint8)

    @property
    def _fill_value(self) -> int:
        return constants.DQF_FILL

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.int8, copy=False).view(np.uint8)
