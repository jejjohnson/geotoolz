"""Himawari AHI L1b reader: one band of one scan from its HSD segments.

An AHI observation area (the full disk, a Japan area, a target area) is
split into horizontal segments, one HSD file each (10 for the full disk).
:class:`Reader` places the segments it is given on the area's
geostationary grid — missing segments simply read as fill — and decodes
only the segments a window touches.

The grid is the CGMS normalised geostationary projection of the header's
``CFAC`` / ``LFAC`` / ``COFF`` / ``LOFF`` (evenly spaced ``+proj=geos``
metres, ``sweep=y``; see
:meth:`geoproducts._src.geostationary.FixedGrid.from_cgms`).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from geoproducts._src.base import ProductReader, Track
from geoproducts._src.geostationary import FixedGrid
from geoproducts.himawari._src.hsd import HSDHeader, read_header, read_lines


__all__ = ["Calibration", "Reader"]

Calibration = Literal["counts", "radiance", "reflectance", "brightness_temperature"]
Source = str | os.PathLike[str]

# Header fields every segment of one band / one scan shares.
_SHARED = (
    "satellite",
    "area",
    "band",
    "columns",
    "lines",
    "total_segments",
    "cfac",
    "lfac",
    "coff",
    "loff",
    "sub_lon",
)


class Reader(ProductReader):
    """Himawari-8 / -9 AHI L1b reader (one band, one scan, any segments).

    ``load`` / ``read_from_window`` / ``read_from_bounds`` return a
    ``(1, H, W)`` ``GeoTensor`` on the observation area's grid, the band
    named ``"B01"`` … ``"B16"``. Pixels in segments that were not given,
    outside the scan, flagged as errors or — for calibrated outputs — off
    the Earth's disk hold the fill: ``NaN`` for every calibrated output,
    ``65535`` for ``"counts"`` (which are otherwise the stored values).

    Calibration uses the coefficients in each file (block 5 of the header):

    - ``"counts"`` — the stored counts (``uint16``).
    - ``"radiance"`` — ``counts · gain + offset`` (W m-2 sr-1 µm-1). Bands
      1-6 use the updated visible calibration when the file carries one.
    - ``"reflectance"`` — ``radiance · albedo coefficient``, the albedo
      (reflectance factor; bands 1-6 only).
    - ``"brightness_temperature"`` — the effective temperature of the
      radiance at the band's central wavelength (Planck), corrected by the
      file's ``c0 + c1 T + c2 T²`` (bands 7-16 only).

    ``.bz2`` segments are decompressed on read (the most recent one is
    kept in memory); for many small windows over a large file,
    decompress once with ``himawari.aws.download(..., decompress=True)``
    and read the ``.DAT`` files, which are memory-mapped.

    Args:
        segments: HSD segment files (``.DAT`` or ``.DAT.bz2``) of one band
            of one scan, in any order — all of them, or only those covering
            the area of interest.
        calibration: Output quantity; see above. Default ``"radiance"``.

    Raises:
        ValueError: No segments, segments of different bands / scans /
            areas, a duplicated segment, or a ``calibration`` that does
            not apply to the band.

    Examples:
        The full disk of band 13 (10.4 µm), brightness temperature::

            paths = sorted(Path("ahi").glob("HS_H09_*_B13_FLDK_R20_S*.DAT.bz2"))
            reader = himawari.Reader(paths, calibration="brightness_temperature")
            reader.shape  # (1, 5500, 5500) · 2 km
    """

    def __init__(
        self,
        segments: Source | Sequence[Source],
        *,
        calibration: Calibration = "radiance",
    ) -> None:
        if calibration not in get_args(Calibration):
            raise ValueError(
                f"calibration must be one of {get_args(Calibration)}; "
                f"got {calibration!r}."
            )
        paths = [segments] if isinstance(segments, str | os.PathLike) else segments
        if not paths:
            raise ValueError("Reader needs at least one HSD segment file.")
        headers = [read_header(p) for p in paths]
        first = headers[0]
        for path, header in zip(paths, headers, strict=True):
            for name in _SHARED:
                if getattr(header, name) != getattr(first, name):
                    raise ValueError(
                        f"{Path(path).name} differs from the first segment in "
                        f"{name!r}: segments must be one band of one scan."
                    )
        numbers = [h.segment for h in headers]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"duplicated segments: {sorted(numbers)}.")
        if calibration == "reflectance" and first.is_emissive:
            raise ValueError(
                f"reflectance applies to bands 1-6; {self._name(first)} is emissive."
            )
        if calibration == "brightness_temperature" and not first.is_emissive:
            raise ValueError(
                f"brightness_temperature applies to bands 7-16; "
                f"{self._name(first)} is reflective."
            )
        order = np.argsort(numbers)
        self._paths: tuple[Path, ...] = tuple(Path(paths[i]) for i in order)
        self._headers: tuple[HSDHeader, ...] = tuple(headers[i] for i in order)
        self.calibration = calibration
        h = self._headers[0]
        self._grid = FixedGrid.from_cgms(
            columns=h.columns,
            lines=h.area_lines,
            cfac=h.cfac,
            lfac=h.lfac,
            coff=h.coff,
            loff=h.loff,
            lon_0=h.sub_lon,
            height_m=(h.distance_km - h.equatorial_radius_km) * 1000.0,
            semi_major_m=h.equatorial_radius_km * 1000.0,
            semi_minor_m=h.polar_radius_km * 1000.0,
        )
        self._cache: tuple[int, np.ndarray] | None = None

    @staticmethod
    def _name(header: HSDHeader) -> str:
        return f"B{header.band:02d}"

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = None  # decompressed segments stay with the process
        return state

    def __repr__(self) -> str:
        return (
            f"himawari.Reader(band={self.band!r}, area={self.area!r}, "
            f"segments={self.segments}, calibration={self.calibration!r}, "
            f"shape={self.shape})"
        )

    # -- metadata ---------------------------------------------------------

    @property
    def header(self) -> HSDHeader:
        """The decoded header of the first segment."""
        return self._headers[0]

    @property
    def band(self) -> str:
        """AHI band name (``"B01"`` … ``"B16"``)."""
        return self._name(self.header)

    @property
    def wavelength_um(self) -> float:
        """Band central wavelength (µm)."""
        return self.header.wavelength_um

    @property
    def satellite(self) -> str:
        """``"Himawari-9"`` / ``"Himawari-8"``."""
        return self.header.satellite

    @property
    def area(self) -> str:
        """Observation area (``"FLDK"``, ``"JP01"``, ``"R301"``, …)."""
        return self.header.area

    @property
    def segments(self) -> tuple[int, ...]:
        """The segment numbers given (1-based)."""
        return tuple(h.segment for h in self._headers)

    @property
    def paths(self) -> tuple[Path, ...]:
        """The segment files, in segment order."""
        return self._paths

    @property
    def start_time(self) -> datetime:
        """Observation start (UTC) of the earliest segment."""
        return min(h.start_time for h in self._headers)

    @property
    def end_time(self) -> datetime:
        """Observation end (UTC) of the latest segment."""
        return max(h.end_time for h in self._headers)

    @property
    def satellite_lon_deg(self) -> float:
        """Sub-satellite longitude of the projection (degrees east)."""
        return self.header.sub_lon

    @property
    def satellite_height_m(self) -> float:
        """Satellite height above the equator (metres)."""
        return self._grid.height_m

    @property
    def units(self) -> str:
        """Units of the calibrated output."""
        return {
            "counts": "1",
            "radiance": "W m-2 sr-1 um-1",
            "reflectance": "1",
            "brightness_temperature": "K",
        }[self.calibration]

    # -- georeferencing ---------------------------------------------------

    @property
    def _crs(self) -> CRS:
        return self._grid.crs

    @property
    def _transform(self) -> Affine:
        return self._grid.transform

    @property
    def _shape(self) -> tuple[int, ...]:
        return (1, self._grid.height, self._grid.width)

    @property
    def _dtype(self) -> Any:
        return np.dtype(np.uint16 if self.calibration == "counts" else np.float32)

    @property
    def _fill_value(self) -> float | int:
        return self.header.error_count if self.calibration == "counts" else np.nan

    @property
    def _bands(self) -> tuple[str, ...]:
        return (self.band,)

    @property
    def _track(self) -> Track:
        return "A"

    def _band_attrs(self) -> dict[str, Any]:
        return {
            "units": (self.units,),
            "wavelengths": (round(self.wavelength_um * 1000.0, 6),),
            "calibration": self.calibration,
        }

    # -- windowed I/O -----------------------------------------------------

    def _segment_counts(self, index: int, rows: slice, cols: slice) -> np.ndarray:
        path, header = self._paths[index], self._headers[index]
        if path.suffix != ".bz2":
            return read_lines(path, header, rows, cols)
        if self._cache is None or self._cache[0] != index:
            whole = read_lines(path, header, slice(0, header.lines), slice(None))
            self._cache = (index, whole)
        return self._cache[1][rows, cols]

    def _read_window(self, window: Window) -> np.ndarray:
        def read(rows: slice, cols: slice) -> np.ndarray:
            counts = np.full(
                (rows.stop - rows.start, cols.stop - cols.start),
                self.header.error_count,
                dtype=np.uint16,
            )
            for i, h in enumerate(self._headers):
                seg_start = h.first_line - 1
                lo = max(rows.start, seg_start)
                hi = min(rows.stop, seg_start + h.lines)
                if lo >= hi:
                    continue
                counts[lo - rows.start : hi - rows.start] = self._segment_counts(
                    i, slice(lo - seg_start, hi - seg_start), cols
                )
            return self._calibrate(counts, rows, cols)[None]

        return self._read_boundless(window, read)

    def _calibrate(self, counts: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        h = self.header
        if self.calibration == "counts":
            return counts
        invalid = (counts == h.error_count) | (counts == h.outside_count)
        invalid |= ~self._grid.on_earth(rows, cols)
        gain, offset = h.gain, h.offset
        if not h.is_emissive and np.isfinite(h.updated_gain):
            gain, offset = h.updated_gain, h.updated_offset
        radiance = counts.astype(np.float64) * gain + offset
        radiance[invalid] = np.nan
        if self.calibration == "radiance":
            return radiance.astype(np.float32)
        if self.calibration == "reflectance":
            return (radiance * h.albedo_coefficient).astype(np.float32)
        return _brightness_temperature(radiance, h).astype(np.float32)


def _brightness_temperature(radiance: np.ndarray, h: HSDHeader) -> np.ndarray:
    """Radiance (W m-2 sr-1 µm-1) → brightness temperature (K)."""
    c, planck, boltzmann = h.planck_constants
    wavelength_m = h.wavelength_um * 1e-6
    spectral = radiance * 1e6  # per µm → per m
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = np.where(spectral > 0, spectral, np.nan)
        effective = (planck * c / (boltzmann * wavelength_m)) / np.log(
            2.0 * planck * c**2 / (wavelength_m**5 * positive) + 1.0
        )
    c0, c1, c2 = h.tb_coefficients
    return c0 + c1 * effective + c2 * effective**2
