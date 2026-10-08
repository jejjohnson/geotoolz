"""GOES-R ABI L1b radiance reader.

One L1b file (``OR_ABI-L1b-Rad{F,C,M1,M2}-M6Cnn_Gnn_s..._e..._c....nc``)
holds one channel of one scan on the geostationary fixed grid. The reader
calibrates with the coefficients stored in that same file: ``kappa0`` for
reflectance and the ``planck_*`` set for brightness temperature.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import numpy as np

from geoproducts.goes import constants
from geoproducts.goes._src.base import ABIFile, Source
from geoproducts.goes._src.metadata import abi_scalar


__all__ = ["Calibration", "QualityReader", "Reader"]

Calibration = Literal["counts", "radiance", "reflectance", "brightness_temperature"]


class Reader(ABIFile):
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
            reader.channel, reader.shape  # ('C13', (1, 1500, 2500))
            bt = reader.read_from_window(Window(0, 0, 512, 512))  # (1, 512, 512) K
    """

    def __init__(
        self, source: Source, *, calibration: Calibration = "radiance"
    ) -> None:
        if calibration not in get_args(Calibration):
            raise ValueError(
                f"calibration must be one of {get_args(Calibration)}; "
                f"got {calibration!r}."
            )
        try:
            self._init_file(source, ["Rad"])
        except ValueError as exc:
            raise ValueError(
                f"not a GOES-R ABI L1b radiance file ({exc}); L2 products "
                "are read with goes.L2Reader."
            ) from exc
        self.calibration = calibration
        with self._open() as f:
            self._band_id = int(np.asarray(f["band_id"][()]).reshape(-1)[0])
            self._wavelength_um = round(
                float(np.asarray(f["band_wavelength"][()]).reshape(-1)[0]), 6
            )
            self._coefficients = {
                name: abi_scalar(f, name)
                for name in (
                    "kappa0",
                    "planck_fk1",
                    "planck_fk2",
                    "planck_bc1",
                    "planck_bc2",
                )
            }
        c = self._coefficients
        if calibration == "reflectance" and not np.isfinite(c["kappa0"]):
            raise ValueError(
                f"reflectance applies to the reflective channels "
                f"{constants.REFLECTIVE_CHANNELS[0]}-{constants.REFLECTIVE_CHANNELS[-1]};"
                f" {self.channel} has no kappa0."
            )
        planck = [c["planck_fk1"], c["planck_fk2"], c["planck_bc1"], c["planck_bc2"]]
        if calibration == "brightness_temperature" and not np.all(np.isfinite(planck)):
            raise ValueError(
                f"brightness_temperature applies to the emissive channels "
                f"{constants.EMISSIVE_CHANNELS[0]}-{constants.EMISSIVE_CHANNELS[-1]};"
                f" {self.channel} has no Planck coefficients."
            )

    def __repr__(self) -> str:
        return (
            f"goes.Reader({self._describe()!r}, channel={self.channel!r}, "
            f"calibration={self.calibration!r}, shape={self.shape})"
        )

    @property
    def channel(self) -> str:
        """ABI channel name (``"C01"`` … ``"C16"``)."""
        return f"C{self._band_id:02d}"

    @property
    def wavelength_um(self) -> float:
        """Band central wavelength (µm)."""
        return self._wavelength_um

    @property
    def quality(self) -> QualityReader:
        """The file's per-pixel data-quality flags (``DQF``) on the same grid."""
        return QualityReader._sharing(self)

    @property
    def units(self) -> str:
        """Units of the calibrated output."""
        return {
            "counts": "1",
            "radiance": self._vars[0].units,
            "reflectance": "1",
            "brightness_temperature": "K",
        }[self.calibration]

    @property
    def calibration_coefficients(self) -> dict[str, float]:
        """The file's scaling and calibration coefficients (``NaN`` = n/a)."""
        rad = self._vars[0]
        return {
            "scale_factor": rad.scale if rad.scale is not None else float("nan"),
            "add_offset": rad.offset if rad.offset is not None else float("nan"),
            **self._coefficients,
        }

    @property
    def _bands(self) -> tuple[str, ...]:
        return (self.channel,)

    @property
    def _dtype(self) -> Any:
        return np.dtype(np.uint16 if self.calibration == "counts" else np.float32)

    @property
    def _fill_value(self) -> float | int:
        return self._vars[0].fill if self.calibration == "counts" else np.nan

    def _band_attrs(self) -> dict[str, Any]:
        return {
            "units": (self.units,),
            "wavelengths": (self._wavelength_um * 1000.0,),
            "calibration": self.calibration,
        }

    def _decode(self, raw: list[np.ndarray]) -> np.ndarray:
        rad = self._vars[0]
        if self.calibration == "counts":
            return rad.raw(raw[0])[None]
        radiance = rad.decode(raw[0])
        c = self._coefficients
        if self.calibration == "radiance":
            return radiance[None]
        if self.calibration == "reflectance":
            return (radiance * np.float32(c["kappa0"]))[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = np.where(radiance > 0, radiance, np.nan)
            bt = (
                c["planck_fk2"] / np.log(c["planck_fk1"] / positive + 1.0)
                - c["planck_bc1"]
            ) / c["planck_bc2"]
        return bt.astype(np.float32)[None]


class QualityReader(ABIFile):
    """The ``DQF`` data-quality flags of an ABI file, as unsigned integers.

    For L1b files the values follow :data:`geoproducts.goes.constants.DQF_FLAGS`
    (``0`` good, ``1`` conditionally usable, ``2`` out of range, ``3`` no
    value, ``4`` focal-plane temperature exceeded); ``255`` marks pixels with
    no flag and reads outside the file's extent. L2 products define their
    own meanings: read them with :meth:`flags`. Usually reached through
    ``Reader.quality`` / ``L2Reader.quality``.

    Args:
        source: Path to an ABI ``.nc`` file, or a binary file-like object.
        variables: Quality variables to read, one band each. Default
            ``("DQF",)``.

    Raises:
        ImportError: ``h5py`` is not installed (the ``[goes]`` extra).
        ValueError: ``source`` is not an ABI file or lacks a variable.
    """

    def __init__(
        self, source: Source, *, variables: tuple[str, ...] = ("DQF",)
    ) -> None:
        self._init_file(source, variables)

    @classmethod
    def _sharing(
        cls, reader: ABIFile, variables: tuple[str, ...] = ("DQF",)
    ) -> QualityReader:
        """A quality reader on ``reader``'s file, reusing its parsed grid."""
        out = cls.__new__(cls)
        out._share(reader, variables)
        return out

    def __repr__(self) -> str:
        return f"goes.QualityReader({self._describe()!r}, variables={self.variables})"

    @property
    def _bands(self) -> tuple[str, ...]:
        return self.variables
