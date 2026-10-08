"""GOES-R ABI L2 product reader.

Every ABI L2 product (``OR_ABI-L2-<code><sector>-M6..._G.._s..._e..._c....nc``)
puts its retrievals on the same fixed grid as the L1b radiances, packed
with the same CF conventions, so one reader covers them all: the clear sky
mask (ACM), cloud-top height (ACHA), land surface temperature (LST), fire
detection (FDC), stability indices (DSI), total precipitable water (TPW),
cloud and moisture imagery (CMIP, and MCMIP with all 16 channels on one
2 km grid), and the rest.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from geoproducts.goes import constants
from geoproducts.goes._src.base import ABIFile, Source, grid_variables
from geoproducts.goes.l1b import QualityReader


__all__ = ["L2Reader", "product_code"]

_PRODUCT_RE = re.compile(r"OR_ABI-L2-(?P<code>[A-Za-z0-9]+?)(?:[FC]|M[12])-M\d")
_CMI_RE = re.compile(r"CMI_(?P<channel>C\d{2})")


def product_code(dataset_name: str) -> str:
    """The L2 product code of a file name: ``"ACM"``, ``"MCMIP"``, ``"ACHA2KM"``.

    Returns ``""`` when the name is not an ABI L2 file name.
    """
    match = _PRODUCT_RE.search(dataset_name)
    return match["code"] if match else ""


class L2Reader(ABIFile):
    """GOES-R ABI L2 product reader: chosen variables as bands.

    Each band is one 2-D variable of the file. Unpacked integer variables
    (masks, class codes such as the clear sky mask's ``BCM`` / ``ACM`` or
    the fire ``Mask``) are kept as integers with the file's fill; anything
    packed or floating (cloud-top height, LST, CMI reflectance and brightness
    temperature, fire power) is decoded to ``float32`` with ``NaN`` fill.

    Band names are the variable names, except that cloud and moisture
    imagery is named by channel (``CMI_C13`` / CMIP's ``CMI`` → ``"C13"``),
    so an MCMIP file feeds the channel-named presets and RGB recipes
    directly.

    Args:
        source: Path to an L2 ``.nc`` file, or a binary file-like object.
        variables: Variable name(s) to read. Default: the product's primary
            variables — ``BCM`` + ``ACM`` for ACM, ``Mask`` for FDC, ``LST``
            for LST, otherwise every 2-D variable that is not a quality
            flag (all 16 ``CMI_Cnn`` for MCMIP, the five indices for DSI,
            ``HT`` for ACHA, …). ``available_variables`` lists them all.

    Raises:
        ImportError: ``h5py`` is not installed (the ``[goes]`` extra).
        ValueError: ``source`` is not an ABI file, or a variable is missing
            or not on the fixed grid.

    Examples:
        Cloud-top height and the clear sky mask of one mesoscale scan::

            height = goes.L2Reader(acha_path)            # bands ('HT',), metres
            mask = goes.L2Reader(acm_path)               # ('BCM', 'ACM') uint8
            mask.flags("ACM")  # {0: 'clear', 1: 'probably_clear', ...}

        All 16 channels, already calibrated, on one 2 km grid::

            cmi = goes.L2Reader(mcmip_path).load()       # (16, 500, 500) float32
            cmi.attrs["band_names"][:3]                  # ('C01', 'C02', 'C03')
    """

    def __init__(
        self, source: Source, *, variables: str | Sequence[str] | None = None
    ) -> None:
        self._init_file(source, [])
        self._product = product_code(self._info.dataset_name)
        with self._open() as f:
            names = self._resolve(f, variables)
            self._vars = tuple(self._variable(f, name) for name in names)
            self._channel_of_cmi = (
                f"C{int(np.asarray(f['band_id'][()]).reshape(-1)[0]):02d}"
                if "CMI" in names and "band_id" in f
                else None
            )
            self._available = grid_variables(f)
            self._wavelengths = self._cmi_wavelengths(f)

    def _resolve(self, f: Any, variables: str | Sequence[str] | None) -> list[str]:
        if isinstance(variables, str):
            return [variables]
        if variables is not None:
            names = list(variables)
            if not names:
                raise ValueError("variables must name at least one variable.")
            return names
        default = constants.L2_DEFAULT_VARIABLES.get(self._product)
        if default is not None:
            return list(default)
        names = [n for n in grid_variables(f) if not n.startswith("DQF")]
        if not names:
            raise ValueError(
                f"{self._repr_name()} has no 2-D data variables on its grid."
            )
        return names

    def _cmi_wavelengths(self, f: Any) -> tuple[float, ...] | None:
        """Band-centre wavelengths (nm) when every band is CMI imagery."""
        out = []
        for name in self._bands:
            key = f"band_wavelength_{name}" if f"band_wavelength_{name}" in f else None
            if key is None and name == self._channel_of_cmi:
                key = "band_wavelength"
            if key is None or key not in f:
                return None
            um = float(np.asarray(f[key][()]).reshape(-1)[0])
            out.append(round(um, 6) * 1000.0)
        return tuple(out)

    def __repr__(self) -> str:
        return (
            f"goes.L2Reader({self._repr_name()!r}, product={self.product!r}, "
            f"bands={self.bands}, shape={self.shape})"
        )

    @property
    def product(self) -> str:
        """The L2 product code (``"ACM"``, ``"ACHA"``, ``"MCMIP"``, …)."""
        return self._product

    @property
    def available_variables(self) -> tuple[str, ...]:
        """Every 2-D variable of the file on the fixed grid (data and flags)."""
        return self._available

    @property
    def quality(self) -> QualityReader:
        """The quality flags matching the bands, on the same grid.

        ``CMI_Cnn`` bands map to ``DQF_Cnn``; everything else to the
        product's ``DQF`` (``DQF_Overall`` for the sounding products).
        """
        names: list[str] = []
        for var in self._vars:
            match = _CMI_RE.fullmatch(var.name)
            if match and f"DQF_{match['channel']}" in self._available:
                name = f"DQF_{match['channel']}"
            elif "DQF" in self._available:
                name = "DQF"
            elif "DQF_Overall" in self._available:
                name = "DQF_Overall"
            else:
                raise ValueError(f"{self._repr_name()} has no quality flags.")
            if name not in names:
                names.append(name)
        return QualityReader._sharing(self, tuple(names))

    @property
    def _bands(self) -> tuple[str, ...]:
        out = []
        for var in self._vars:
            match = _CMI_RE.fullmatch(var.name)
            if match:
                out.append(match["channel"])
            elif var.name == "CMI" and self._channel_of_cmi:
                out.append(self._channel_of_cmi)
            else:
                out.append(var.name)
        return tuple(out)

    def _band_attrs(self) -> dict[str, Any]:
        attrs = {**super()._band_attrs(), "product": self._product}
        if self._wavelengths is not None:
            attrs["wavelengths"] = self._wavelengths
        return attrs
