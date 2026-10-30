"""Layer-1 ``Operator`` wrappers around the :mod:`xrops.mask._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`pipekit.Operator`, so they compose with
:class:`~pipekit.Sequential`, the ``|`` pipe, and the functional
:class:`~pipekit.Graph` API.
"""

from __future__ import annotations

from typing import Any

from pipekit import Signature

from xrops._src import signature as _sig
from xrops._src.operator import Operator
from xrops.mask._src import (
    masks as _masks,
)


class AddLandMask(Operator):
    """Add a boolean land mask derived from ``regionmask``.

    Args:
        name: Name of the mask variable to add.

    Returns:
        The input dataset with a boolean land-mask variable.
    """

    def __init__(self, name: str = "land_mask"):
        self.name = name

    def _apply(self, ds):
        return _masks.add_land_mask(ds, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class AddOceanMask(Operator):
    """Add a boolean ocean/basin mask derived from ``regionmask``.

    Args:
        ocean: Ocean/basin selector (e.g. ``"global"``).
        name: Name of the mask variable to add.

    Returns:
        The input dataset with a boolean ocean-mask variable.
    """

    def __init__(self, ocean: str = "global", name: str = "ocean_mask"):
        self.ocean = ocean
        self.name = name

    def _apply(self, ds):
        return _masks.add_ocean_mask(ds, ocean=self.ocean, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"ocean": self.ocean, "name": self.name}


class AddCountryMask(Operator):
    """Add a boolean country mask derived from ``regionmask``.

    Args:
        country: Country name/selector to mask.
        name: Name of the mask variable to add.

    Returns:
        The input dataset with a boolean country-mask variable.
    """

    def __init__(self, country: str, name: str = "country_mask"):
        self.country = country
        self.name = name

    def _apply(self, ds):
        return _masks.add_country_mask(ds, country=self.country, name=self.name)

    def get_config(self) -> dict[str, Any]:
        return {"country": self.country, "name": self.name}


class ApplyMask(Operator):
    """Apply a boolean mask to a dataset.

    Args:
        mask: A boolean DataArray, or the name of a mask variable already
            in the dataset.
        drop: If ``True``, drop fully-masked slices
            (``where(..., drop=True)``); otherwise keep them as NaN.

    Returns:
        The masked dataset.
    """

    def __init__(self, mask, drop: bool = False):
        self.mask = mask
        self.drop = drop

    def _apply(self, ds):
        return _masks.apply_mask(ds, self.mask, drop=self.drop)

    def get_config(self) -> dict[str, Any]:
        mask_repr = self.mask if isinstance(self.mask, str) else "<DataArray>"
        return {"mask": mask_repr, "drop": self.drop}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        if not self.drop:
            return input_signature
        return _sig.make(
            {name: None for name in _sig.sizes(input_signature)},
            dtype=input_signature.dtype,
        )


__all__ = [
    "AddCountryMask",
    "AddLandMask",
    "AddOceanMask",
    "ApplyMask",
]
