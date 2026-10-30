"""Layer-1 ``Operator`` wrappers around the :mod:`xrops.cf._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`pipekit.Operator`, so they compose with
:class:`~pipekit.Sequential`, the ``|`` pipe, and the functional
:class:`~pipekit.Graph` API.
"""

from __future__ import annotations

from typing import Any, Literal

from pipekit import Signature

from xrops._src import signature as _sig
from xrops._src.operator import Operator
from xrops.cf._src import (
    validation as _validation,
)


def _validate_signature(
    input_signature: Signature, *, aliases: tuple[str, ...], canonical: str
) -> Signature:
    """Mirror the runtime rename done by ``validate_longitude`` /
    ``validate_latitude``: if any alias is present in the input dims,
    rename it to ``canonical``. Shape-preserving otherwise."""
    for alias in aliases:
        if alias in _sig.sizes(input_signature):
            return _sig.rename_dims(input_signature, {alias: canonical})
    return input_signature


class ValidateLongitude(Operator):
    """Validate and CF-normalise the longitude coordinate.

    Renames a recognised longitude alias to ``lon`` and checks the values
    lie in range, raising on malformed input.

    Returns:
        The input dataset with a validated ``lon`` coordinate.
    """

    def _apply(self, ds):
        return _validation.validate_longitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _validate_signature(
            input_signature,
            aliases=_validation._LONGITUDE_ALIASES,
            canonical="lon",
        )


class ValidateLatitude(Operator):
    """Validate and CF-normalise the latitude coordinate.

    Renames a recognised latitude alias to ``lat`` and checks the values
    lie in ``[-90, 90]``, raising on malformed input.

    Returns:
        The input dataset with a validated ``lat`` coordinate.
    """

    def _apply(self, ds):
        return _validation.validate_latitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _validate_signature(
            input_signature,
            aliases=_validation._LATITUDE_ALIASES,
            canonical="lat",
        )


class ValidateCoords(Operator):
    """Validate both longitude and latitude in one pass.

    Applies :class:`ValidateLongitude` then :class:`ValidateLatitude`.

    Returns:
        The input dataset with validated ``lon`` and ``lat`` coordinates.
    """

    def _apply(self, ds):
        ds = _validation.validate_longitude(ds)
        return _validation.validate_latitude(ds)

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        signature = _validate_signature(
            input_signature,
            aliases=_validation._LONGITUDE_ALIASES,
            canonical="lon",
        )
        return _validate_signature(
            signature,
            aliases=_validation._LATITUDE_ALIASES,
            canonical="lat",
        )


class DecodeCFTime(Operator):
    """Decode a CF-encoded time coordinate to ``datetime64``.

    Args:
        time: Name of the time coordinate to decode.
        units: CF time units (e.g. ``"days since 1990-01-01"``); ``None``
            reads them from the coordinate's ``units`` attribute.

    Returns:
        The input dataset with ``time`` decoded to ``datetime64``.
    """

    def __init__(self, *, time: str = "time", units: str | None = None):
        self.time = time
        self.units = units

    def _apply(self, ds):
        return _validation.decode_cf_time(ds, time=self.time, units=self.units)

    def get_config(self) -> dict[str, Any]:
        return {"time": self.time, "units": self.units}


class ValidateTime(Operator):
    """Validate and standardise a time coordinate.

    Args:
        time: Name of the time coordinate.
        unit: Expected time unit; ``None`` infers it.
        origin: Reference epoch for numeric times (default ``"unix"``).

    Returns:
        The input dataset with a validated ``time`` coordinate.
    """

    def __init__(
        self,
        *,
        time: str = "time",
        unit: str | None = None,
        origin: str = "unix",
    ):
        self.time = time
        self.unit = unit
        self.origin = origin

    def _apply(self, ds):
        return _validation.validate_time(
            ds, time=self.time, unit=self.unit, origin=self.origin
        )

    def get_config(self) -> dict[str, Any]:
        return {"time": self.time, "unit": self.unit, "origin": self.origin}


class RenameCoords(Operator):
    """Rename coordinates by an explicit mapping.

    Args:
        mapping: ``{old_name: new_name}`` coordinate renames.

    Returns:
        The input dataset with the coordinates renamed.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_coords(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return _sig.rename_dims(input_signature, self.mapping)


class RenameVariables(Operator):
    """Rename data variables by an explicit mapping.

    Args:
        mapping: ``{old_name: new_name}`` data-variable renames.

    Returns:
        The input dataset with the data variables renamed.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def _apply(self, ds):
        return _validation.rename_variables(ds, self.mapping)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


class RenameToCFStandardNames(Operator):
    """Wrap :func:`xrops.cf.rename_to_cf_standard_names`.

    Renames each variable / coord that carries a ``standard_name`` attr
    to that attr value. Raises ``ValueError`` on collision (two source
    vars mapping to the same ``standard_name``).

    Args:
        include_coords: If ``True`` (default), rename coords too.
    """

    def __init__(self, *, include_coords: bool = True) -> None:
        self.include_coords = include_coords

    def _apply(self, ds):
        return _validation.rename_to_cf_standard_names(
            ds, include_coords=self.include_coords
        )

    def get_config(self) -> dict[str, Any]:
        return {"include_coords": self.include_coords}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


class RenameFromCFStandardNames(Operator):
    """Wrap :func:`xrops.cf.rename_from_cf_standard_names`.

    Renames CF ``standard_name``-shaped variables to their xrops
    canonical names using the :mod:`xrproducts.types.Variable` registry.

    Args:
        fallback: ``"passthrough"`` (default) leaves unrecognized names
            unchanged. ``"raise"`` raises ``KeyError`` on unknown names.
        include_coords: If ``True`` (default), rename coords too.
    """

    def __init__(
        self,
        *,
        fallback: Literal["passthrough", "raise"] = "passthrough",
        include_coords: bool = True,
    ) -> None:
        self.fallback = fallback
        self.include_coords = include_coords

    def _apply(self, ds):
        return _validation.rename_from_cf_standard_names(
            ds,
            fallback=self.fallback,
            include_coords=self.include_coords,
        )

    def get_config(self) -> dict[str, Any]:
        return {"fallback": self.fallback, "include_coords": self.include_coords}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        return input_signature


__all__ = [
    "DecodeCFTime",
    "RenameCoords",
    "RenameFromCFStandardNames",
    "RenameToCFStandardNames",
    "RenameVariables",
    "ValidateCoords",
    "ValidateLatitude",
    "ValidateLongitude",
    "ValidateTime",
]
