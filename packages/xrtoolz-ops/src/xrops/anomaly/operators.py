"""Layer-1 ``Operator`` wrappers around the :mod:`xrops.anomaly._src` primitives.

Each class is a thin adapter: store configuration, implement
``_apply``, return a JSON-serializable ``get_config``. They all inherit
from :class:`pipekit.Operator`, so they compose with
:class:`~pipekit.Sequential`, the ``|`` pipe, and the functional
:class:`~pipekit.Graph` API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit import Signature

from xrops._src import signature as _sig
from xrops._src.operator import Operator
from xrops.anomaly._src import (
    detrend as _detrend,
)


def _replace_dim(
    input_signature: Signature,
    *,
    old: str,
    new: str,
    size: int | None,
) -> Signature:
    dims = {}
    for name, dim_size in _sig.sizes(input_signature).items():
        if name == old:
            dims[new] = size
        else:
            dims[name] = dim_size
    return _sig.make(dims, dtype=input_signature.dtype)


class CalculateClimatology(Operator):
    """Compute a climatology grouped at ``freq`` from the time axis.

    Args:
        freq: Climatology grouping — e.g. ``"day"`` (day-of-year),
            ``"month"``, ``"season"`` (one of ``CLIMATOLOGY_DIMS``).
        time: Name of the time coordinate.

    Returns:
        A dataset of the per-group climatological means, with ``time``
        replaced by the grouping dimension.
    """

    def __init__(self, freq: str = "day", time: str = "time"):
        if freq not in _detrend.CLIMATOLOGY_DIMS:
            raise ValueError(
                f"Unsupported climatology frequency {freq!r}; expected one of "
                f"{sorted(_detrend.CLIMATOLOGY_DIMS)}."
            )
        self.freq = freq
        self.time = time

    def _apply(self, ds):
        return _detrend.calculate_climatology(ds, freq=self.freq, time=self.time)

    def get_config(self) -> dict[str, Any]:
        return {"freq": self.freq, "time": self.time}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        dim = _detrend.CLIMATOLOGY_DIMS[self.freq]
        return _replace_dim(input_signature, old=self.time, new=dim, size=None)


class CalculateClimatologySmoothed(Operator):
    """Compute a day-of-year climatology smoothed with a rolling window.

    Args:
        window: Width (in days) of the centred rolling-mean smoothing
            applied around the day-of-year cycle.
        time: Name of the time coordinate.

    Returns:
        A smoothed day-of-year climatology dataset.
    """

    def __init__(self, window: int = 60, time: str = "time"):
        self.window = window
        self.time = time

    def _apply(self, ds):
        return _detrend.calculate_climatology_smoothed(
            ds, window=self.window, time=self.time
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window, "time": self.time}

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        # Smoothed climatology always groups by day-of-year; pull the
        # canonical dim name from the same source the runtime uses
        # rather than hard-coding the literal here.
        new_dim = _detrend.CLIMATOLOGY_DIMS["day"]
        return _replace_dim(input_signature, old=self.time, new=new_dim, size=None)


class RemoveMean(Operator):
    """Subtract the mean over ``dim`` (a cheap anomaly without a climatology).

    Args:
        dim: Dimension or dimensions to average over before subtracting.

    Returns:
        The input dataset with the per-``dim`` mean removed.
    """

    def __init__(self, dim: str | Sequence[str]):
        self.dim = (dim,) if isinstance(dim, str) else tuple(dim)

    def _apply(self, ds):
        return _detrend.remove_mean(ds, self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"dim": list(self.dim)}


class RemoveClimatology(Operator):
    """Subtract a precomputed climatology to form anomalies.

    Args:
        climatology: Climatology dataset (e.g. from
            :class:`CalculateClimatology`) to subtract.
        time: Name of the time coordinate.

    Returns:
        The anomaly dataset (input minus the matched climatology).
    """

    def __init__(self, climatology, time: str = "time"):
        self.climatology = climatology
        self.time = time

    def _apply(self, ds):
        return _detrend.remove_climatology(ds, self.climatology, time=self.time)

    def get_config(self) -> dict[str, Any]:
        # climatology is rich state — referenced rather than serialized
        return {"climatology": "<xr object>", "time": self.time}


class AddClimatology(Operator):
    """Add a climatology back to anomalies (inverse of :class:`RemoveClimatology`).

    Args:
        climatology: Climatology dataset to add.
        time: Name of the time coordinate.

    Returns:
        The reconstructed dataset (anomaly plus climatology).
    """

    def __init__(self, climatology, time: str = "time"):
        self.climatology = climatology
        self.time = time

    def _apply(self, ds):
        return _detrend.add_climatology(ds, self.climatology, time=self.time)

    def get_config(self) -> dict[str, Any]:
        return {"climatology": "<xr object>", "time": self.time}


__all__ = [
    "AddClimatology",
    "CalculateClimatology",
    "CalculateClimatologySmoothed",
    "RemoveClimatology",
    "RemoveMean",
]
