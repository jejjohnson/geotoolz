"""Coord forwarding for labeled einx results.

The single public helper, :func:`forward_coords`, builds the coordinate
dict for the output of a labeled einx call. Only **dimension
coordinates** (1-D coords whose name equals their dim) are forwarded;
auxiliary / multidimensional coords are dropped (a reduction or
restructure rarely keeps them meaningful, and forwarding them risks a
shape mismatch). Callers needing more control pass ``extra``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr


def forward_coords(
    inputs: Sequence[xr.DataArray],
    output_dims: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the coord dict for the output of a labeled einx call.

    Args:
        inputs: input DataArrays in pattern order.
        output_dims: dim names of the output, in result order.
        extra: explicit coord overrides supplied by the caller. Wins
            over inferred coords.

    Returns:
        Coord dict suitable for ``xr.DataArray(..., coords=coords)``.

    Rules:
        - For each output dim, the first input carrying it as a 1-D
          dimension coordinate donates its coord.
        - ``extra`` overrides inferred coords.
        - Output dims not present on any input get coords only if
          ``extra`` provides them; otherwise they remain unindexed.
    """
    coords: dict[str, Any] = {}
    for dim in output_dims:
        for inp in inputs:
            if dim in inp.coords and dim in inp.dims and inp.coords[dim].ndim == 1:
                coords[dim] = inp.coords[dim].values
                break
    if extra:
        for name, value in extra.items():
            coords[name] = value
    return coords


__all__ = ["forward_coords"]
