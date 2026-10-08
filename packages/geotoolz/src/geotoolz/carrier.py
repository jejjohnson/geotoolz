"""The `GeoTensor` carrier contract, as public helpers.

Every geotoolz operator keeps the same carrier rules: a `GeoTensor` in
comes back as a `GeoTensor` on the input's grid (a plain array as a plain
array), with a fresh ``attrs``, per-band metadata that matches the output
bands, and a ``fill_value_default`` that marks its nodata. These are the
helpers the built-in operators use to keep them; an operator you write on
top of the stack uses the same ones, so it composes with everything else.
The rules themselves are in the repository's ``AGENTS.md`` ("The two
contracts") and ``docs/concepts.md``.

Rewrap:
    `wrap_like` puts a result back on the input's carrier (``transform=``
    for a new grid); `INHERIT` is its "take the input's fill" default.

Nodata:
    `valid_pixels` / `invalid_values` (a pixel is invalid when any band is
    non-finite or equals the fill), `mask_invalid_to_nan` before NaN-aware
    statistics, `restore_fill` / `wrap_filled` to write the output's fill
    back, and `carried_fill` for an output that carries the input's values.

Bands:
    `resolve_band` / `resolve_bands` (an integer position or a name looked
    up in ``band_names``, ``descriptions``, ``bands``), `band_names`.

Shape:
    `band_axis`, `require_ndim`, `map_frames` / `over_frames` (run a
    per-scene function on each frame of a ``(T, C, H, W)`` stack) and
    `keep_band_axis` (a band-collapsing result keeps its time axis).

Grids:
    `require_geotensor`, `require_projected_crs`, `require_grid_match`,
    `grid_matches`.

Dtypes and fitting:
    `as_float` (promote integer DNs before arithmetic) and `fit_once`
    (fit on first call, once, without races).

Example:
    An operator that keeps the carrier contract::

        import numpy as np
        from pipekit import Operator

        from geotoolz.carrier import mask_invalid_to_nan, over_frames, wrap_like


        class ZScore(Operator):
            def __init__(self, *, mean: float, std: float) -> None:
                self.mean = mean
                self.std = std

            @over_frames  # (T, C, H, W) stacks run frame by frame
            def _apply(self, gt):  # (C, H, W) -> (C, H, W) float
                values = mask_invalid_to_nan(gt)  # nodata -> NaN, whole pixel
                out = (values - self.mean) / self.std
                return wrap_like(gt, out, fill_value_default=np.nan)

    Check it with `geotoolz.testing.check_operator`.
"""

from __future__ import annotations

from geotoolz._src.bands import band_names, resolve_band, resolve_bands
from geotoolz._src.dtype import as_float
from geotoolz._src.fitted import fit_once
from geotoolz._src.geo import (
    grid_matches,
    require_geotensor,
    require_grid_match,
    require_projected_crs,
)
from geotoolz._src.shape import (
    band_axis,
    keep_band_axis,
    map_frames,
    over_frames,
    require_ndim,
)
from geotoolz._src.valid import (
    carried_fill,
    invalid_values,
    mask_invalid_to_nan,
    restore_fill,
    valid_pixels,
    wrap_filled,
)
from geotoolz._src.wrap import INHERIT, wrap_like


__all__ = [
    "INHERIT",
    "as_float",
    "band_axis",
    "band_names",
    "carried_fill",
    "fit_once",
    "grid_matches",
    "invalid_values",
    "keep_band_axis",
    "map_frames",
    "mask_invalid_to_nan",
    "over_frames",
    "require_geotensor",
    "require_grid_match",
    "require_ndim",
    "require_projected_crs",
    "resolve_band",
    "resolve_bands",
    "restore_fill",
    "valid_pixels",
    "wrap_filled",
    "wrap_like",
]
