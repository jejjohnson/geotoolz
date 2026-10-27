"""The `GeoArray` carrier contract, as public helpers.

Every geotoolz operator keeps the same carrier rules: a `geoarray.GeoArray`
in comes back as a `GeoArray` on the input's grid (a plain array as a
plain array), with per-band metadata that matches the output bands and a
``fill_value`` that marks its nodata; a georeader ``GeoTensor`` is refused,
naming ``geoarray.interop.georeader.from_georeader``. These are the
helpers the built-in operators use to keep them; an operator you write on
top of the stack uses the same ones, so it composes with everything else.
The rules themselves are in the repository's ``AGENTS.md`` ("The two
contracts") and ``docs/operators/concepts.md``.

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
    up in ``band_names``, ``descriptions``, ``bands``), `band_names`;
    `BandRef` (the type of a band reference) and `SENTINEL2_L2A_BANDS`
    (the canonical Sentinel-2 L2A band order).

Shape:
    `band_axis`, `band_dim` / `dim_axis` (an operator's ``dim=``, and
    where that dim sits in a carrier), `require_ndim`, `map_frames` /
    `over_frames` (run a
    per-scene function on each frame of a ``(T, C, H, W)`` stack),
    `keep_band_axis` (a band-collapsing result keeps its time axis) and
    `single_band` (pick one band as a 2-D plane).

Samples:
    `cube_to_samples` / `samples_to_cube` turn a ``(c, h, w)`` cube into
    an ``(n, c)`` sample matrix for a per-pixel estimator and back;
    `sample_layout` / `SampleLayout` describe the round trip.

Grids:
    `require_geoarray`, `reject_geotensor` (a georeader ``GeoTensor``
    carrier), `reject_georeader` (any georeader object, e.g. a raster an
    operator holds in its constructor), `require_projected_crs`,
    `require_grid_match`, `grid_matches`, `affine_of` (the transform as an
    ``affine.Affine``) and `pixel_xy` (map coordinates of pixel centres).

Dtypes and fitting:
    `as_float` (promote integer DNs before arithmetic) and `fit_once`
    (fit on first call, once, without races).

Config:
    The operator side of the contract: `jsonable` coerces a
    ``get_config()`` value to JSON builtins, `as_tuple` turns lists back
    into tuples on reload, `mapping_to_pairs` / `mapping_from_pairs`
    carry non-string-keyed mappings, `nested_config` / `callable_name`
    describe a nested operator or a callable, and `reject_config_summary`
    refuses a debug summary passed back to a constructor.

Optional dependencies:
    `import_optional` imports an extra's module inside the function that
    needs it, with an error naming the extra; `optional_name` is the
    ``from a import b`` form and `optional_root` the ``import a.b`` one.

Example:
    An operator that keeps the carrier contract::

        import numpy as np
        from pipekit import Operator

        from geoops.carrier import mask_invalid_to_nan, over_frames, wrap_like


        class ZScore(Operator):
            backends = ("numpy",)  # refuses JAX carriers (pipekit's check)

            def __init__(self, *, mean: float, std: float) -> None:
                self.mean = mean
                self.std = std

            @over_frames  # (T, C, H, W) stacks run frame by frame
            def _apply(self, gt):  # (C, H, W) -> (C, H, W) float
                values = mask_invalid_to_nan(gt)  # nodata -> NaN, whole pixel
                out = (values - self.mean) / self.std
                return wrap_like(gt, out, fill_value=np.nan)

    Check it with `geoops.testing.check_operator`.
"""

from __future__ import annotations

from geoops._src.bands import (
    SENTINEL2_L2A_BANDS,
    BandRef,
    band_names,
    resolve_band,
    resolve_bands,
)
from geoops._src.config import (
    as_tuple,
    callable_name,
    jsonable,
    mapping_from_pairs,
    mapping_to_pairs,
    nested_config,
    reject_config_summary,
)
from geoops._src.dtype import as_float
from geoops._src.fitted import fit_once
from geoops._src.geo import (
    affine_of,
    grid_matches,
    pixel_xy,
    reject_georeader,
    reject_geotensor,
    require_geoarray,
    require_grid_match,
    require_projected_crs,
)
from geoops._src.optional import import_optional, optional_name, optional_root
from geoops._src.samples import (
    SampleLayout,
    cube_to_samples,
    sample_layout,
    samples_to_cube,
)
from geoops._src.shape import (
    band_axis,
    band_dim,
    dim_axis,
    keep_band_axis,
    map_frames,
    over_frames,
    require_ndim,
    single_band,
)
from geoops._src.valid import (
    carried_fill,
    invalid_values,
    mask_invalid_to_nan,
    restore_fill,
    valid_pixels,
    wrap_filled,
)
from geoops._src.wrap import INHERIT, wrap_like


__all__ = [
    "INHERIT",
    "SENTINEL2_L2A_BANDS",
    "BandRef",
    "SampleLayout",
    "affine_of",
    "as_float",
    "as_tuple",
    "band_axis",
    "band_dim",
    "band_names",
    "callable_name",
    "carried_fill",
    "cube_to_samples",
    "dim_axis",
    "fit_once",
    "grid_matches",
    "import_optional",
    "invalid_values",
    "jsonable",
    "keep_band_axis",
    "map_frames",
    "mapping_from_pairs",
    "mapping_to_pairs",
    "mask_invalid_to_nan",
    "nested_config",
    "optional_name",
    "optional_root",
    "over_frames",
    "pixel_xy",
    "reject_config_summary",
    "reject_georeader",
    "reject_geotensor",
    "require_geoarray",
    "require_grid_match",
    "require_ndim",
    "require_projected_crs",
    "resolve_band",
    "resolve_bands",
    "restore_fill",
    "sample_layout",
    "samples_to_cube",
    "single_band",
    "valid_pixels",
    "wrap_filled",
    "wrap_like",
]
