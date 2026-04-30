"""Value resampling — gap-fill, regrid, coarsen, refine, bin, resample, smooth.

Single conceptual home for *value resampling* (D12). Sub-organized by
source/target structure under :mod:`._src`:

- :mod:`._src.gap_fill` — ``fillnan_spatial``, ``fillnan_temporal``, ``fillnan_rbf``
- :mod:`._src.grid_to_grid` — ``coarsen``, ``refine``
- :mod:`._src.resample` — ``resample_time``
- :mod:`._src.binning` — ``Grid``, ``Period``, ``SpaceTimeGrid``, ``bin_2d``,
  ``histogram_2d``
- :mod:`._src.points_to_grid` — ``points_to_grid``
- :mod:`._src.grid_to_points`, :mod:`._src.coord_remap`, :mod:`._src.smooth`,
  :mod:`._src.downscale` — placeholder submodules for upcoming work
  (D12, issues #34/#35/#36)

Layer-1 ``Operator`` wrappers live in :mod:`xr_toolz.interpolate.operators`.
"""

from __future__ import annotations

from xr_toolz.interpolate._src.binning import (
    Grid,
    Period,
    SpaceTimeGrid,
    bin_2d,
    histogram_2d,
)
from xr_toolz.interpolate._src.gap_fill import (
    fillnan_rbf,
    fillnan_spatial,
    fillnan_temporal,
)
from xr_toolz.interpolate._src.grid_to_grid import coarsen, refine
from xr_toolz.interpolate._src.points_to_grid import points_to_grid
from xr_toolz.interpolate._src.resample import resample_time


__all__ = [
    "Grid",
    "Period",
    "SpaceTimeGrid",
    "bin_2d",
    "coarsen",
    "fillnan_rbf",
    "fillnan_spatial",
    "fillnan_temporal",
    "histogram_2d",
    "points_to_grid",
    "refine",
    "resample_time",
]
