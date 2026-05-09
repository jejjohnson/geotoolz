"""Value resampling — gap-fill, regrid, coarsen, refine, bin, resample, smooth.

Single conceptual home for *value resampling* (D12). Sub-organized by
source/target structure under :mod:`._src`:

- :mod:`._src.gap_fill` — ``fillnan_spatial``, ``fillnan_temporal``,
  ``fillnan_climatology``, ``fillnan_laplacian``, ``fillnan_rbf``,
  ``fillnan_idw``
- :mod:`._src.grid_to_grid` — ``coarsen``, ``coarsen_conservative``,
  ``refine``, ``refine_2d``
- :mod:`._src.resample` — ``resample_time``
- :mod:`._src.binning` — ``Grid``, ``Period``, ``SpaceTimeGrid``, ``bin_2d``,
  ``histogram_2d``
- :mod:`._src.points_to_grid` — ``points_to_grid``, ``kde_to_grid``
- :mod:`._src.smooth` — ``moving_average``, ``gaussian_smooth``,
  ``lowpass_filter``, ``fir_filter``
- :mod:`._src.coord_remap` — ``remap_axis``, ``to_phase``
- :mod:`._src.grid_to_points`, :mod:`._src.downscale` — placeholder
  submodules for upcoming work (D12, issue #36)

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
from xr_toolz.interpolate._src.coord_remap import remap_axis, to_phase
from xr_toolz.interpolate._src.gap_fill import (
    fillnan_climatology,
    fillnan_idw,
    fillnan_laplacian,
    fillnan_rbf,
    fillnan_spatial,
    fillnan_temporal,
)
from xr_toolz.interpolate._src.grid_to_grid import (
    coarsen,
    coarsen_conservative,
    refine,
    refine_2d,
    regrid_like,
)
from xr_toolz.interpolate._src.knn import idw_to_grid, idw_to_points
from xr_toolz.interpolate._src.points_to_grid import kde_to_grid, points_to_grid
from xr_toolz.interpolate._src.resample import resample_time
from xr_toolz.interpolate._src.smooth import (
    fir_filter,
    gaussian_smooth,
    lowpass_filter,
    moving_average,
)
from xr_toolz.interpolate.operators import KDEToGrid


__all__ = [
    "Grid",
    "KDEToGrid",
    "Period",
    "SpaceTimeGrid",
    "bin_2d",
    "coarsen",
    "coarsen_conservative",
    "fillnan_climatology",
    "fillnan_idw",
    "fillnan_laplacian",
    "fillnan_rbf",
    "fillnan_spatial",
    "fillnan_temporal",
    "fir_filter",
    "gaussian_smooth",
    "histogram_2d",
    "idw_to_grid",
    "idw_to_points",
    "kde_to_grid",
    "lowpass_filter",
    "moving_average",
    "points_to_grid",
    "refine",
    "refine_2d",
    "regrid_like",
    "remap_axis",
    "resample_time",
    "to_phase",
]
