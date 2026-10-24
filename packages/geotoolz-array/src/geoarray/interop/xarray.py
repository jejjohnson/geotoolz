"""`geoarray.interop.xarray` — lossless converters to and from xarray.

Behind ``geotoolz-array[xarray]``. The CF ``grid_mapping`` + GDAL
``GeoTransform`` convention (GA-Q2), written without rioxarray:

- `to_xarray` / `from_xarray` — a `GeoArray` or `GeoSwath` ↔ ``DataArray``,
  a `GeoPoints` ↔ a CF Discrete Sampling Geometry ``Dataset``; the carrier
  methods ``to_xarray()`` / ``from_xarray()`` call these.
- `to_dataset` / `from_dataset` — several carriers on one grid.
- `to_datatree` / `from_datatree` — several grids in one tree.

Importing this module also registers the ``DataArray`` kind reader, so
`geoarray.types.matches` reads a lazy cube (``time lat lon``) as a
``GeoMapSeries``.

Example:
    >>> import numpy as np
    >>> from geoarray import GeoArray
    >>> from geoarray.interop.xarray import from_xarray, to_xarray
    >>> g = GeoArray(np.zeros((2, 3)), (20, 0, 600000, 0, -20, 3550000), 32613,
    ...              dims="y x")
    >>> from_xarray(to_xarray(g)).identical(g)
    True
"""

from __future__ import annotations

from geoarray._src.xarray import (
    from_dataset,
    from_datatree,
    from_xarray,
    to_dataset,
    to_datatree,
    to_xarray,
)


__all__ = [
    "from_dataset",
    "from_datatree",
    "from_xarray",
    "to_dataset",
    "to_datatree",
    "to_xarray",
]
