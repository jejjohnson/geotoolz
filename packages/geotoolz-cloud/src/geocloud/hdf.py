"""`geocloud.hdf` — HDF5 / NetCDF / HDF4 from any location, and CF decoding.

- `read_hdf` / `read_netcdf` — one dataset / variable as a ``GeoTensor``,
  from a local path or any URI the pool accepts. HDF5 and NetCDF-4 are
  read with ranged requests; NetCDF-3 and HDF4 through a
  `geocloud.cache` copy.
- `open_hdf5` — an ``h5py.File`` over ranged reads, for product readers.
- The CF decoders product readers share: `PackedVariable` (``_Unsigned``,
  ``_FillValue``, ``scale_factor`` / ``add_offset``), `attr`, `scalar`,
  `unpacked`, `time_attr`, `grid_variables`, `fill_value_from_attrs`,
  `affine_from_geotransform`, and the 1-based band selection
  `select_indexes` / `read_indexes`.

Extras: ``[hdf5]`` (h5py), ``[netcdf]`` (netCDF4), ``[hdf4]`` (pyhdf).

```python
from georeader.geotensor import GeoTensor

from geocloud.hdf import open_hdf5, read_netcdf

ch4: GeoTensor = read_netcdf("s3://bucket/L2.nc", "methane", group="PRODUCT")
with open_hdf5("s3://noaa-goes19/ABI-L1b-RadC/.../OR_ABI-L1b-RadC-M6C02.nc") as f:
    radiance = f["Rad"][:500, :500]          # only these chunks are fetched
```
"""

from __future__ import annotations

from geocloud._src.hdf import (
    HDF4_SIGNATURE,
    HDF5_SIGNATURE,
    HdfSource,
    PackedVariable,
    affine_from_geotransform,
    attr,
    fill_value_from_attrs,
    grid_variables,
    open_hdf5,
    read_hdf,
    read_indexes,
    read_netcdf,
    scalar,
    select_indexes,
    time_attr,
    unpacked,
)


__all__ = [
    "HDF4_SIGNATURE",
    "HDF5_SIGNATURE",
    "HdfSource",
    "PackedVariable",
    "affine_from_geotransform",
    "attr",
    "fill_value_from_attrs",
    "grid_variables",
    "open_hdf5",
    "read_hdf",
    "read_indexes",
    "read_netcdf",
    "scalar",
    "select_indexes",
    "time_attr",
    "unpacked",
]
