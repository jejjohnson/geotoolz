# Read HDF and NetCDF

Read one variable of a NetCDF, HDF5 or HDF4 file as a `GeoTensor`, from a
local path or any URI the pool reads. `geocloud.hdf` is the stack's one
implementation: `gz.io.ReadNetCDF` / `ReadHDF` and the GOES and Himawari
readers all use it.

```bash
pip install 'geotoolz-cloud[hdf5,netcdf]'   # h5py, netCDF4
pip install 'geotoolz-cloud[hdf4]'          # pyhdf, for HDF4 / HDF-EOS (MODIS)
```

| Format | How a remote file is read |
| --- | --- |
| HDF5, NetCDF-4 (`read_hdf`, `open_hdf5`) | h5py over ranged requests: the header, then only the chunks read |
| NetCDF-3 and -4 (`read_netcdf`) | netCDF4 on a [cached local copy](cache.md) |
| HDF4 / HDF-EOS (`read_hdf`) | pyhdf on a cached local copy |

## Read a NetCDF variable

`read_netcdf` applies the CF mask and `scale_factor` / `add_offset`, takes
the CRS from the variable's `grid_mapping` and the transform from GDAL's
`GeoTransform` attribute.

```python
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
from georeader.geotensor import GeoTensor
from obstore.store import MemoryStore
from pyproj import CRS

from geocloud import files
from geocloud.hdf import read_netcdf
from geocloud.store import mount

path: Path = Path(tempfile.mkdtemp()) / "l2.nc"
with netCDF4.Dataset(path, "w") as ds:                      # a small packed L2 file
    ds.createDimension("y", 20)
    ds.createDimension("x", 30)
    crs = ds.createVariable("crs", "i4")
    crs.crs_wkt = CRS("EPSG:32610").to_wkt()
    crs.GeoTransform = "750000 10 0 4350000 0 -10"
    ch4 = ds.createVariable("ch4", "i2", ("y", "x"), fill_value=-999)
    ch4.scale_factor, ch4.add_offset, ch4.grid_mapping = 0.1, 1800.0, "crs"
    ch4[:] = np.random.default_rng(0).uniform(1850, 1950, (20, 30))  # (20, 30) → packed int16

mount("s3://demo-hdf", MemoryStore())                       # a stand-in bucket
files.upload(path, "s3://demo-hdf/l2.nc")
methane: GeoTensor = read_netcdf("s3://demo-hdf/l2.nc", "ch4")  # (20, 30) float64 · ppb · NaN = fill
assert methane.crs is not None and methane.transform.a == 10
```

## Read an HDF5 dataset without downloading it

`open_hdf5` hands you an `h5py.File` over ranged reads, for code that
walks a product's own layout. `read_hdf` reads one dataset as a
`GeoTensor`, with 1-based band `indexes` and optional swath `geolocation`.

```python
import tempfile
from pathlib import Path

import h5py
import numpy as np
from georeader.geotensor import GeoTensor
from obstore.store import MemoryStore

from geocloud import files
from geocloud.hdf import open_hdf5, read_hdf
from geocloud.store import mount

path: Path = Path(tempfile.mkdtemp()) / "swath.h5"
with h5py.File(path, "w") as f:
    f["radiance"] = np.random.default_rng(0).random((3, 200, 300)).astype("f4")  # (3, 200, 300) float32

mount("s3://demo-h5", MemoryStore())
files.upload(path, "s3://demo-h5/swath.h5")
with open_hdf5("s3://demo-h5/swath.h5") as f:
    corner: np.ndarray = f["radiance"][0, :10, :10]           # (10, 10) float32, one chunk fetched
bands: GeoTensor = read_hdf("s3://demo-h5/swath.h5", "radiance", indexes=[3, 1])  # (2, 200, 300) float32
```

## Decode CF packing yourself

Product readers decode variable by variable with `PackedVariable`:
`_Unsigned` integers, `_FillValue` (as `NaN`) and `scale_factor` /
`add_offset`, to `float32`. `attr`, `scalar`, `unpacked` and `time_attr`
read attributes and coordinates as plain Python values.

## Pitfalls

- **NetCDF-3 and HDF4 copy the file.** Their libraries need a real file;
  the copy lands in the [cache](cache.md), so a second read is local.
- **Credentials** come from `geocloud.credentials`, like every other read
  on the pool.
- **The result has an identity transform** unless the file carries a
  `GeoTransform`; swath products ship latitude / longitude arrays instead
  (`read_hdf(..., geolocation=("lat", "lon"))`).
