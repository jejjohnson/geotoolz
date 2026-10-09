# Read HDF and NetCDF files

Read one dataset or variable from an HDF5, HDF4 / HDF-EOS or NetCDF-CF
file as a `GeoTensor`, ready for any operator. `gz.io.ReadHDF` and
`gz.io.ReadNetCDF` are source operators: build them with a path or any URI
geotoolz-cloud reads (`s3://`, `gs://`, `az://`, signed `https://`), then
call them with no input. They wrap [`geocloud.hdf`](../cloud/how-to/hdf.md):
HDF5 is read with ranged requests, NetCDF and HDF4 from a cached local copy.

Install the backend you need; the extras are listed on the
[landing page](index.md#install):

```bash
pip install 'geotoolz[hdf5,netcdf]'     # h5py, netCDF4
pip install 'geotoolz[hdf4]'            # pyhdf, for HDF4 / HDF-EOS (MODIS)
```

## Read an HDF dataset

Pick a dataset by path inside the file and, optionally, 1-based band
`indexes`. `geolocation=` names the latitude and longitude datasets of a
swath; they travel in `attrs["geolocation"]`.

```python
import tempfile
from pathlib import Path

import h5py
import numpy as np
from georeader.geotensor import GeoTensor

import geotoolz as gz

# A small swath file: 3 radiance bands plus per-pixel latitude / longitude.
path: Path = Path(tempfile.mkdtemp()) / "swath.h5"
rng: np.random.Generator = np.random.default_rng(0)
lat: np.ndarray
lon: np.ndarray
lat, lon = np.meshgrid(np.linspace(39.3, 38.9, 20), np.linspace(-120.2, -119.9, 30), indexing="ij")  # (20, 30) float64 each
with h5py.File(path, "w") as f:
    f["radiance"] = rng.random((3, 20, 30)).astype(np.float32)   # (3, 20, 30) float32
    f["lat"], f["lon"] = lat, lon

reader: gz.ReadHDF = gz.io.ReadHDF(path=path, dataset="radiance", indexes=[1, 3], geolocation=("lat", "lon"))
bands: GeoTensor = reader()                                       # (2, 20, 30) float32 · bands 1 and 3
```

HDF5 files go through `h5py`. HDF4 / HDF-EOS files are recognised by
their signature and need `pyhdf`; without it the call raises an
`ImportError` that names `geotoolz[hdf4]`.

## Read a NetCDF-CF variable

`ReadNetCDF` applies CF masking and `scale_factor` / `add_offset` by
default, walks nested groups with `group="A/B"`, and recovers the CRS
from the variable's CF `grid_mapping`. The transform comes from the
GDAL `GeoTransform` attribute when the file has one.

```python
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
from georeader.geotensor import GeoTensor
from pyproj import CRS

import geotoolz as gz

# A small Level-2 file: packed int16 methane on a UTM grid, inside group PRODUCT.
path: Path = Path(tempfile.mkdtemp()) / "l2.nc"
with netCDF4.Dataset(path, "w") as ds:
    product: netCDF4.Group = ds.createGroup("PRODUCT")
    product.createDimension("y", 20)
    product.createDimension("x", 30)
    crs: netCDF4.Variable = product.createVariable("crs", "i4")
    crs.crs_wkt = CRS("EPSG:32610").to_wkt()
    crs.GeoTransform = "750000 10 0 4350000 0 -10"
    ch4: netCDF4.Variable = product.createVariable("ch4", "i2", ("y", "x"), fill_value=-999)
    ch4.scale_factor, ch4.add_offset, ch4.grid_mapping = 0.1, 1800.0, "crs"
    ch4[:] = np.random.default_rng(0).uniform(1850, 1950, (20, 30))  # (20, 30) float64 → packed int16

reader: gz.ReadNetCDF = gz.io.ReadNetCDF(path=path, variable="ch4", group="PRODUCT")
methane: GeoTensor = reader()                                     # (20, 30) float64 · ppb · NaN = fill
assert methane.crs is not None and methane.transform.a == 10
```

Pass `decode_cf=False` to get the raw packed integers, and
`use_cf_grid_mapping=False` to skip the CRS lookup.

## Further reading

- [IO reference](api/io.md): window readers, STAC and Earth Engine
  sources, and the GeoTIFF / COG / Zarr writers.
- Sensor-specific readers (GOES, Himawari, Carbon Mapper) live in
  [geoproducts](../products/index.md).
