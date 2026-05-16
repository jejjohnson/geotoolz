# Multi-format readers

`geotoolz.io` includes source operators for sensor-agnostic HDF, NetCDF-CF,
and xRIT segment reads. Install only the backend extras you need:

```bash
uv pip install -e '.[hdf5,netcdf]'
```

## HDF / HDF-EOS / HDF5

```python
import geotoolz as gz

b01 = gz.io.ReadHDF(
    path="MOD021KM.A2024196.1855.061.2024197144543.hdf",
    dataset="EV_1KM_RefSB",
    indexes=[1],
    geolocation=("Latitude", "Longitude"),
)()
```

HDF5 files use `h5py`. HDF4 / HDF-EOS files dispatch by file signature and
raise a clear `ImportError` asking for `geotoolz[hdf4]` when `pyhdf` is not
available.

## NetCDF-CF

```python
import geotoolz as gz

ch4 = gz.io.ReadNetCDF(
    path="S5P_OFFL_L2__CH4____20240715T093041_...nc",
    variable="methane_mixing_ratio_bias_corrected",
    group="PRODUCT",
    decode_cf=True,
)()
```

`ReadNetCDF` uses `netCDF4` directly, applies CF mask/scale decoding by
default, supports nested group paths, and recovers the CRS from a CF
`grid_mapping` variable when present.

## xRIT / LRIT / HRIT segments

```python
import glob
import geotoolz as gz

ir108 = gz.io.ReadXRIT(
    paths=sorted(glob.glob("H-000-MSG4__-MSG4________-IR_108___-*")),
    channel="IR_108",
    decompress=True,
)()
```

`ReadXRIT` assembles ordered segment payloads along the scan-direction axis.
Compressed HRIT wavelet payload decoding is reserved for a future codec backend;
uncompressed arrays and raw byte segments can already be composed by downstream
sensor-specific readers.
