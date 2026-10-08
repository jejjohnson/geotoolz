# GOES-R ABI

`geoproducts.goes` reads GOES-R Advanced Baseline Imager (ABI) L1b radiances —
GOES-16, -17, -18 and -19 — and finds them in NOAA's public AWS buckets.

```bash
pip install 'geotoolz-products[goes]'        # h5py: ABI files are NetCDF-4 / HDF5
pip install 'geotoolz-products[goes,operators]'  # + geotoolz presets (NDVI, synthetic green, parallax)
```

Without the extra, `goes.Reader` raises an `ImportError` naming it; the
bucket helpers in `goes.aws` are standard library and always work.

## What a file is

One L1b file (`OR_ABI-L1b-Rad{F,C,M1,M2}-M6Cnn_Gnn_s…_e…_c….nc`) is **one
channel of one scan** on the ABI fixed grid: `x` / `y` are scan angles in
radians, so scaling them by the perspective-point height gives a clean
`+proj=geos` affine grid. Each channel ships at its native resolution:

| Channels | Resolution | Calibrations |
|---|---|---|
| C02 (red, 0.64 µm) | 0.5 km | counts · radiance · reflectance |
| C01, C03, C05 | 1 km | counts · radiance · reflectance |
| C04, C06 | 2 km | counts · radiance · reflectance |
| C07–C16 (3.9–13.3 µm) | 2 km | counts · radiance · brightness temperature |

| Sector | `product` | Size (2 km channel) | Cadence |
|---|---|---|---|
| Full disk | `ABI-L1b-RadF` | 5424 × 5424 | 10 min |
| CONUS / PACUS | `ABI-L1b-RadC` | 1500 × 2500 | 5 min |
| Mesoscale (M1, M2) | `ABI-L1b-RadM` | 500 × 500 | 1 min |

The full table is `goes.BANDS` (name, wavelength, resolution, reflective /
emissive).

## Find, fetch, read

```python
from datetime import datetime
from pathlib import Path

from georeader.geotensor import GeoTensor

from geoproducts import goes
from geoproducts.goes import aws

files: list[aws.ABIFile] = aws.list_files(
    satellite="G19",                      # GOES-East
    product="ABI-L1b-RadC",               # CONUS sector
    start=datetime(2026, 10, 7, 18, 0),   # naive → UTC
    end=datetime(2026, 10, 7, 18, 10),
    channel=13,                           # clean longwave window
)                                         # 2 scans, 5 min apart, ~4 MB each
path: Path = aws.download(files[0], "data/goes")  # streamed to .part, then renamed

reader: goes.Reader = goes.Reader(path, calibration="brightness_temperature")
reader.shape                              # (1, 1500, 2500) · +proj=geos · 2004 m pixels

aoi = (-100.0, 30.0, -95.0, 35.0)         # lon/lat box over east Texas
bt: GeoTensor = reader.read_from_bounds(aoi, crs_bounds="EPSG:4326")      # (1, 219, 266) float32, K
flags: GeoTensor = reader.quality.read_from_bounds(aoi, crs_bounds="EPSG:4326")  # (1, 219, 266) uint8 DQF
```

Only the HDF5 chunks under the window are decompressed, so a small AOI out
of a full-disk file stays cheap. `bt.attrs` carries `band_names=("C13",)`,
`wavelengths=(10300.0,)` (nm), `units="K"` and `calibration`; off-disk and
missing pixels are `NaN`.

`list_files` and `download` talk to `https://noaa-goes{16..19}.s3.amazonaws.com`
anonymously and retry transient failures. To skip the download, pass any
binary file object instead of a path, e.g.
`goes.Reader(fsspec.open("s3://noaa-goes19/…", anon=True).open())`.

## Calibration

Every coefficient comes from the file being read:

| `calibration=` | Output | Formula |
|---|---|---|
| `"counts"` | `uint16` | the stored detector counts (read unsigned) |
| `"radiance"` *(default)* | `float32` | `counts · scale_factor + add_offset` |
| `"reflectance"` | `float32` | `kappa0 · radiance` (C01–C06) |
| `"brightness_temperature"` | `float32`, K | `(fk2 / ln(fk1 / L + 1) − bc1) / bc2` (C07–C16) |

`reflectance` is NOAA's reflectance factor: it folds in the Earth–Sun
distance and band solar irradiance, not the solar zenith angle. Asking for a
calibration a channel does not support raises `ValueError`.

`reader.quality` reads the `DQF` layer on the same grid:
`goes.constants.DQF_FLAGS` maps `0` good, `1` conditionally usable, `2` out
of range, `3` no value, `4` focal-plane temperature exceeded; `255` is the
fill outside the file.

## Multi-channel products

ABI ships one channel per file at its own resolution, so put channels on one
grid before combining them. With the `[operators]` extra:

```python
import geotoolz as gz

def reflectance(channel: int) -> GeoTensor:
    item: aws.ABIFile = aws.list_files(satellite="G19", start=datetime(2026, 10, 7, 18),
                                       end=datetime(2026, 10, 7, 18, 5), channel=channel)[0]
    reader = goes.Reader(aws.download(item, "data/goes"), calibration="reflectance")
    return reader.read_from_bounds(aoi, crs_bounds="EPSG:4326")

blue: GeoTensor = reflectance(1)                     # (1, 437, 530)  1 km
red: GeoTensor = reflectance(2)                      # (1, 874, 1059) 0.5 km
veggie: GeoTensor = reflectance(3)                   # (1, 437, 530)  1 km

red_1km: GeoTensor = gz.geom.ReprojectLike(like=blue, resampling="average")(red)  # (1, 437, 530)
stack: GeoTensor = gz.spectral.StackBands()([blue, red_1km, veggie])  # (3, 437, 530) C01, C02, C03

ndvi: GeoTensor = goes.NDVI()(stack)                 # (437, 530) float32
green: GeoTensor = goes.SyntheticGreen()(stack)      # (437, 530) — 0.45·C02 + 0.10·C03 + 0.45·C01
rgb: GeoTensor = gz.spectral.StackBands()([red_1km, green, blue])  # (3, 437, 530) true colour
```

`goes.ParallaxCorrect` binds `gz.geom.GeostationaryParallaxCorrect` to the
GOES geometry. It works on lat/lon grids, so reproject first:

```python
bt_ll: GeoTensor = gz.geom.Reproject(dst_crs="EPSG:4326", resolution=(0.01, 0.01))(bt)  # (1, 552, 834)
cloud_tops: GeoTensor = goes.ParallaxCorrect(
    satellite_lon_deg=reader.satellite_lon_deg,      # -75.2 for GOES-East
    target_height_m=10_000.0,                        # or a cloud-top-height field
)(bt_ll)                                             # (1, 552, 834), features moved to nadir
```

## Module layout

| Module | Contents |
|---|---|
| `goes.reader` | `Reader` (radiance / reflectance / BT), `QualityReader` (DQF) |
| `goes.aws` | `list_files`, `download`, `parse_key`, `ABIFile`, `bucket`, `url` |
| `goes.constants` | `BANDS`, `CHANNELS`, `DQF_FLAGS`, satellite positions, synthetic-green weights |
| `goes.presets` | `NDVI`, `SyntheticGreen`, `ParallaxCorrect` (`[operators]` extra) |

L2 products (clear-sky mask, cloud-top height) and the remaining RGB
composites are not covered yet.
