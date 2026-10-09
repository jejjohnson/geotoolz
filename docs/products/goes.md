# GOES-R ABI

`geoproducts.goes` reads the Advanced Baseline Imager on GOES-16 … 19.
It finds L1b radiances and every L2 product in NOAA's public buckets,
puts them on one grid and builds the standard RGB composites.

![Four RGB recipes from one GOES-19 mesoscale MCMIP file: true colour, natural colour, day cloud phase and fire temperature of a hurricane near the Yucatán](../assets/figures/goes-recipes.jpg)

```bash
pip install 'geotoolz-products[goes]'            # readers + goes.aws
pip install 'geotoolz-products[goes,operators]'  # + presets and RGB operators
```

Without `[goes]`, the readers and `goes.aws` raise an `ImportError` naming
the extra; `goes.recipes` always works.

## What a file is

Every ABI file holds one product of one scan, on the **ABI fixed grid**.
Its `x` / `y` are scan angles; scaled by the satellite height they give a
clean `+proj=geos` affine grid. A window decompresses only the HDF5 chunks
under it.

| Level | File | Reader | Bands |
|---|---|---|---|
| L1b | `OR_ABI-L1b-Rad{F,C,M1,M2}-M6Cnn_…` — one channel | `goes.Reader` | the channel (`"C13"`), calibrated |
| L2 | `OR_ABI-L2-<code>{F,C,M1,M2}-M6_…` — one product | `goes.L2Reader` | the product's variables (`"HT"`, `"BCM"`, …) |
| L2 imagery | `MCMIP` (16 channels) / `CMIP` (one) | `goes.L2Reader` | `"C01"` … `"C16"`, already calibrated |

| Sector | `product` suffix | Size (2 km grid) | Cadence |
|---|---|---|---|
| Full disk | `F` (`ABI-L1b-RadF`, `ABI-L2-ACMF`) | 5424 × 5424 | 10 min |
| CONUS / PACUS | `C` | 1500 × 2500 | 5 min |
| Mesoscale | `M` (sectors `M1`, `M2`) | 500 × 500 | 1 min |

Channels keep their native resolution: C02 at 0.5 km; C01, C03 and C05 at
1 km; the rest at 2 km. L2 products have their own (cloud-top height
4 km, stability indices 10 km). `goes.BANDS` is the 16-channel table.

## Find and download

`goes.aws` lists and downloads unsigned through
[`geocloud.files`](../cloud/how-to/move-files.md), on the stack's shared
connection pool. File names parse offline:

```python
from geoproducts.goes import aws

item: aws.ABIFile = aws.parse_key(
    "ABI-L1b-RadC/2026/280/18/"
    "OR_ABI-L1b-RadC-M6C13_G19_s20262801801178_e20262801803551_c20262801803578.nc"
)                                                    # satellite 'G19' · channel 'C13' · start 18:01:17 UTC
bucket: str = aws.bucket("G19")                      # 'noaa-goes19'
```

```python
from datetime import datetime
from pathlib import Path

from geoproducts.goes import aws

files: list[aws.ABIFile] = aws.list_files(
    satellite="G19",                                 # GOES-East
    product="ABI-L1b-RadC",                          # CONUS sector
    start=datetime(2026, 10, 7, 18, 0),              # naive → UTC
    end=datetime(2026, 10, 7, 18, 10),
    channel=13,                                      # clean longwave window
)                                                    # 2 scans, 5 min apart, ~4.3 MB each
path: Path = aws.download(files[0], "data/goes")     # ranged reads to .part, then renamed
```

## Read and calibrate

```python
from datetime import datetime
from pathlib import Path

from georeader.geotensor import GeoTensor

from geoproducts import goes
from geoproducts.goes import aws

item: aws.ABIFile = aws.list_files(
    satellite="G19", product="ABI-L1b-RadC", channel=13,
    start=datetime(2026, 10, 7, 18, 0), end=datetime(2026, 10, 7, 18, 5),
)[0]
path: Path = aws.download(item, "data/goes")
reader: goes.Reader = goes.Reader(path, calibration="brightness_temperature")  # (1, 1500, 2500) · 2004 m

aoi: tuple[float, float, float, float] = (-100.0, 30.0, -95.0, 35.0)  # lon/lat box over east Texas
bt: GeoTensor = reader.read_from_bounds(aoi, crs_bounds="EPSG:4326")            # (1, 219, 266) float32 · K
flags: GeoTensor = reader.quality.read_from_bounds(aoi, crs_bounds="EPSG:4326")  # (1, 219, 266) uint8 · DQF
```

Every coefficient comes from the file being read:

| `calibration=` | Output | Formula |
|---|---|---|
| `"counts"` | `uint16` | the stored detector counts |
| `"radiance"` *(default)* | `float32` | `counts · scale_factor + add_offset` |
| `"reflectance"` | `float32` | `kappa0 · radiance` (C01–C06) |
| `"brightness_temperature"` | `float32`, K | `(fk2 / ln(fk1 / L + 1) − bc1) / bc2` (C07–C16) |

`reflectance` is NOAA's reflectance factor: it folds in the Earth–Sun
distance and band irradiance, not the solar zenith angle. Off-disk and
missing pixels are `NaN`; `reader.quality.flags()` maps `DQF` values to
their meanings.

## L2 products

`goes.L2Reader` reads any L2 product, one variable per band. Masks and
class codes stay integer; other variables decode to `float32` with `NaN`
fill. Pass `variables=` to choose bands.

```python
from datetime import datetime
from pathlib import Path

from geoproducts import goes
from geoproducts.goes import aws


def first(product: str) -> Path:
    """Download the first mesoscale-1 file of ``product`` at 18:00 UTC."""
    item: aws.ABIFile = aws.list_files(
        satellite="G19", product=product, sector="M1",
        start=datetime(2026, 10, 7, 18, 0), end=datetime(2026, 10, 7, 18, 2),
    )[0]
    return aws.download(item, "data/goes")


cmi: goes.L2Reader = goes.L2Reader(first("ABI-L2-MCMIPM"))   # (16, 500, 500) float32 · C01 … C16
acm: goes.L2Reader = goes.L2Reader(first("ABI-L2-ACMM"))     # (2, 500, 500) uint8 · BCM, ACM
acha: goes.L2Reader = goes.L2Reader(first("ABI-L2-ACHAM"))   # (1, 250, 250) float32 · HT (m), 4 km
codes: dict[int, str] = acm.flags("ACM")   # {0: 'clear', 1: 'probably_clear', 2: 'probably_cloudy', 3: 'cloudy'}
```

| Product | Default bands | Notes |
|---|---|---|
| `ACM` clear sky mask | `BCM`, `ACM` | `uint8`; binary and four-level cloud mask |
| `ACHA` / `ACHA2KM` cloud-top height | `HT` | metres |
| `LST` land surface temperature | `LST` | K |
| `FDC` fire detection | `Mask` | `int16` classes; `variables="Power"` for MW |
| `DSI` stability indices | `LI`, `CAPE`, `TT`, `SI`, `KI` | 10 km; quality in `DQF_Overall` |
| `MCMIP` / `CMIP` imagery | `C01` … `C16` / one channel | reflectance factor (C01–C06), K (C07–C16) |
| anything else (`TPW`, `AOD`, …) | every non-`DQF` variable | `available_variables` lists them |

## One grid

[`geoproducts.stack`](concepts.md#one-grid-stack) puts L1b channels at
different resolutions — or L1b and L2 — on one grid:

```python
from datetime import datetime

from georeader.geotensor import GeoTensor

import geoproducts
from geoproducts import goes
from geoproducts.goes import aws

channels: list[goes.Reader] = [
    goes.Reader(
        aws.download(
            aws.list_files(satellite="G19", product="ABI-L1b-RadC", channel=c,
                           start=datetime(2026, 10, 7, 18, 0), end=datetime(2026, 10, 7, 18, 5))[0],
            "data/goes",
        ),
        calibration="reflectance",
    )
    for c in (1, 2, 3)                                # 1 km, 0.5 km, 1 km
]
rgb_in: GeoTensor = geoproducts.stack(
    channels, bounds=(-100.0, 30.0, -95.0, 35.0), crs_bounds="EPSG:4326"
)                                                     # (3, 437, 530) float32 · C02 averaged to 1 km
```

## RGB recipes and presets

`goes.recipes` holds the operational RGBs as data; with `[operators]`
each preset is a [`gz.viz.RGBRecipe`](../operators/api/viz.md) (see
[Recipes and presets](concepts.md#recipes-and-presets)).

```python
from datetime import datetime

import numpy as np
from georeader.geotensor import GeoTensor

import geoproducts
import geotoolz as gz
from geoproducts import goes
from geoproducts.goes import aws


def m1(product: str) -> goes.L2Reader:
    """The first mesoscale-1 file of ``product`` at 18:00 UTC."""
    item: aws.ABIFile = aws.list_files(
        satellite="G19", product=product, sector="M1",
        start=datetime(2026, 10, 7, 18, 0), end=datetime(2026, 10, 7, 18, 2),
    )[0]
    return goes.L2Reader(aws.download(item, "data/goes"))


cmi: goes.L2Reader = m1("ABI-L2-MCMIPM")
scene: GeoTensor = geoproducts.stack([cmi, m1("ABI-L2-ACMM"), m1("ABI-L2-ACHAM")])  # (19, 500, 500) float32

true_color: GeoTensor = goes.TrueColor()(scene)       # (19, 500, 500) → (3, 500, 500) float32 in [0, 1]
cloudy: GeoTensor = goes.MaskClouds()(scene)          # (19, 500, 500) → (500, 500) bool · BCM cloudy
# HT is NaN under clear sky: mask only the imagery, or every clear pixel becomes nodata.
imagery: GeoTensor = gz.spectral.SelectBands(bands=[*goes.constants.CHANNELS, "BCM"])(scene)  # → (17, 500, 500)
clear_sky: GeoTensor = gz.mask.ApplyMask(mask=goes.MaskClouds())(imagery)  # (17, 500, 500) · clouds → NaN

# Parallax: move cloud tops back over the ground they sit above.
lonlat: GeoTensor = gz.geom.Reproject(dst_crs="EPSG:4326", resolution=(0.02, 0.02))(scene)  # → (19, 535, 624)
bt: GeoTensor = gz.spectral.SelectBands(bands=["C13"])(lonlat)      # (19, 535, 624) → (1, 535, 624)
height: GeoTensor = gz.spectral.SelectBands(bands=["HT"])(lonlat)   # (19, 535, 624) → (1, 535, 624) m · NaN = clear
moved: GeoTensor = goes.ParallaxCorrect(
    satellite_lon_deg=cmi.satellite_lon_deg,                        # -75.2 for GOES-East
    target_height_m=np.nan_to_num(np.asarray(height)[0]),           # clear sky → height 0
)(bt)                                                               # (1, 535, 624) float32
```

| Preset | Channels | Recipe |
|---|---|---|
| `goes.TrueColor()` | C01, C02, C03 | red C02 · CIMSS synthetic green · blue C01, γ 2.2 |
| `goes.NaturalColor()` | C02, C03, C05 | CIRA day land cloud |
| `goes.DayCloudPhase()` | C02, C05, C13 | CIRA day cloud phase distinction |
| `goes.FireTemperature()` | C05, C06, C07 | CIRA fire temperature |

`goes.presets.Recipe("…")` builds any entry of `goes.recipes.RECIPES`.
The other presets are `NDVI`, `SyntheticGreen`, `MaskClouds`
(`conservative=True` adds "probably clear") and `ParallaxCorrect`.

## Module layout

| Module | Contents |
|---|---|
| `goes.l1b` | `Reader` (counts / radiance / reflectance / BT), `QualityReader` (DQF) |
| `goes.l2` | `L2Reader`, `product_code` |
| `goes.aws` | `list_files`, `download`, `parse_key`, `ABIFile`, `bucket`, `url` |
| `goes.recipes` | `Recipe`, `RECIPES`, `TRUE_COLOR`, `NATURAL_COLOR`, `DAY_CLOUD_PHASE`, `FIRE_TEMPERATURE` |
| `goes.presets` | `NDVI`, `SyntheticGreen`, `MaskClouds`, `ParallaxCorrect`, `Recipe` and the four named recipes (`[operators]`) |
| `goes.constants` | `BANDS`, `CHANNELS`, `DQF_FLAGS`, ACM / BCM codes, `GOES_EAST_LON_DEG`, `GOES_WEST_LON_DEG` |
