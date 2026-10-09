# Read COGs

`geocloud.cog` reads windows of a Cloud-Optimized GeoTIFF straight from
the pool, without GDAL. A batch of windows fetches each tile it touches
once. It needs the `[cog]` extra (`pip install 'geotoolz-cloud[cog]'`).

## Read a batch of windows

`CogSource.open` fetches the header once. `read_windows` collects every
tile the windows touch, fetches each once in concurrent groups, and crops
each window out of the decoded tiles.

```python
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geocloud import credentials
from geocloud.cog import CogDomain, CogSource

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")

src: CogSource = CogSource.open(
    "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"
)
domain: CogDomain = src.domain        # shape (1, 10980, 10980) · EPSG:32610 · 10 m · nodata 0
chip: GeoTensor = src.read_window(Window(0, 0, 512, 512))  # (1, 512, 512) uint16
chips: list[GeoTensor] = src.read_windows(
    [Window(0, 0, 512, 512), Window(256, 0, 512, 512)]
)                                     # 2 × (1, 512, 512) uint16 · shared tiles fetched once
```

`aread_window` / `aread_windows` (and `CogSource.aopen`) are the
`await`-able versions. A source pickles by URL, so it ships to
process-pool workers and reopens there.

## Read like georeader, asynchronously

`AsyncCogReader` is the georeader-shaped face of a source: `GeoData`
metadata, lazy window views and `read_*` coroutines that mirror
`georeader.read` pixel for pixel.

```python
from georeader.geotensor import GeoTensor

from geocloud import credentials
from geocloud.cog import AsyncCogReader, read_from_bounds, read_from_tile

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")
uri: str = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"

# in a notebook / async function
reader: AsyncCogReader = await AsyncCogReader.open(uri)            # (1, 10980, 10980) uint16
tile: GeoTensor = await read_from_tile(reader, x=160, y=396, z=10)  # (1, 256, 256) uint16 · EPSG:3857
box: GeoTensor | None = await read_from_bounds(
    reader, (-123.0, 37.5, -122.95, 37.55), crs_bounds="EPSG:4326"
)                                                                  # (1, 555, 442) uint16 · UTM 10N
coarse: AsyncCogReader = reader.reader_overview(2)                 # (1, 1373, 1373) — 8× overview
```

Use an overview for low zoom levels: `reader.overviews()` lists the
factors (`[2, 4, 8, 16]` here). The other mirrors are `read_from_window`, `read_from_polygon`,
`read_from_center_coords`, `read_reproject`, `read_reproject_like` and
`read_to_crs`.

## Patch a COG

[`geopatcher.fields.CogField`](../../patcher/api/fields.md) is `CogSource`
with the patcher's `Field` interface; install `geotoolz-patcher[cog]`.

## Pitfalls

- **Lossless codecs** decode bit-for-bit like GDAL.
- **JPEG tiles** are decoded by async-tiff, not libjpeg, so values can
  differ by a few DN: up to ±3 for `PHOTOMETRIC=YCbCr`.
- **Timeouts.** `open` waits 120 s by default for the header
  (`timeout=`).
