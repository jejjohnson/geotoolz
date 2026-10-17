# Read COGs

`geocloud.cog` reads windows of a Cloud-Optimized GeoTIFF straight from
the pool, without GDAL, as `geoarray.GeoArray` chips. A batch of windows
fetches each tile it touches once. It needs the `[cog]` extra
(`pip install 'geotoolz-cloud[cog]'`).

## Read a batch of windows

`CogSource.open` fetches the header once. `read_windows` collects every
tile the windows touch, fetches each once in concurrent groups, and crops
each window out of the decoded tiles.

```python
from geoarray import GeoArray, Grid, PixelWindow

from geocloud import credentials
from geocloud.cog import CogDomain, CogSource

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")

src: CogSource = CogSource.open(
    "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"
)
domain: CogDomain = src.domain        # shape (1, 10980, 10980) · EPSG:32610 · 10 m · nodata 0
grid: Grid = src.grid                 # (10980, 10980) · the same georeferencing, hashable
window: PixelWindow = PixelWindow(col_off=0, row_off=0, width=512, height=512)
chip: GeoArray = src.read_window(window)   # (1, 512, 512) uint16 · fill_value = nodata
chips: list[GeoArray] = src.read_windows([
    window,
    PixelWindow(col_off=256, row_off=0, width=512, height=512),
])                                    # 2 × (1, 512, 512) uint16 · shared tiles fetched once
edge: GeoArray = src.read_window(
    PixelWindow(col_off=10800, row_off=0, width=512, height=512), boundless=False
)                                     # (1, 512, 180) uint16 · clipped to the image
```

A window may also be a `rasterio.windows.Window`; fractional windows snap
outward to whole pixels. With the default `boundless=True` a window past
the image comes back whole, its outside pixels set to `fill_value`.
`aread_window` / `aread_windows` (and `CogSource.aopen`) are the
`await`-able versions. A source pickles by URL, so it ships to
process-pool workers and reopens there.

## Use a COG wherever a raster source goes

A `CogSource` is a `geoarray.RasterSource` (`grid`, `shape`, `dtype`,
`fill_value`, `attrs`, `read_window`, `load`) and an
`geoarray.AsyncRasterSource` (`aread_window`, `aload`). Code written for
any lazy raster reads it window by window — `write_cog` stages it strip by
strip, for one.

```python
from geoarray import AsyncRasterSource, GeoArray, PixelWindow, RasterSource

from geocloud import credentials
from geocloud.cog import CogSource

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")


def corner(source: RasterSource) -> GeoArray:
    window = PixelWindow(col_off=0, row_off=0, width=256, height=256)
    return source.read_window(window)                    # (..., 256, 256)


src: CogSource = CogSource.open(
    "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"
)
assert isinstance(src, RasterSource) and isinstance(src, AsyncRasterSource)
chip: GeoArray = corner(src)                             # (1, 256, 256) uint16
```

## Read like georeader, asynchronously

`AsyncCogReader` is the `await`-native face of a source: a
`geoarray.AsyncRasterSource` (`grid`, `aread_window`, `aload`, …) that
keeps georeader's metadata surface (an `affine.Affine` `transform`,
`bounds`, `footprint`), lazy window views and `read_*` coroutines that
mirror `georeader.read` pixel for pixel. georeader does the window maths
and the warping; every result is a `GeoArray`.

```python
from geoarray import GeoArray

from geocloud import credentials
from geocloud.cog import AsyncCogReader, read_from_bounds, read_from_tile

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")
uri: str = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"

# in a notebook / async function
reader: AsyncCogReader = await AsyncCogReader.open(uri)            # (1, 10980, 10980) uint16
tile: GeoArray = await read_from_tile(reader, x=160, y=396, z=10)  # (1, 256, 256) uint16 · EPSG:3857
box: GeoArray | None = await read_from_bounds(
    reader, (-123.0, 37.5, -122.95, 37.55), crs_bounds="EPSG:4326"
)                                                                  # (1, 555, 442) uint16 · UTM 10N
coarse: AsyncCogReader = reader.reader_overview(2)                 # (1, 1373, 1373) — 8× overview
```

Use an overview for low zoom levels: `reader.overviews()` lists the
factors (`[2, 4, 8, 16]` here). The other mirrors are `read_from_window`, `read_from_polygon`,
`read_from_center_coords`, `read_reproject`, `read_reproject_like` (onto
the grid of any `GeoArray` or raster source) and `read_to_crs`.

## Patch a COG

[`geopatcher.fields.CogField`](../../patcher/api/fields.md) is `CogSource`
with the patcher's `Field` interface; install `geotoolz-patcher[cog]`.

## Pitfalls

- **Lossless codecs** decode bit-for-bit like GDAL.
- **JPEG tiles** are decoded by async-tiff, not libjpeg, so values can
  differ by a few DN: up to ±3 for `PHOTOMETRIC=YCbCr`.
- **Timeouts.** `open` waits 120 s by default for the header
  (`timeout=`).
- **No nodata declared** means `fill_value` is `0`: pixels outside the
  image (and masked ones) read as `0` and count as nodata in
  `GeoArray.valid_mask`, as in a masked GDAL read.
- **A georeader `GeoTensor`** passed as `read_reproject_like`'s
  `data_like` raises `TypeError`; convert it with
  `geoarray.interop.georeader.from_georeader`.
