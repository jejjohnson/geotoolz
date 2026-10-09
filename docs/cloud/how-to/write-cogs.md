# Write COGs

`geocloud.cog.write_cog` writes a `GeoTensor`, or a lazy georeader
reader, as a validated COG to a local path or any bucket. It runs on the
base install (GDAL's `COG` driver through rasterio). In a pipeline, use
the operator `gz.WriteCOG`.

## Write a float, a class map and a mask

The example writes small synthetic rasters to an in-memory bucket (see
[Mounts](../concepts.md#mounts)) and a temporary directory. It runs
offline.

```python
import tempfile
from pathlib import Path

import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor
from obstore.store import MemoryStore

from geocloud import files
from geocloud.cog import write_cog
from geocloud.store import mount

mount("s3://demo-bucket", MemoryStore())
rng: np.random.Generator = np.random.default_rng(0)
grid: Affine = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_200_000.0)   # 10 m, UTM 10N

ndvi: GeoTensor = GeoTensor(
    rng.uniform(-1, 1, (1, 1024, 1024)).astype(np.float32),
    transform=grid, crs="EPSG:32610", fill_value_default=np.nan,
)                                                    # (1, 1024, 1024) float32 · NaN = no data
classes: GeoTensor = GeoTensor(
    rng.integers(0, 5, (1024, 1024), dtype=np.uint8),
    transform=grid, crs="EPSG:32610", fill_value_default=0,
)                                                    # (1024, 1024) uint8 · 0 = no data
mask: GeoTensor = GeoTensor(
    np.asarray(ndvi)[0] < 0, transform=grid, crs="EPSG:32610", fill_value_default=False,
)                                                    # (1024, 1024) bool

out: Path = Path(tempfile.mkdtemp())
write_cog(ndvi, "s3://demo-bucket/products/ndvi.tif")      # deflate + float predictor, average overviews
write_cog(classes, out / "landcover.tif", compress="zstd")  # nearest overviews keep class values
write_cog(mask, out / "water.tif")                          # bool → uint8 0 / 1, no nodata
size: int = files.info("s3://demo-bucket/products/ndvi.tif").size  # ≈ 4.7 MB
```

A lazy reader (a georeader `RasterioReader`, a geoproducts reader) is
staged strip by strip, so a scene larger than memory never loads whole.

## How a write runs

1. **Stage.** The pixels go to a tiled GeoTIFF in a private temporary
   directory, with nodata, band descriptions and tags.
2. **Translate.** GDAL's `COG` driver lays out the tiles, builds the
   overviews and puts the header first.
3. **Check.** With `validate=True` (the default), the result must read
   back as `LAYOUT=COG` with the expected shape, dtype, tiles and
   overviews.
4. **Place.** A local file is renamed into place, so a failed write never
   leaves a truncated file. A cloud `dest` is uploaded through
   [`geocloud.files`](move-files.md) with that bucket's
   [credentials](credentials.md).

## Options

| Keyword | Default | Notes |
| --- | --- | --- |
| `compress` | `"deflate"` | also `zstd`, `lzw`, `lerc*`, `webp` / `jpeg` (8-bit), `none` |
| `level` | GDAL's | DEFLATE 1–12, ZSTD 1–22 |
| `predictor` | `True` | GDAL picks horizontal (ints) or floating-point (floats) for DEFLATE / LZW / ZSTD |
| `blocksize` | `512` | a power of two |
| `overviews` | `True` | down to one tile |
| `resampling` | auto | `nearest` for integer / bool data, `average` for floats |
| `nodata` | `"auto"` | the data's `fill_value_default` if the dtype can hold it; never for a bool mask |
| `descriptions` | auto | from `attrs["band_names"]` (or `descriptions`) when it has one entry per band |
| `tags`, `creation_options` | — | dataset tags; raw GDAL `COG` options that win over the keywords |
| `overwrite`, `validate` | `True` | `overwrite=False` raises `FileExistsError` |

## Compared with `save_cog`

`georeader.save.save_cog` also writes COGs. `write_cog` differs where a
pipeline needs it to:

| | `save_cog` | `write_cog` |
| --- | --- | --- |
| Options | a free-form profile dict, modified in place | explicit keywords with checked values |
| Defaults | LZW, no predictor, cubic-spline overviews for every dtype | values that suit the dtype |
| Validation | none | the output must be a COG |
| Local write | non-atomic | atomic |
| Cloud write | fsspec, via a temp file in the current directory | the shared pool and the credentials registry |
| Lazy readers | no | yes |
| Unsupported dtypes (bool, float16) | fail | converted (uint8, float32) |
| A nodata the dtype cannot hold | written as is | dropped (`"auto"`) or rejected (explicit value) |
