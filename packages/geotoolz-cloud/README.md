# geotoolz-cloud

Cloud object storage for the geotoolz stack (import name `geocloud`): one
process-wide [obstore](https://developmentseed.org/obstore/) client pool,
file verbs on top of it (list, download, upload, copy, sync, sign),
credentials registered once per bucket, and batched, async
Cloud-Optimized GeoTIFF reads.

geopatcher's `CogField` and geoproducts' cloud byte reads take their client
from `geocloud.store`, so a process talking to one bucket through them builds
**one** client and **one** HTTP/2 connection pool for it. (geocatalog does not
use the pool: staging downloads through fsspec and the builders read through
rasterio / GDAL.) `geocloud.cog` reads COG tiles straight
from that pool with async-geotiff: no GDAL, every tile a batch of windows
touches fetched once, groups of tiles fetched concurrently.

## Install

```bash
pip install geotoolz-cloud            # the obstore client pool
pip install 'geotoolz-cloud[cog]'     # + COG reads (async-geotiff)
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | obstore, georeader, rasterio, numpy | `geocloud.store`, `geocloud.files`, `geocloud.credentials` |
| `[cog]` | async-geotiff | `geocloud.cog` (`CogSource`, `AsyncCogReader`, `read_*`) |

## The client pool — `geocloud.store`

```python
from obstore.store import ObjectStore

from geocloud.store import get_obstore, object_key

uri = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2026/10/S2A_10SDG_20261007_0_L2A/B04.tif"
store: ObjectStore = get_obstore(uri)          # built once per bucket, then reused
head: bytes = bytes(store.get_range(object_key(uri), start=0, length=16_384))
```

`s3://`, `gs://`, `az://` / `abfs[s]://`, `http(s)://` (pre-signed URLs and
SAS tokens included) and `hf://` (Hugging Face Hub) URIs are supported.
`storage_options` are part of the pool key; `clear_obstore_pool()` drops every
client (it also runs automatically in a forked child).

## Moving files — `geocloud.files`

```python
from pathlib import Path

from geocloud import files

scenes: list[files.ObjectInfo] = files.ls("s3://bucket/scenes/2026/10/")
local: Path = files.download(scenes[0].uri, "data/")         # streamed, atomic
files.upload("out/ndvi.tif", "az://account/results/ndvi/")   # multipart
files.copy("s3://bucket/a.tif", "gs://other/a.tif")          # across clouds
files.sync("s3://bucket/scenes/", "data/scenes/")            # resumable mirror
url: str = files.sign("s3://bucket/a.tif")                   # pre-signed HTTPS
```

Any URI the pool understands or a local path works on either side; a copy
inside one bucket is server-side, and remote ends share the pooled clients.

## Credentials — `geocloud.credentials`

```python
from geocloud import credentials

credentials.set_credentials("s3://noaa-goes19", anonymous=True, region="us-east-1")
credentials.set_credentials("az://myaccount/raw", sas_token=sas)   # expiry + scope checked
credentials.load_credentials("team-credentials.toml")                   # `${VAR}` reads the env

path, env = credentials.gdal_access("az://myaccount/raw/scene.tif")  # for rasterio
```

Registered once per bucket / container / host, picked up by every call on
the pool (and by GDAL through `gdal_access`); nothing is written to
`os.environ`. `~/.config/geocloud/credentials.toml` (or
`$GEOCLOUD_CREDENTIALS`) loads by itself on first use.

## COG reads — `geocloud.cog`

```python
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geocloud.cog import AsyncCogReader, CogSource, read_from_tile

src: CogSource = CogSource.open(uri)                    # one header fetch
src.domain.shape                                        # (1, 10980, 10980)
chips: list[GeoTensor] = src.read_windows(              # tiles shared by the
    [Window(0, 0, 512, 512), Window(256, 0, 512, 512)]  # windows fetched once
)                                                       # 2 × (1, 512, 512) uint16

reader: AsyncCogReader = await AsyncCogReader.open(uri)  # georeader GeoData face
tile: GeoTensor = await read_from_tile(reader, x=163, y=395, z=10)  # (1, 256, 256)
```

`CogSource` pickles by URL, so it ships to process-pool workers;
`AsyncCogReader`'s `read_*` coroutines mirror `georeader.read` pixel for
pixel. Lossless codecs decode bit-for-bit identical to GDAL.

## Next steps

- **Docs:** [geotoolz-cloud](https://jejjohnson.github.io/geotoolz/cloud/).
- **Patching a COG:** `geopatcher.fields.CogField` is `CogSource` as a
  geopatcher `Field` (`pip install 'geotoolz-patcher[cog]'`).

## License

MIT — see [LICENSE](../../LICENSE).
