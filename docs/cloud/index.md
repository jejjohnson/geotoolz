# geotoolz-cloud

> **One object-storage client per bucket, shared by the whole stack.**
> List, move and sign files by URI, register credentials once, and read
> or write Cloud-Optimized GeoTIFFs — imported as `geocloud`.

## Where it comes from

A pipeline touches object storage in many places. A catalog stages
files, a reader fetches byte ranges, a patcher reads COG windows, and you
upload the result. When each builds its own client, one process holds a
dozen connection pools to the same bucket, and credentials leak into
every call and error message.

`geocloud` gives the process **one pool**. `geocloud.store.get_obstore`
builds an [obstore](https://developmentseed.org/obstore/) client the first
time a bucket is touched, then hands every later caller the same one. Credentials are registered once per store
root in `geocloud.credentials`, so code downstream handles nothing but
URIs. `geocloud.files` (whole-object verbs) and `geocloud.cog` (COG reads
and writes) sit on that pool, and the other packages build on them.

![geocloud's layers: geopatcher's CogField, geoproducts' readers and bucket helpers, geocatalog's stage and your code call geocloud.files and geocloud.cog; both take their client from the geocloud.store pool, which merges the grants in geocloud.credentials (also handed to GDAL by gdal_access) and keeps one client per s3, gs, Azure, https or hf root](../assets/diagrams/cloud-pool.png)

## Install

```bash
pip install geotoolz-cloud            # the client pool, file verbs, credentials, write_cog
pip install 'geotoolz-cloud[cog]'     # + COG reads (async-geotiff)
pip install 'geotoolz-cloud[fsspec]'  # + geocloud.fs, for xarray / zarr / pyarrow
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | obstore, georeader, rasterio, numpy | `geocloud.store`, `geocloud.files`, `geocloud.credentials`, `geocloud.cog.write_cog` |
| `[cog]` | async-geotiff | `geocloud.cog.CogSource`, `AsyncCogReader` and the async `read_*` functions |
| `[fsspec]` | fsspec (no s3fs / gcsfs / adlfs) | `geocloud.fs`, the fsspec filesystem on the pool |
| `[hdf5]` / `[netcdf]` / `[hdf4]` | h5py / netCDF4 / pyhdf | `geocloud.hdf` readers for HDF5 + NetCDF-4 / NetCDF-3 / HDF4 |

The other packages reach geocloud through their own extras; see
[How the packages interlock](../geostack.md).

## Quickstart

List a Sentinel-2 scene in the public `sentinel-cogs` bucket, then read
two overlapping windows of its red band. Their shared tiles are fetched
once.

```python
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geocloud import credentials, files
from geocloud.cog import CogSource

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")

scene: str = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/"
objects: list[files.ObjectInfo] = files.ls(scene)        # B01.tif … TCI.tif, metadata
src: CogSource = CogSource.open(scene + "B04.tif")       # one header fetch
src.domain.shape                                          # (1, 10980, 10980) · EPSG:32610, 10 m
chips: list[GeoTensor] = src.read_windows(
    [Window(0, 0, 512, 512), Window(256, 0, 512, 512)]
)                                                         # 2 × (1, 512, 512) uint16
```

## What's inside

| Namespace | Use it to… | Guide |
|---|---|---|
| `geocloud.store` | get the pooled client for a URI or local path (`get_obstore`, `object_key`), tell local from remote (`local_path`), mount a test store | [Concepts](concepts.md) |
| `geocloud.files` | list, read, download, upload, copy, sync, delete and pre-sign objects | [Move files](how-to/move-files.md) |
| `geocloud.credentials` | register credentials per bucket / container / host, load a TOML file, hand them to GDAL, redact logs | [Credentials](how-to/credentials.md) |
| `geocloud.hdf` | read HDF5, NetCDF and HDF4 from any location; decode CF packing | [Read HDF and NetCDF](how-to/hdf.md) |
| `geocloud.cache` | get a local file for any URI, downloaded once (`localize`, `LocalCache`) | [Cache remote files](how-to/cache.md) |
| `geocloud.fs` | hand xarray, zarr, pyarrow or geopandas an fsspec filesystem on the pool | [Use fsspec libraries](how-to/fsspec.md) |
| `geocloud.cog` | read COG windows in batches, sync or async (`CogSource`, `AsyncCogReader`) | [Read COGs](how-to/read-cogs.md) |
| `geocloud.cog.write_cog` | write a validated COG locally or to a bucket | [Write COGs](how-to/write-cogs.md) |

## Advanced — one pool across packages

Every package asks the same pool, so geocatalog staging, geopatcher
fields and geoproducts downloads share **one** client per bucket.
Register the bucket once and all of them use it:

```python
import geopatcher as gp
from geocloud import credentials

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")

field: gp.fields.CogField = gp.fields.CogField.open(
    "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"
)                                                         # the pooled client for s3://sentinel-cogs
field.domain.shape                                        # (1, 10980, 10980)
```

## Next steps

- [Concepts](concepts.md) — the pool, store roots, `mount` and credential precedence.
- How-to: [move files](how-to/move-files.md) · [credentials](how-to/credentials.md) ·
  [read COGs](how-to/read-cogs.md) · [write COGs](how-to/write-cogs.md).
- Tutorial: [a tour of geocloud](notebooks/geocloud_tour.ipynb) — register buckets, read COG windows, write and validate a COG, test offline.
- [API reference](api.md).
- To patch a COG, [`geopatcher.fields.CogField`](../patcher/api/fields.md) is
  `CogSource` with the `Field` interface.
