# Use fsspec libraries on the pool

Hand xarray, zarr, pyarrow or geopandas a filesystem whose every read and
write goes through `geocloud.store`. They get the pooled clients and the
credentials registered in `geocloud.credentials`, with no s3fs, gcsfs or
adlfs and no second set of credentials. Install the `[fsspec]` extra.

```bash
pip install 'geotoolz-cloud[fsspec]'
```

## Open a file

`geocloud.fs.filesystem()` returns a `GeoCloudFileSystem`. Its `open`
gives a seekable file whose reads are ranged GETs, cached in blocks.

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from obstore.store import MemoryStore

from geocloud.fs import GeoCloudFileSystem, filesystem
from geocloud.store import mount

mount("s3://demo", MemoryStore())                          # a stand-in bucket
fs: GeoCloudFileSystem = filesystem()

with fs.open("s3://demo/table.parquet", "wb") as fh:      # the object appears on close
    pq.write_table(pa.table({"band": np.arange(3)}), fh)
with fs.open("s3://demo/table.parquet") as fh:            # ranged reads, pooled client
    table: pa.Table = pq.read_table(fh)                   # 3 rows
```

Paths are full locations in any form the pool accepts: `s3://`, `gs://`,
`az://account/container/…`, signed `https://` URLs, `hf://` and local
paths. The filesystem keeps them as written.

## Open a Zarr store

Zarr is a tree of keys, not one file, so hand it a mapper.

```python
import numpy as np
import xarray as xr
from obstore.store import MemoryStore

from geocloud.fs import GeoCloudFileSystem, filesystem
from geocloud.store import mount

mount("s3://demo-zarr", MemoryStore())
fs: GeoCloudFileSystem = filesystem()

cube: xr.Dataset = xr.Dataset({"t2m": (("y", "x"), np.zeros((4, 5)))})  # (4, 5) float64
cube.to_zarr(fs.get_mapper("s3://demo-zarr/cube.zarr"), mode="w")
back: xr.Dataset = xr.open_zarr(fs.get_mapper("s3://demo-zarr/cube.zarr"))
```

## Pass store options

`filesystem(storage_options)` forwards them to
`geocloud.store.get_obstore` for every remote path, over the registered
credentials. They are obstore options, not s3fs ones.

```python
from geocloud.fs import GeoCloudFileSystem, filesystem

public: GeoCloudFileSystem = filesystem({"skip_signature": True})  # unsigned requests
```

## Pitfalls

- **Register credentials once** with `geocloud.credentials.set_credentials`
  rather than per call; see [Register credentials](credentials.md).
- **The filesystem is not registered under a protocol.** `fsspec.open("s3://…")`
  still looks for s3fs; pass `fs.open(…)` or `filesystem=fs` instead.
- **Listing** works where `geocloud.files.ls` does: S3, GCS, Azure and local
  directories, not plain `https://` or `hf://`.
