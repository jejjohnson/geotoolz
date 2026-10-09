# Cache remote files locally

Get a complete local file for any URI with `geocloud.cache.localize`. The
first call downloads it through `geocloud.files`; later calls return the
cached copy. A local path comes back in place, so code can call it on
any location.

Use it for readers that need a real file on disk: HDF4 (`pyhdf`),
memory-mapped binaries such as Himawari HSD, and GDAL drivers without
`/vsi*/` support. For ranged reads without a copy, use
[`geocloud.files.open`](move-files.md) or [`geocloud.fs`](fsspec.md).

## Localize a file

```python
import tempfile
from pathlib import Path

from obstore.store import MemoryStore

from geocloud import files
from geocloud.cache import LocalCache, localize
from geocloud.store import mount

mount("s3://demo-cache", MemoryStore())                      # a stand-in bucket
files.write_bytes("s3://demo-cache/scene.hdf", b"\x0e\x03\x13\x01")

cache: LocalCache = LocalCache(root=tempfile.mkdtemp())
path: Path = localize("s3://demo-cache/scene.hdf", cache=cache)  # downloaded
again: Path = cache.fetch("s3://demo-cache/scene.hdf")           # cache hit, no request
assert path == again and path.read_bytes() == b"\x0e\x03\x13\x01"
```

## Tune the cache

| Argument | Default | Effect |
| --- | --- | --- |
| `root` | `$GEOCLOUD_CACHE`, else `~/.cache/geocloud` | the cache directory |
| `ttl_days` | `None` (forever) | older copies are fetched again; `prune()` deletes them |
| `timeout` | `60.0` s | per request; a download moves in 16 MiB ranges |

Files land at `{root}/{key[:2]}/{key}{ext}`. The key is the SHA-256 of
`cache_key(uri)`: the URI without its expiring signature (Azure SAS,
`X-Amz-*`, `X-Goog-*`, CloudFront), so a re-signed URL hits the same file.
Downloads go to a hidden `.part` file renamed into place, so a slot only
ever holds a complete file.

## Pitfalls

- **Keyed by location, not content.** Two URIs with the same bytes get
  two slots; a replaced object keeps its stale copy until `ttl_days`.
- **`prune()`** removes expired copies and abandoned `*.part` downloads;
  run it from a maintenance job, not inside a reader.
- **Catalogs** stage every row through the same cache with
  [`geocatalog.staging.stage`](../../catalog/how-to/staging.md).
