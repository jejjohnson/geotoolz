# Concepts

geocloud keeps one object-storage client per store root and lets every
package share it. Four ideas explain how: the pool, store roots, mounts
and credential precedence.

## The pool

`geocloud.store.get_obstore(uri)` returns the client for a URI. The first
call for a bucket builds an obstore store; later calls get the same
instance, so the HTTP/2 connections and TLS sessions are reused.
`object_key(uri)` gives the key to request from that client.

```python
from obstore.store import ObjectStore

from geocloud import credentials
from geocloud.store import get_obstore, object_key

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")

uri: str = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"
store: ObjectStore = get_obstore(uri)                     # built once per bucket, then reused
same: bool = get_obstore(uri.replace("B04", "B08")) is store   # True
head: bytes = bytes(store.get_range(object_key(uri), start=0, length=16_384))  # the TIFF header
```

- **Bounded.** The pool is an LRU of 64 clients by default
  (`set_obstore_pool_maxsize`).
- **Keyed by options.** `storage_options` passed to a call are part of the
  key, so different options give a different client.
- **Fork-safe.** `clear_obstore_pool()` drops every client. It runs by
  itself in a forked child; call it after rotating credentials.

## Store roots

A **store root** is what one client serves: an S3 or GCS bucket, an Azure
container, or an HTTP origin. Pooled stores never carry a prefix, so the
key is the full path inside the root. `geocloud.store.SUPPORTED_SCHEMES`
lists the schemes.

| URI | Pooled store | `object_key` |
| --- | --- | --- |
| `s3://bucket/key`, `gs://bucket/key` | `S3Store` / `GCSStore(bucket)` | `key` |
| `az://account/container/key` | `AzureStore(container, account)` | `key` |
| `abfs[s]://container@account.dfs.core.windows.net/key` | `AzureStore(container, account)` | `key` |
| `https://account.blob.core.windows.net/container/key` | `AzureStore(container, account)` | `key` |
| `http[s]://host/path?query` (pre-signed / SAS) | `HTTPStore(origin?query)` | `path` |
| `hf://[datasets/]org/repo[@rev]/path` | `HTTPStore` on the Hub (token from `$HF_TOKEN`) | `…/resolve/<rev>/path` |
| a local path, `C:\data\x.tif`, `file:///data/x.tif` | `LocalStore` at the filesystem anchor (`/`, `C:\`) | the absolute path below it |

An `http(s)` query string stays on the store, so signed URLs are
requested signed. Each distinct signature therefore gets its own client.

Local paths go through the pool too, so `geocloud.files`, `CogSource` and
`get_range_bytes` read a local file and a bucket the same way.
`geocloud.store.local_path` is the one rule that tells them apart: a
`Path`, a string without `scheme://` and a `file://` URI are local.
`SUPPORTED_SCHEMES` lists only the remote schemes.

```python
from pathlib import Path

from geocloud.store import local_path

path: Path | None = local_path("file:///data/a%20b.tif")  # Path('/data/a b.tif')
remote: Path | None = local_path("s3://bucket/a.tif")       # None
```

## Mounts

`geocloud.store.mount(root, store)` serves one root from a store you
built. Code under test keeps its real URIs, and every package on the pool
sees the stand-in.

```python
from obstore.store import MemoryStore

from geocloud import files
from geocloud.store import mount, unmount

mount("s3://demo-bucket", MemoryStore())                  # every s3://demo-bucket/… hits memory
files.write_bytes("s3://demo-bucket/runs/1.json", b"{}")
listed: list[files.ObjectInfo] = files.ls("s3://demo-bucket/")  # [ObjectInfo('s3://demo-bucket/runs/1.json', 2)]
unmount("s3://demo-bucket")
```

A mount wins over the pool and over any `storage_options`.
`clear_obstore_pool` leaves mounts in place.

## Credential precedence

Credentials are registered per store root with
`geocloud.credentials.set_credentials` (or a TOML file), and
`get_obstore` merges them into every call. When several apply, the most
specific wins:

1. **Call over registry.** `storage_options` passed to a call win over the
   registered ones.
2. **Container over account.** An Azure container's entry wins over its
   account's entry.
3. **`client_options` merge one level deep.** A call's `timeout` keeps the
   registered `default_headers`, and the reverse.

With nothing registered, obstore's default chains apply: environment
variables, instance metadata, Azure managed identity. geocloud never
writes `os.environ`. The [credentials how-to](how-to/credentials.md) has
the recipes.
