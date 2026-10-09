# Move files

`geocloud.files` lists, reads, downloads, uploads, copies, syncs, deletes
and pre-signs objects with one set of verbs. Either end can be any URI
the [pool](../concepts.md#store-roots) understands or a local path.

## List a public bucket

```python
from geocloud import credentials, files

credentials.set_credentials("s3://noaa-goes19", anonymous=True, region="us-east-1")

hour: str = "s3://noaa-goes19/ABI-L1b-RadC/2026/280/18/"         # day 280 = 7 Oct, 18 UTC
scans: list[files.ObjectInfo] = files.ls(hour)                   # 192 objects, sorted by URI
top: list[files.ObjectInfo] = files.ls("s3://noaa-goes19/ABI-L1b-RadC/2026/", recursive=False)
size: int = scans[0].size                                        # 12_876_810 bytes
```

`recursive=False` lists one level, with sub-"directories" as `is_dir`
entries. Prefixes match whole segments: `2026` does not match `2026-old`.

## Download, upload, copy

The example runs offline: it mounts a `MemoryStore` at a bucket root (see
[Mounts](../concepts.md#mounts)) and works in a temporary directory.

```python
import tempfile
from pathlib import Path

from obstore.store import MemoryStore

from geocloud import files
from geocloud.store import mount

mount("s3://demo-bucket", MemoryStore())
work: Path = Path(tempfile.mkdtemp())
(work / "ndvi.tif").write_bytes(b"\x00" * 1024)

files.upload(work / "ndvi.tif", "s3://demo-bucket/results/")     # keeps the name
files.copy("s3://demo-bucket/results/ndvi.tif", "s3://demo-bucket/archive/")  # server-side
local: Path = files.download("s3://demo-bucket/archive/ndvi.tif", work / "data/")
files.sync("s3://demo-bucket/results/", work / "mirror/")         # skips same-size files
there: bool = files.exists("s3://demo-bucket/archive/ndvi.tif")   # True
meta: files.ObjectInfo = files.info("s3://demo-bucket/archive/ndvi.tif")  # size=1024
files.rm("s3://demo-bucket/results/", recursive=True)
```

A trailing `/` on a destination, or an existing local directory, keeps
the source's file name. `overwrite=False` on `copy`, `download` and
`upload` leaves an existing destination alone and returns it.

## Small objects and file handles

```python
from __future__ import annotations

import obstore
from obstore.store import MemoryStore

from geocloud import files
from geocloud.store import mount

mount("s3://demo-bucket", MemoryStore())

files.write_bytes("s3://demo-bucket/runs/run.json", b'{"ok": true}')
data: bytes = files.read_bytes("s3://demo-bucket/runs/run.json")  # b'{"ok": true}'
with files.open("s3://demo-bucket/runs/log.txt", "wb") as fh:
    fh.write(b"done")
reader: obstore.ReadableFile = files.open("s3://demo-bucket/runs/log.txt", "rb")  # seekable
first: bytes = bytes(reader.read(4))                             # b'done'
reader.close()
```

A `"wb"` handle is a context manager; a `"rb"` handle is not, so close
it yourself. Hand a `"rb"` handle to h5py or `zipfile` to read only the
ranges they ask for.

## Share without credentials

`files.sign` computes a pre-signed HTTPS URL locally, with no request.
The store must hold real credentials, so this sketch reads them from the
environment:

```python
import os
from datetime import timedelta

from geocloud import credentials, files

credentials.set_credentials(
    "s3://my-results",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region="eu-west-1",
)
url: str = files.sign("s3://my-results/ndvi.tif", expires=timedelta(hours=6))
```

S3, GCS and Azure sign; local paths, plain `https://` and `hf://` raise.

## Verbs

| Verb | Does | Notes |
| --- | --- | --- |
| `ls(prefix, recursive=True)` | list objects (sorted `ObjectInfo`) | whole-segment prefixes |
| `info` / `exists` | one object's size, mtime, etag | a prefix alone does not "exist" |
| `read_bytes` / `write_bytes` | whole object in memory | writes are atomic in the store |
| `open(uri, "rb" \| "wb")` | seekable reader / buffered writer | range requests |
| `download(uri, dest)` | object → local file | 16 MiB ranged reads into a hidden `.part`, size-checked, renamed |
| `upload(path, uri)` | local file → object | multipart for large files |
| `copy(src, dst)` | any → any | server-side in one bucket; ranged reads into a multipart upload between stores |
| `sync(src, dst)` | mirror a prefix / directory | skips same-size objects; never deletes |
| `rm(uri, recursive=False)` | delete | batched bulk deletes |
| `sign(uri, expires=…)` | pre-signed HTTPS URL | computed locally |

## Pitfalls

- **Two ends, two grants.** `storage_options` on a verb go to both ends.
  For a `copy` or `sync` across accounts, register each root with
  [`geocloud.credentials`](credentials.md).
- **Large objects move in 16 MiB ranges.** obstore's 30 s timeout bounds
  one range, not the whole object.
- **Replaced mid-transfer.** Every range is pinned to the first entity tag,
  so an object replaced during a transfer raises rather than mixing
  versions.
