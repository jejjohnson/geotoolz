"""`geocloud.files` — list, read, write and move objects by URI.

One set of verbs for every location the stack touches: ``s3://``,
``gs://``, ``az://`` / ``abfs[s]://``, Azure ``https://``, signed
``http(s)://``, ``hf://`` and local paths. Remote objects go through the
`geocloud.store` client pool, so files moved here share connections with
every COG read and product download in the process.

```python
from pathlib import Path

from geocloud import files

scenes: list[files.ObjectInfo] = files.ls("s3://bucket/scenes/2026/10/")
local: Path = files.download(scenes[0].uri, "data/")         # atomic, streamed
files.upload("out/ndvi.tif", "az://account/results/ndvi/")   # multipart
files.copy("s3://bucket/a.tif", "gs://other/a.tif")          # streamed across clouds
files.sync("s3://bucket/scenes/", "data/scenes/")            # resumable mirror
url: str = files.sign("s3://bucket/a.tif")                   # pre-signed HTTPS
```
"""

from __future__ import annotations

from geocloud._src.files import (
    ObjectInfo,
    copy,
    download,
    exists,
    info,
    ls,
    open,
    read_bytes,
    rm,
    sign,
    sync,
    upload,
    write_bytes,
)


__all__ = [
    "ObjectInfo",
    "copy",
    "download",
    "exists",
    "info",
    "ls",
    "open",
    "read_bytes",
    "rm",
    "sign",
    "sync",
    "upload",
    "write_bytes",
]
