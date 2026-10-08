"""`geocloud.store` — the stack's one process-wide ``obstore`` client pool.

`geocloud.cog`, geopatcher's `CogField` and geoproducts' cloud byte reads
take their object-store client from here, so every package in the
process shares one client (and one HTTP/2 connection pool) per bucket /
container. ``s3://``, ``gs://``, ``az://`` / ``abfs[s]://``,
``http(s)://`` (signed URLs included) and ``hf://`` URIs are supported;
`mount` serves one bucket from a store you built (a ``MemoryStore`` in
tests).

```python
from geocloud.store import get_obstore, object_key

store = get_obstore("az://account/container/path/scene.tif")
data = store.get_range(object_key("az://account/container/path/scene.tif"),
                       start=0, length=16384)
```
"""

from __future__ import annotations

from geocloud._src.store import (
    SUPPORTED_SCHEMES,
    clear_obstore_pool,
    get_obstore,
    get_range_bytes,
    mount,
    object_key,
    set_obstore_pool_maxsize,
    unmount,
)


__all__ = [
    "SUPPORTED_SCHEMES",
    "clear_obstore_pool",
    "get_obstore",
    "get_range_bytes",
    "mount",
    "object_key",
    "set_obstore_pool_maxsize",
    "unmount",
]
