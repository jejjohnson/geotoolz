"""`geocloud.cache` — complete local copies of remote objects.

For readers that need a real file on disk (HDF4, memory-mapped binaries,
GDAL drivers without ``/vsi*/``) and for staging data next to compute:
`localize` returns a local path for any location — in place when it is
already local, else downloaded once into a `LocalCache` and served from
there afterwards. `cache_key` strips expiring signatures, so a re-signed
URL hits the same slot.

```python
from pathlib import Path

from geocloud.cache import LocalCache, localize

path: Path = localize("s3://bucket/MOD021KM.A2024001.hdf")     # downloaded once
cache = LocalCache(root="/scratch/cache", ttl_days=7)
again: Path = cache.fetch("s3://bucket/MOD021KM.A2024001.hdf")
```
"""

from __future__ import annotations

from geocloud._src.cache import LocalCache, cache_key, localize


__all__ = ["LocalCache", "cache_key", "localize"]
