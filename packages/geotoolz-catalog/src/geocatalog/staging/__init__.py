"""`geocatalog.staging` — copy a catalog's remote assets to local disk.

`stage` fetches every asset a catalog points at into a
`geocloud.cache.LocalCache` (downloads through `geocloud.files`, skipping
files the cache already holds) and returns the same catalog with its paths
rewritten to the local copies. Hand the result to
`geocatalog.patch.field_for` or the `geocatalog.load` loaders.

```python
from geocloud.cache import LocalCache

from geocatalog.staging import stage

local = stage(catalog, cache=LocalCache("./.cache", ttl_days=7))
```
"""

from __future__ import annotations

from geocatalog._src.staging.stage import stage


__all__ = ["stage"]
