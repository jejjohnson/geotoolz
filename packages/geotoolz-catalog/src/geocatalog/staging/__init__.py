"""`geocatalog.staging` — copy a catalog's remote assets to local disk.

`stage` downloads (through fsspec) every asset a catalog points at,
skipping files a `LocalCache` already holds, and returns the same
catalog with its paths rewritten to the local copies. Hand the result to
`geocatalog.patch.field_for` or the `geocatalog.load` loaders.

```python
from geocatalog.staging import LocalCache, stage

local = stage(catalog, cache=LocalCache("./.cache", ttl_days=7))
```
"""

from __future__ import annotations

from geocatalog._src.staging import (
    LocalCache,
    stage,
)


__all__ = [
    "LocalCache",
    "stage",
]
