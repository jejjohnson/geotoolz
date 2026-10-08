"""`geocatalog.patch` — hand a catalog to geopatcher.

- `field_for` — the rows a `geocatalog.GeoSlice` selects, as a
  `geopatcher.RasterField` ready to ``split`` (``[patch]`` extra).
- `CatalogDomain` — a geopatcher `Domain` over a catalog at a fixed
  resolution, so a patcher can iterate the catalog's footprint.

```python
from geocatalog.patch import field_for
from geocatalog.staging import stage

local = stage(catalog, dest="./.cache")      # remote assets -> local files
field = field_for(local, aoi)                # geopatcher.RasterField
patches = list(patcher.split(field))
```
"""

from __future__ import annotations

from geocatalog._src.patch.domain import CatalogDomain
from geocatalog._src.patch.field_for import field_for


__all__ = [
    "CatalogDomain",
    "field_for",
]
