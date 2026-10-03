"""Public alias for `geopatcher._src.objstore` — the stack's one obstore pool.

``geotoolz`` and geopatcher's ``ObstoreCogField`` take their object-store
client from here (``geotoolz-catalog[obstore]`` installs it), so every
package in the process shares one client (and one HTTP/2 connection
pool) per bucket / container. Needs ``obstore``:
``pip install 'geotoolz-patcher[obstore]'``.

```python
from geopatcher.objstore import get_obstore, object_key

store = get_obstore("az://account/container/path/scene.tif")
data = store.get_range(object_key("az://account/container/path/scene.tif"),
                       start=0, length=16384)
```
"""

from __future__ import annotations

from geopatcher._src.objstore import (
    SUPPORTED_SCHEMES,
    clear_obstore_pool,
    get_obstore,
    get_range_bytes,
    object_key,
    set_obstore_pool_maxsize,
)


__all__ = [
    "SUPPORTED_SCHEMES",
    "clear_obstore_pool",
    "get_obstore",
    "get_range_bytes",
    "object_key",
    "set_obstore_pool_maxsize",
]
