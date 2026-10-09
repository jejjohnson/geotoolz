"""`geocloud.fs` — an fsspec filesystem on the shared pool (``[fsspec]`` extra).

Libraries that only speak fsspec (xarray, zarr, pyarrow, geopandas) get
the stack's one engine: every path goes through `geocloud.store`, with the
credentials registered in `geocloud.credentials`, and local paths work
the same way. No s3fs / gcsfs / adlfs client is involved.

```python
import xarray as xr

from geocloud.fs import filesystem

fs = filesystem()
with fs.open("s3://bucket/scene.nc") as fh:          # ranged reads, pooled client
    ds = xr.open_dataset(fh, engine="h5netcdf")
cube = xr.open_zarr(fs.get_mapper("gs://bucket/cube.zarr"))
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geocloud._src.extras import missing_extra


if TYPE_CHECKING:
    from geocloud._src.fs import GeoCloudFileSystem, filesystem


__all__ = ["GeoCloudFileSystem", "filesystem"]


def __getattr__(name: str) -> Any:
    """Resolve the names lazily, so `import geocloud` works without fsspec."""
    if name not in __all__:
        raise AttributeError(f"module 'geocloud.fs' has no attribute {name!r}")
    try:
        from geocloud._src import fs as impl
    except ImportError as exc:
        raise missing_extra(f"geocloud.fs.{name}", "fsspec") from exc
    return getattr(impl, name)
