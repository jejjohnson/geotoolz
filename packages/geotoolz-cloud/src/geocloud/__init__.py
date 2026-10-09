"""`geocloud` — cloud object storage for the geotoolz stack.

Six modules, each the one home of its names:

- `geocloud.store` — the process-wide ``obstore`` client pool:
  `get_obstore` / `object_key` for ``s3://``, ``gs://``, ``az://``,
  ``http(s)://`` and ``hf://`` URIs, one client (and HTTP/2 connection
  pool) per bucket shared by every package in the process.
- `geocloud.files` — list, read, write, download, upload, copy, sync,
  delete and pre-sign objects by URI (local paths too), on that pool.
- `geocloud.credentials` — credentials registered once per bucket /
  container / host (anonymous, SAS, keys, providers; a TOML file), used by
  the pool and handed to GDAL by `gdal_access`; `redact` for logs.
- `geocloud.cache` — complete local copies of remote objects
  (`localize`, `LocalCache`), for readers that need a real file.
- `geocloud.fs` — an fsspec filesystem on that pool, for libraries that
  only speak fsspec (xarray, zarr, pyarrow, geopandas; ``[fsspec]``).
- `geocloud.cog` — Cloud-Optimized GeoTIFF reads on that pool:
  `CogSource` (batched window reads, each tile fetched once) and
  `AsyncCogReader` with the async ``read_*`` mirrors of
  ``georeader.read`` (``[cog]`` extra).

geopatcher's `CogField` and geoproducts' cloud byte reads (its
``[obstore]`` extra) both build on it.
"""

from __future__ import annotations

from geocloud import cache, cog, credentials, files, fs, store


__version__ = "0.3.0"  # x-release-please-version

__all__ = ["__version__", "cache", "cog", "credentials", "files", "fs", "store"]
