# `geocatalog` — API Reference

Curated mkdocstrings reference, one page per namespace. For the
conceptual walkthrough see [Concepts](../concepts.md); for a worked
example see the [Quickstart](../quickstart.md).

The root holds the names every workflow touches; everything else lives
in one namespace per step, and each public name has exactly one home.

| Step | Namespace | Page |
|---|---|---|
| core | `geocatalog` — `GeoCatalog`, `GeoSlice`, `open_catalog`, `query` / `intersect` / `union` | [Core](core.md) |
| discover | `geocatalog.sources` — STAC / CMR / earthaccess / Earth Engine adapters | [Sources](sources.md) |
| index | `geocatalog.build` — raster / xarray / vector catalog builders | [Build](build.md) |
| hold | `geocatalog.backends` — in-memory and DuckDB catalogs, errors | [Backends](backends.md) |
| join | `geocatalog.matchup` — the matchup engine and its strategies | [Matchup](matchup.md) |
| read | `geocatalog.load` — raster / xarray / vector loaders | [Load](load.md) |
| save / share | `geocatalog.storage` — GeoParquet, STAC export, bundles | [Storage](storage.md) |
| stage | `geocatalog.staging` — remote assets to a local cache | [Staging](staging.md) |
| patch | `geocatalog.patch` — the geopatcher bridge | [Patch](patch.md) |
| grids | `geocatalog.grid` — windows, alignment, exact pixel counts | [Grid](grid.md) |
| helpers | `geocatalog.utils` — URIs, retries, UTC time | [Utils](utils.md) |

```python
from geocatalog import GeoSlice, open_catalog
from geocatalog.load import load_raster
from geocatalog.matchup import Intersects, NearestInTime, matchup
```
