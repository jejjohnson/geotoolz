# geostack API reference

The API of every geostack package, generated from its docstrings. The concepts, how-to guides, tutorials and design pages are in the [main docs](https://jejjohnson.github.io/geotoolz/).

| Package | Import | What it holds |
|---|---|---|
| [geotoolz-ops](../operators/api/core.md) | `geoops` | Operators: the pipeline core, carriers, indices, masks, radiometry, geometry, learning and the rest of the operator families |
| [geopatcher](../patcher/api/reference.md) | `geopatcher` | Tiling a scene into patches, running operators per patch, and stitching the results |
| [geocatalog](../catalog/api/reference.md) | `geocatalog` | Indexing archives, querying them, matching sources and loading the matches |
| [geocloud](../cloud/api.md) | `geocloud` | Object-store paths, credentials, caching and cloud-optimised reads and writes |
| [geoproducts](../products/api.md) | `geoproducts` | Readers for satellite products (GOES, Himawari, Carbon Mapper, …) |
| [geoarray](../array/api.md) | `geoarray` | The georeferenced array carrier and its coordinates |
| [xarray lane](../xarray/api/index.md) | `xrops`, `xrproducts` | Operators and archive readers on `xarray` objects |
