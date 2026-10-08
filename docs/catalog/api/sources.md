# Sources — `geocatalog.sources`

Discover scenes in external archives: the `Source` protocol, the
STAC / CMR / earthaccess / Earth Engine adapters, and building a
catalog straight from STAC. Adapters are extras-gated and imported
lazily.

## Protocol

::: geocatalog.sources.Source
::: geocatalog.sources.SourceRow
::: geocatalog.sources.AuthStatus

### STAC *(extras: `[stac]`)*

::: geocatalog.sources.STACSource

### earthaccess *(extras: `[earthaccess]`)*

::: geocatalog.sources.EarthAccessSource

### CMR

::: geocatalog.sources.CMRSource

### Earth Engine *(extras: `[gee]`)*

::: geocatalog.sources.GEESource

## Catalogs from STAC *(extras: `[stac]`)*

::: geocatalog.sources.from_stac_items
::: geocatalog.sources.from_stac_search

To export a catalog as a STAC collection, see
[`to_stac_collection`](storage.md#stac-export-extras-stac).

## Source rows to catalog rows

::: geocatalog.sources.source_row_to_gdf_row
