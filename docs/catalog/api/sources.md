# Discovery sources & bundle

STAC / earthaccess / CMR discovery adapters, STAC conversion, and
the provenance-recording `CatalogBundle`.

## Discovery sources

The `Source` Protocol and its adapters live under
`geocatalog.sources`. Adapters are extras-gated and imported lazily.

::: geocatalog.sources.Source
::: geocatalog.sources.SourceRow
::: geocatalog.sources.AuthStatus

### STAC *(extras: `[stac]`)*

::: geocatalog.sources.STACSource

### earthaccess *(extras: `[earthaccess]`)*

::: geocatalog.sources.EarthAccessSource

### CMR

::: geocatalog.sources.CMRSource

## STAC conversion *(extras: `[stac]`)*

::: geocatalog._src.stac.from_stac_items
::: geocatalog._src.stac.from_stac_search
::: geocatalog._src.stac.to_stac_collection

## Bundle

::: geocatalog.bundle.CatalogBundle
::: geocatalog.bundle.QueryRecord
::: geocatalog.bundle.source_row_to_gdf_row
