# Catalogs & backends

The `GeoCatalog` protocol, the in-memory and DuckDB backends, the
`open_catalog` factory, and cross-catalog set algebra.

## Protocol

::: geocatalog._src.base.GeoCatalog
::: geocatalog._src.base.CatalogRow

## Backends

::: geocatalog._src.memory.InMemoryGeoCatalog

### DuckDB *(extras: `[duckdb]`)*

::: geocatalog._src.duckdb_backend.DuckDBGeoCatalog

## Factory

::: geocatalog.open_catalog

## Set algebra

::: geocatalog._src.ops.query
::: geocatalog._src.ops.intersect
::: geocatalog._src.ops.union
