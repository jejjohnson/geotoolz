# Backends — `geocatalog.backends`

The catalog implementations behind the `GeoCatalog` protocol, the row
they yield, and the errors they raise. `geocatalog.open_catalog` picks
a backend for you.

## In memory

::: geocatalog.backends.InMemoryGeoCatalog

## DuckDB *(extras: `[duckdb]`)*

::: geocatalog.backends.DuckDBGeoCatalog

## Rows

::: geocatalog.backends.CatalogRow

## Errors

Errors about a catalog's state or artifacts derive from
`GeoCatalogError` and from the builtin callers caught before
(`ValueError` / `RuntimeError`). Invalid arguments raise the builtin
`ValueError` / `TypeError`; see the
[parameter vocabulary](../vocabulary.md#errors).

::: geocatalog.backends.GeoCatalogError
::: geocatalog.backends.CatalogSchemaError
::: geocatalog.backends.CatalogMetadataError
::: geocatalog.backends.CatalogClosedError
