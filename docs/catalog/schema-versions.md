# Catalog schema versions

Every GeoParquet artifact written by `geocatalog` carries a reserved
`_schema_version` column. Readers (`from_geoparquet`, `DuckDBGeoCatalog.open`)
check it on load and dispatch on three cases:

| Case                         | Behaviour                                              |
| ---------------------------- | ------------------------------------------------------ |
| `v_artifact == v_current`    | Load directly.                                         |
| `v_artifact <  v_current`    | Chain forward migrations (in-memory backend only).     |
| `v_artifact >  v_current`    | Raise [`CatalogSchemaError`][geocatalog.CatalogSchemaError]. |

The current reader version is
[`SCHEMA_VERSION_CURRENT`][geocatalog.SCHEMA_VERSION_CURRENT]. Bump it
whenever the on-disk schema changes substantively (new required column,
renamed reserved column, changed semantics of an existing field).
Bumping also requires registering a forward migration in
`geocatalog._src.parquet._MIGRATIONS`.

## Why bother

Without `_schema_version`, an artifact written by tomorrow's library
loads silently in today's library — the columns line up by name but the
*meaning* may have shifted. The Delta Lake protocol-versioning model
([reference](https://github.com/delta-io/delta/blob/master/PROTOCOL.md#protocol))
catches that explicitly: writers stamp the version; readers refuse
versions newer than they understand.

## Migration policy

* **Forward-only.** Migrations bring older artifacts up to the current
  version. Downgrading is not supported — once you've written at v2 you
  can't write back to v1.
* **Pure.** Each migration is a `gpd.GeoDataFrame → gpd.GeoDataFrame`
  function with no I/O. The reader composes the chain in-memory.
* **Idempotent.** Re-running a migration over an already-migrated gdf
  is a no-op or a clean overwrite — never destructive.
* **DuckDB backend opts out.** Forward migrations require materialising
  the relation through pandas, which defeats the lazy backend. Use
  `geocatalog migrate <path>` to bring the artifact up to date out of
  band; then reopen.

## CLI

```console
$ geocatalog migrate legacy.parquet --to-version 0
legacy.parquet already at v0

$ geocatalog migrate v0_artifact.parquet
v0_artifact.parquet already at v0
```

(In the future, once ``SCHEMA_VERSION_CURRENT`` is bumped to 1, the
second invocation will print ``wrote v0_artifact.parquet (v0 -> v1)``
because the rewrite actually changes the version.)

`--to-version` defaults to the reader's current version. The command
exits with status 2 if the artifact is *newer* than the reader (you
need to upgrade `geocatalog`).

## Version history

| Version | Released | Change |
| ------- | -------- | ------ |
| 0       | initial  | First versioned schema. ``filepath`` / ``geometry`` / ``start_time`` / ``end_time`` columns; reserved ``_backend`` / ``_schema_version``. |

This table grows as the schema evolves.
