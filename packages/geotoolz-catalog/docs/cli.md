# Command-line interface

`pip install geocatalog` puts a `geocatalog` command on your `PATH`. It
maps thin wrappers over the same functions the Python API exposes —
nothing in the library is CLI-only — so you can mix it with cron, CI,
or a shell session without writing a Python script.

```console
$ geocatalog --help
Usage: geocatalog COMMAND

Spatiotemporal catalog over geospatial files.
```

## Subcommands

### `build`

Builds a catalog from a glob of files and writes it as GeoParquet.

```console
$ geocatalog build raster \
    --input-glob "data/sentinel2/*.tif" \
    --regex "S2_(?P<date>\d{8})_.+\.tif" \
    --out catalog.parquet
wrote catalog.parquet (1247 rows)

$ geocatalog build vector \
    --input-glob "labels/*.gpkg" \
    --layer fields \
    --out vector.parquet

$ geocatalog build xarray \
    --input-glob "data/*.nc" \
    --time-var time \
    --out xarray.parquet
```

`build xarray` requires the `[xarray-raster]` extra. The CLI surfaces a
clear error if the extra isn't installed.

### `query`

Filter a catalog by bbox + time window and print the matching row count.

```console
$ geocatalog query catalog.parquet \
    --bbox "-5.2,36.1,-4.5,36.8" \
    --start 2024-06-01 \
    --end 2024-06-30
source  catalog.parquet
rows    42
bbox    (-5.2, 36.1, -4.5, 36.8)
time    ['2024-06-01', '2024-06-30']
```

Pass `--crs EPSG:32629` (or any other identifier) to interpret `--bbox`
in a non-default CRS. The library reprojects internally.

`--start` and `--end` are paired — pass either both or neither.
Passing only one is rejected with exit code 1.

### `stats`

Top-line metadata about an artifact — rows, bounds, temporal extent,
backend tag, CRS.

```console
$ geocatalog stats catalog.parquet
rows             1247891
bounds           [-15.2, 36.1, 4.5, 47.2]
temporal_start   2021-01-01 00:00:00
temporal_end     2024-06-29 23:59:59
backend          raster
crs              EPSG:4326
```

### `info`

Inspect a single row.

```console
$ geocatalog info catalog.parquet --row 0
filepath     /archive/sentinel-2/S2_20210101_T29SND.tif
geometry     POLYGON ((...))
start_time   2021-01-01 00:00:00
end_time     2021-01-01 23:59:59.999999
```

### `convert`

Rewrite an existing GeoParquet artifact, optionally into a Hive-partitioned
directory. When `--out` is omitted, `single.parquet` converts to `single/`.

```console
$ geocatalog convert single.parquet --partition-by year,month
source  single.parquet
out     single
rows    1247891
```

### `migrate`

Rewrite an artifact in place at a target schema version. A thin
wrapper over `migrate_geoparquet`; see
[Schema versions](schema-versions.md) for the migration policy.

```console
$ geocatalog migrate catalog.parquet
catalog.parquet already at v0

$ geocatalog migrate catalog.parquet --to-version 0
catalog.parquet already at v0
```

`--to-version` defaults to the reader's current schema version
(`SCHEMA_VERSION_CURRENT`, currently 0 — so today every invocation
reports "already at v0"; once the schema version is bumped, older
artifacts print `wrote catalog.parquet (v0 -> v1)`). Migrations are
idempotent — re-running against an already-migrated artifact leaves
the file untouched.

Exit codes follow the shared table below: `3` when the source is
missing or unreadable, `2` on a corrupt artifact or schema mismatch
(`CatalogSchemaError`).

## Output formats

Every subcommand accepts `--json` to switch from the human-readable
table to a JSON object. This is the right choice when piping into
`jq`, a workflow scheduler, or another tool.

```console
$ geocatalog stats catalog.parquet --json | jq .rows
1247891

$ geocatalog build raster --input-glob "data/*.tif" \
    --regex "S2_(?P<date>\d{8})_.+\.tif" \
    --out catalog.parquet --json
{"out": "catalog.parquet", "rows": 42}
```

## Exit codes

| Code | Meaning                                                       |
| ---- | ------------------------------------------------------------- |
| 0    | Success.                                                      |
| 1    | User error — bad args, glob matched nothing, missing extra.   |
| 2    | Catalog error — corrupt artifact or unrecognised schema.      |
| 3    | I/O error — source path doesn't exist or can't be read.       |

## Coming soon

Subcommands deferred to follow-on PRs:

- `geocatalog compact catalog.parquet` (#8).
