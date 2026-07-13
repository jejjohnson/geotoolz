# Changelog

## [0.0.3](https://github.com/jejjohnson/geocatalog/compare/v0.0.2...v0.0.3) (2026-06-09)


### Features

* **builder:** obstore client pool + async concurrency for build_raster_catalog ([#62](https://github.com/jejjohnson/geocatalog/issues/62)) ([10c187a](https://github.com/jejjohnson/geocatalog/commit/10c187a0c9e52b679fffa63efedf798a44317c45))
* exact grid alignment for GeoSlice (divide_evenly, align modes, is_grid_aligned) ([#64](https://github.com/jejjohnson/geocatalog/issues/64)) ([02a5e37](https://github.com/jejjohnson/geocatalog/commit/02a5e37289f9fa646c171d49cc5062eb6eab8fce))

## [0.0.2](https://github.com/jejjohnson/geocatalog/compare/v0.0.1...v0.0.2) (2026-05-25)


### Features

* **bundle:** catalog.ingest(Source) + persistence layer ([#56](https://github.com/jejjohnson/geocatalog/issues/56)) ([d23145f](https://github.com/jejjohnson/geocatalog/commit/d23145ffc01777e94ad260a85e88f9458af498ef))
* **cli:** geocatalog CLI via cyclopts ([#36](https://github.com/jejjohnson/geocatalog/issues/36)) ([362568c](https://github.com/jejjohnson/geocatalog/commit/362568c456e5e108c2939247a8fd4bcc02083081))
* **duckdb:** auto-load remote URI extensions ([#48](https://github.com/jejjohnson/geocatalog/issues/48)) ([ab3f6ad](https://github.com/jejjohnson/geocatalog/commit/ab3f6ad6411bd2a7159e9947f8a573de22e52ebe))
* **io:** add fsspec URI resolution ([#42](https://github.com/jejjohnson/geocatalog/issues/42)) ([1187158](https://github.com/jejjohnson/geocatalog/commit/11871589c46930788249bd577b6fdbb9afcdd61e))
* **io:** retry/backoff on remote i/o ([#51](https://github.com/jejjohnson/geocatalog/issues/51)) ([327dd6d](https://github.com/jejjohnson/geocatalog/commit/327dd6d0a01074ae755007a9742abfd1a72adcb3))
* **matchup:** implement spatial + temporal strategies + matchup engine ([#55](https://github.com/jejjohnson/geocatalog/issues/55)) ([73d34c6](https://github.com/jejjohnson/geocatalog/commit/73d34c64402ace6aa09efc65f513fb680afa71f3))
* **parquet:** hive-partitioned archives + incremental append_files() ([#41](https://github.com/jejjohnson/geocatalog/issues/41)) ([55a994a](https://github.com/jejjohnson/geocatalog/commit/55a994aa340ad6964aa0057b0a700309e78c526d))
* query → matchup → stage pipeline + scaffolding ([#53](https://github.com/jejjohnson/geocatalog/issues/53)) ([16378ec](https://github.com/jejjohnson/geocatalog/commit/16378ec355749232eefc947f63ebe3aadad4581d))
* schema migration framework on _schema_version ([#39](https://github.com/jejjohnson/geocatalog/issues/39)) ([a16c6ce](https://github.com/jejjohnson/geocatalog/commit/a16c6ce9664fc84d5a0f957637ecea108293395a))
* **sources:** implement EarthAccessSource + CMRSource ([#57](https://github.com/jejjohnson/geocatalog/issues/57)) ([87ff655](https://github.com/jejjohnson/geocatalog/commit/87ff655e2d51ac5a3ec19031e445dc105f17fcc7))
* **sources:** implement STACSource against pystac-client ([#54](https://github.com/jejjohnson/geocatalog/issues/54)) ([e087bf7](https://github.com/jejjohnson/geocatalog/commit/e087bf7839292dc39bb0be95cc2ee203b126c0e5))
* **stac:** catalog builders + collection export ([#43](https://github.com/jejjohnson/geocatalog/issues/43)) ([14145d0](https://github.com/jejjohnson/geocatalog/commit/14145d0c901f57a13ded2da751678d74927a01db))
* **staging:** field_for() helper for geopatcher Field construction ([#59](https://github.com/jejjohnson/geocatalog/issues/59)) ([2b42346](https://github.com/jejjohnson/geocatalog/commit/2b42346d89f89dca2d1a32eb8c06a4c1194ccd72))
* **staging:** implement stage() + LocalCache (fsspec-backed) ([#58](https://github.com/jejjohnson/geocatalog/issues/58)) ([39e7dd3](https://github.com/jejjohnson/geocatalog/commit/39e7dd3f81b651d51b4e97d01b99215af3fe60f9))


### Bug Fixes

* **duckdb:** add DuckDBGeoCatalog connection lifecycle ([#47](https://github.com/jejjohnson/geocatalog/issues/47)) ([bf6bfaf](https://github.com/jejjohnson/geocatalog/commit/bf6bfaf4e5867b1423af2f8664ef9883f3c10068))
* **staging:** address PR [#59](https://github.com/jejjohnson/geocatalog/issues/59) review comments on field_for() ([#60](https://github.com/jejjohnson/geocatalog/issues/60)) ([44ca8a4](https://github.com/jejjohnson/geocatalog/commit/44ca8a4edc79bb8af001286d86214957441340b2))
* **streaming:** deterministic order in parallel row extraction ([#49](https://github.com/jejjohnson/geocatalog/issues/49)) ([08c7333](https://github.com/jejjohnson/geocatalog/commit/08c7333ad909310c1696d3aac3f64c5a3690e16d))


### Performance Improvements

* **duckdb:** cache scalar properties on DuckDBGeoCatalog ([#45](https://github.com/jejjohnson/geocatalog/issues/45)) ([6c79de3](https://github.com/jejjohnson/geocatalog/commit/6c79de3dd13c39733747870c652d197bb2327121))
* **memory:** use spatial join in InMemoryGeoCatalog.intersect ([#46](https://github.com/jejjohnson/geocatalog/issues/46)) ([1115fb7](https://github.com/jejjohnson/geocatalog/commit/1115fb7e27fa95c84e2223ecbe8cc510339a7bad))
* **memory:** vectorise InMemoryGeoCatalog.iter_rows ([#44](https://github.com/jejjohnson/geocatalog/issues/44)) ([5315c4f](https://github.com/jejjohnson/geocatalog/commit/5315c4f54e2ee3016bd5327dd9874e5aa20b62e3))
* **raster:** concurrent per-day loads in load_raster_timeseries ([#50](https://github.com/jejjohnson/geocatalog/issues/50)) ([3d22a81](https://github.com/jejjohnson/geocatalog/commit/3d22a8181a61b6bc8c180ed2a12f058bc3c700b8))

## 0.0.1 (2026-05-21)


### Features

* port geocatalog package from geotoolz ([#1](https://github.com/jejjohnson/geocatalog/issues/1)) ([9823273](https://github.com/jejjohnson/geocatalog/commit/98232736d9b1120d2da6b26b93c00280f7a37076))

## Changelog
