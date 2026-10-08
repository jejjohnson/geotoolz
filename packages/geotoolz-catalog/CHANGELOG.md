# Changelog

## [0.3.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.2.3...geotoolz-catalog-v0.3.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* geopatcher.temporal.stencils.divide_evenly is now exact_quotient, geocatalog.grid.divide_evenly is now count_steps, and geotoolz.patch_ops.SpatialTriangular is now TriangularWindow. No aliases; saved patcher configs naming geotoolz.patch_ops.SpatialTriangular must be regenerated.
* **catalog:** the flat root re-exports and the geocatalog.catalog / .types / .io / .bundle modules are gone, with no aliases. Import each name from its namespace (e.g. geocatalog.load.load_raster, geocatalog.backends.InMemoryGeoCatalog, geocatalog.storage.to_geoparquet, geocatalog.patch.field_for, geocatalog.grid.slice_to_window, geocatalog.utils.parse_uri). geocatalog.matchup is no longer callable: use geocatalog.matchup.matchup(...).
* **patcher:** no aliases are kept. The prefixed axis names (SpatialHann, TemporalMean, ...) and the root re-exports of axes, runners, hooks and caches are gone; geopatcher.time is geopatcher.temporal; geopatcher.objstore / geopatcher.cog become geocloud.store / geocloud.cog; geopatcher.runners / dask / jax / hooks fold into geopatcher.run and geopatcher.observe; ObstoreCogField becomes geopatcher.fields.CogField; the patcher extras obstore / obstore-cog are replaced by [cog]; and geotoolz-catalog[obstore] is removed (install geotoolz-cloud). Saved config envelopes that use the old class names must be regenerated.

### Code Refactoring

* **catalog:** organise geocatalog by workflow step, one home per name ([#433](https://github.com/jejjohnson/geotoolz/issues/433)) ([94621fd](https://github.com/jejjohnson/geotoolz/commit/94621fd2ae2f007627b7443caa29925f7cea58a8))
* give the divide-evenly helpers distinct names and drop the last Spatial* prefix ([#434](https://github.com/jejjohnson/geotoolz/issues/434)) ([d8155d2](https://github.com/jejjohnson/geotoolz/commit/d8155d2f96439c3339d93fae219b82103ea6ed75))
* **patcher:** organise geopatcher by task and split object storage into geotoolz-cloud ([#431](https://github.com/jejjohnson/geotoolz/issues/431)) ([a7a4d6e](https://github.com/jejjohnson/geotoolz/commit/a7a4d6e4fbec075260f8f4a9fd371a791ff13b30))

## [0.2.3](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.2.2...geotoolz-catalog-v0.2.3) (2026-10-05)


### Bug Fixes

* **catalog:** `geocatalog --version` and `__version__` report the distribution version ([#396](https://github.com/jejjohnson/geotoolz/issues/396)) ([6f470b3](https://github.com/jejjohnson/geotoolz/commit/6f470b30c33d5ed63fdf996810d0bf3efb09ee47)), closes [#250](https://github.com/jejjohnson/geotoolz/issues/250)
* **catalog:** extras install exactly what geocatalog uses ([#395](https://github.com/jejjohnson/geotoolz/issues/395)) ([1b96443](https://github.com/jejjohnson/geotoolz/commit/1b9644387daca6235ced738503c81a8bbc7c9327)), closes [#253](https://github.com/jejjohnson/geotoolz/issues/253)

## [0.2.2](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.2.1...geotoolz-catalog-v0.2.2) (2026-10-03)


### Features

* **catalog:** one meaning per parameter name, one error hierarchy ([#373](https://github.com/jejjohnson/geotoolz/issues/373)) ([094dd61](https://github.com/jejjohnson/geotoolz/commit/094dd61638944895528028903b012d37ec678836))
* **catalog:** one public surface — facades partition the top level; public paths for streaming, I/O and time helpers; one lazy table ([#372](https://github.com/jejjohnson/geotoolz/issues/372)) ([0086e83](https://github.com/jejjohnson/geotoolz/commit/0086e83865a8a15c238df039795198a155a915d1))


### Bug Fixes

* **catalog,patcher:** get_config()["crs"] is one string on both backends and across round trips; pool key covers every endpoint variable; hf:// in the pool ([#371](https://github.com/jejjohnson/geotoolz/issues/371)) ([6b8b3ff](https://github.com/jejjohnson/geotoolz/commit/6b8b3ff34489958ed89ff210a3e4abb7961dd501))

## [0.2.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.2.0...geotoolz-catalog-v0.2.1) (2026-10-03)


### Bug Fixes

* **catalog:** bundle ids are a primary key; one granule mapper; adapter retry and CMR query fixes ([#361](https://github.com/jejjohnson/geotoolz/issues/361)) ([f0e875b](https://github.com/jejjohnson/geotoolz/commit/f0e875ba79d51eaab80e6d0252d407980e916bb4))
* **catalog:** field_for mosaics a slice into one GeoTensor RasterField in the catalog CRS ([#366](https://github.com/jejjohnson/geotoolz/issues/366)) ([1deffca](https://github.com/jejjohnson/geotoolz/commit/1deffcad1f5b8b58d944c74838e92219c9f3b14e))
* **catalog:** limit=0 yields nothing on every source; negative limits raise ([#358](https://github.com/jejjohnson/geotoolz/issues/358)) ([76dd06c](https://github.com/jejjohnson/geotoolz/commit/76dd06ca63d2a4364fdac6281798545efe56c160))
* **catalog:** matchup accepts catalogs and bundles; deterministic ids and tie-breaks; tz/NaT/CRS-safe ([#364](https://github.com/jejjohnson/geotoolz/issues/364)) ([ee46dc5](https://github.com/jejjohnson/geotoolz/commit/ee46dc534a9e4547f928aa6f133c50e81c66c9f6))
* **catalog:** matchup engine searches each strategy's envelope, so CentroidWithin(buffer&gt;0) matches ([#363](https://github.com/jejjohnson/geotoolz/issues/363)) ([df2f342](https://github.com/jejjohnson/geotoolz/commit/df2f3427acc7509c3f304f283398de41077146ee)), closes [#241](https://github.com/jejjohnson/geotoolz/issues/241)
* **catalog:** one STAC item decoder — real footprints, proj:* CRS, UTC times, absolute hrefs, densified reprojection ([#359](https://github.com/jejjohnson/geotoolz/issues/359)) ([88f95a9](https://github.com/jejjohnson/geotoolz/commit/88f95a92d3cefdb5d57d50808b8087689b485c26))
* **catalog:** reject bad --crs / --target-crs and non-vector inputs in the CLI with one line; tests for every verb; cli.md matches ([#367](https://github.com/jejjohnson/geotoolz/issues/367)) ([cf667c6](https://github.com/jejjohnson/geotoolz/commit/cf667c68b50060ec42d3b6c24d85ca1e69f1077c))
* **catalog:** run behaviour tests on every backend and fix the divergences they found ([#355](https://github.com/jejjohnson/geotoolz/issues/355)) ([9b4cc70](https://github.com/jejjohnson/geotoolz/commit/9b4cc70ab23495af81ec7706883a2459f88aab24)), closes [#235](https://github.com/jejjohnson/geotoolz/issues/235)
* **catalog:** split UMM-G footprints at the antimeridian; handle ExclusiveZone, Lines, open ranges and null names ([#357](https://github.com/jejjohnson/geotoolz/issues/357)) ([86aa722](https://github.com/jejjohnson/geotoolz/commit/86aa72289f456086880f1fcfc02e09b687666038))
* **catalog:** stage() works without fsspec for local files, follows the primary asset, downloads atomically once per URI ([#365](https://github.com/jejjohnson/geotoolz/issues/365)) ([5c4c469](https://github.com/jejjohnson/geotoolz/commit/5c4c4690e835d7e0081ede20fc51f5112fa62530))
* **catalog:** streaming writer infers its schema over the first batch, takes schema=, keeps ns UTC times ([#353](https://github.com/jejjohnson/geotoolz/issues/353)) ([0729c60](https://github.com/jejjohnson/geotoolz/commit/0729c60df38c2139d5365a050047bdbb9cefe329)), closes [#233](https://github.com/jejjohnson/geotoolz/issues/233)
* **catalog:** unwrap STAC polygons whose hole crosses ±180°, route every line through lonlat_line, give empty bundles the ingested columns ([#369](https://github.com/jejjohnson/geotoolz/issues/369)) ([dd6b5f8](https://github.com/jejjohnson/geotoolz/commit/dd6b5f8705fb334f142f566e39230bc9777e1319))
* **catalog:** writers are atomic, partitioned replace keeps unrelated files, append_files is idempotent ([#354](https://github.com/jejjohnson/geotoolz/issues/354)) ([635c18d](https://github.com/jejjohnson/geotoolz/commit/635c18dd0b4726b4acc4012d0b1a56fcbce63322)), closes [#234](https://github.com/jejjohnson/geotoolz/issues/234)

## [0.2.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.1.2...geotoolz-catalog-v0.2.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **objstore:** geotoolz.readers._src.obstore and geocatalog._src.objstore are removed (use geopatcher.objstore; the geotoolz read_byte_range helper is now get_range_bytes). The [obstore] extras of geotoolz and geotoolz-catalog install geotoolz-patcher. Pool keys changed shape, and object_key / get_obstore raise ValueError for malformed Azure URIs and abfs:// URIs without container@account.

### Bug Fixes

* **objstore:** one obstore pool in geopatcher with correct Azure and signed-URL handling ([#334](https://github.com/jejjohnson/geotoolz/issues/334)) ([946a514](https://github.com/jejjohnson/geotoolz/commit/946a51466a3a73d71ecc57f984cf44c672585d55)), closes [#180](https://github.com/jejjohnson/geotoolz/issues/180)

## [0.1.2](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.1.1...geotoolz-catalog-v0.1.2) (2026-09-26)


### Bug Fixes

* **catalog:** bring DuckDB catalogs to parity with InMemory ([#270](https://github.com/jejjohnson/geotoolz/issues/270)) ([3345987](https://github.com/jejjohnson/geotoolz/commit/3345987548e86b3f0306631ac980ee6502a43dea)), closes [#225](https://github.com/jejjohnson/geotoolz/issues/225)
* **catalog:** build DuckDB set algebra without named views ([#265](https://github.com/jejjohnson/geotoolz/issues/265)) ([abe154a](https://github.com/jejjohnson/geotoolz/commit/abe154a900b1d3509a50ce3f849813d9a31221f8)), closes [#222](https://github.com/jejjohnson/geotoolz/issues/222)
* **catalog:** derive timeseries steps from the slice; parse regex dates reliably ([#275](https://github.com/jejjohnson/geotoolz/issues/275)) ([7f11f24](https://github.com/jejjohnson/geotoolz/commit/7f11f2489c7850d772b5882e9ae56aef7839fb47)), closes [#219](https://github.com/jejjohnson/geotoolz/issues/219)
* **catalog:** drop housekeeping columns when materialising a DuckDB catalog ([#264](https://github.com/jejjohnson/geotoolz/issues/264)) ([06d5811](https://github.com/jejjohnson/geotoolz/commit/06d58116dc8003f8f1d451f596133110f02aabd0)), closes [#223](https://github.com/jejjohnson/geotoolz/issues/223)
* **catalog:** honour Resampling.nearest, validate nodata, unify builder CRS ([#273](https://github.com/jejjohnson/geotoolz/issues/273)) ([23beab0](https://github.com/jejjohnson/geotoolz/commit/23beab0e71baed9f8e6038283031e6ad8822e704)), closes [#217](https://github.com/jejjohnson/geotoolz/issues/217)
* **catalog:** reproject load_raster sources into the slice CRS ([#272](https://github.com/jejjohnson/geotoolz/issues/272)) ([ab7678f](https://github.com/jejjohnson/geotoolz/commit/ab7678fb8d213b8013e5011e5bd9197398cd67e0))
* **catalog:** select the right window in load_xarray and cover pixel edges ([#274](https://github.com/jejjohnson/geotoolz/issues/274)) ([48b96ce](https://github.com/jejjohnson/geotoolz/commit/48b96ce1ed40810b2e931a5a5be9dab8cd20178c)), closes [#218](https://github.com/jejjohnson/geotoolz/issues/218)
* **catalog:** serialise work on a shared DuckDB connection across threads ([#267](https://github.com/jejjohnson/geotoolz/issues/267)) ([6adedd7](https://github.com/jejjohnson/geotoolz/commit/6adedd7228a9931df8ddd374a0b031af4761e710))


### Performance Improvements

* **catalog:** open DuckDB catalogs lazily and prune with the bbox column ([#269](https://github.com/jejjohnson/geotoolz/issues/269)) ([c57c578](https://github.com/jejjohnson/geotoolz/commit/c57c578502a8257ecd538291e53a467398fa1429)), closes [#221](https://github.com/jejjohnson/geotoolz/issues/221)
* **catalog:** range-read remote rasters via GDAL; fail fast on missing files ([#276](https://github.com/jejjohnson/geotoolz/issues/276)) ([23a5249](https://github.com/jejjohnson/geotoolz/commit/23a52494def705a2e2798265944d8456f4b12faf)), closes [#220](https://github.com/jejjohnson/geotoolz/issues/220)

## [0.1.1](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.1.0...geotoolz-catalog-v0.1.1) (2026-09-26)


### Bug Fixes

* **catalog:** align the GeoCatalog Protocol, CatalogDomain and iter_slices with both backends ([#262](https://github.com/jejjohnson/geotoolz/issues/262)) ([ae74904](https://github.com/jejjohnson/geotoolz/commit/ae74904008d5ad70c3f06fb12bb017e4596587ef)), closes [#232](https://github.com/jejjohnson/geotoolz/issues/232)
* **catalog:** hold catalog times in naive UTC across constructors, queries and backends ([#261](https://github.com/jejjohnson/geotoolz/issues/261)) ([b0ab340](https://github.com/jejjohnson/geotoolz/commit/b0ab3405bdf1ac09a2b0952e04a0998a5ec3e323)), closes [#231](https://github.com/jejjohnson/geotoolz/issues/231)
* **catalog:** keep GeometryCollection overlaps and sync time columns in intersect ([#260](https://github.com/jejjohnson/geotoolz/issues/260)) ([329935a](https://github.com/jejjohnson/geotoolz/commit/329935a653cc8bf0ff4c14393fa1208fdbabd1f7)), closes [#230](https://github.com/jejjohnson/geotoolz/issues/230)
* **catalog:** keep GeoSlice hash consistent with equality; validate bounds ([#257](https://github.com/jejjohnson/geotoolz/issues/257)) ([d93c6dc](https://github.com/jejjohnson/geotoolz/commit/d93c6dce311a38c4c88301151ad0c3a0d8db61d5)), closes [#227](https://github.com/jejjohnson/geotoolz/issues/227)
* **catalog:** make grid-alignment tolerance pixel-relative; round shape half up ([#256](https://github.com/jejjohnson/geotoolz/issues/256)) ([f22acda](https://github.com/jejjohnson/geotoolz/commit/f22acdad78aef2811336a0d2a92c361129d49e1e)), closes [#226](https://github.com/jejjohnson/geotoolz/issues/226)
* **catalog:** return None from temporal_extent on empty catalogs ([#255](https://github.com/jejjohnson/geotoolz/issues/255)) ([f500ccb](https://github.com/jejjohnson/geotoolz/commit/f500ccbb27073d79f7149ce1e57f1101072a2b81)), closes [#228](https://github.com/jejjohnson/geotoolz/issues/228)
* **catalog:** split antimeridian-crossing query AOIs instead of inverting them ([#258](https://github.com/jejjohnson/geotoolz/issues/258)) ([3146e9c](https://github.com/jejjohnson/geotoolz/commit/3146e9c1c11d15e017c9eceb78aa8f04464ac592)), closes [#229](https://github.com/jejjohnson/geotoolz/issues/229)

## [0.1.0](https://github.com/jejjohnson/geotoolz/compare/geotoolz-catalog-v0.0.3...geotoolz-catalog-v0.1.0) (2026-07-13)


### ⚠ BREAKING CHANGES

* merge geopatcher + geocatalog into a uv workspace monorepo ([#97](https://github.com/jejjohnson/geotoolz/issues/97))

### Features

* merge geopatcher + geocatalog into a uv workspace monorepo ([#97](https://github.com/jejjohnson/geotoolz/issues/97)) ([202286c](https://github.com/jejjohnson/geotoolz/commit/202286c2e3ea976a18142bdbaba906211e17cba1))

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
