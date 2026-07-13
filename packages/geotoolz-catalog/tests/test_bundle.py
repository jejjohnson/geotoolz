"""End-to-end tests for `CatalogBundle` — ingest + matchup + persistence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyproj
import pytest
from shapely.geometry import box

from geocatalog._src.matchup import (
    Intersects,
    NearestInTime,
    matchup,
)
from geocatalog._src.sources._base import (
    AuthStatus,
    Source,
    SourceRow,
)
from geocatalog.bundle import CatalogBundle, source_row_to_gdf_row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeSource(Source):
    """In-test `Source` that yields a fixed set of SourceRows.

    Captures the kwargs `query()` was called with so tests can assert
    the bundle forwarded them correctly.
    """

    name = "fake"

    def __init__(self, rows: list[SourceRow]) -> None:
        self._rows = rows
        self.last_query_kwargs: dict[str, Any] | None = None

    def query(
        self,
        bounds,
        interval=None,
        *,
        collection=None,
        filters=None,
        limit=None,
    ) -> Iterator[SourceRow]:
        self.last_query_kwargs = {
            "bounds": bounds,
            "interval": interval,
            "collection": collection,
            "filters": filters,
            "limit": limit,
        }
        yield from self._rows

    def auth_status(self) -> AuthStatus:
        return AuthStatus(source=self.name, authenticated=True)


def _src_row(
    id_: str,
    *,
    source: str = "fake",
    collection: str = "test-collection",
    bbox: tuple[float, float, float, float] = (-9.0, 38.0, -8.0, 39.0),
    time: datetime,
    duration: timedelta = timedelta(0),
    assets: dict[str, str] | None = None,
    properties: dict[str, Any] | None = None,
) -> SourceRow:
    geom = box(*bbox)
    # `assets is None` rather than `assets or` so callers can pass
    # `assets={}` and have it respected (empty dict is falsy under `or`).
    resolved_assets = {"data": f"s3://bucket/{id_}.tif"} if assets is None else assets
    return SourceRow(
        id=id_,
        source=source,
        collection=collection,
        geometry=geom,
        interval=pd.Interval(
            pd.Timestamp(time), pd.Timestamp(time + duration), closed="both"
        ),
        assets=resolved_assets,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# source_row_to_gdf_row
# ---------------------------------------------------------------------------


class TestSourceRowToGdfRow:
    def test_basic_columns(self) -> None:
        row = _src_row(
            "abc",
            time=datetime(2024, 6, 15, 12, tzinfo=UTC),
            assets={"red": "s3://red.tif", "nir": "s3://nir.tif"},
            properties={"eo:cloud_cover": 12.4},
        )
        d = source_row_to_gdf_row(row, target_crs=pyproj.CRS.from_epsg(4326))
        assert d["id"] == "abc"
        assert d["source"] == "fake"
        assert d["collection"] == "test-collection"
        # First asset key wins by default (dict iteration order).
        assert d["filepath"] == "s3://red.tif"
        # JSON-encoded dicts so Parquet doesn't have to deal with nested.
        assert json.loads(d["assets"]) == {
            "red": "s3://red.tif",
            "nir": "s3://nir.tif",
        }
        assert json.loads(d["properties"])["eo:cloud_cover"] == 12.4
        assert d["geometry"].bounds == (-9.0, 38.0, -8.0, 39.0)

    def test_primary_asset_kwarg_picks_specific_key(self) -> None:
        row = _src_row(
            "x",
            time=datetime(2024, 6, 15, tzinfo=UTC),
            assets={"red": "r", "nir": "n", "scl": "s"},
        )
        d = source_row_to_gdf_row(
            row, target_crs=pyproj.CRS.from_epsg(4326), primary_asset="nir"
        )
        assert d["filepath"] == "n"

    def test_primary_asset_missing_falls_back_to_first(self) -> None:
        row = _src_row(
            "x",
            time=datetime(2024, 6, 15, tzinfo=UTC),
            assets={"red": "r"},
        )
        # Asking for "nir" but it's not in the assets → falls back
        # to the first asset, not crashes.
        d = source_row_to_gdf_row(
            row, target_crs=pyproj.CRS.from_epsg(4326), primary_asset="nir"
        )
        assert d["filepath"] == "r"

    def test_empty_assets_yields_empty_filepath(self) -> None:
        row = _src_row("x", time=datetime(2024, 6, 15, tzinfo=UTC), assets={})
        d = source_row_to_gdf_row(row, target_crs=pyproj.CRS.from_epsg(4326))
        assert d["filepath"] == ""

    def test_reprojects_geometry_to_target_crs(self) -> None:
        # Build a row with a geometry near Iberia in EPSG:4326; reproject
        # to UTM 29N. The resulting geometry's bounds should be far
        # from (-9, 38) — typical UTM Y is ~4_000_000.
        row = _src_row(
            "x",
            bbox=(-9.0, 38.0, -8.5, 38.5),
            time=datetime(2024, 6, 15, tzinfo=UTC),
        )
        d = source_row_to_gdf_row(row, target_crs=pyproj.CRS.from_epsg(32629))
        xmin, ymin, _xmax, _ymax = d["geometry"].bounds
        # Latitude 38 ~ 4.2M N in UTM; reasonable sanity check.
        assert 4_000_000 < ymin < 5_000_000
        # The original WGS84 bbox is _not_ what came out.
        assert xmin != -9.0


# ---------------------------------------------------------------------------
# CatalogBundle.empty + ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_empty_bundle_has_zero_items(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        assert bundle.n_items == 0
        assert bundle.queries == []
        assert bundle.matchups == []
        assert bundle.target_crs == pyproj.CRS.from_epsg(4326)

    def test_ingest_appends_rows(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src = _FakeSource(
            [
                _src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC)),
                _src_row("b", time=datetime(2024, 6, 15, tzinfo=UTC)),
                _src_row("c", time=datetime(2024, 6, 16, tzinfo=UTC)),
            ]
        )
        query_id = bundle.ingest(
            src,
            bounds=(-10, 35, 5, 45),
            interval=pd.Interval(
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"
            ),
            collection="test-collection",
            tag="iberia_test",
        )
        assert bundle.n_items == 3
        assert len(bundle.queries) == 1
        assert bundle.queries[0].query_id == query_id
        assert bundle.queries[0].n_returned == 3
        assert bundle.queries[0].tag == "iberia_test"

    def test_ingest_forwards_kwargs_to_source(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src = _FakeSource([])
        bundle.ingest(
            src,
            bounds=(-10, 35, 5, 45),
            collection="sentinel-2-l2a",
            filters={"eo:cloud_cover": {"lt": 20}},
            limit=5,
        )
        assert src.last_query_kwargs is not None
        assert src.last_query_kwargs["bounds"] == (-10, 35, 5, 45)
        assert src.last_query_kwargs["collection"] == "sentinel-2-l2a"
        assert src.last_query_kwargs["filters"] == {"eo:cloud_cover": {"lt": 20}}
        assert src.last_query_kwargs["limit"] == 5

    def test_ingest_stamps_query_id_on_row_provenance(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src = _FakeSource([_src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC))])
        query_id = bundle.ingest(src, bounds=(-10, 35, 5, 45), tag="run1")

        # The provenance column is JSON-encoded; pull and decode.
        prov_json = bundle.catalog.gdf["provenance"].iloc[0]
        prov = json.loads(prov_json)
        assert prov["query_id"] == query_id
        assert prov["query_tag"] == "run1"

    def test_ingest_preserves_existing_provenance(self) -> None:
        # If the adapter already set provenance fields, ingest should
        # not blow them away — only fill in `query_id` if missing.
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        row = _src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC))
        # Mutate one row's provenance via dataclasses.replace.
        import dataclasses

        rich_row = dataclasses.replace(
            row, provenance={"query_id": "preset", "extra_field": "yes"}
        )
        src = _FakeSource([rich_row])
        new_id = bundle.ingest(src, bounds=(-10, 35, 5, 45))
        prov = json.loads(bundle.catalog.gdf["provenance"].iloc[0])
        # Adapter's query_id is preserved (not overwritten).
        assert prov["query_id"] == "preset"
        assert prov["extra_field"] == "yes"
        # The bundle's QueryRecord still records the new call's UUID.
        assert bundle.queries[0].query_id == new_id

    def test_ingest_does_not_overwrite_adapter_query_tag(self) -> None:
        # If an adapter already stamped `query_tag` on the row's
        # provenance, the user's `tag` argument must not clobber it
        # — same "do not overwrite" contract as `query_id`.
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        import dataclasses as _dc

        adapter_tagged = _dc.replace(
            _src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC)),
            provenance={"query_tag": "adapter_set"},
        )
        src = _FakeSource([adapter_tagged])
        bundle.ingest(src, bounds=(-10, 35, 5, 45), tag="user_tag")
        prov = json.loads(bundle.catalog.gdf["provenance"].iloc[0])
        # Adapter's tag wins; bundle-level tag still lives in
        # QueryRecord.tag for the queries.parquet table.
        assert prov["query_tag"] == "adapter_set"
        assert bundle.queries[0].tag == "user_tag"

    def test_two_ingests_accumulate(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src_a = _FakeSource([_src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC))])
        src_b = _FakeSource(
            [
                _src_row("b1", time=datetime(2024, 6, 15, tzinfo=UTC)),
                _src_row("b2", time=datetime(2024, 6, 16, tzinfo=UTC)),
            ]
        )
        bundle.ingest(src_a, bounds=(-10, 35, 5, 45))
        bundle.ingest(src_b, bounds=(-10, 35, 5, 45))
        assert bundle.n_items == 3
        assert len(bundle.queries) == 2

    def test_ingest_reprojects_to_bundle_crs(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:32629")
        src = _FakeSource(
            [
                _src_row(
                    "iberia",
                    bbox=(-9.0, 38.0, -8.5, 38.5),
                    time=datetime(2024, 6, 14, tzinfo=UTC),
                )
            ]
        )
        bundle.ingest(src, bounds=(-10, 35, 5, 45))
        # Catalog CRS is UTM 29N; geometries should be in UTM units.
        assert str(bundle.catalog.gdf.crs).startswith("EPSG:32629")
        _xmin, ymin, _, _ = bundle.catalog.gdf.geometry.iloc[0].bounds
        assert 4_000_000 < ymin < 5_000_000


# ---------------------------------------------------------------------------
# write_matchups
# ---------------------------------------------------------------------------


class TestWriteMatchups:
    def test_writes_iterable_of_matchup_rows(self) -> None:
        # Build a small matchup against two fake sources and write
        # the rows into the bundle.

        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        primary = [
            _src_row("p1", time=datetime(2024, 6, 15, 12, tzinfo=UTC)),
            _src_row("p2", time=datetime(2024, 6, 16, 12, tzinfo=UTC)),
        ]
        secondaries = [
            _src_row(
                "s1",
                source="other",
                time=datetime(2024, 6, 15, 13, tzinfo=UTC),
            ),
            _src_row(
                "s2",
                source="other",
                time=datetime(2024, 6, 16, 13, tzinfo=UTC),
            ),
        ]
        rows = matchup(
            primary=primary,
            secondary=secondaries,
            spatial=Intersects(),
            temporal=NearestInTime(dt="6h"),
        )
        n = bundle.write_matchups(rows, tag="round_one")
        assert n == 2
        assert len(bundle.matchups) == 2
        assert all(m.query_set == "round_one" for m in bundle.matchups)


# ---------------------------------------------------------------------------
# Round-trip — to_directory / from_directory
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    def test_roundtrip_items_only(self, tmp_path: Path) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src = _FakeSource(
            [
                _src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC)),
                _src_row("b", time=datetime(2024, 6, 15, tzinfo=UTC)),
            ]
        )
        bundle.ingest(src, bounds=(-10, 35, 5, 45), tag="t1")
        bundle.to_directory(tmp_path / "cat")

        # Directory layout: items.parquet + queries.parquet + _meta.json.
        files = {p.name for p in (tmp_path / "cat").iterdir()}
        assert {"items.parquet", "queries.parquet", "_meta.json"} <= files
        # matchups.parquet not written because matchups is empty.
        assert "matchups.parquet" not in files

        # Round-trip.
        reloaded = CatalogBundle.from_directory(tmp_path / "cat")
        assert reloaded.n_items == 2
        assert len(reloaded.queries) == 1
        assert reloaded.queries[0].tag == "t1"
        assert reloaded.target_crs == pyproj.CRS.from_epsg(4326)

    def test_roundtrip_with_matchups(self, tmp_path: Path) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        primary = [_src_row("p", time=datetime(2024, 6, 15, 12, tzinfo=UTC))]
        secondaries = [
            _src_row(
                "s",
                source="other",
                time=datetime(2024, 6, 15, 13, tzinfo=UTC),
            )
        ]
        rows = list(
            matchup(
                primary=primary,
                secondary=secondaries,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
                tag="my_pairs",
            )
        )
        bundle.write_matchups(rows)
        bundle.to_directory(tmp_path / "cat")

        reloaded = CatalogBundle.from_directory(tmp_path / "cat")
        assert len(reloaded.matchups) == 1
        m = reloaded.matchups[0]
        assert m.member_ids == ("p", "s")
        assert m.query_set == "my_pairs"
        # Strategy + tolerance survived the JSON round-trip.
        assert "Intersects" in m.strategy or "NearestInTime" in m.strategy
        assert m.tolerance["join"] == "all"

    def test_from_directory_rejects_file(self, tmp_path: Path) -> None:
        # A plain Parquet file path is not a bundle.
        plain_file = tmp_path / "items.parquet"
        plain_file.write_bytes(b"not actually a parquet file")
        with pytest.raises(NotADirectoryError, match="expects a directory"):
            CatalogBundle.from_directory(plain_file)

    def test_from_directory_rejects_missing_meta(self, tmp_path: Path) -> None:
        # A directory without _meta.json is not a bundle.
        d = tmp_path / "not-a-bundle"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match=r"_meta\.json"):
            CatalogBundle.from_directory(d)

    def test_stale_sidecar_files_cleaned_on_rewrite(self, tmp_path: Path) -> None:
        # First write: bundle has queries + matchups → sibling files
        # present.
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src_a = _FakeSource([_src_row("a", time=datetime(2024, 6, 14, tzinfo=UTC))])
        bundle.ingest(src_a, bounds=(-10, 35, 5, 45))
        bundle.write_matchups(
            matchup(
                primary=[_src_row("p", time=datetime(2024, 6, 15, tzinfo=UTC))],
                secondary=[
                    _src_row("s", source="o", time=datetime(2024, 6, 15, 1, tzinfo=UTC))
                ],
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
            )
        )
        bundle.to_directory(tmp_path / "cat")
        assert (tmp_path / "cat" / "queries.parquet").exists()
        assert (tmp_path / "cat" / "matchups.parquet").exists()

        # Second write: clear the in-memory queries + matchups; rewrite.
        # The stale sibling files must disappear so a subsequent
        # `from_directory()` doesn't resurrect them.
        bundle.queries.clear()
        bundle.matchups.clear()
        bundle.to_directory(tmp_path / "cat")
        assert not (tmp_path / "cat" / "queries.parquet").exists()
        assert not (tmp_path / "cat" / "matchups.parquet").exists()
        reloaded = CatalogBundle.from_directory(tmp_path / "cat")
        assert len(reloaded.queries) == 0
        assert len(reloaded.matchups) == 0

    def test_from_directory_rejects_unknown_schema_version(
        self, tmp_path: Path
    ) -> None:
        # Tamper _meta.json to claim a future version we don't know
        # how to read; the loader must fail fast rather than silently
        # misinterpret the layout.
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        bundle.to_directory(tmp_path / "cat")
        meta_path = tmp_path / "cat" / "_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["bundle_schema_version"] = 999
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="bundle_schema_version"):
            CatalogBundle.from_directory(tmp_path / "cat")

    def test_from_directory_rejects_missing_schema_version(
        self, tmp_path: Path
    ) -> None:
        # Pre-versioning bundle (`bundle_schema_version` field absent).
        d = tmp_path / "cat"
        CatalogBundle.empty(target_crs="EPSG:4326").to_directory(d)
        meta_path = d / "_meta.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("bundle_schema_version")
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match=r"missing.*bundle_schema_version"):
            CatalogBundle.from_directory(d)


# ---------------------------------------------------------------------------
# queries_df / matchups_df DataFrame views
# ---------------------------------------------------------------------------


class TestDataFrameAccessors:
    def test_empty_returns_dataframe_with_columns(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        df = bundle.queries_df()
        assert "query_id" in df.columns
        assert "tag" in df.columns
        assert len(df) == 0

    def test_after_ingest_populated(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        src = _FakeSource([_src_row("a", time=datetime(2024, 6, 15, tzinfo=UTC))])
        bundle.ingest(src, bounds=(-10, 35, 5, 45), tag="t")
        df = bundle.queries_df()
        assert len(df) == 1
        assert df["tag"].iloc[0] == "t"

    def test_matchups_df_serializes_geometry_to_wkt(self) -> None:
        bundle = CatalogBundle.empty(target_crs="EPSG:4326")
        primary = [_src_row("p", time=datetime(2024, 6, 15, tzinfo=UTC))]
        secondary = [
            _src_row("s", source="o", time=datetime(2024, 6, 15, 1, tzinfo=UTC))
        ]
        bundle.write_matchups(
            matchup(
                primary=primary,
                secondary=secondary,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
            )
        )
        df = bundle.matchups_df()
        # geometry_intersect column carries a WKT string for analysis.
        assert isinstance(df["geometry_intersect"].iloc[0], str)
        assert df["geometry_intersect"].iloc[0].startswith("POLYGON")
