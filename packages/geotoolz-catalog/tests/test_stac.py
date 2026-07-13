"""Tests for STAC catalog import/export helpers. Skipped without [stac]."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry


pystac = pytest.importorskip("pystac")

import geocatalog as gc


def _item(
    item_id: str,
    *,
    href: str,
    datetime: str | None = "2024-06-01T10:00:00Z",
    start_datetime: str | None = None,
    end_datetime: str | None = None,
) -> pystac.Item:
    props: dict[str, object] = {
        "proj:epsg": 32629,
        "eo:cloud_cover": 12.5,
    }
    if start_datetime is not None:
        props["start_datetime"] = start_datetime
    if end_datetime is not None:
        props["end_datetime"] = end_datetime
    item = pystac.Item(
        id=item_id,
        geometry=shapely.geometry.mapping(shapely.geometry.box(-3.8, 40.3, -3.6, 40.5)),
        bbox=(-3.8, 40.3, -3.6, 40.5),
        datetime=pd.Timestamp(datetime).to_pydatetime()
        if datetime is not None
        else None,
        properties=props,
        collection="sentinel-2-l2a",
    )
    item.add_asset("B04", pystac.Asset(href=href))
    return item


def test_from_stac_items_preserves_asset_and_extra_properties() -> None:
    cat = gc.from_stac_items(
        [_item("s2-a", href="s3://bucket/B04.tif")],
        asset_key="B04",
        extra_properties=("eo:cloud_cover",),
    )

    assert len(cat) == 1
    assert cat.backend == "raster"
    assert cat.gdf.crs == "EPSG:4326"
    row = next(cat.iter_rows())
    assert row.filepath == "s3://bucket/B04.tif"
    assert row.interval.left == pd.Timestamp("2024-06-01T10:00:00Z").tz_convert(None)
    assert row.extras["asset_key"] == "B04"
    assert row.extras["crs"] == "EPSG:32629"
    assert row.extras["eo:cloud_cover"] == 12.5
    assert tuple(round(value, 1) for value in cat.total_bounds) == (
        -3.8,
        40.3,
        -3.6,
        40.5,
    )


def test_from_stac_items_uses_date_range_when_datetime_is_missing() -> None:
    cat = gc.from_stac_items(
        [
            _item(
                "s2-range",
                href="https://example.test/B04.tif",
                datetime=None,
                start_datetime="2024-06-01T00:00:00Z",
                end_datetime="2024-06-30T23:59:59Z",
            )
        ],
        asset_key="B04",
    )

    assert cat.temporal_extent == pd.Interval(
        pd.Timestamp("2024-06-01T00:00:00Z").tz_convert(None),
        pd.Timestamp("2024-06-30T23:59:59Z").tz_convert(None),
        closed="both",
    )


def test_from_stac_items_can_expand_all_assets() -> None:
    item = _item("s2-multi", href="https://example.test/B04.tif")
    item.add_asset("B08", pystac.Asset(href="https://example.test/B08.tif"))

    cat = gc.from_stac_items([item], asset_key="*")

    assert len(cat) == 2
    assert set(cat.gdf["asset_key"]) == {"B04", "B08"}
    assert set(cat.gdf["filepath"]) == {
        "https://example.test/B04.tif",
        "https://example.test/B08.tif",
    }


def test_from_stac_items_empty_returns_empty_catalog() -> None:
    cat = gc.from_stac_items([], asset_key="B04")

    assert len(cat) == 0
    assert cat.backend == "raster"
    assert cat.gdf.crs == "EPSG:4326"


def test_from_stac_items_prefers_per_asset_proj_epsg() -> None:
    item = _item("s2-asset-crs", href="s3://bucket/B04.tif")
    # STAC projection extension: per-asset proj fields override item-level.
    item.assets["B04"].extra_fields["proj:epsg"] = 32630

    cat = gc.from_stac_items([item], asset_key="B04")

    row = next(cat.iter_rows())
    assert row.extras["crs"] == "EPSG:32630"


def test_to_stac_collection_round_trips_with_pystac(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.box(-3.8, 40.3, -3.6, 40.5)],
            "start_time": [pd.Timestamp("2024-06-01")],
            "end_time": [pd.Timestamp("2024-06-01")],
            "filepath": ["https://example.test/B04.tif"],
            "asset_key": ["B04"],
            "stac_item_id": ["s2-a"],
            "crs": ["EPSG:32629"],
            "eo:cloud_cover": [12.5],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    catalog = gc.InMemoryGeoCatalog(gdf, backend="raster")

    collection = gc.to_stac_collection(
        catalog,
        collection_id="example",
        description="Example collection",
    )
    collection.normalize_hrefs(str(tmp_path))
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    reopened = pystac.Collection.from_file(tmp_path / "collection.json")
    items = list(reopened.get_items())
    assert reopened.id == "example"
    assert len(items) == 1
    assert items[0].id == "s2-a"
    assert items[0].assets["B04"].href == "https://example.test/B04.tif"
    assert items[0].properties["proj:epsg"] == 32629
    assert items[0].properties["eo:cloud_cover"] == 12.5


def test_from_stac_items_prefers_temporal_range_over_nominal_datetime() -> None:
    """Regression for the P1 bug where an item carrying both
    ``start_datetime``/``end_datetime`` AND ``datetime`` collapsed to a
    zero-width interval at ``datetime``. Per STAC 1.0 the range
    describes the acquisition window and should win.
    """
    item = _item(
        "s2-with-range-and-datetime",
        href="https://example.test/B04.tif",
        datetime="2024-06-15T12:00:00Z",
        start_datetime="2024-06-01T00:00:00Z",
        end_datetime="2024-06-30T23:59:59Z",
    )

    cat = gc.from_stac_items([item], asset_key="B04")

    row = next(cat.iter_rows())
    assert row.interval.left == pd.Timestamp("2024-06-01T00:00:00Z").tz_convert(None)
    assert row.interval.right == pd.Timestamp("2024-06-30T23:59:59Z").tz_convert(None)


def test_to_stac_collection_emits_rfc3339_datetimes(tmp_path: Path) -> None:
    """Regression: ``start_datetime``/``end_datetime`` (and the item's
    ``datetime`` field) must be tz-aware RFC3339, not the naive
    ``datetime.isoformat()`` output that strict STAC validators reject.
    """
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                shapely.geometry.box(-3.8, 40.3, -3.6, 40.5),
                shapely.geometry.box(-3.8, 40.3, -3.6, 40.5),
            ],
            # Row 0: a real time range; row 1: a single instant.
            "start_time": [
                pd.Timestamp("2024-06-01T00:00:00"),
                pd.Timestamp("2024-06-15T12:00:00"),
            ],
            "end_time": [
                pd.Timestamp("2024-06-30T23:59:59"),
                pd.Timestamp("2024-06-15T12:00:00"),
            ],
            "filepath": [
                "https://example.test/A.tif",
                "https://example.test/B.tif",
            ],
            "asset_key": ["B04", "B04"],
            "stac_item_id": ["range-row", "instant-row"],
            "crs": ["EPSG:4326", "EPSG:4326"],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    catalog = gc.InMemoryGeoCatalog(gdf, backend="raster")

    collection = gc.to_stac_collection(
        catalog,
        collection_id="rfc3339",
        description="RFC3339 datetime regression",
    )
    items = {item.id: item for item in collection.get_items()}

    range_item = items["range-row"]
    start = range_item.properties["start_datetime"]
    end = range_item.properties["end_datetime"]
    assert isinstance(start, str) and isinstance(end, str)
    assert start.endswith("Z") or start.endswith("+00:00")
    assert end.endswith("Z") or end.endswith("+00:00")
    # Item-level datetime is None when a range is in play.
    assert range_item.datetime is None

    instant_item = items["instant-row"]
    # When start == end we collapse to item.datetime; it must be tz-aware.
    assert instant_item.datetime is not None
    assert instant_item.datetime.tzinfo is not None

    # Round-trip through pystac's own JSON serialisation to confirm
    # canonical output. pystac formats tz-aware UTC datetimes as `Z`.
    collection.normalize_hrefs(str(tmp_path))
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    reopened_items = {
        item.id: item
        for item in pystac.Collection.from_file(
            tmp_path / "collection.json"
        ).get_items()
    }
    rt_start = reopened_items["range-row"].properties["start_datetime"]
    rt_end = reopened_items["range-row"].properties["end_datetime"]
    assert rt_start.endswith("Z") or rt_start.endswith("+00:00")
    assert rt_end.endswith("Z") or rt_end.endswith("+00:00")
