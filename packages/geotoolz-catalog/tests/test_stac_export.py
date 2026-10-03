"""`to_stac_collection` emits one item per scene and valid JSON (#239)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely


pystac = pytest.importorskip("pystac")

from geocatalog import InMemoryGeoCatalog, from_stac_items, to_stac_collection
from tests.conftest import assert_catalogs_equal


def _scene(item_id: str, x: float, day: int, bands: dict[str, str]) -> Any:
    item = pystac.Item(
        id=item_id,
        geometry=shapely.geometry.mapping(shapely.box(x, 0, x + 1, 1)),
        bbox=[x, 0, x + 1, 1],
        datetime=datetime(2024, 6, day, 10, tzinfo=UTC),
        properties={"proj:code": "EPSG:32630", "eo:cloud_cover": 10.0 + day},
        collection="s2",
    )
    for key, href in bands.items():
        item.add_asset(key, pystac.Asset(href=href))
    return item


def _two_scenes() -> list[Any]:
    return [
        _scene("A", 0, 1, {"B04": "https://x/A_B04.tif", "B08": "https://x/A_B08.tif"}),
        _scene("B", 5, 2, {"B04": "https://x/B_B04.tif", "B08": "https://x/B_B08.tif"}),
    ]


def test_one_item_per_scene_with_all_its_assets() -> None:
    catalog = from_stac_items(_two_scenes(), asset_key="*")
    assert len(catalog) == 4

    items = {
        i.id: i for i in to_stac_collection(catalog, collection_id="s2").get_items()
    }

    assert sorted(items) == ["A", "B"]
    assert set(items["A"].assets) == {"B04", "B08"}
    assert items["A"].assets["B08"].href == "https://x/A_B08.tif"


def test_round_trip_through_stac_is_lossless() -> None:
    catalog = from_stac_items(
        _two_scenes(), asset_key="*", extra_properties=("eo:cloud_cover",)
    )
    collection = to_stac_collection(catalog, collection_id="s2")
    back = from_stac_items(
        collection.get_items(), asset_key="*", extra_properties=("eo:cloud_cover",)
    )
    assert_catalogs_equal(back, catalog)


def test_ids_come_from_the_bundle_id_column() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1)],
            "start_time": [pd.Timestamp("2024-06-01")],
            "end_time": [pd.Timestamp("2024-06-01")],
            "filepath": ["https://x/a.tif"],
            "id": ["granule-123"],
        },
        crs="EPSG:4326",
    )
    collection = to_stac_collection(
        InMemoryGeoCatalog(gdf, backend="raster"), collection_id="c"
    )
    (item,) = collection.get_items()
    assert item.id == "granule-123"
    assert "id" not in item.properties


def test_rows_without_an_id_get_distinct_ids() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
            "start_time": [pd.Timestamp("2024-06-01")] * 2,
            "end_time": [pd.Timestamp("2024-06-01")] * 2,
            "filepath": ["https://x/a.tif", "https://x/b.tif"],
        },
        crs="EPSG:4326",
    )
    collection = to_stac_collection(
        InMemoryGeoCatalog(gdf, backend="raster"), collection_id="c"
    )
    assert sorted(i.id for i in collection.get_items()) == ["c-0", "c-1"]


def test_duplicate_asset_in_one_item_raises() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1)] * 2,
            "start_time": [pd.Timestamp("2024-06-01")] * 2,
            "end_time": [pd.Timestamp("2024-06-01")] * 2,
            "filepath": ["https://x/a.tif", "https://x/b.tif"],
            "stac_item_id": ["same", "same"],
            "asset_key": ["B04", "B04"],
        },
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match=r"'same'.*'B04'"):
        to_stac_collection(InMemoryGeoCatalog(gdf, backend="raster"), collection_id="c")


def test_properties_are_json_serialisable() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
            "start_time": [pd.Timestamp("2024-06-01")] * 2,
            "end_time": [pd.Timestamp("2024-06-02")] * 2,
            "filepath": ["https://x/a.tif", "https://x/b.tif"],
            "orbit": np.array([7, 8], dtype=np.int64),
            "cloud": [np.nan, 3.5],
            "acquired": [pd.NaT, pd.Timestamp("2024-06-01T10:00", tz="UTC")],
            "bands": [np.array([1, 2]), np.array([3, 4])],
            "meta": [{"gain": np.float32(1.5), "missing": None}, {"gain": 2.0}],
        },
        crs="EPSG:4326",
    )
    collection = to_stac_collection(
        InMemoryGeoCatalog(gdf, backend="raster"), collection_id="c"
    )
    for item in collection.get_items():
        json.dumps(item.to_dict(), allow_nan=False)
    json.dumps(collection.to_dict(), allow_nan=False)

    first, second = sorted(collection.get_items(), key=lambda i: i.id)
    assert first.properties["orbit"] == 7
    assert "cloud" not in first.properties  # NaN omitted
    assert "acquired" not in first.properties  # NaT omitted
    assert first.properties["bands"] == [1, 2]
    assert first.properties["meta"] == {"gain": 1.5}
    assert second.properties["acquired"] == "2024-06-01T10:00:00Z"


# ---------------------------------------------------------------------------
# Review follow-ups (#360)
# ---------------------------------------------------------------------------


def test_same_item_id_in_two_collections_stays_two_items() -> None:
    a = _scene("X", 0, 1, {"B04": "https://x/s2_B04.tif"})
    b = _scene("X", 5, 2, {"B04": "https://x/l8_B04.tif"})
    b.collection_id = "l8"
    catalog = from_stac_items([a, b], asset_key="*")

    items = {
        i.id: i for i in to_stac_collection(catalog, collection_id="mix").get_items()
    }

    assert sorted(items) == ["l8:X", "s2:X"]
    assert items["l8:X"].assets["B04"].href == "https://x/l8_B04.tif"


def test_numpy_datetimes_and_0d_arrays_are_serialised() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1)],
            "start_time": [pd.Timestamp("2024-06-01")],
            "end_time": [pd.Timestamp("2024-06-01")],
            "filepath": ["https://x/a.tif"],
            "times": [
                np.array(["2024-06-01T10:00:00.123456789"], dtype="datetime64[ns]")
            ],
            "scalar": [np.array(7)],
        },
        crs="EPSG:4326",
    )
    (item,) = to_stac_collection(
        InMemoryGeoCatalog(gdf, backend="raster"), collection_id="c"
    ).get_items()
    json.dumps(item.to_dict(), allow_nan=False)
    assert item.properties["times"][0].startswith("2024-06-01T10:00:00")
    assert item.properties["scalar"] == 7


def _gdf(**columns: Any) -> InMemoryGeoCatalog:
    n = len(next(iter(columns.values())))
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.box(i, 0, i + 1, 1) for i in range(n)],
            "start_time": [pd.Timestamp("2024-06-01")] * n,
            "end_time": [pd.Timestamp("2024-06-01")] * n,
            **columns,
        },
        crs="EPSG:4326",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


def test_bundle_scope_separates_reused_ids() -> None:
    catalog = _gdf(
        filepath=["https://x/a.tif", "https://x/b.tif"],
        id=["X", "X"],
        source=["stac.pc", "stac.pc"],
        collection=["s2", "l8"],
    )
    ids = sorted(
        i.id for i in to_stac_collection(catalog, collection_id="c").get_items()
    )
    assert ids == ["stac.pc/l8:X", "stac.pc/s2:X"]


def test_anonymous_rows_never_merge_with_an_explicit_id() -> None:
    catalog = _gdf(filepath=["https://x/a.tif", "https://x/b.tif"], id=[None, "c-0"])
    items = {
        i.id: i for i in to_stac_collection(catalog, collection_id="c").get_items()
    }
    assert sorted(items) == ["c-0", "c-0~2"]
    assert items["c-0"].assets["data"].href == "https://x/b.tif"  # explicit id kept


def test_qualified_ids_do_not_collide_with_existing_ids() -> None:
    catalog = _gdf(
        filepath=["https://x/1.tif", "https://x/2.tif", "https://x/3.tif"],
        stac_item_id=["X", "X", "s:X"],
        stac_collection=["s", "t", "u"],
    )
    ids = [i.id for i in to_stac_collection(catalog, collection_id="c").get_items()]
    assert len(ids) == len(set(ids)) == 3


def test_item_crs_only_when_every_asset_shares_it() -> None:
    catalog = _gdf(
        filepath=["https://x/B04.tif", "https://x/B08.tif"],
        stac_item_id=["S", "S"],
        asset_key=["B04", "B08"],
        crs=["EPSG:32633", None],
    )
    (item,) = to_stac_collection(catalog, collection_id="c").get_items()
    assert not any(k.startswith("proj:") for k in item.properties)
    assert item.assets["B04"].extra_fields == {"proj:epsg": 32633}
    assert item.assets["B08"].extra_fields == {}


# ---------------------------------------------------------------------------
# Review follow-ups (#360)
# ---------------------------------------------------------------------------


def test_scopes_that_flatten_alike_stay_separate_scenes() -> None:
    catalog = _gdf(
        filepath=["https://x/1.tif", "https://x/2.tif"],
        id=["X", "X"],
        source=["a", "a/b"],
        collection=["b/c", "c"],
    )
    items = list(to_stac_collection(catalog, collection_id="c").get_items())
    assert len(items) == 2
    assert {len(i.assets) for i in items} == {1}


def test_crs_stored_as_a_projjson_mapping_is_exported() -> None:
    import pyproj

    projjson = pyproj.CRS("EPSG:32633").to_json_dict()
    catalog = _gdf(
        filepath=["https://x/B04.tif", "https://x/B08.tif"],
        stac_item_id=["S", "S"],
        asset_key=["B04", "B08"],
        crs=[projjson, "EPSG:32633"],
    )
    (item,) = to_stac_collection(catalog, collection_id="c").get_items()
    assert item.properties["proj:epsg"] == 32633  # equal CRSs are shared
    assert item.assets["B04"].extra_fields == {}


def test_carried_projection_survives_without_a_crs_column() -> None:
    catalog = _gdf(filepath=["https://x/a.tif"], stac_item_id=["S"])
    catalog.gdf["proj:epsg"] = [32633]
    (item,) = to_stac_collection(catalog, collection_id="c").get_items()
    assert item.properties["proj:epsg"] == 32633


def test_missing_scope_positions_keep_scenes_apart() -> None:
    catalog = _gdf(
        filepath=["https://x/1.tif", "https://x/2.tif"],
        id=["X", "X"],
        source=["a", None],
        collection=[None, "a"],
    )
    items = list(to_stac_collection(catalog, collection_id="c").get_items())
    assert len(items) == 2
