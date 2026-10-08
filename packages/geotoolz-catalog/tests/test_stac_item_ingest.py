"""Both STAC ingest paths read an item the same way (#238).

`from_stac_items` (catalog builder) and `STACSource` (bundle adapter)
share one decoder: the real footprint (not the bbox), split at the
antimeridian; the asset CRS from ``proj:*``; UTC times; absolute hrefs
with expiring signatures flagged; densified reprojection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyproj
import pytest
import shapely
import shapely.geometry
import shapely.ops


pystac = pytest.importorskip("pystac")

from geocatalog._src._stac_item import is_signed_href, reproject_geometry
from geocatalog._src.bundle._catalog_bundle import source_row_to_gdf_row
from geocatalog._src.sources.stac import _item_to_source_row
from geocatalog.sources import from_stac_items


NOTCHED = shapely.Polygon([(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)])


def _item(
    *,
    geometry: shapely.geometry.base.BaseGeometry | None = NOTCHED,
    bbox: list[float] | None = None,
    dt: datetime | None = datetime(2024, 6, 15, 10, tzinfo=UTC),
    properties: dict[str, Any] | None = None,
    href: str = "https://example.com/B04.tif",
    asset_fields: dict[str, Any] | None = None,
    collection: str | None = "s2",
) -> Any:
    item = pystac.Item(
        id="item-1",
        geometry=None if geometry is None else shapely.geometry.mapping(geometry),
        bbox=bbox
        if bbox is not None
        else (list(geometry.bounds) if geometry else None),
        datetime=dt,
        properties=dict(properties or {}),
        collection=collection,
    )
    item.add_asset("data", pystac.Asset(href=href, extra_fields=asset_fields or {}))
    return item


def _source_row(item: Any) -> Any:
    return _item_to_source_row(
        item,
        source_name="stac.test",
        query_id="q",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_version="v",
    )


def _only_row(item: Any, **kwargs: Any) -> Any:
    catalog = from_stac_items([item], **kwargs)
    assert len(catalog) == 1
    return next(catalog.iter_rows())


# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------


def test_footprint_is_the_geometry_not_the_bbox() -> None:
    row = _only_row(_item())
    assert row.geometry.equals(NOTCHED)
    assert _source_row(_item()).geometry.equals(NOTCHED)


def test_antimeridian_ring_is_split() -> None:
    ring = shapely.Polygon([(178, -5), (-178, -5), (-178, 5), (178, 5)])
    item = _item(geometry=ring, bbox=[178, -5, -178, 5])
    for geom in (_only_row(item).geometry, _source_row(item).geometry):
        assert geom.geom_type == "MultiPolygon"
        assert geom.area == pytest.approx(40)


def test_antimeridian_bbox_without_geometry_is_split() -> None:
    item = _item(geometry=None, bbox=[179, -10, -179, 10])
    geom = _only_row(item).geometry
    assert geom.area == pytest.approx(40)
    assert not geom.covers(shapely.Point(0, 0))


# ---------------------------------------------------------------------------
# CRS from the projection extension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("properties", "asset_fields", "expected"),
    [
        ({"proj:code": "EPSG:32633"}, {}, "EPSG:32633"),
        ({"proj:epsg": 32630}, {}, "EPSG:32630"),
        ({"proj:code": "EPSG:32633"}, {"proj:code": "EPSG:32634"}, "EPSG:32634"),
        (
            {"proj:projjson": pyproj.CRS.from_epsg(3035).to_json_dict()},
            {},
            "EPSG:3035",
        ),
        ({}, {}, "EPSG:4326"),
    ],
    ids=["proj:code", "proj:epsg", "asset-overrides-item", "proj:projjson", "none"],
)
def test_asset_crs(properties: dict, asset_fields: dict, expected: str) -> None:
    row = _only_row(_item(properties=properties, asset_fields=asset_fields))
    assert row.extras["crs"] == expected


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_times_are_utc_on_both_paths() -> None:
    local = datetime.fromisoformat("2024-06-15T12:00:00+02:00")
    item = _item(dt=local)
    row = _only_row(item)
    assert row.interval.left == pd.Timestamp("2024-06-15T10:00")  # naive UTC
    assert _source_row(item).interval.left == pd.Timestamp("2024-06-15T10:00", tz="UTC")


def test_catalog_has_no_time_columns_beside_the_index() -> None:
    catalog = from_stac_items([_item()])
    assert "start_time" not in catalog.gdf.columns
    assert "end_time" not in catalog.gdf.columns


def test_empty_catalog_has_the_same_columns() -> None:
    full = from_stac_items([_item()])
    empty = from_stac_items([])
    assert list(empty.gdf.columns) == list(full.gdf.columns)
    assert "stac_collection" in empty.gdf.columns


def test_extra_properties_cannot_overwrite_core_columns() -> None:
    with pytest.raises(ValueError, match=r"\['filepath'\]"):
        from_stac_items([_item()], extra_properties=("filepath",))


# ---------------------------------------------------------------------------
# Hrefs
# ---------------------------------------------------------------------------


def test_relative_href_is_resolved_against_the_item() -> None:
    item = _item(href="./B04.tif")
    item.set_self_href("https://example.com/items/item-1.json")
    assert _only_row(item).filepath == "https://example.com/items/B04.tif"
    assert _source_row(item).assets["data"] == "https://example.com/items/B04.tif"


SIGNED = (
    "https://acct.blob.core.windows.net/c/B04.tif"
    "?st=2024-06-15T09%3A00Z&se=2024-06-15T10%3A00Z&sp=rl&sv=2021&sr=c&sig=abc%3D"
)


def test_signed_hrefs_are_flagged() -> None:
    assert is_signed_href(SIGNED)
    assert not is_signed_href("https://example.com/B04.tif?version=2")
    assert bool(_only_row(_item(href=SIGNED)).extras["href_signed"]) is True
    assert bool(_only_row(_item()).extras["href_signed"]) is False

    gdf_row = source_row_to_gdf_row(
        _source_row(_item(href=SIGNED)), crs=pyproj.CRS.from_epsg(4326)
    )
    assert gdf_row["href_signed"] is True


# ---------------------------------------------------------------------------
# Densified reprojection
# ---------------------------------------------------------------------------


def test_reprojection_is_densified() -> None:
    tile = shapely.box(0, 40, 15, 50)  # 15° x 10°
    to_laea = pyproj.Transformer.from_crs(4326, 3035, always_xy=True).transform
    reference = shapely.ops.transform(to_laea, shapely.segmentize(tile, 0.01))
    corners_only = shapely.ops.transform(to_laea, tile)
    got = reproject_geometry(tile, "EPSG:4326", "EPSG:3035")

    assert abs(corners_only.area / reference.area - 1) > 0.005  # the bug
    assert got.area == pytest.approx(reference.area, rel=1e-3)

    row = _only_row(_item(geometry=tile), crs="EPSG:3035")
    assert row.geometry.area == pytest.approx(reference.area, rel=1e-3)


# ---------------------------------------------------------------------------
# Review follow-ups (#359)
# ---------------------------------------------------------------------------


def test_any_asset_projection_field_overrides_the_item() -> None:
    # v2 `proj:code` on the item, v1 `proj:epsg` on the asset: asset wins.
    row = _only_row(
        _item(properties={"proj:code": "EPSG:32633"}, asset_fields={"proj:epsg": 32634})
    )
    assert row.extras["crs"] == "EPSG:32634"


def test_3d_antimeridian_polygon_is_split() -> None:
    ring = [(178, -5, 10), (-178, -5, 10), (-178, 5, 10), (178, 5, 10), (178, -5, 10)]
    item = _item(geometry=shapely.Polygon(ring), bbox=[178, -5, -178, 5])
    for geom in (_only_row(item).geometry, _source_row(item).geometry):
        assert geom.geom_type == "MultiPolygon"
        assert geom.area == pytest.approx(40)
        assert not geom.has_z


def test_antimeridian_line_is_split() -> None:
    line = shapely.LineString([(179, 0), (-179, 1)])
    item = _item(geometry=line, bbox=[179, 0, -179, 1])
    geom = _source_row(item).geometry
    assert geom.geom_type == "MultiLineString"
    assert geom.length == pytest.approx((2**2 + 1**2) ** 0.5)


def test_bundle_row_carries_the_asset_crs() -> None:
    item = _item(
        properties={"proj:code": "EPSG:32633"}, asset_fields={"proj:code": "EPSG:32601"}
    )
    gdf_row = source_row_to_gdf_row(_source_row(item), crs=pyproj.CRS.from_epsg(4326))
    assert gdf_row["crs"] == "EPSG:32601"


def test_split_footprints_are_densified_per_part() -> None:
    parts = shapely.MultiPolygon(
        [shapely.box(170, 40, 180, 55), shapely.box(-180, 40, -170, 55)]
    )
    to_polar = pyproj.Transformer.from_crs(4326, 3995, always_xy=True).transform
    reference = shapely.MultiPolygon(
        [
            shapely.ops.transform(to_polar, shapely.segmentize(p, 0.01))
            for p in parts.geoms
        ]
    )
    got = reproject_geometry(parts, "EPSG:4326", "EPSG:3995")
    assert got.area == pytest.approx(reference.area, rel=1e-3)


def test_export_projection_fields_follow_the_asset_crs() -> None:
    from geocatalog.storage import to_stac_collection

    item = _item(
        properties={"proj:code": "EPSG:32633"}, asset_fields={"proj:epsg": 32634}
    )
    catalog = from_stac_items([item], extra_properties=("proj:code",))
    (exported,) = to_stac_collection(catalog, collection_id="c").get_items()
    proj = {k: v for k, v in exported.properties.items() if k.startswith("proj:")}
    assert proj == {"proj:epsg": 32634}


def test_explicit_null_asset_crs_overrides_the_item() -> None:
    row = _only_row(
        _item(properties={"proj:code": "EPSG:32633"}, asset_fields={"proj:code": None})
    )
    assert row.extras["crs"] is None


def test_nested_multipart_crossings_are_split() -> None:
    lines = shapely.MultiLineString([[(179, 0), (-179, 1)], [(10, 0), (11, 1)]])
    collection = shapely.GeometryCollection([lines, shapely.Point(5, 5)])
    item = _item(geometry=collection, bbox=[-180, 0, 180, 5])
    geom = _source_row(item).geometry
    assert geom.length == pytest.approx(2 * (2**2 + 1) ** 0.5 / 2 + 2**0.5, rel=0.01)
    assert geom.bounds[2] - geom.bounds[0] <= 360
    assert not geom.intersects(shapely.box(0, 0.4, 5, 0.6))


# ---------------------------------------------------------------------------
# Review follow-ups (#359)
# ---------------------------------------------------------------------------


def test_nested_multipart_members_are_densified_per_part() -> None:
    parts = shapely.MultiPolygon(
        [shapely.box(170, 40, 180, 55), shapely.box(-180, 40, -170, 55)]
    )
    nested = shapely.GeometryCollection([parts])
    got = reproject_geometry(nested, "EPSG:4326", "EPSG:3995")
    flat = reproject_geometry(parts, "EPSG:4326", "EPSG:3995")
    assert got.area == pytest.approx(flat.area, rel=1e-9)


def test_export_of_an_unlocated_asset_drops_carried_projection() -> None:
    from geocatalog.storage import to_stac_collection

    item = _item(
        properties={"proj:code": "EPSG:32633"}, asset_fields={"proj:code": None}
    )
    catalog = from_stac_items([item], extra_properties=("proj:code",))
    (exported,) = to_stac_collection(catalog, collection_id="c").get_items()
    assert not [k for k in exported.properties if k.startswith("proj:")]


def test_empty_catalog_keeps_the_boolean_signed_flag() -> None:
    assert from_stac_items([]).gdf["href_signed"].dtype == bool
    assert from_stac_items([_item()]).gdf["href_signed"].dtype == bool


def test_hole_crossing_the_antimeridian_is_unwrapped() -> None:
    # Only the hole jumps across ±180°; the shell stays east of it.
    polygon = shapely.Polygon(
        [(170, -10), (179.9, -10), (179.9, 10), (170, 10)],
        [[(179, -1), (-179, -1), (-179, 1), (179, 1)]],
    )
    geom = _source_row(_item(geometry=polygon, bbox=[170, -10, 180, 10])).geometry
    assert not geom.covers(shapely.Point(0, 0))  # not the 358° complement
    assert not geom.covers(shapely.Point(179.5, 0))  # inside the hole


def test_line_on_the_seam_is_on_both_edges() -> None:
    line = shapely.LineString([(180, -10), (180, 10)])
    geom = _source_row(_item(geometry=line, bbox=[180, -10, 180, 10])).geometry
    assert geom.intersects(shapely.box(179, -1, 180, 1))
    assert geom.intersects(shapely.box(-180, -1, -179, 1))


def test_segment_180_degrees_long_goes_over_the_pole() -> None:
    line = shapely.LineString([(0, 80), (180, 80)])
    geom = _source_row(_item(geometry=line, bbox=[0, 80, 180, 80])).geometry
    assert geom.intersects(shapely.Point(0, 89.5))
    assert not geom.intersects(shapely.box(80, 79, 100, 81))


def test_empty_bundle_has_the_ingested_schema() -> None:
    from geocatalog.storage import CatalogBundle

    empty = CatalogBundle.empty(crs="EPSG:4326").catalog.gdf
    row = source_row_to_gdf_row(_source_row(_item()), crs=pyproj.CRS("EPSG:4326"))
    assert set(row) - {"geometry", "start_time", "end_time"} <= set(empty.columns)
    assert empty["href_signed"].dtype == bool
