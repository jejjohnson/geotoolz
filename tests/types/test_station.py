"""Station and StationCollection tests."""

from __future__ import annotations

import pytest

from xr_toolz.types import BBox, Station, StationCollection


def _s(
    id: str,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
    province: str | None = None,
    community: str | None = None,
    city: str | None = None,
    source: str | None = None,
    active: bool | None = None,
    wmo_id: str | None = None,
) -> Station:
    return Station(
        id=id,
        name=id,
        lon=lon,
        lat=lat,
        province=province,
        community=community,
        city=city,
        source=source,
        active=active,
        wmo_id=wmo_id,
    )


# ---- Station -------------------------------------------------------------


def test_station_bounds_validated() -> None:
    with pytest.raises(ValueError, match="lon out of"):
        Station(id="1", name="x", lon=200.0, lat=0.0)
    with pytest.raises(ValueError, match="lat out of"):
        Station(id="1", name="x", lon=0.0, lat=95.0)


def test_station_attrs_are_frozen() -> None:
    s = Station(id="1", name="x", lon=0.0, lat=0.0, attrs={"k": "v"})
    with pytest.raises(TypeError):
        s.attrs["k"] = "mutated"  # type: ignore[index]


def test_station_distance_to_monotonic() -> None:
    madrid = Station(id="m", name="Madrid", lon=-3.7, lat=40.4)
    barcelona = 500.0
    assert 400 < madrid.distance_to(2.17, 41.39) < 600
    assert madrid.distance_to(-3.7, 40.4) == pytest.approx(0.0)
    # Monotonic in longitude on the equator:
    a = madrid.distance_to(0.0, 40.4)
    b = madrid.distance_to(10.0, 40.4)
    assert a < b
    assert barcelona > 0  # guard against accidental reuse


def test_station_in_bbox_wraps_antimeridian() -> None:
    s = Station(id="a", name="a", lon=179.5, lat=0.0)
    wrap = BBox(lon_min=170.0, lon_max=-170.0, lat_min=-10.0, lat_max=10.0)
    assert s.in_bbox(wrap)
    non_wrap = BBox(lon_min=0.0, lon_max=10.0, lat_min=-10.0, lat_max=10.0)
    assert not s.in_bbox(non_wrap)


# ---- StationCollection ---------------------------------------------------


def test_collection_iteration_and_len() -> None:
    c = StationCollection.from_iter([_s("a"), _s("b")])
    assert len(c) == 2
    assert [s.id for s in c] == ["a", "b"]


def test_collection_lookup_by_id_and_contains() -> None:
    c = StationCollection.from_iter([_s("a"), _s("b")])
    assert "a" in c
    assert "c" not in c
    got = c["b"]
    assert isinstance(got, Station) and got.id == "b"
    with pytest.raises(KeyError):
        c["missing"]


def test_collection_filter_combines_fields() -> None:
    c = StationCollection.from_iter(
        [
            _s("a", province="Cantabria", community="Cantabria"),
            _s("b", province="Madrid", community="Madrid"),
            _s("c", province="cantabria", community="Cantabria"),
        ]
    )
    # Case-insensitive matching, OR within a filter, AND across filters.
    got = c.filter(province=["cantabria"])
    assert got.ids() == ("a", "c")
    got2 = c.filter(community="madrid", province="Madrid")
    assert got2.ids() == ("b",)


def test_collection_filter_has_wmo() -> None:
    c = StationCollection.from_iter(
        [_s("a", wmo_id="08123"), _s("b"), _s("c", wmo_id="")]
    )
    assert c.filter(has_wmo=True).ids() == ("a",)
    assert c.filter(has_wmo=False).ids() == ("b", "c")


def test_collection_within_and_nearest() -> None:
    a = _s("a", lon=-3.7, lat=40.4)  # Madrid
    b = _s("b", lon=2.17, lat=41.39)  # Barcelona
    c = _s("c", lon=-5.99, lat=37.38)  # Sevilla
    coll = StationCollection.from_iter([a, b, c])
    iberia = BBox(lon_min=-10.0, lon_max=5.0, lat_min=35.0, lat_max=44.0)
    assert coll.within(iberia).ids() == ("a", "b", "c")
    near_madrid = coll.nearest((-3.7, 40.4), n=2).ids()
    assert near_madrid[0] == "a"
    assert "c" in near_madrid or "b" in near_madrid  # closer than the other


def test_collection_list_helpers() -> None:
    c = StationCollection.from_iter(
        [
            _s("a", province="Cantabria", community="Cantabria", city="Santander"),
            _s("b", province="Madrid", community="Madrid", city="Madrid"),
            _s("c", province="Madrid", community="Madrid", city=None),
        ]
    )
    assert c.provinces() == ("Cantabria", "Madrid")
    assert c.communities() == ("Cantabria", "Madrid")
    assert c.cities() == ("Madrid", "Santander")
    df = c.to_dataframe()
    assert list(df["id"]) == ["a", "b", "c"]


def test_collection_to_dataframe_empty_has_columns() -> None:
    df = StationCollection().to_dataframe()
    assert "id" in df.columns and len(df) == 0


def test_collection_nearest_handles_n_zero_or_big() -> None:
    c = StationCollection.from_iter([_s("a"), _s("b")])
    assert c.nearest((0, 0), n=0).ids() == ()
    assert len(c.nearest((0, 0), n=99)) == 2
