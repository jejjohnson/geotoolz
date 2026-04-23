"""Cache key serialization — covers the _json_default fallback branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from xr_toolz.data._src.cache import _json_default, cache_path


@dataclass
class _Sample:
    x: int
    y: str


def test_json_default_dataclass_is_serialized_as_dict():
    out = _json_default(_Sample(x=1, y="a"))
    assert out == {"x": 1, "y": "a"}


def test_json_default_path_is_stringified():
    assert _json_default(Path("/tmp/foo")) == "/tmp/foo"


def test_json_default_set_is_listified():
    out = _json_default({3, 1, 2})
    assert sorted(out) == [1, 2, 3]


def test_json_default_tuple_is_listified():
    assert _json_default((1, 2, 3)) == [1, 2, 3]


def test_json_default_timestamp_uses_isoformat():
    import pandas as pd

    ts = pd.Timestamp("2020-01-01", tz="UTC")
    assert _json_default(ts).startswith("2020-01-01")


def test_json_default_unknown_falls_back_to_str():
    class Foo:
        def __str__(self) -> str:
            return "<foo>"

    assert _json_default(Foo()) == "<foo>"


def test_cache_path_respects_suffix(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    p = cache_path("cmems", "foo", {"a": 1}, suffix=".zarr")
    assert p.suffix == ".zarr"


def test_cache_path_survives_exotic_request_payloads(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    # Dataclass + Path + set + tuple all in one request — must hash deterministically.
    req = {
        "ds": _Sample(x=1, y="a"),
        "p": Path("/tmp/x"),
        "s": {1, 2, 3},
        "t": (4, 5),
    }
    p1 = cache_path("cmems", "foo", req)
    p2 = cache_path("cmems", "foo", req)
    assert p1 == p2


def test_cache_path_rejects_unserializable_would_raise(monkeypatch, tmp_path):
    # Bytes are serializable via str() fallback; confirm no crash.
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    assert cache_path("cmems", "foo", {"b": b"raw"}) is not None


@pytest.mark.parametrize(
    "source,dataset_id",
    [
        ("cmems", "cmems_mod_glo_phy_my_0.083deg_P1D-m"),
        ("cds", "reanalysis-era5-single-levels"),
        ("cmems", "METOFFICE-GLO-SST-L4-REP-OBS-SST"),
    ],
)
def test_cache_path_creates_nested_directories(
    monkeypatch, tmp_path, source, dataset_id
):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    p = cache_path(source, dataset_id, {"a": 1})
    assert p.parent.exists()
    assert p.parent.parent.name == source
