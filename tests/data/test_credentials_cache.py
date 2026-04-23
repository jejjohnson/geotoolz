"""Credentials loader + cache key determinism."""

from __future__ import annotations

from pathlib import Path

from xr_toolz.data._src.cache import cache_path
from xr_toolz.data._src.credentials import load_cds, load_cmems


def test_load_cmems_prefers_explicit_args(monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    c = load_cmems(username="u", password="p", path=Path("/nonexistent"))
    assert c is not None and c.username == "u" and c.password == "p"


def test_load_cmems_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "eu")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "ep")
    c = load_cmems(path=Path("/nonexistent"))
    assert c is not None and c.username == "eu"


def test_load_cmems_falls_back_to_file(tmp_path, monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    cfg = tmp_path / "cmems"
    cfg.write_text("username: fileu\npassword: filep\n")
    c = load_cmems(path=cfg)
    assert c is not None and c.username == "fileu" and c.password == "filep"


def test_load_cmems_returns_none_when_nothing_found(monkeypatch, tmp_path):
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    assert load_cmems(path=tmp_path / "no_such") is None


def test_load_cds_dotrc(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    cfg = tmp_path / "cdsapirc"
    cfg.write_text("url: https://cds.test\nkey: abc:def\n")
    c = load_cds(path=cfg)
    assert c is not None
    assert c.url == "https://cds.test"
    assert c.key == "abc:def"


def test_cache_path_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    req = {"variables": ["t2m"], "bbox": [0, 1, 2, 3]}
    p1 = cache_path("cds", "era5", req)
    p2 = cache_path("cds", "era5", req)
    assert p1 == p2
    assert p1.parent.parent == tmp_path / "cds"


def test_cache_path_changes_when_request_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    p1 = cache_path("cds", "era5", {"variables": ["t2m"]})
    p2 = cache_path("cds", "era5", {"variables": ["u10"]})
    assert p1 != p2


def test_cache_path_sanitizes_dataset_id(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    p = cache_path("cmems", "cmems/mod:phy?id", {})
    assert "/" not in p.parent.name and ":" not in p.parent.name
