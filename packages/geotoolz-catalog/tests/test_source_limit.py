"""``limit`` means the same on every source adapter (#237).

``0`` yields nothing and never reaches the upstream client (whose own
``0`` means "no limit" for pystac-client and earthaccess), ``None`` is
unbounded, and a negative limit is rejected.
"""

from __future__ import annotations

from typing import Any

import pytest

from geocatalog._src.sources._base import wants_no_rows


BOUNDS = (-10.0, 35.0, -5.0, 45.0)


class _Boom:
    """Stands in for an upstream client; any use fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"upstream touched via {name!r} for limit=0")


def _stac_source() -> Any:
    pytest.importorskip("pystac_client")
    from geocatalog._src.sources.stac import STACSource

    src = STACSource(endpoint="https://fake", name="stac.fake")
    src._client = _Boom()  # type: ignore[assignment]
    return src


def _earthaccess_source(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("earthaccess")
    from geocatalog._src.sources import earthaccess as ea_mod

    def search_data(**kwargs: Any) -> Any:
        raise AssertionError(f"earthaccess.search_data called with {kwargs}")

    monkeypatch.setattr(ea_mod.earthaccess, "search_data", search_data)
    return ea_mod.EarthAccessSource()


def _cmr_source(monkeypatch: pytest.MonkeyPatch) -> Any:
    from geocatalog._src.sources import cmr as cmr_mod

    def fetch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CMR request made for limit=0")

    monkeypatch.setattr(cmr_mod, "_fetch_page", fetch)
    return cmr_mod.CMRSource()


def _gee_source() -> Any:
    pytest.importorskip("ee")
    from geocatalog._src.sources.gee import GEESource

    # Still scaffolding: any query that could return rows raises
    # NotImplementedError, but the limit contract already holds.
    return GEESource()


@pytest.fixture(params=["stac", "earthaccess", "cmr", "gee"])
def source(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    if request.param == "stac":
        return _stac_source()
    if request.param == "earthaccess":
        return _earthaccess_source(monkeypatch)
    if request.param == "gee":
        return _gee_source()
    return _cmr_source(monkeypatch)


def test_limit_zero_yields_nothing_without_a_request(source: Any) -> None:
    assert list(source.query(BOUNDS, collection="X", limit=0)) == []


def test_negative_limit_is_rejected(source: Any) -> None:
    with pytest.raises(ValueError, match="limit must be >= 0"):
        list(source.query(BOUNDS, collection="X", limit=-1))


@pytest.mark.parametrize(
    ("limit", "expected"), [(None, False), (0, True), (1, False), (10_000, False)]
)
def test_wants_no_rows(limit: int | None, expected: bool) -> None:
    assert wants_no_rows(limit) is expected


def test_wants_no_rows_rejects_bool() -> None:
    with pytest.raises(TypeError):
        wants_no_rows(False)  # type: ignore[arg-type]
