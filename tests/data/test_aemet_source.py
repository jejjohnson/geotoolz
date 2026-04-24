"""AemetSource tests with a fake HTTP client (no network I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from xr_toolz.data import AemetAuthError, AEMETCredentials, AemetSource
from xr_toolz.data._src.aemet.source import AemetRateLimitError
from xr_toolz.types import TimeRange


# ---- fake HTTP ----------------------------------------------------------


@dataclass
class _Resp:
    status_code: int
    body: Any = None
    headers: dict[str, str] | None = None
    text: str = ""

    def json(self):
        return self.body

    @property
    def content(self) -> bytes:
        import json

        if self.body is None:
            return b""
        return json.dumps(self.body).encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeClient:
    """In-memory router: path → list of responses (each get() pops one)."""

    def __init__(self, routes: dict[str, list[_Resp]]):
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, headers=None, timeout=None):
        self.calls.append((url, headers))
        for path, responses in self.routes.items():
            if url.endswith(path) or url == path:
                if not responses:
                    raise AssertionError(f"no more responses queued for {path}")
                return responses.pop(0)
        raise AssertionError(f"unexpected URL: {url}")


def _env_ok(datos: str, *, remaining: int = 100) -> _Resp:
    return _Resp(
        status_code=200,
        body={"estado": 200, "descripcion": "exito", "datos": datos, "metadatos": ""},
        headers={"Remaining-request-count": str(remaining)},
    )


# ---- fixtures -----------------------------------------------------------


@pytest.fixture
def source_and_fake():
    fake = FakeClient(routes={})
    src = AemetSource(
        credentials=AEMETCredentials(api_key="test"),
        client=fake,
        max_retries=2,
        max_workers=1,
    )
    return src, fake


# ---- auth ---------------------------------------------------------------


def test_requires_api_key(monkeypatch, tmp_path):
    # Isolate from the developer's real .env / env so the autoload
    # doesn't resolve a key from the enclosing shell.
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    src = AemetSource(credentials=None, client=FakeClient({}))
    with pytest.raises(AemetAuthError):
        src._require_key()


def test_401_raises_auth_error(source_and_fake):
    src, fake = source_and_fake
    fake.routes = {
        "/valores/climatologicos/inventarioestaciones/todasestaciones": [
            _Resp(status_code=401, text="bad key")
        ]
    }
    with pytest.raises(AemetAuthError):
        src.list_stations()


# ---- rate limit ----------------------------------------------------------


def test_429_retries_then_raises(source_and_fake, monkeypatch):
    src, fake = source_and_fake
    path = "/valores/climatologicos/inventarioestaciones/todasestaciones"
    fake.routes = {
        path: [_Resp(status_code=429), _Resp(status_code=429), _Resp(status_code=429)]
    }
    # Disable the minute-scale global 429 pause for speed, and stub
    # ``time.sleep`` so the per-attempt backoff doesn't burn wall time.
    src.max_retries = 2
    src.rate_limit_pause_scale = 0.0
    monkeypatch.setattr("xr_toolz.data._src.aemet.source.time.sleep", lambda _s: None)
    with pytest.raises(AemetRateLimitError):
        src.list_stations()


# ---- stations -----------------------------------------------------------


def test_list_stations_parses_dms(source_and_fake):
    src, fake = source_and_fake
    envelope_path = "/valores/climatologicos/inventarioestaciones/todasestaciones"
    data_url = "https://fake.aemet/data1"
    fake.routes = {
        envelope_path: [_env_ok(data_url)],
        data_url: [
            _Resp(
                status_code=200,
                body=[
                    {
                        "indicativo": "3195",
                        "nombre": "MADRID, RETIRO",
                        "provincia": "MADRID",
                        "latitud": "402358N",
                        "longitud": "034041W",
                        "altitud": "667",
                        "indsinop": "08222",
                    }
                ],
            )
        ],
    }
    stations = src.list_stations()
    assert len(stations) == 1
    s = stations["3195"]
    assert s.name == "MADRID, RETIRO"
    assert 40.0 < s.lat < 40.5
    assert -4.0 < s.lon < -3.5
    assert s.altitude == 667.0
    assert s.wmo_id == "08222"
    assert s.source == "aemet"


# ---- daily --------------------------------------------------------------


def _daily_row(fecha: str, **kwargs: str) -> dict[str, str]:
    base = {
        "fecha": fecha,
        "indicativo": "3195",
        "tmed": "10,5",
        "tmin": "5,0",
        "tmax": "15,0",
        "prec": "0,3",
    }
    base.update(kwargs)
    return base


def test_get_daily_stitches_chunked_windows(source_and_fake):
    # The parent fixture is unused here; this test builds its own
    # path-matching client to handle the variable daily URL.
    del source_and_fake
    # Two 180-day chunks for a single station.
    # The mock returns the same two-row payload for both envelope URLs.
    env = _env_ok("https://fake.aemet/daily1")
    env2 = _env_ok("https://fake.aemet/daily2")

    # We don't pin exact URL paths because the date strings vary; match prefix.
    class PathMatcher(FakeClient):
        def get(self, url: str, headers=None, timeout=None):
            self.calls.append((url, headers))
            if "/valores/climatologicos/diarios/datos/" in url:
                return self.routes["envelope"].pop(0)
            return self.routes[url].pop(0)

    fake2 = PathMatcher(
        routes={
            "envelope": [env, env2],
            "https://fake.aemet/daily1": [
                _Resp(
                    status_code=200,
                    body=[
                        _daily_row("2024-01-01"),
                        _daily_row("2024-01-02", tmed="11,0"),
                    ],
                )
            ],
            "https://fake.aemet/daily2": [
                _Resp(
                    status_code=200,
                    body=[_daily_row("2024-07-10", tmed="22,5")],
                )
            ],
        }
    )
    src2 = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake2,
        max_workers=1,
    )
    tr = TimeRange.parse("2024-01-01", "2024-07-15")
    ds = src2.get_daily(["3195"], time=tr)
    assert ds.sizes["station"] == 1
    # Full daily index over the window
    assert ds.sizes["time"] >= 190
    tmed = ds["air_temperature_daily_mean"].sel(station="3195")
    # First two days observed, middle is NaN, then the July value is observed.
    values = tmed.values
    assert not np.isnan(values[0])
    assert not np.isnan(values[1])
    assert np.isnan(values[100])  # a gap somewhere
    assert ds.attrs["source"] == "aemet"
    assert ds.attrs["featureType"] == "timeSeries"
    assert ds.attrs["endpoint"] == "daily"
    # CF attrs wired through
    attrs = ds["air_temperature_daily_mean"].attrs
    assert attrs.get("standard_name") == "air_temperature"


def test_get_daily_requires_time(source_and_fake):
    src, _ = source_and_fake
    with pytest.raises(ValueError, match="TimeRange"):
        src.get_daily(["3195"])


def test_get_daily_rejects_empty_stations(source_and_fake):
    src, _ = source_and_fake
    tr = TimeRange.parse("2024-01-01", "2024-01-02")
    with pytest.raises(ValueError, match="at least one station"):
        src.get_daily([], time=tr)


# ---- dataset-level dispatch ---------------------------------------------


def test_open_stations_preset_returns_inventory(source_and_fake):
    src, fake = source_and_fake
    env_path = "/valores/climatologicos/inventarioestaciones/todasestaciones"
    data_url = "https://fake.aemet/inv"
    fake.routes = {
        env_path: [_env_ok(data_url)],
        data_url: [
            _Resp(
                status_code=200,
                body=[
                    {
                        "indicativo": "AAA",
                        "nombre": "A",
                        "provincia": "X",
                        "latitud": "400000N",
                        "longitud": "030000W",
                        "altitud": "10",
                    }
                ],
            )
        ],
    }
    ds = src.open("aemet_stations")
    assert "lon" in ds and "lat" in ds
    assert "AAA" in ds["station"].values


def test_unknown_dataset_raises(source_and_fake):
    src, _ = source_and_fake
    with pytest.raises(ValueError, match="unknown AEMET dataset"):
        src.open("aemet_nonsense")


# ---- subset by variables ------------------------------------------------


def test_get_monthly_trims_to_requested_window():
    """Monthly output must not leak months outside the requested range.

    AEMET's endpoint returns whole years, but callers asking for
    ``TimeRange.parse("2024-06-01", "2024-09-30")`` expect Jun-Sep
    only. Without trimming, incremental archive syncs would rewrite
    entire years on every run.
    """

    # Build a monthly row helper
    def _row(year: int, month: int, tm: str = "10.0") -> dict[str, str]:
        return {"fecha": f"{year}-{month}", "indicativo": "3195", "tm_mes": tm}

    # Build 12 months of 2024 as the AEMET payload
    payload = [_row(2024, m) for m in range(1, 13)]

    # Fake client: any monthly envelope → one data URL with the payload
    env_body = {"estado": 200, "descripcion": "x", "datos": "https://fake/d"}
    fake = FakeClient(routes={})
    src = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake,
        max_retries=0,
        max_workers=1,
        min_interval_s=0.0,
    )

    # Override the fake to match any monthly-data URL
    class PathMatcher(FakeClient):
        def get(self, url, headers=None, timeout=None):
            self.calls.append((url, headers))
            if "/valores/climatologicos/mensualesanuales/" in url:
                return _Resp(status_code=200, body=env_body)
            return _Resp(status_code=200, body=payload)

    src._client = PathMatcher(routes={})
    tr = TimeRange.parse("2024-06-01", "2024-09-30")
    ds = src.get_monthly(["3195"], time=tr)
    months = pd.to_datetime(ds["time"].values).month.tolist()
    assert months == [6, 7, 8, 9], f"expected Jun-Sep only, got months={months}"


def test_trip_rate_limit_blocks_all_workers():
    """A 429 should install a global pause that ``_rate_limit`` honours.

    Without this, concurrent workers keep AEMET's minute bucket hot
    while the 429'd worker is backing off, and no one makes progress.
    """
    import time

    fake = FakeClient(routes={})
    src = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake,
        min_interval_s=0.0,
    )
    src._trip_rate_limit(0.3)
    t0 = time.monotonic()
    src._rate_limit()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.25, f"expected ≥0.25s pause, got {elapsed:.3f}s"


def test_normals_preserves_zero_values():
    """``_normals_to_dataset`` must keep legitimate 0.0 values.

    The previous ``parse_spanish_float(...) or np.nan`` idiom treated
    zeros as falsy and turned them into NaN — silently erasing zero
    precipitation / sunshine months in the climate normals output.
    """
    import numpy as np

    from xr_toolz.data._src.aemet.source import _normals_to_dataset

    rows_per: dict[str, list[dict[str, Any]]] = {
        "s1": [
            # January with a zero precipitation month — must survive as 0.
            {"mes": "1", "tm_mes": "0", "p_mes": "0", "inso": "0"},
            # February with a real positive value to prove we didn't
            # accidentally zero everything.
            {"mes": "2", "tm_mes": "5.5", "p_mes": "12.3", "inso": "4.2"},
        ]
    }
    ds = _normals_to_dataset(rows_per, ("s1",))
    tmean = ds["tm_mes"].sel(station="s1").values
    precip = ds["p_mes"].sel(station="s1").values
    sun = ds["inso"].sel(station="s1").values
    # Jan (index 0) must be 0.0, not NaN.
    assert tmean[0] == 0.0
    assert precip[0] == 0.0
    assert sun[0] == 0.0
    # Feb (index 1) keeps its positive values.
    assert tmean[1] == 5.5
    assert precip[1] == 12.3
    # Mar (index 2) and onwards are missing → NaN.
    assert np.isnan(tmean[2])
    assert np.isnan(precip[2])


def test_chunk_days_range_includes_single_day():
    """Single-day windows must emit one chunk, not zero."""
    from datetime import UTC, datetime

    from xr_toolz.data._src.aemet.source import _chunk_days_range

    day = datetime(2024, 1, 1, tzinfo=UTC)
    chunks = _chunk_days_range("sid", day, day, chunk_days=180)
    assert len(chunks) == 1
    assert chunks[0] == ("sid", day, day)

    # And a two-day window still emits exactly one chunk (fits in 180 days).
    next_day = datetime(2024, 1, 2, tzinfo=UTC)
    chunks = _chunk_days_range("sid", day, next_day, chunk_days=180)
    assert len(chunks) == 1
    assert chunks[0] == ("sid", day, next_day)


def test_rate_limit_spaces_requests():
    """Two back-to-back fetches should honour ``min_interval_s``."""
    import time

    fake = FakeClient(
        routes={
            "/valores/climatologicos/inventarioestaciones/todasestaciones": [
                _env_ok("https://fake.aemet/d1"),
                _env_ok("https://fake.aemet/d2"),
            ],
            "https://fake.aemet/d1": [_Resp(status_code=200, body=[])],
            "https://fake.aemet/d2": [_Resp(status_code=200, body=[])],
        }
    )
    src = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake,
        max_retries=0,
        max_workers=1,
        min_interval_s=0.2,
    )
    t0 = time.monotonic()
    src.list_stations()
    src.list_stations()
    elapsed = time.monotonic() - t0
    # Four hops (2 envelope + 2 data) × 0.2s gap = ≥0.6s between first
    # and last. We only need evidence the gate fired, not exact timing.
    assert elapsed >= 0.5, f"expected ≥0.5s, got {elapsed:.3f}s"


def test_rate_limit_zero_is_no_op():
    """``min_interval_s=0`` should not add artificial delay."""
    import time

    fake = FakeClient(
        routes={
            "/valores/climatologicos/inventarioestaciones/todasestaciones": [
                _env_ok("https://fake.aemet/d"),
            ],
            "https://fake.aemet/d": [_Resp(status_code=200, body=[])],
        }
    )
    src = AemetSource(
        credentials=AEMETCredentials(api_key="t"),
        client=fake,
        max_retries=0,
        max_workers=1,
        min_interval_s=0.0,
    )
    t0 = time.monotonic()
    src.list_stations()
    assert time.monotonic() - t0 < 0.1


def test_variable_subset_drops_others(source_and_fake):
    """Passing ``variables=[x]`` keeps only ``x`` in the result dataset."""
    del source_and_fake
    # Build a minimal dataset via get_hourly machinery with no rows; instead
    # call the subset helper directly.
    import xarray as xr

    from xr_toolz.data._src.aemet.source import _subset_variables

    ds = xr.Dataset(
        {
            "air_temperature": (("time",), [1.0]),
            "precipitation_amount": (("time",), [2.0]),
        },
        coords={"time": [0]},
    )
    out = _subset_variables(ds, ["air_temperature"])
    assert "air_temperature" in out
    assert "precipitation_amount" not in out
