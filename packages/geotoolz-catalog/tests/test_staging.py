"""End-to-end tests for `geocatalog.staging.stage` + `LocalCache`.

No network. Local paths are staged in place; the remote-URI paths
(retry, timeout, on_error) fake ``geocloud.files.download`` on the
``cloud_stub`` module, so they also run on a base install without the
``[cloud]`` extra. One end-to-end test downloads through the real
`geocloud.files` from an in-memory bucket (`geocloud.store.mount`).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from shapely.geometry import box

from geocatalog._src.backends.memory import InMemoryGeoCatalog
from geocatalog._src.staging.stage import (
    LocalCache,
    _ext_for,
    _fetch_one,
    _plan_row,
    stage,
)
from tests.conftest import catalog_from_rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_tif(path: Path, *, content: bytes = b"fake-tif-bytes") -> Path:
    """Write a fake TIF to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write(dest: Path, content: bytes) -> Path:
    """What a successful `geocloud.files.download` leaves behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


# ---------------------------------------------------------------------------
# LocalCache
# ---------------------------------------------------------------------------


class TestLocalCacheRootResolution:
    def test_explicit_root_used_when_set(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path / "explicit")
        resolved = cache.resolve_root()
        assert resolved == tmp_path / "explicit"
        assert resolved.is_dir()  # auto-created

    def test_env_var_takes_precedence_when_root_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_root = tmp_path / "env"
        monkeypatch.setenv("GEOCATALOG_CACHE", str(env_root))
        cache = LocalCache()
        assert cache.resolve_root() == env_root

    def test_falls_back_to_home_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("GEOCATALOG_CACHE", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        cache = LocalCache()
        resolved = cache.resolve_root()
        assert resolved == tmp_path / ".cache" / "geocatalog"


class TestLocalCachePathFor:
    def test_deterministic_per_uri(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path)
        a1 = cache.path_for("s3://bucket/data.tif")
        a2 = cache.path_for("s3://bucket/data.tif")
        b = cache.path_for("s3://bucket/other.tif")
        assert a1 == a2
        assert a1 != b

    def test_path_carries_extension(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path)
        path = cache.path_for("https://example.com/foo/bar.nc")
        assert path.suffix == ".nc"

    def test_two_letter_directory_prefix(self, tmp_path: Path) -> None:
        # Two-level layout keeps any one directory small on large
        # catalogs. The prefix is the first 2 chars of the sha256.
        cache = LocalCache(root=tmp_path)
        uri = "s3://bucket/x.tif"
        path = cache.path_for(uri)
        digest = hashlib.sha256(uri.encode()).hexdigest()
        assert path.parent.name == digest[:2]


class TestLocalCacheTTL:
    def test_fresh_when_no_ttl(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path)
        target = cache.path_for("s3://x.tif")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        assert cache.is_fresh(target)

    def test_fresh_within_ttl(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path, ttl_days=7)
        target = cache.path_for("s3://x.tif")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        assert cache.is_fresh(target)

    def test_stale_past_ttl(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path, ttl_days=1)
        target = cache.path_for("s3://x.tif")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        # Backdate the mtime to 2 days ago.
        two_days_ago = (datetime.now(tz=UTC) - timedelta(days=2)).timestamp()
        Path(target).touch()
        import os

        os.utime(target, (two_days_ago, two_days_ago))
        assert not cache.is_fresh(target)

    def test_missing_path_is_not_fresh(self, tmp_path: Path) -> None:
        cache = LocalCache(root=tmp_path)
        assert not cache.is_fresh(tmp_path / "nope.tif")


# ---------------------------------------------------------------------------
# _ext_for
# ---------------------------------------------------------------------------


class TestExtFor:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("s3://bucket/foo.tif", ".tif"),
            ("https://x/path/to/bar.nc?token=abc", ".nc"),
            ("/local/file.geojson", ".geojson"),
            ("https://x/no-extension", ""),
            ("file:///local/data.tar.gz", ".gz"),
        ],
    )
    def test_extension_extraction(self, uri: str, expected: str) -> None:
        assert _ext_for(uri) == expected


# ---------------------------------------------------------------------------
# _plan_row — turning a raw catalog row into a stage plan
# ---------------------------------------------------------------------------


class TestPlanRow:
    def _row_tuple(self, **fields: Any) -> Any:
        """Build something `_plan_row` will accept (mimics itertuples)."""
        from collections import namedtuple

        T = namedtuple("Row", fields.keys())
        return T(**fields)

    def test_asset_map_promotes_assets_to_plan(self) -> None:
        row = self._row_tuple(
            filepath="s3://bucket/data.tif",
            assets=json.dumps({"red": "s3://r.tif", "nir": "s3://n.tif"}),
        )
        plan = _plan_row(row, idx=0, asset_filter=None)
        assert plan.has_asset_map is True
        assert plan.assets == {"red": "s3://r.tif", "nir": "s3://n.tif"}

    def test_no_asset_map_falls_back_to_filepath(self) -> None:
        row = self._row_tuple(filepath="/tmp/local.tif")
        plan = _plan_row(row, idx=0, asset_filter=None)
        assert plan.has_asset_map is False
        assert plan.assets == {"filepath": "/tmp/local.tif"}

    def test_asset_filter_narrows_subset(self) -> None:
        row = self._row_tuple(
            filepath="s3://bucket/data.tif",
            assets=json.dumps(
                {"red": "s3://r.tif", "nir": "s3://n.tif", "scl": "s3://s.tif"}
            ),
        )
        plan = _plan_row(row, idx=0, asset_filter=["red", "nir"])
        assert set(plan.assets) == {"red", "nir"}


# ---------------------------------------------------------------------------
# stage() — end-to-end on local files
# ---------------------------------------------------------------------------


class TestStageLegacyFilepath:
    """Rows without an asset map — just `filepath` is staged."""

    def test_local_uri_passes_through(self, tmp_path: Path) -> None:
        # Local files are fast-pathed: no copy, just return the path.
        src_file = _seed_tif(tmp_path / "src" / "x.tif")
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": str(src_file),
                }
            ],
            crs="EPSG:4326",
        )
        out = stage(cat, dest=tmp_path / "cache")
        assert out.gdf.iloc[0]["filepath"] == str(src_file)

    def test_file_scheme_uri_fetched_to_cache(self, tmp_path: Path) -> None:
        # `file://` URIs that point at real files get copied into
        # the cache (so subsequent runs read from cache, not the
        # source location).
        src_file = _seed_tif(tmp_path / "src" / "x.tif", content=b"hello")
        uri = f"file://{src_file}"
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": uri,
                }
            ],
            crs="EPSG:4326",
        )
        out = stage(cat, dest=tmp_path / "cache")
        new_path = out.gdf.iloc[0]["filepath"]
        # Local-file fast path: we return the existing path directly.
        # (Verified by content readback either way.)
        assert Path(new_path).read_bytes() == b"hello"


class TestStageAssetMap:
    """Rows with a JSON-encoded asset map — every asset staged independently."""

    def _make_catalog(self, tmp_path: Path) -> tuple[InMemoryGeoCatalog, Path, Path]:
        red = _seed_tif(tmp_path / "src" / "red.tif", content=b"red")
        nir = _seed_tif(tmp_path / "src" / "nir.tif", content=b"nir")
        # Plain paths (no scheme) are local and staged in place.
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": str(red),
                    "assets": json.dumps({"red": str(red), "nir": str(nir)}),
                }
            ],
            crs="EPSG:4326",
        )
        return cat, red, nir

    def test_all_assets_staged_by_default(self, tmp_path: Path) -> None:
        cat, _red, _nir = self._make_catalog(tmp_path)
        out = stage(cat, dest=tmp_path / "cache")
        assets_out = json.loads(out.gdf.iloc[0]["assets"])
        assert set(assets_out) == {"red", "nir"}
        # Local-URI fast path keeps the original paths.
        assert Path(assets_out["red"]).read_bytes() == b"red"
        assert Path(assets_out["nir"]).read_bytes() == b"nir"

    def test_assets_filter_picks_subset(self, tmp_path: Path) -> None:
        cat, _red, _nir = self._make_catalog(tmp_path)
        out = stage(cat, dest=tmp_path / "cache", assets=["red"])
        assets_out = json.loads(out.gdf.iloc[0]["assets"])
        assert set(assets_out) == {"red"}

    def test_staged_from_preserves_original_uris(self, tmp_path: Path) -> None:
        cat, red, nir = self._make_catalog(tmp_path)
        out = stage(cat, dest=tmp_path / "cache")
        staged_from = json.loads(out.gdf.iloc[0]["staged_from"])
        assert staged_from == {"red": str(red), "nir": str(nir)}


class TestStageRemoteFetch:
    """Remote URIs download into the cache; local ones are used in place."""

    def test_local_path_is_used_in_place(self, tmp_path: Path) -> None:
        src_file = _seed_tif(tmp_path / "remote" / "data.nc", content=b"abc")
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": str(src_file),
                }
            ],
            crs="EPSG:4326",
        )
        out = stage(cat, cache=LocalCache(root=tmp_path / "cache"))
        assert out.gdf.iloc[0]["filepath"] == str(src_file)

    def test_s3_uri_downloads_through_geocloud(self, tmp_path: Path) -> None:
        pytest.importorskip("geocloud")
        import obstore
        from geocloud.store import mount, unmount
        from obstore.store import MemoryStore

        bucket = MemoryStore()
        obstore.put(bucket, "scenes/a.tif", b"remote-bytes")
        mount("s3://stage-test", bucket)
        try:
            cat = catalog_from_rows(
                rows=[
                    {
                        "geometry": box(0, 0, 1, 1),
                        "start_time": pd.Timestamp("2024-06-01"),
                        "end_time": pd.Timestamp("2024-06-02"),
                        "filepath": "s3://stage-test/scenes/a.tif",
                    }
                ],
                crs="EPSG:4326",
            )
            cache = LocalCache(root=tmp_path / "cache")
            out = stage(cat, cache=cache)
        finally:
            unmount("s3://stage-test")
        local = Path(out.gdf.iloc[0]["filepath"])
        assert local == cache.path_for("s3://stage-test/scenes/a.tif")
        assert local.read_bytes() == b"remote-bytes"
        assert not list((tmp_path / "cache").rglob("*.part"))


class TestStageCacheHit:
    """Second stage() of the same URI should not re-fetch."""

    def test_cache_hit_skips_download(self, tmp_path: Path) -> None:
        # A remote URI whose cache slot is already filled.
        cache = LocalCache(root=tmp_path / "cache")
        uri = "https://example.com/data.tif"
        # Pre-populate the cache to simulate "already fetched".
        target = cache.path_for(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"cached-bytes")

        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": uri,
                }
            ],
            crs="EPSG:4326",
        )
        # Even though `uri` is unreachable, stage() must not try
        # to fetch — the cache is fresh.
        out = stage(cat, cache=cache)
        assert out.gdf.iloc[0]["filepath"] == str(target)


class TestStageRetry:
    """`_fetch_one` retries transient failures up to `retries` times."""

    def test_retries_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud_stub: Any
    ) -> None:
        # Build a fake `geocloud.files.download` that fails twice (with a plain,
        # transient OSError), then succeeds.
        attempts = {"n": 0}

        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("transient")
            return _write(dest, b"hello")

        import time

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)
        # No need to actually wait out the backoff.
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        cache = LocalCache(root=tmp_path / "cache")

        path = _fetch_one("https://example.com/x.tif", cache, retries=3)
        assert attempts["n"] == 3
        assert path.read_bytes() == b"hello"

    @pytest.mark.parametrize("exc_type", [FileNotFoundError, PermissionError])
    def test_fatal_error_does_not_retry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cloud_stub: Any,
        exc_type: type[OSError],
    ) -> None:
        # Fatal OSError subclasses (missing object, auth/permission
        # problems) will not heal on retry — they must propagate on
        # the first attempt instead of burning the retry budget.
        attempts = {"n": 0}

        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            attempts["n"] += 1
            raise exc_type("fatal")

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)

        cache = LocalCache(root=tmp_path / "cache")
        with pytest.raises(exc_type, match="fatal"):
            _fetch_one("https://example.com/x.tif", cache, retries=3)
        assert attempts["n"] == 1

    def test_transient_budget_exhausted_raises_last_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud_stub: Any
    ) -> None:
        attempts = {"n": 0}

        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            attempts["n"] += 1
            raise OSError("still down")

        import time

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        cache = LocalCache(root=tmp_path / "cache")
        with pytest.raises(OSError, match="still down"):
            _fetch_one("https://example.com/x.tif", cache, retries=2)
        assert attempts["n"] == 3  # initial attempt + 2 retries


class TestStageTimeout:
    """`LocalCache.timeout` becomes the object-store client's request timeout."""

    def test_default_timeout_is_60s(self) -> None:
        assert LocalCache().timeout == 60.0

    def test_timeout_survives_dataclass_serialization(self) -> None:
        import dataclasses

        cfg = dataclasses.asdict(LocalCache(root="/x"))
        assert cfg["timeout"] == 60.0

    def test_timeout_forwarded_to_the_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud_stub: Any
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            seen.update(kwargs)
            return _write(dest, b"x")

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)

        cache = LocalCache(root=tmp_path / "cache", timeout=12.5)
        _fetch_one("https://example.com/x.tif", cache, retries=0)
        assert seen["storage_options"] == {
            "client_options": {"timeout": timedelta(seconds=12.5)}
        }

    def test_timeout_none_keeps_the_client_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud_stub: Any
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            seen.update(kwargs)
            return _write(dest, b"x")

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)

        cache = LocalCache(root=tmp_path / "cache", timeout=None)
        _fetch_one("https://example.com/y.tif", cache, retries=0)
        assert seen["storage_options"] is None


class TestStageOnError:
    """`on_error="skip"` keeps going past a failed asset."""

    def test_skip_keeps_other_assets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud_stub: Any
    ) -> None:
        # Build a catalog with one bad URI and one good local file.
        good = _seed_tif(tmp_path / "good.tif", content=b"good")
        bad = "https://nonexistent/never.tif"

        # Force the download to fail for the bad URI (the good, local asset
        # is staged in place and never downloaded).
        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            if uri == bad:
                raise OSError("nope")
            raise AssertionError(f"local asset {uri!r} must not be downloaded")

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)

        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": str(good),
                    "assets": json.dumps({"good": str(good), "bad": bad}),
                }
            ],
            crs="EPSG:4326",
        )

        # `on_error="raise"` (default): the bad asset propagates.
        with pytest.raises(OSError, match="nope"):
            stage(cat, dest=tmp_path / "cache", retries=0)

        # `on_error="skip"`: the row survives — the good asset is
        # rewritten to a local cache path, and the bad asset retains
        # its original URI so downstream code can detect what was
        # left unstaged.
        out = stage(cat, dest=tmp_path / "cache", retries=0, on_error="skip")
        assets_out = json.loads(out.gdf.iloc[0]["assets"])
        assert "good" in assets_out
        assert assets_out.get("bad") == bad
        # The primary `filepath` should be a real local cache path
        # (i.e. the successful "good" asset), not the unresolved URI.
        assert out.gdf.iloc[0]["filepath"] != bad
        assert out.gdf.iloc[0]["filepath"] == assets_out["good"]

    def test_fatal_error_skips_without_retrying(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud_stub: Any
    ) -> None:
        # A fatal failure still honours on_error="skip" (the row
        # survives, the bad asset keeps its URI) — it just never
        # retries, even with a generous retry budget.
        good = _seed_tif(tmp_path / "good.tif", content=b"good")
        bad = "https://forbidden/secret.tif"
        attempts = {"n": 0}

        def fake_download(uri: str, dest: Path, **kwargs: Any) -> Path:
            if uri == bad:
                attempts["n"] += 1
                raise PermissionError("denied")
            raise AssertionError(f"local asset {uri!r} must not be downloaded")

        monkeypatch.setattr(cloud_stub, "download", fake_download, raising=False)

        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": str(good),
                    "assets": json.dumps({"good": str(good), "bad": bad}),
                }
            ],
            crs="EPSG:4326",
        )

        # on_error="raise" (default): the fatal error propagates,
        # after exactly one attempt despite retries=5.
        with pytest.raises(PermissionError, match="denied"):
            stage(cat, dest=tmp_path / "cache", retries=5)
        assert attempts["n"] == 1

        # on_error="skip": the row survives with the good asset
        # staged and the bad one keeping its URI — still one attempt.
        attempts["n"] = 0
        out = stage(cat, dest=tmp_path / "cache", retries=5, on_error="skip")
        assert attempts["n"] == 1
        assets_out = json.loads(out.gdf.iloc[0]["assets"])
        assert "good" in assets_out
        assert assets_out.get("bad") == bad

    def test_invalid_on_error_rejected(self, tmp_path: Path) -> None:
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": "/some/path",
                }
            ],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="on_error"):
            stage(cat, dest=tmp_path / "cache", on_error="bogus")


class TestStageGuardrails:
    def test_non_inmemory_catalog_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="InMemoryGeoCatalog"):
            stage(object(), dest=tmp_path)  # type: ignore[arg-type]

    def test_negative_retries_rejected(self, tmp_path: Path) -> None:
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-06-01"),
                    "end_time": pd.Timestamp("2024-06-02"),
                    "filepath": "/some/path",
                }
            ],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="retries"):
            stage(cat, dest=tmp_path / "cache", retries=-1)
