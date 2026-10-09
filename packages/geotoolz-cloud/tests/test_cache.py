"""`geocloud.cache`: local copies of remote objects (no network)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from obstore.store import MemoryStore

from geocloud import files
from geocloud.cache import LocalCache, cache_key, localize
from geocloud.store import mount, unmount


ROOT = "s3://cache-test"


@pytest.fixture
def bucket() -> Iterator[str]:
    mount(ROOT, MemoryStore())
    yield ROOT
    unmount(ROOT)


# --- where files land -----------------------------------------------------------


def test_explicit_root_is_created(tmp_path: Path):
    assert LocalCache(root=tmp_path / "c").resolve_root() == tmp_path / "c"
    assert (tmp_path / "c").is_dir()


def test_env_var_then_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEOCLOUD_CACHE", str(tmp_path / "env"))
    assert LocalCache().resolve_root() == tmp_path / "env"
    monkeypatch.delenv("GEOCLOUD_CACHE")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert LocalCache().resolve_root() == tmp_path / ".cache" / "geocloud"


def test_slots_are_per_uri_with_extension_and_prefix(tmp_path: Path):
    cache = LocalCache(root=tmp_path)
    uri = "s3://bucket/x.tif"
    path = cache.path_for(uri)
    digest = hashlib.sha256(uri.encode()).hexdigest()
    assert path == tmp_path / digest[:2] / f"{digest}.tif"
    assert cache.path_for(uri) == path
    assert cache.path_for("s3://bucket/y.tif") != path
    assert cache.path_for("https://h/a.nc?token=1").suffix == ".nc"
    assert cache.path_for("https://h/no-extension").suffix == ""
    # A local path is not a URL: `#` and `?` are ordinary characters.
    assert cache.path_for("/data/run#3/scene.tif").suffix == ".tif"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "https://h/a.tif?X-Amz-Signature=1&X-Amz-Date=2",
            "https://h/a.tif?X-Amz-Signature=9",
        ),
        (
            "https://acct.blob.core.windows.net/c/a.tif?sig=1&se=2",
            "https://acct.blob.core.windows.net/c/a.tif?sig=3&se=4",
        ),
        (
            "https://cdn/a.tif?Expires=1&Signature=2&Key-Pair-Id=3",
            "https://cdn/a.tif?Expires=9&Signature=8&Key-Pair-Id=3",
        ),
    ],
)
def test_resigned_urls_share_a_slot(a: str, b: str, tmp_path: Path):
    cache = LocalCache(root=tmp_path)
    assert cache.path_for(a) == cache.path_for(b)


def test_content_selecting_query_keeps_its_own_slot(tmp_path: Path):
    cache = LocalCache(root=tmp_path)
    assert cache.path_for("https://x/a?sp=3") != cache.path_for("https://x/a?sp=4")
    assert cache_key("https://x/a?sp=3") == "https://x/a?sp=3"  # no `sig`: not SAS
    assert cache_key("/data/a?b.tif") == "/data/a?b.tif"


# --- freshness and pruning ----------------------------------------------------------


def _age(path: Path, days: float) -> None:
    then = (datetime.now(tz=UTC) - timedelta(days=days)).timestamp()
    os.utime(path, (then, then))


def test_ttl_freshness(tmp_path: Path):
    target = LocalCache(root=tmp_path).path_for("s3://b/x.tif")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    assert LocalCache(root=tmp_path).is_fresh(target)
    assert LocalCache(root=tmp_path, ttl_days=7).is_fresh(target)
    _age(target, 2)
    assert not LocalCache(root=tmp_path, ttl_days=1).is_fresh(target)
    assert not LocalCache(root=tmp_path).is_fresh(tmp_path / "nope.tif")


def test_prune_evicts_expired_files_and_partials(tmp_path: Path):
    cache = LocalCache(root=tmp_path, ttl_days=1)
    old = cache.path_for("s3://b/old.tif")
    new = cache.path_for("s3://b/new.tif")
    for path in (old, new):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    _age(old, 2)
    part = old.with_name(f".{old.name}.{'0' * 32}.part")
    part.write_bytes(b"partial")
    assert cache.prune() == 2
    assert new.exists() and not old.exists() and not part.exists()
    assert LocalCache(root=tmp_path).prune() == 0  # no TTL: only partials


# --- fetching -----------------------------------------------------------------------


def test_local_paths_are_used_in_place(tmp_path: Path):
    src = tmp_path / "a.tif"
    src.write_bytes(b"a")
    cache = LocalCache(root=tmp_path / "cache")
    assert cache.fetch(src) == src
    assert cache.fetch(src.as_uri()) == src
    assert localize(str(src), cache=cache) == src
    with pytest.raises(FileNotFoundError):
        cache.fetch(tmp_path / "gone.tif")


def test_remote_objects_download_once(bucket: str, tmp_path: Path, monkeypatch):
    files.write_bytes(f"{bucket}/scene.nc", b"netcdf")
    calls: list[str] = []
    real = files.download

    def counting(uri: str, dest: Any, **kwargs: Any) -> Path:
        calls.append(uri)
        return real(uri, dest, **kwargs)

    monkeypatch.setattr(files, "download", counting)
    cache = LocalCache(root=tmp_path)
    first = cache.fetch(f"{bucket}/scene.nc")
    second = localize(f"{bucket}/scene.nc", cache=cache)
    assert first == second == cache.path_for(f"{bucket}/scene.nc")
    assert first.read_bytes() == b"netcdf"
    assert calls == [f"{bucket}/scene.nc"]


def test_timeout_reaches_the_client(tmp_path: Path, monkeypatch):
    seen: list[Any] = []

    def fake(uri: str, dest: Path, storage_options: Any = None) -> Path:
        seen.append(storage_options)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(files, "download", fake)
    LocalCache(root=tmp_path, timeout=12.5).fetch("s3://b/a.tif")
    LocalCache(root=tmp_path / "2", timeout=None).fetch("s3://b/a.tif")
    assert seen == [{"client_options": {"timeout": timedelta(seconds=12.5)}}, None]


def test_default_timeout_is_60s_and_serialisable():
    import dataclasses

    assert LocalCache().timeout == 60.0
    assert dataclasses.asdict(LocalCache(root="/x"))["timeout"] == 60.0
