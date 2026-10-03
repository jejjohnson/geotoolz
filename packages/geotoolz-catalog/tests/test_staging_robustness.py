"""`stage()` robustness: base install, primary asset, atomic and deduped
downloads, cancellation, `staged_from`, asset-key validation and cache
keys (#243).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from shapely.geometry import box

from geocatalog._src.staging._base import (
    LocalCache,
    _ext_for,
    cache_key,
    stage,
)
from tests.conftest import catalog_from_rows


def _cat(*rows: dict[str, Any]) -> Any:
    return catalog_from_rows(
        rows=[
            {
                "geometry": box(0, 0, 1, 1),
                "start_time": pd.Timestamp("2024-06-01"),
                "end_time": pd.Timestamp("2024-06-02"),
                **row,
            }
            for row in rows
        ],
        crs="EPSG:4326",
    )


class _Remote:
    """Stand-in for `fsspec.open` serving bytes per URI and counting opens."""

    def __init__(
        self, content: dict[str, bytes], *, size: dict[str, int] | None = None
    ):
        self.content = content
        self.size = size or {}
        self.opens: list[str] = []
        self.lock = threading.Lock()
        self.fail: dict[str, BaseException] = {}
        self.fail_after_write: dict[str, BaseException] = {}
        self.delay = 0.0  # seconds each successful open takes

    def __call__(self, uri: str, mode: str = "rb", **kwargs: Any) -> Any:
        with self.lock:
            self.opens.append(uri)
        if uri in self.fail:
            raise self.fail[uri]
        if self.delay:
            threading.Event().wait(self.delay)  # `time.sleep` is patched out
        remote = self

        class _File:
            size = remote.size.get(uri, len(remote.content[uri]))
            done = False

            def read(self, _n: int = -1) -> bytes:
                if self.done:
                    if uri in remote.fail_after_write:
                        raise remote.fail_after_write[uri]
                    return b""
                self.done = True
                return remote.content[uri]

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        return _File()


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> _Remote:
    import fsspec

    fake = _Remote({})
    monkeypatch.setattr(fsspec, "open", fake)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return fake


def _leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.rglob("*") if p.name.endswith(".part"))


# ---------------------------------------------------------------------------
# Base install
# ---------------------------------------------------------------------------


def test_local_catalog_stages_without_fsspec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "a.tif"
    src.write_bytes(b"a")
    monkeypatch.setitem(sys.modules, "fsspec", None)  # `import fsspec` fails
    out = stage(_cat({"filepath": str(src)}), dest=tmp_path / "cache")
    assert out.gdf.iloc[0]["filepath"] == str(src)
    out = stage(_cat({"filepath": src.as_uri()}), dest=tmp_path / "cache")
    assert out.gdf.iloc[0]["filepath"] == str(src)


def test_remote_uri_without_fsspec_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ModuleNotFoundError, match=r"geotoolz-catalog\[fsspec\]"):
        stage(_cat({"filepath": "s3://b/x.tif"}), dest=tmp_path / "cache")


def test_missing_local_file_is_a_fatal_error(tmp_path: Path) -> None:
    cat = _cat({"filepath": str(tmp_path / "gone.tif")})
    with pytest.raises(FileNotFoundError, match=r"gone\.tif"):
        stage(cat, dest=tmp_path / "cache", retries=3)


# ---------------------------------------------------------------------------
# filepath follows the primary asset
# ---------------------------------------------------------------------------


def test_filepath_follows_the_primary_asset(tmp_path: Path, remote: _Remote) -> None:
    b04, b08 = "https://x/B04.tif", "https://x/B08.tif"
    remote.content.update({b04: b"red", b08: b"nir"})
    cat = _cat({"filepath": b08, "assets": json.dumps({"B04": b04, "B08": b08})})
    out = stage(cat, dest=tmp_path / "cache")
    assert Path(out.gdf.iloc[0]["filepath"]).read_bytes() == b"nir"


def test_filepath_keeps_its_uri_when_the_primary_is_filtered_out(
    tmp_path: Path, remote: _Remote
) -> None:
    b04, b08 = "https://x/B04.tif", "https://x/B08.tif"
    remote.content.update({b04: b"red", b08: b"nir"})
    cat = _cat({"filepath": b08, "assets": json.dumps({"B04": b04, "B08": b08})})
    out = stage(cat, dest=tmp_path / "cache", assets=["B04"])
    assert out.gdf.iloc[0]["filepath"] == b08
    assert set(json.loads(out.gdf.iloc[0]["assets"])) == {"B04"}


# ---------------------------------------------------------------------------
# Atomic downloads
# ---------------------------------------------------------------------------


def test_interrupted_download_is_not_a_cache_hit(
    tmp_path: Path, remote: _Remote
) -> None:
    uri = "https://x/a.tif"
    remote.content[uri] = b"payload"
    remote.fail_after_write[uri] = PermissionError("connection dropped")
    cache = LocalCache(root=tmp_path / "cache")
    with pytest.raises(PermissionError):
        stage(_cat({"filepath": uri}), cache=cache)
    assert not cache.path_for(uri).exists()
    assert _leftovers(tmp_path / "cache") == []

    del remote.fail_after_write[uri]
    out = stage(_cat({"filepath": uri}), cache=cache)
    assert Path(out.gdf.iloc[0]["filepath"]).read_bytes() == b"payload"
    assert len(remote.opens) == 2


def test_short_read_is_retried_then_fails(tmp_path: Path, remote: _Remote) -> None:
    uri = "https://x/a.tif"
    remote.content[uri] = b"abc"
    remote.size[uri] = 10
    cache = LocalCache(root=tmp_path / "cache")
    with pytest.raises(OSError, match="short read"):
        stage(_cat({"filepath": uri}), cache=cache, retries=2)
    assert len(remote.opens) == 3
    assert not cache.path_for(uri).exists()
    assert _leftovers(tmp_path / "cache") == []


# ---------------------------------------------------------------------------
# One download per URI; on_error="raise" cancels the rest
# ---------------------------------------------------------------------------


def test_rows_sharing_a_uri_download_it_once(tmp_path: Path, remote: _Remote) -> None:
    uri = "https://x/shared.tif"
    remote.content[uri] = b"shared"
    rows = [{"filepath": uri} for _ in range(5)]
    out = stage(_cat(*rows), dest=tmp_path / "cache", parallel=8)
    assert remote.opens == [uri]
    assert len(set(out.gdf["filepath"])) == 1


def test_shared_uri_failure_marks_every_user(tmp_path: Path, remote: _Remote) -> None:
    uri = "https://x/shared.tif"
    remote.fail[uri] = PermissionError("denied")
    rows = [{"filepath": "x", "assets": json.dumps({"k": uri})} for _ in range(2)]
    out = stage(_cat(*rows), dest=tmp_path / "cache", on_error="skip")
    assert [json.loads(a) for a in out.gdf["assets"]] == [{"k": uri}, {"k": uri}]


def test_raise_cancels_downloads_not_yet_started(
    tmp_path: Path, remote: _Remote
) -> None:
    uris = [f"https://x/{i}.tif" for i in range(6)]
    remote.content.update({u: b"x" for u in uris})
    remote.fail[uris[0]] = PermissionError("denied")
    remote.delay = 0.2
    with pytest.raises(PermissionError):
        stage(
            _cat(*[{"filepath": u} for u in uris]),
            dest=tmp_path / "cache",
            parallel=1,
        )
    # The single worker may already have taken the next URI when the
    # failure is seen; everything after it is cancelled.
    assert remote.opens[0] == uris[0]
    assert len(remote.opens) <= 2


# ---------------------------------------------------------------------------
# staged_from, asset keys and column shapes
# ---------------------------------------------------------------------------


def test_staged_from_is_a_visible_extra(tmp_path: Path, remote: _Remote) -> None:
    uri = "https://x/a.tif"
    remote.content[uri] = b"a"
    out = stage(_cat({"filepath": uri}), dest=tmp_path / "cache")
    (row,) = out.iter_rows()
    assert json.loads(row.extras["staged_from"]) == {"filepath": uri}


def test_empty_assets_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"assets=\[\]"):
        stage(_cat({"filepath": "x"}), dest=tmp_path, assets=[])


def test_unknown_asset_key_raises(tmp_path: Path) -> None:
    cat = _cat({"filepath": "a", "assets": json.dumps({"red": "a", "nir": "b"})})
    with pytest.raises(ValueError, match=r"\['swir'\].*known keys"):
        stage(cat, dest=tmp_path, assets=["red", "swir"])


def test_legacy_rows_gain_no_assets_column(tmp_path: Path) -> None:
    src = tmp_path / "a.tif"
    src.write_bytes(b"a")
    out = stage(_cat({"filepath": str(src)}), dest=tmp_path / "cache")
    assert "assets" not in out.gdf.columns


def test_mixed_rows_keep_legacy_asset_cells(tmp_path: Path) -> None:
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    out = stage(
        _cat(
            {"filepath": str(a), "assets": None},
            {"filepath": str(b), "assets": json.dumps({"data": str(b)})},
        ),
        dest=tmp_path / "cache",
    )
    assert pd.isna(out.gdf.iloc[0]["assets"])
    assert json.loads(out.gdf.iloc[1]["assets"]) == {"data": str(b)}


def test_dict_asset_map_is_an_asset_map(tmp_path: Path, remote: _Remote) -> None:
    uri = "https://x/a.tif"
    remote.content[uri] = b"a"
    cat = _cat({"filepath": uri, "assets": "placeholder"})
    cat.gdf["assets"] = [{"data": uri}]
    out = stage(cat, dest=tmp_path / "cache")
    staged = out.gdf.iloc[0]["assets"]
    assert isinstance(staged, dict)
    assert Path(staged["data"]).read_bytes() == b"a"
    assert out.gdf.iloc[0]["filepath"] == staged["data"]


# ---------------------------------------------------------------------------
# Cache keys and eviction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "https://acct.blob.core.windows.net/c/a.tif?st=1&se=2&sp=r&sv=3&sr=b&sig=AAA",
            "https://acct.blob.core.windows.net/c/a.tif?st=4&se=5&sp=r&sv=3&sr=b&sig=BBB",
        ),
        (
            "https://b.s3.amazonaws.com/a.tif?X-Amz-Signature=1&X-Amz-Date=2",
            "https://b.s3.amazonaws.com/a.tif?X-Amz-Signature=3&X-Amz-Date=4",
        ),
    ],
    ids=["azure-sas", "aws"],
)
def test_resigned_urls_share_a_cache_slot(a: str, b: str, tmp_path: Path) -> None:
    cache = LocalCache(root=tmp_path)
    assert cache.path_for(a) == cache.path_for(b)
    assert cache.path_for(a).suffix == ".tif"


def test_content_selecting_query_keeps_its_own_slot(tmp_path: Path) -> None:
    cache = LocalCache(root=tmp_path)
    assert cache.path_for("https://x/a?band=1") != cache.path_for("https://x/a?band=2")
    # `sp` only signs when an Azure `sig` is present.
    assert cache_key("https://x/a?sp=3") == "https://x/a?sp=3"


def test_local_path_with_hash_keeps_its_extension() -> None:
    assert _ext_for("/data/run#3/scene.tif") == ".tif"
    assert _ext_for("/data/scene#1.nc") == ".nc"


def test_prune_evicts_expired_files_and_partials(tmp_path: Path) -> None:
    cache = LocalCache(root=tmp_path, ttl_days=1)
    old = cache.path_for("https://x/old.tif")
    new = cache.path_for("https://x/new.tif")
    for p in (old, new):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    part = new.with_name(f".{new.name}.{'a' * 32}.part")
    part.write_bytes(b"partial")
    two_days_ago = time.time() - 2 * 86400
    os.utime(old, (two_days_ago, two_days_ago))

    assert cache.prune() == 2
    assert not old.exists() and not part.exists()
    assert new.exists()
    assert LocalCache(root=tmp_path).prune() == 0  # no TTL: only partials


# ---------------------------------------------------------------------------
# Review follow-ups (#365)
# ---------------------------------------------------------------------------


def test_primary_follows_a_selected_alias_of_its_uri(
    tmp_path: Path, remote: _Remote
) -> None:
    b04, b08 = "https://x/B04.tif", "https://x/B08.tif"
    remote.content.update({b04: b"red", b08: b"nir"})
    assets = {"visual": b04, "B04": b04, "B08": b08}  # `visual` aliases B04
    cat = _cat({"filepath": b04, "assets": json.dumps(assets)})
    out = stage(cat, dest=tmp_path / "cache", assets=["B04"])
    assert Path(out.gdf.iloc[0]["filepath"]).read_bytes() == b"red"


def test_asset_filter_on_a_filepath_only_catalog_stages_the_filepath(
    tmp_path: Path, remote: _Remote
) -> None:
    uri = "https://x/a.tif"
    remote.content[uri] = b"a"
    out = stage(_cat({"filepath": uri}), dest=tmp_path / "cache", assets=["red"])
    assert Path(out.gdf.iloc[0]["filepath"]).read_bytes() == b"a"


def test_file_uri_authority_is_kept() -> None:
    from geocatalog._src.staging._base import _local_path

    assert _local_path("file://server/share/x.tif") == Path("//server/share/x.tif")
    assert _local_path("file://localhost/data/x.tif") == Path("/data/x.tif")
    assert _local_path("file:///data/a%20b.tif") == Path("/data/a b.tif")


def test_prune_keeps_a_complete_download_named_part(
    tmp_path: Path, remote: _Remote
) -> None:
    uri = "https://x/archive.part"
    remote.content[uri] = b"whole"
    cache = LocalCache(root=tmp_path / "cache")
    stage(_cat({"filepath": uri}), cache=cache)
    slot = cache.path_for(uri)
    temp = slot.with_name(f".{slot.name}.{'0' * 32}.part")
    temp.write_bytes(b"partial")
    assert cache.prune() == 1
    assert slot.read_bytes() == b"whole" and not temp.exists()
