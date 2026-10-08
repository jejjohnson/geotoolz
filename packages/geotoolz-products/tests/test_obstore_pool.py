"""geoproducts has no obstore pool of its own — it uses geopatcher's.

The pool itself (keys, Azure construction, signed URLs, LRU) is tested
in ``packages/geotoolz-patcher/tests/test_objstore.py``; these smoke
tests pin that the sensor-reader byte path goes through it.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import pytest


pytest.importorskip("obstore")

import geopatcher.objstore as shared_pool

from geoproducts._src import base


def test_private_pool_module_is_gone():
    assert importlib.util.find_spec("geoproducts._src.obstore") is None


class _RecordingStore:
    """Stand-in client that records the key each range request asks for."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    async def get_range_async(self, key: str, *, start: int, length: int) -> Any:
        self.keys.append(key)
        return b"x" * length


def test_reader_byte_path_uses_geopatcher_object_key(monkeypatch):
    calls: list[str] = []
    real = shared_pool.object_key

    def spy(uri: str) -> str:
        calls.append(uri)
        return real(uri)

    monkeypatch.setattr(shared_pool, "object_key", spy)
    store = _RecordingStore()
    uri = "az://acct/container/scene/b04.tif"
    got = base._run_coroutine_safely(base._get_range_async(store, uri, 0, 3))
    assert got == b"xxx"
    assert calls == [uri]
    assert store.keys == ["scene/b04.tif"]


def test_reader_remote_schemes_cover_the_pool():
    assert shared_pool.SUPPORTED_SCHEMES == base._REMOTE_SCHEMES
