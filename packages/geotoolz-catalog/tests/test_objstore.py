"""geocatalog has no obstore pool of its own — its ``[obstore]`` extra is geopatcher's.

The pool (keys, Azure construction, signed URLs, LRU, ``set_obstore_pool_maxsize``)
is tested in ``packages/geotoolz-patcher/tests/test_objstore.py``.
"""

from __future__ import annotations

import importlib.util
import re
from importlib.metadata import requires

import pytest


def test_private_pool_module_is_gone():
    assert importlib.util.find_spec("geocatalog._src.objstore") is None


def test_obstore_extra_installs_geopatcher_pool():
    reqs = requires("geotoolz-catalog") or []
    obstore_reqs = [r for r in reqs if re.search(r"extra\s*==\s*['\"]obstore['\"]", r)]
    assert any(r.startswith("geotoolz-patcher[obstore]") for r in obstore_reqs), (
        obstore_reqs
    )


def test_shared_pool_is_importable():
    pytest.importorskip("obstore")
    from geopatcher.objstore import clear_obstore_pool, get_obstore

    store = get_obstore("https://example.com/catalog/item.tif")
    assert get_obstore("https://example.com/other.tif") is store
    clear_obstore_pool()
    assert get_obstore("https://example.com/catalog/item.tif") is not store
    clear_obstore_pool()
