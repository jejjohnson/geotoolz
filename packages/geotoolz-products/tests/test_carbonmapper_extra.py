"""``geoproducts.carbonmapper`` fails at import, naming its extra, when missing."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest


def test_missing_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def no_requests(name: str, *args, **kwargs):
        return None if name == "requests" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", no_requests)
    for name in [m for m in sys.modules if m.startswith("geoproducts.carbonmapper")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r"geotoolz-products\[carbonmapper\]"):
        importlib.import_module("geoproducts.carbonmapper")
