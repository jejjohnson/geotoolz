"""Shared fixtures: no test reads the developer's own credentials file."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from geocloud._src import credentials


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GEOCLOUD_CREDENTIALS", "")
    credentials._clear()
    yield
    credentials._clear()
