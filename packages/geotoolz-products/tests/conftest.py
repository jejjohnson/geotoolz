"""Shared fixtures: ``fake_s3`` installs :class:`_s3_fake.FakeS3`."""

from __future__ import annotations

import urllib.request

import pytest
from _s3_fake import FakeS3


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch):
    """Factory: ``fake_s3(pages=..., bodies=...)`` patches ``urlopen``."""

    def install(pages=None, bodies=None) -> FakeS3:
        fake = FakeS3(pages or {}, bodies or {})
        monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
        return fake

    return install
