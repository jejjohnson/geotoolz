"""Shared fixtures: ``fake_s3`` serves public buckets from memory."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import pytest


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator:
    """Factory: ``fake_s3(objects, bucket=...)`` → a `FakeS3` with them in it.

    Skips without geotoolz-cloud (a base install of this package).
    """
    pytest.importorskip("geocloud")
    from _s3_fake import FakeS3

    fake = FakeS3(monkeypatch)

    def install(
        objects: Mapping[str, bytes] | Iterable[str] = (),
        *,
        bucket: str = "noaa-goes19",
    ) -> FakeS3:
        fake.put(bucket, objects)
        return fake

    yield install
    fake.close()
