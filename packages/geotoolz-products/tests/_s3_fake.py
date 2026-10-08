"""Public buckets served from memory for the bucket-helper tests (no network).

`FakeS3` mounts an obstore ``MemoryStore`` at every ``s3://`` bucket the
code under test touches (`geocloud.store.mount`), so the real listing and
download path runs; ``calls`` records the URI of every ``ls`` /
``download`` it made.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest


class FakeS3:
    """In-memory buckets behind `geocloud.files`; records every call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from geocloud import files
        from obstore.store import MemoryStore

        self._memory = MemoryStore
        self.stores: dict[str, Any] = {}
        self.calls: list[str] = []
        real_ls, real_download = files.ls, files.download

        def ls(uri: str, **kwargs: Any) -> Any:
            self._record(uri)
            return real_ls(uri, **kwargs)

        def download(uri: str, dest: Any, **kwargs: Any) -> Any:
            self._record(uri)
            return real_download(uri, dest, **kwargs)

        monkeypatch.setattr(files, "ls", ls)
        monkeypatch.setattr(files, "download", download)

    def bucket(self, name: str) -> Any:
        """The bucket's ``MemoryStore``, mounted on first use."""
        from geocloud.store import mount

        if name not in self.stores:
            self.stores[name] = self._memory()
            mount(f"s3://{name}", self.stores[name])
        return self.stores[name]

    def put(self, bucket: str, objects: Mapping[str, bytes] | Iterable[str]) -> None:
        """Add objects; bare keys get a 100-byte body."""
        import obstore

        items = (
            objects.items()
            if isinstance(objects, Mapping)
            else ((key, b"x" * 100) for key in objects)
        )
        for key, body in items:
            obstore.put(self.bucket(bucket), key, body)

    def _record(self, uri: str) -> None:
        self.calls.append(uri)
        self.bucket(uri.removeprefix("s3://").split("/", 1)[0])

    def close(self) -> None:
        from geocloud.store import unmount

        for name in self.stores:
            unmount(f"s3://{name}")
