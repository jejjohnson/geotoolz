"""An fsspec filesystem on the shared pool, for libraries that only speak fsspec.

xarray, zarr, pyarrow and geopandas read remote data through fsspec.
`GeoCloudFileSystem` answers them with the stack's own engine: every path
resolves through `geocloud.store` (one client per bucket, the credentials
registered in `geocloud.credentials`, local paths through the pooled
``LocalStore``), so no s3fs / gcsfs / adlfs / huggingface-hub client and
no second set of credentials is involved.

Paths are full locations in any form the pool accepts (``s3://b/k``,
``az://account/container/k``, signed ``https://`` URLs, ``hf://``, local
paths); the filesystem keeps them as written rather than stripping a
protocol. It is not registered under a protocol: build one with
`filesystem` and hand it to the library (``fs.open(uri)``,
``fs.get_mapper(uri)``, ``filesystem=fs``).
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from typing import Any

import obstore
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem

from geocloud._src.files import (
    _resolve,
    info,
    ls,
    open as open_object,
    read_bytes,
    rm,
    write_bytes,
)
from geocloud._src.store import local_path


__all__ = ["GeoCloudFileSystem", "filesystem"]


class _ObjectWriter(io.RawIOBase):
    """A write-only file over an obstore writer, for libraries that check
    ``closed`` / ``tell`` (pyarrow); the object appears on `close`."""

    def __init__(self, writer: Any) -> None:
        super().__init__()
        self._writer = writer
        self._pos = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        chunk = bytes(data)
        self._writer.write(chunk)
        self._pos += len(chunk)
        return len(chunk)

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        if not self.closed:
            self._writer.close()
        super().close()


class _ObjectFile(AbstractBufferedFile):
    """A read-only, seekable fsspec file whose reads are ranged GETs."""

    def _fetch_range(self, start: int, end: int) -> bytes:
        target = _resolve(self.path, self.fs._options())
        if end <= start:
            return b""
        return bytes(obstore.get_range(target.store, target.key, start=start, end=end))


class GeoCloudFileSystem(AbstractFileSystem):
    """fsspec over `geocloud.store`: any URI or local path, one engine.

    Reads are ranged GETs through fsspec's block cache; writes go
    through `geocloud.files.open` (``"wb"``), so the object appears
    when the file is closed. Listing works where `geocloud.files.ls`
    does (S3, GCS, Azure, local directories).

    Args:
        storage_options: Forwarded to `geocloud.store.get_obstore` for
            every remote path (over the registered credentials).

    Examples:
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "x.bin"
        >>> fs = GeoCloudFileSystem()
        >>> fs.pipe_file(str(path), b"abcdef")
        >>> with fs.open(str(path)) as fh:
        ...     _ = fh.seek(2)
        ...     fh.read(3)
        b'cde'
    """

    protocol = ("geocloud",)
    root_marker = ""
    cachable = False  # the pool already caches the clients

    def __init__(
        self, storage_options: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._pool_options = dict(storage_options or {})

    @classmethod
    def _strip_protocol(cls, path: Any) -> str:
        """Keep the location as written: the scheme says which store."""
        if isinstance(path, list):  # fsspec passes lists through here too
            return [cls._strip_protocol(p) for p in path]  # type: ignore[return-value]
        text = os.fspath(path)
        return text.rstrip("/") if text.rstrip("/") else text

    def _parent(self, path: str) -> str:
        return path.rstrip("/").rsplit("/", 1)[0]

    def _options(self) -> Mapping[str, Any] | None:
        return self._pool_options or None

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        path = self._strip_protocol(path)
        try:
            meta = info(path, storage_options=self._options())
        except FileNotFoundError:
            if self._is_dir(path):
                return {"name": path, "size": 0, "type": "directory"}
            raise
        return {
            "name": path,
            "size": meta.size,
            "type": "file",
            "mtime": meta.last_modified,
            "ETag": meta.e_tag,
        }

    def _is_dir(self, path: str) -> bool:
        local = local_path(path)
        if local is not None:
            return local.is_dir()
        try:
            return bool(ls(path, recursive=False, storage_options=self._options()))
        except (FileNotFoundError, ValueError):
            return False

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        path = self._strip_protocol(path)
        entries = [
            {
                "name": e.uri.rstrip("/"),
                "size": e.size,
                "type": "directory" if e.is_dir else "file",
            }
            for e in ls(path, recursive=False, storage_options=self._options())
        ]
        if not entries and not self._is_dir(path):
            # `ls` of one file names the file itself, as fsspec expects.
            entries = [self.info(path)]
        return entries if detail else [e["name"] for e in entries]

    def cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        path = self._strip_protocol(path)
        if start is None and end is None:
            return read_bytes(path, storage_options=self._options())
        size = self.size(path)
        start = 0 if start is None else (start if start >= 0 else size + start)
        end = size if end is None else (end if end >= 0 else size + end)
        target = _resolve(path, self._options())
        if end <= start:
            return b""
        return bytes(obstore.get_range(target.store, target.key, start=start, end=end))

    def pipe_file(
        self, path: str, value: bytes, mode: str = "overwrite", **kwargs: Any
    ) -> None:
        path = self._strip_protocol(path)
        local = local_path(path)
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(path, value, storage_options=self._options())

    def rm_file(self, path: str) -> None:
        rm(self._strip_protocol(path), storage_options=self._options())

    def _rm(self, path: str) -> None:
        self.rm_file(path)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        local = local_path(path)
        if local is not None:  # object stores have no directories
            local.mkdir(parents=create_parents, exist_ok=True)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        local = local_path(path)
        if local is not None:
            local.mkdir(parents=True, exist_ok=exist_ok)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        path = self._strip_protocol(path)
        if mode == "rb":
            return _ObjectFile(
                self,
                path,
                mode,
                block_size=block_size or "default",
                cache_options=cache_options,
                size=self.size(path),
            )
        if mode == "wb":
            local = local_path(path)
            if local is not None:
                local.parent.mkdir(parents=True, exist_ok=True)
            return _ObjectWriter(
                open_object(path, "wb", storage_options=self._options())
            )
        raise ValueError(
            f"GeoCloudFileSystem: mode must be 'rb' or 'wb'; got {mode!r}."
        )


def filesystem(storage_options: Mapping[str, Any] | None = None) -> GeoCloudFileSystem:
    """An fsspec filesystem that reads and writes through the shared pool.

    Args:
        storage_options: Forwarded to `geocloud.store.get_obstore` for
            every remote path, over the registered credentials.

    Returns:
        A `GeoCloudFileSystem`.

    Examples:
        >>> fs = filesystem()
        >>> fs.exists("/no/such/file.bin")
        False
    """
    return GeoCloudFileSystem(storage_options)
