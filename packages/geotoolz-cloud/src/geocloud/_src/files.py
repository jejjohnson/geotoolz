"""Whole-object file operations on the pool: list, read, write, move, sign.

Every function takes URIs in any form `get_obstore` accepts (``s3://``,
``gs://``, ``az://`` / ``abfs[s]://``, Azure ``https://``, signed
``http(s)://``, ``hf://``) or a **local path** (``str`` / ``Path`` /
``file://``), so one call covers cloud → local, local → cloud and cloud →
cloud. Remote objects go through the shared client pool; local paths go
through an obstore ``LocalStore`` rooted at the filesystem anchor, so every
transfer runs on the same engine.

How a transfer runs:

- same store (two keys of one bucket / container) → a server-side copy,
  no bytes through this process;
- local file → remote → a multipart upload straight from the file;
- remote → local → streamed in chunks to a hidden ``.part`` sibling, size
  checked, then renamed into place (a cut-off download never leaves a
  truncated file under the final name);
- remote → another remote store → streamed through memory chunk by chunk
  (obstore's async ``get`` feeding its async multipart ``put``).

Listing (`ls`, `sync`, recursive `rm`) needs a store that can list: S3,
GCS, Azure or a local directory. Plain ``http(s)://`` and ``hf://`` URLs
can be read, downloaded and copied from, not listed.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlsplit

import obstore
from obstore.store import LocalStore

from geocloud._src.aio import _run_coroutine_safely
from geocloud._src.store import _locate, get_obstore, object_key


if TYPE_CHECKING:
    from obstore import ReadableFile, WritableFile
    from obstore.store import ObjectStore


__all__ = [
    "ObjectInfo",
    "copy",
    "download",
    "exists",
    "info",
    "ls",
    "open",
    "read_bytes",
    "rm",
    "sign",
    "sync",
    "upload",
    "write_bytes",
]

#: A URI string, a local path string, or a ``Path``.
Location = str | os.PathLike[str]

# Streaming chunk size: big enough for S3's 5 MiB multipart minimum, small
# enough that a copy never holds more than a few chunks in memory.
_CHUNK = 8 * 1024 * 1024
# Keys per bulk delete request (S3's DeleteObjects limit).
_DELETE_BATCH = 1000


@dataclass(frozen=True)
class ObjectInfo:
    """One object (or, from a non-recursive `ls`, one sub-directory).

    Attributes:
        uri: Where it is, in the form of the listed URI: a full URI for
            remote objects, a local path for local ones.
        size: Size in bytes (``0`` for a directory entry).
        last_modified: Last modification time, UTC; ``None`` for a
            directory entry.
        e_tag: The store's entity tag, when it reports one.
        is_dir: ``True`` for a common prefix returned by
            ``ls(..., recursive=False)``.

    Examples:
        >>> ObjectInfo("s3://bucket/a.tif", 10).is_dir
        False
    """

    uri: str
    size: int
    last_modified: datetime | None = None
    e_tag: str | None = None
    is_dir: bool = False


@dataclass(frozen=True)
class _Target:
    """A location resolved to ``(store, key)`` plus how to name its children."""

    store: ObjectStore
    key: str
    root: str  # prefix that turns a store key back into a URI / path
    local: bool
    listable: bool

    def child(self, key: str) -> str:
        """URI (or local path) of ``key`` in the same store."""
        if self.local:
            return str(Path(self.root) / key)
        return self.root + key

    @property
    def name(self) -> str:
        """The last path segment of the key (the file name)."""
        return PurePosixPath(self.key).name

    def describe(self, meta: Mapping[str, Any]) -> ObjectInfo:
        return ObjectInfo(
            uri=self.child(meta["path"]),
            size=int(meta["size"]),
            last_modified=meta.get("last_modified"),
            e_tag=meta.get("e_tag"),
        )


_LOCAL_STORES: dict[str, LocalStore] = {}
_LOCAL_LOCK = threading.Lock()


def _local_path(location: Location) -> Path | None:
    """The filesystem path ``location`` names; ``None`` for a remote URI."""
    if isinstance(location, os.PathLike):
        return Path(location)
    text = str(location)
    if text.startswith("file://"):
        return Path(unquote(urlsplit(text).path))
    if "://" in text:
        return None
    return Path(text)


def _local_store(anchor: str) -> LocalStore:
    """One ``LocalStore`` per filesystem anchor (``/``, ``C:\\``)."""
    with _LOCAL_LOCK:
        store = _LOCAL_STORES.get(anchor)
        if store is None:
            store = _LOCAL_STORES[anchor] = LocalStore(anchor)
        return store


def _resolve(location: Location, storage_options: Mapping[str, Any] | None) -> _Target:
    """Resolve ``location`` to its store, key and child-naming root."""
    local = _local_path(location)
    if local is not None:
        # `abspath` also folds `..`, which a LocalStore key may not hold.
        path = Path(os.path.abspath(local.expanduser()))
        key = path.relative_to(path.anchor).as_posix()
        return _Target(
            _local_store(path.anchor),
            "" if key == "." else key,
            path.anchor,
            local=True,
            listable=True,
        )
    uri = str(location)
    key = object_key(uri)
    # Plain HTTP stores "list" through WebDAV, which object hosts don't speak.
    listable = _locate(uri).backend in ("s3", "gcs", "azure")
    base = uri.split("?", 1)[0].split("#", 1)[0]
    # The root is the URI minus its key; for `hf://` (whose key is a
    # rewritten Hub route) there is none, and children cannot be named.
    root = base[: len(base) - len(key)] if base.endswith(key) else ""
    if root and not root.endswith("/"):
        root += "/"  # a bare root: `s3://bucket`, `az://account/container`
    store = get_obstore(uri, storage_options=storage_options)
    return _Target(store, key, root, local=False, listable=listable)


def _prefix(target: _Target) -> str | None:
    """``target.key`` as a listing prefix (``None`` lists the whole store)."""
    return target.key.rstrip("/") or None


def _require_listable(target: _Target, location: Location) -> None:
    if not target.listable:
        raise ValueError(
            f"{location!r} cannot be listed: listing needs an s3://, gs://, "
            "az:// / abfs[s]:// URI or a local directory."
        )


def _require_object(target: _Target, location: Location, role: str) -> None:
    """Reject a bucket / container root or a ``.../`` prefix as one object."""
    if not target.key or target.key.endswith("/"):
        raise ValueError(
            f"copy: {role} {location!r} names a prefix, not an object; use "
            "`sync` for prefixes, or end the destination with `/` to keep the "
            "source's name."
        )


def _into(dst: Location, name: str) -> Location:
    """``dst``, or ``dst/name`` when ``dst`` is a directory (trailing ``/``)."""
    if not name:
        return dst
    local = _local_path(dst)
    if local is not None:
        text = os.fspath(dst)
        if local.is_dir() or text.endswith(("/", os.sep)):
            return local / name
        return dst
    text = str(dst)
    return text + name if text.endswith("/") else dst


def _join(prefix: Location, rel: str) -> Location:
    """The location ``rel`` below the directory-like ``prefix``."""
    local = _local_path(prefix)
    if local is not None:
        return local / rel
    return str(prefix).rstrip("/") + "/" + rel


# --- inspect ---------------------------------------------------------------


def ls(
    prefix: Location,
    *,
    recursive: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> list[ObjectInfo]:
    """List the objects under ``prefix``, sorted by URI.

    The prefix is matched by whole path segments, like a directory:
    ``s3://bucket/2026`` lists ``2026/…`` but not ``2026-old/…``.

    Args:
        prefix: A bucket / container root, a "directory" URI, or a local
            directory.
        recursive: List everything below ``prefix`` (default); ``False``
            lists one level, with sub-directories as ``is_dir`` entries.
        storage_options: Forwarded to `get_obstore` for a remote prefix.

    Returns:
        One `ObjectInfo` per object (and directory, when not recursive).

    Raises:
        ValueError: ``prefix`` is a plain ``http(s)://`` or ``hf://`` URL,
            which cannot be listed.

    Examples:
        >>> import tempfile, pathlib
        >>> root = pathlib.Path(tempfile.mkdtemp())
        >>> write_bytes(root / "a" / "x.bin", b"abc")
        >>> [(p.uri.removeprefix(str(root)), p.size) for p in ls(root)]
        [('/a/x.bin', 3)]
    """
    target = _resolve(prefix, storage_options)
    _require_listable(target, prefix)
    if recursive:
        found = [
            target.describe(meta)
            for page in obstore.list(target.store, prefix=_prefix(target))
            for meta in page
        ]
    else:
        level = obstore.list_with_delimiter(target.store, prefix=_prefix(target))
        found = [target.describe(meta) for meta in level["objects"]]
        found += [
            ObjectInfo(target.child(path) + "/", 0, is_dir=True)
            for path in level["common_prefixes"]
        ]
    return sorted(found, key=lambda item: item.uri)


def info(
    uri: Location, *, storage_options: Mapping[str, Any] | None = None
) -> ObjectInfo:
    """Size, modification time and entity tag of one object.

    Raises:
        FileNotFoundError: No object at ``uri``.

    Examples:
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "x.bin"
        >>> write_bytes(path, b"abc")
        >>> info(path).size
        3
    """
    target = _resolve(uri, storage_options)
    return target.describe(obstore.head(target.store, target.key))


def exists(uri: Location, *, storage_options: Mapping[str, Any] | None = None) -> bool:
    """Whether an object exists at ``uri`` (a prefix alone does not count).

    Examples:
        >>> exists("/no/such/file.bin")
        False
    """
    try:
        info(uri, storage_options=storage_options)
    except FileNotFoundError:
        return False
    return True


# --- read / write ------------------------------------------------------------


def read_bytes(
    uri: Location, *, storage_options: Mapping[str, Any] | None = None
) -> bytes:
    """The whole object at ``uri``, in memory.

    Raises:
        FileNotFoundError: No object at ``uri``.

    Examples:
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "x.bin"
        >>> write_bytes(path, b"abc")
        >>> read_bytes(path)
        b'abc'
    """
    target = _resolve(uri, storage_options)
    return bytes(obstore.get(target.store, target.key).bytes())


def write_bytes(
    uri: Location,
    data: bytes | bytearray | memoryview,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> None:
    """Write ``data`` as the object at ``uri``, replacing any existing one.

    Local parent directories are created. The store writes the object
    atomically: readers see the old object or the new one, never a part.
    """
    target = _resolve(uri, storage_options)
    obstore.put(target.store, target.key, bytes(data))


def open(  # `geocloud.files.open`, like `fsspec.open`
    uri: Location,
    mode: Literal["rb", "wb"] = "rb",
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> ReadableFile | WritableFile:
    """A file-like handle on the object at ``uri``.

    ``"rb"`` returns a seekable reader that fetches byte ranges on demand
    (hand it to h5py, ``zipfile``, …; ``close()`` it when done); ``"wb"``
    a buffered writer, used as a context manager, whose object appears
    when it is closed.

    Args:
        uri: The object.
        mode: ``"rb"`` or ``"wb"``.
        storage_options: Forwarded to `get_obstore` for a remote URI.

    Raises:
        ValueError: Any other ``mode``.
        FileNotFoundError: ``"rb"`` and no object at ``uri``.

    Examples:
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "x.bin"
        >>> with open(path, "wb") as fh:
        ...     _ = fh.write(b"abcdef")
        >>> fh = open(path)
        >>> _ = fh.seek(2)
        >>> bytes(fh.read(2))
        b'cd'
        >>> fh.close()
    """
    target = _resolve(uri, storage_options)
    if mode == "rb":
        return obstore.open_reader(target.store, target.key)
    if mode == "wb":
        return obstore.open_writer(target.store, target.key)
    raise ValueError(f"open: mode must be 'rb' or 'wb'; got {mode!r}.")


# --- move --------------------------------------------------------------------


def _stream_to_file(source: _Target, dest: Path) -> None:
    """Stream ``source`` into ``dest`` through a size-checked ``.part`` file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.part")
    try:
        result = obstore.get(source.store, source.key)
        expected = int(result.meta["size"])
        written = 0
        with part.open("wb") as fh:
            for chunk in result.stream(min_chunk_size=_CHUNK):
                written += fh.write(chunk)
        if written != expected:
            raise OSError(
                f"short read of {source.child(source.key)!r}: "
                f"{written} of {expected} bytes"
            )
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


async def _stream_between(source: _Target, dest: _Target) -> None:
    result = await obstore.get_async(source.store, source.key)
    await obstore.put_async(dest.store, dest.key, result.stream(min_chunk_size=_CHUNK))


def copy(
    src: Location,
    dst: Location,
    *,
    overwrite: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> Location:
    """Copy one object, between any two locations.

    Server-side within one bucket / container; a multipart upload from a
    local file; a streamed, atomic download to a local file; streamed
    chunk by chunk between two stores (see the module docstring).

    Args:
        src: The object to copy (URI or local path).
        dst: Where to put it. A trailing ``/`` (or an existing local
            directory) keeps the source's file name.
        overwrite: Replace an existing destination (default). ``False``
            leaves an existing destination alone and returns it.
        storage_options: Forwarded to `get_obstore` for each remote end.

    Returns:
        The destination location (with the file name filled in).

    Raises:
        ValueError: ``src`` or ``dst`` names a bucket root or a prefix
            (``.../``) rather than an object.
        FileNotFoundError: No object at ``src``.

    Examples:
        >>> import tempfile, pathlib
        >>> root = pathlib.Path(tempfile.mkdtemp())
        >>> write_bytes(root / "a.bin", b"abc")
        >>> copy(root / "a.bin", f"{root}/out/") == root / "out" / "a.bin"
        True
    """
    source = _resolve(src, storage_options)
    _require_object(source, src, "source")
    dst = _into(dst, source.name)
    dest = _resolve(dst, storage_options)
    _require_object(dest, dst, "destination")
    if not overwrite and exists(dst, storage_options=storage_options):
        return dst
    if dest.local:
        _stream_to_file(source, Path(dest.root) / dest.key)
    elif source.store is dest.store and source.root == dest.root:
        obstore.copy(dest.store, source.key, dest.key)
    elif source.local:
        obstore.put(dest.store, dest.key, Path(source.root) / source.key)
    else:
        _run_coroutine_safely(_stream_between(source, dest))
    return dst


def download(
    uri: Location,
    dest: Location,
    *,
    overwrite: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> Path:
    """Download the object at ``uri`` to the local file ``dest``.

    Streamed in chunks to a hidden ``.part`` sibling, checked against the
    object's size, then renamed into place; parent directories are made.

    Args:
        uri: The object.
        dest: The local file, or a directory (existing, or spelled with a
            trailing ``/``) to download into under the object's name.
        overwrite: Replace an existing file (default); ``False`` keeps it
            and skips the download.
        storage_options: Forwarded to `get_obstore`.

    Returns:
        The local path written (or kept).

    Raises:
        ValueError: ``dest`` is a remote URI (use `copy`).
        FileNotFoundError: No object at ``uri``.
        OSError: The transfer ended short of the object's size.
    """
    if _local_path(dest) is None:
        raise ValueError(f"download: dest must be a local path; got {dest!r}.")
    return Path(copy(uri, dest, overwrite=overwrite, storage_options=storage_options))


def upload(
    path: Location,
    uri: Location,
    *,
    overwrite: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> Location:
    """Upload the local file ``path`` to ``uri`` (multipart when large).

    Args:
        path: The local file.
        uri: Destination; a trailing ``/`` keeps the file's name.
        overwrite: Replace an existing object (default); ``False`` keeps
            it and skips the upload.
        storage_options: Forwarded to `get_obstore`.

    Returns:
        The destination URI.

    Raises:
        ValueError: ``path`` is not a local path (use `copy`).
        FileNotFoundError: ``path`` does not exist.
    """
    if _local_path(path) is None:
        raise ValueError(f"upload: path must be a local file; got {path!r}.")
    return copy(path, uri, overwrite=overwrite, storage_options=storage_options)


def sync(
    src: Location,
    dst: Location,
    *,
    overwrite: bool = False,
    max_concurrency: int = 8,
    storage_options: Mapping[str, Any] | None = None,
) -> list[Location]:
    """Copy every object under ``src`` to the same relative path under ``dst``.

    Mirrors a prefix (or local directory) into another, in any direction.
    Objects already at the destination with the same size are skipped
    unless ``overwrite`` is set, so re-running a sync resumes it. Nothing
    is deleted at the destination.

    Args:
        src: Source prefix or local directory.
        dst: Destination prefix or local directory.
        overwrite: Copy every object even when the destination has one of
            the same size.
        max_concurrency: Objects copied at once.
        storage_options: Forwarded to `get_obstore` for each remote end.

    Returns:
        The destination of every object copied (skipped ones excluded).

    Raises:
        ValueError: ``max_concurrency < 1``, or an end cannot be listed.

    Examples:
        >>> import tempfile, pathlib
        >>> a, b = pathlib.Path(tempfile.mkdtemp()), pathlib.Path(tempfile.mkdtemp())
        >>> write_bytes(a / "x" / "1.bin", b"1")
        >>> len(sync(a, b)), len(sync(a, b))  # the second run skips it
        (1, 0)
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1; got {max_concurrency}.")
    source = _resolve(src, storage_options)
    _require_listable(source, src)
    present: dict[str, tuple[str, int]] = {}
    if not overwrite:
        dest = _resolve(dst, storage_options)
        _require_listable(dest, dst)
        present = _walk(dest)
    todo = [
        (source.child(key), _join(dst, rel))
        for rel, (key, size) in _walk(source).items()
        if overwrite or present.get(rel, ("", -1))[1] != size
    ]
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(
            pool.map(lambda pair: copy(*pair, storage_options=storage_options), todo)
        )


def _walk(target: _Target) -> dict[str, tuple[str, int]]:
    """``{path relative to target: (store key, size)}`` of everything below it."""
    base = _prefix(target)
    cut = len(base) + 1 if base else 0
    return {
        meta["path"][cut:]: (meta["path"], int(meta["size"]))
        for page in obstore.list(target.store, prefix=base)
        for meta in page
    }


def rm(
    uri: Location,
    *,
    recursive: bool = False,
    storage_options: Mapping[str, Any] | None = None,
) -> int:
    """Delete the object at ``uri``, or every object under it.

    Args:
        uri: An object, or with ``recursive`` a prefix / local directory.
        recursive: Delete everything under ``uri`` (batched requests).
            Empty local directories are left in place.
        storage_options: Forwarded to `get_obstore`.

    Returns:
        The number of objects deleted.

    Raises:
        FileNotFoundError: Not ``recursive`` and no object at ``uri``.

    Examples:
        >>> import tempfile, pathlib
        >>> root = pathlib.Path(tempfile.mkdtemp())
        >>> write_bytes(root / "a.bin", b"1"); write_bytes(root / "b.bin", b"2")
        >>> rm(root, recursive=True)
        2
    """
    target = _resolve(uri, storage_options)
    if not recursive:
        obstore.head(target.store, target.key)  # FileNotFoundError if absent
        obstore.delete(target.store, target.key)
        return 1
    _require_listable(target, uri)
    keys = [
        meta["path"]
        for page in obstore.list(target.store, prefix=_prefix(target))
        for meta in page
    ]
    for start in range(0, len(keys), _DELETE_BATCH):
        obstore.delete(target.store, keys[start : start + _DELETE_BATCH])
    return len(keys)


def sign(
    uri: str,
    *,
    expires: timedelta = timedelta(hours=1),
    method: Literal["GET", "PUT"] = "GET",
    storage_options: Mapping[str, Any] | None = None,
) -> str:
    """A pre-signed HTTPS URL for ``uri``, valid for ``expires``.

    Anyone holding the URL can ``GET`` (or ``PUT``) the object without
    credentials until it expires: hand it to ``curl``, a browser, or GDAL
    (``/vsicurl/<url>``). The signature is computed locally from the
    store's credentials; no request is made.

    Args:
        uri: An ``s3://``, ``gs://`` or Azure object URI.
        expires: Lifetime of the URL.
        method: ``"GET"`` to read, ``"PUT"`` to let the holder upload.
        storage_options: Forwarded to `get_obstore`.

    Raises:
        ValueError: The store cannot sign (local paths, plain
            ``http(s)://``, ``hf://``), or has no credentials to sign with.
    """
    if _local_path(uri) is not None:
        raise ValueError(f"sign: {uri!r} is a local path; only cloud objects sign.")
    target = _resolve(uri, storage_options)
    try:
        return obstore.sign(target.store, method, target.key, expires)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"sign: cannot pre-sign {uri.split('?', 1)[0]!r} — only s3://, gs:// "
            f"and Azure stores with credentials sign URLs ({exc})."
        ) from exc
