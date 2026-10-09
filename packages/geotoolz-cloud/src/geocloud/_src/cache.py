"""A local cache of remote objects: one URI → one complete local file.

Some readers need a real file on disk — HDF4 (``pyhdf``), memory-mapped
binaries (Himawari HSD), GDAL drivers without ``/vsi*/`` support — and
catalog staging copies a catalog's assets next to the compute.
`LocalCache.fetch` (and the one-call `localize`) turns any location into
such a file: a local path is returned in place, a remote URI is
downloaded once through `geocloud.files.download` (16 MiB ranged reads
into a ``.part`` file, renamed into place when complete) and served from
the cache afterwards.

Cache slots are keyed by *location*: the SHA-256 of the URI with
expiring signature parameters removed (`cache_key`), so a re-signed URL
hits the same slot, plus the URI's file extension.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from geocloud._src.store import Location, local_path


__all__ = ["LocalCache", "cache_key", "localize"]

# The cache root when neither `root` nor $GEOCLOUD_CACHE is set, resolved
# at first use so a process can set the variable just before calling.
_DEFAULT_SUBDIR = ".cache/geocloud"

# In-progress downloads (`geocloud.files.download`) are named
# `.<slot>.<uuid4 hex>.part`, which no cache slot (a 64-hex digest plus
# the source extension) can match; `LocalCache.prune` removes leftovers.
_PART_NAME = re.compile(r"^\..+\.[0-9a-f]{32}\.part$")

# Query parameters that sign a URL rather than select content: a
# re-signed URL for the same object must hit the same cache slot.
_AZURE_SAS = {
    "sig", "se", "st", "sp", "sv", "sr", "spr", "si", "srt", "ss", "sdd",
    "skoid", "sktid", "skt", "ske", "sks", "skv", "saoid", "suoid", "scid",
}  # fmt: skip
_CLOUDFRONT = {"expires", "signature", "key-pair-id", "policy"}


def cache_key(uri: str) -> str:
    """``uri`` without the parameters of an expiring signature.

    Azure SAS (when ``sig`` is present), AWS / GCS query signing
    (``X-Amz-*`` / ``X-Goog-*``) and CloudFront signed-URL parameters are
    dropped; every other query parameter is kept, so URLs that select
    different content keep different keys. A local path is returned as is.

    Args:
        uri: A URI or local path.

    Returns:
        The URI that names the same content for as long as it exists.

    Examples:
        >>> cache_key("https://h/a.tif?X-Amz-Signature=abc&v=2")
        'https://h/a.tif?v=2'
    """
    if local_path(uri) is not None:
        return uri
    parts = urlsplit(uri)
    params = parse_qsl(parts.query, keep_blank_values=True)
    if not params:
        return uri
    names = {k.lower() for k, _ in params}

    def signing(name: str) -> bool:
        n = name.lower()
        return (
            n.startswith(("x-amz-", "x-goog-"))
            or ("sig" in names and n in _AZURE_SAS)
            or ({"signature", "key-pair-id"} <= names and n in _CLOUDFRONT)
        )

    kept = [(k, v) for k, v in params if not signing(k)]
    if len(kept) == len(params):
        return uri
    return urlunsplit(parts._replace(query=urlencode(kept)))


def _suffix(uri: str) -> str:
    """The file extension of ``uri`` (with the dot), ``""`` when none.

    A local path is not a URL: ``#`` and ``?`` in it are ordinary
    characters.
    """
    local = local_path(uri)
    if local is not None:
        return local.suffix
    return PurePosixPath(urlsplit(uri).path).suffix


@dataclasses.dataclass
class LocalCache:
    """A directory of complete local copies of remote objects.

    Files land at ``{root}/{key[:2]}/{key}{ext}``, where ``key`` is the
    SHA-256 of `cache_key` of the URI. The two-letter prefix keeps any one
    directory under a few thousand entries.

    Args:
        root: The cache directory. ``None`` resolves ``$GEOCLOUD_CACHE``
            when set, else ``~/.cache/geocloud`` (lazily, so a process
            can set the variable before its first fetch).
        ttl_days: Lifetime of a cached copy. Older copies are fetched
            again on their next use, and `prune` deletes them. ``None``
            caches forever.
        timeout: Per-request timeout, in seconds, for the downloads.
            Objects move in 16 MiB ranges, one request each, so this
            bounds one range, not a whole file. ``None`` keeps the
            client's default (30 s).

    Examples:
        >>> import tempfile, pathlib
        >>> cache = LocalCache(root=tempfile.mkdtemp())
        >>> cache.path_for("s3://bucket/a.tif").suffix
        '.tif'
    """

    root: os.PathLike[str] | str | None = None
    ttl_days: int | None = None
    timeout: float | None = 60.0

    def resolve_root(self) -> Path:
        """The cache directory, created on first use."""
        if self.root is not None:
            root = Path(self.root)
        else:
            env = os.environ.get("GEOCLOUD_CACHE")
            root = Path(env) if env else Path.home() / _DEFAULT_SUBDIR
        root.mkdir(parents=True, exist_ok=True)
        return root

    def path_for(self, uri: str) -> Path:
        """The slot ``uri`` is cached in (whether or not it is there yet)."""
        digest = hashlib.sha256(cache_key(uri).encode("utf-8")).hexdigest()
        return self.resolve_root() / digest[:2] / f"{digest}{_suffix(uri)}"

    def is_fresh(self, path: Path) -> bool:
        """Whether the cached ``path`` exists and is within ``ttl_days``."""
        if not path.exists():
            return False
        return self.ttl_days is None or not self._expired(path)

    def fetch(
        self,
        uri: Location,
        *,
        storage_options: Mapping[str, Any] | None = None,
    ) -> Path:
        """A complete local file holding ``uri``'s bytes.

        A local path (or ``file://`` URI) is returned in place: nothing is
        copied. A remote URI is served from the cache when fresh, else
        downloaded with `geocloud.files.download` into its slot.

        Args:
            uri: A URI in any form the pool accepts, or a local path.
            storage_options: Forwarded to `geocloud.store.get_obstore`,
                over the registered credentials and ``timeout``.

        Returns:
            The local path.

        Raises:
            FileNotFoundError: A local path that does not exist, or no
                object at a remote URI.
        """
        local = local_path(uri)
        if local is not None:
            if not local.exists():
                raise FileNotFoundError(f"localize: no local file at {str(uri)!r}")
            return local
        uri = str(uri)
        dest = self.path_for(uri)
        if self.is_fresh(dest):
            return dest
        from geocloud import files  # the public module, so tests can stand in

        options = dict(storage_options or {})
        if self.timeout is not None:
            client = dict(options.get("client_options") or {})
            client.setdefault("timeout", timedelta(seconds=self.timeout))
            options["client_options"] = client
        files.download(uri, dest, storage_options=options or None)
        return dest

    def prune(self) -> int:
        """Delete expired copies and abandoned partial downloads.

        Returns:
            The number of files removed. Without ``ttl_days`` only
            partial downloads (``*.part``) are removed.
        """
        removed = 0
        for path in self.resolve_root().glob("*/*"):
            if not path.is_file():
                continue
            if _PART_NAME.match(path.name) or (
                self.ttl_days is not None and self._expired(path)
            ):
                with contextlib.suppress(OSError):
                    path.unlink()
                    removed += 1
        return removed

    def _expired(self, path: Path) -> bool:
        assert self.ttl_days is not None
        age = datetime.now(tz=UTC) - datetime.fromtimestamp(
            path.stat().st_mtime, tz=UTC
        )
        return age >= timedelta(days=self.ttl_days)


def localize(
    uri: Location,
    *,
    cache: LocalCache | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> Path:
    """A local file holding ``uri``: in place when local, else a cached copy.

    For readers that need a real file (HDF4, memory-mapped binaries, GDAL
    drivers without ``/vsi*/``).

    Args:
        uri: A URI in any form the pool accepts, or a local path.
        cache: The cache to use; ``None`` uses the default `LocalCache`.
        storage_options: Forwarded to `geocloud.store.get_obstore`.

    Returns:
        The local path.

    Raises:
        FileNotFoundError: Nothing at ``uri``.

    Examples:
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "a.bin"
        >>> _ = path.write_bytes(b"x")
        >>> localize(path) == path
        True
    """
    return (cache or LocalCache()).fetch(uri, storage_options=storage_options)
