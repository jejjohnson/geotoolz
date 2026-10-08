"""One parser for paths and URIs.

Every module that needs to know whether a location is local or remote,
what scheme it uses, what its file name or extension is, or how GDAL /
DuckDB / fsspec should reach it, goes through `parse_uri`. Before this
module the question was answered six ways that disagreed on Windows
drive letters, ``s3:foo``, ``file://`` and trailing-slash Zarr stores.

Rules:

* A `pathlib` path, a string without ``://``, and a Windows drive path
  (``C:/x``, ``C:\\x``) are local. ``s3:catalog.parquet`` is therefore a
  local file name, as POSIX reads it — not an S3 URI.
* ``file://`` URIs are local; ``file://server/share/x`` is the UNC path
  ``//server/share/x`` and ``file:///C:/x`` a drive path on Windows.
* Anything else with ``<scheme>://`` is remote. Schemes are compared
  lower-case.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import url2pathname


# Schemes read through fsspec (cloud object stores, HTTP, Hugging Face).
FSSPEC_SCHEMES: frozenset[str] = frozenset(
    {"s3", "s3a", "gs", "gcs", "az", "azure", "abfs", "abfss", "http", "https", "hf"}
)

# Cloud schemes GDAL reads natively through a ``/vsi*/`` virtual file
# system, with ranged requests (a COG header read fetches kilobytes).
_GDAL_VSI_PREFIXES: dict[str, str] = {
    "s3": "/vsis3/",
    "s3a": "/vsis3/",
    "gs": "/vsigs/",
    "gcs": "/vsigs/",
    "az": "/vsiaz/",
    "azure": "/vsiaz/",
    "abfs": "/vsiaz/",
    "abfss": "/vsiaz/",
}

# DuckDB extension each remote scheme needs.
DUCKDB_EXTENSIONS: dict[str, str] = {
    **dict.fromkeys(
        ("s3", "s3a", "s3n", "gs", "gcs", "r2", "hf", "http", "https"), "httpfs"
    ),
    **dict.fromkeys(("az", "azure", "abfss"), "azure"),
}

_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*$")


@dataclass(frozen=True)
class ParsedURI:
    """A location split into the parts the catalog cares about.

    Attributes:
        raw: The input as a string.
        scheme: Lower-case scheme; ``""`` for a plain local path,
            ``"file"`` for a ``file://`` URI.
        netloc: Authority of a URI (bucket, host, container); ``""`` for
            local paths.
        path: The path component — the URL path of a URI, the input
            itself for a plain local path.
        query: The raw query string of a URI (``""`` for local paths).
    """

    raw: str
    scheme: str
    netloc: str
    path: str
    query: str = ""

    @property
    def is_local(self) -> bool:
        """True for plain paths and ``file://`` URIs."""
        return self.scheme in ("", "file")

    @property
    def is_remote(self) -> bool:
        """True for every ``<scheme>://`` URI except ``file://``."""
        return not self.is_local

    @property
    def name(self) -> str:
        """Final path component (``""`` for a bare bucket or host)."""
        return self.path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str:
        """Extension of `name`, with the dot (``""`` if none)."""
        name = self.name
        return "." + name.rsplit(".", 1)[-1] if "." in name.lstrip(".") else ""

    @property
    def is_zarr(self) -> bool:
        """True when the location is a ``.zarr`` store (trailing slash allowed)."""
        return self.suffix == ".zarr"

    def local_path(self) -> Path | None:
        """The filesystem path of a local location; ``None`` for remote URIs."""
        if self.scheme == "":
            return Path(self.raw)
        if self.scheme == "file":
            path = self.path
            if self.netloc and self.netloc != "localhost":
                path = f"//{self.netloc}{path}"  # UNC share
            return Path(url2pathname(path))
        return None

    def gdal_path(self) -> str | None:
        """GDAL ``/vsi*/`` path for a remote URI GDAL can read, else ``None``.

        ``s3://bucket/key`` → ``/vsis3/bucket/key``; ``http(s)://…`` →
        ``/vsicurl/http(s)://…``; ``abfs[s]://container@account…/key`` →
        ``/vsiaz/container/key``. Credentials come from GDAL's usual
        configuration (``AWS_*``, ``GOOGLE_APPLICATION_CREDENTIALS``,
        ``AZURE_STORAGE_*``, …).
        """
        if self.scheme in ("http", "https"):
            return f"/vsicurl/{self.raw}"
        prefix = _GDAL_VSI_PREFIXES.get(self.scheme)
        if prefix is None:
            return None
        container = self.netloc
        if self.scheme in ("abfs", "abfss"):
            container = container.split("@", 1)[0]
        return f"{prefix}{container}{self.path}"


def parse_uri(uri: str | os.PathLike[str]) -> ParsedURI:
    """Split a path or URI into a `ParsedURI` (see the module rules)."""
    if isinstance(uri, PurePath):
        text = str(uri)
        return ParsedURI(raw=text, scheme="", netloc="", path=text)
    text = os.fspath(uri)
    head, sep, _ = text.partition("://")
    if not sep or _DRIVE.match(text) or not _SCHEME.match(head):
        return ParsedURI(raw=text, scheme="", netloc="", path=text)
    parts = urlsplit(text)
    return ParsedURI(
        raw=text,
        scheme=parts.scheme.lower(),
        netloc=parts.netloc,
        path=parts.path,
        query=parts.query,
    )


def query_params(uri: str) -> list[tuple[str, str]]:
    """The query parameters of a remote URI, in order (blank values kept)."""
    parsed = parse_uri(uri)
    if parsed.is_local or not parsed.query:
        return []
    return parse_qsl(parsed.query, keep_blank_values=True)


def with_query(uri: str, params: list[tuple[str, str]]) -> str:
    """``uri`` with its query string replaced by ``params``."""
    parts = urlsplit(uri)
    return urlunsplit(parts._replace(query=urlencode(params)))


__all__ = [
    "DUCKDB_EXTENSIONS",
    "FSSPEC_SCHEMES",
    "ParsedURI",
    "parse_uri",
    "query_params",
    "with_query",
]
