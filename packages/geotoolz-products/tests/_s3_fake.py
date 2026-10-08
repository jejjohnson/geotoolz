"""A fake anonymous S3 endpoint for the bucket-helper tests (no network).

``FakeS3`` answers ListObjectsV2 pages and object GETs through a patched
``urllib.request.urlopen``; ``listing`` builds a ListObjectsV2 page.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.parse


def listing(keys: list[str], *, token: str | None = None) -> bytes:
    contents = "".join(
        f"<Contents><Key>{key}</Key><Size>{100 + i}</Size></Contents>"
        for i, key in enumerate(keys)
    )
    tail = (
        f"<IsTruncated>true</IsTruncated>"
        f"<NextContinuationToken>{token}</NextContinuationToken>"
        if token
        else "<IsTruncated>false</IsTruncated>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}{tail}</ListBucketResult>"
    ).encode()


class FakeS3:
    """Serves ListObjectsV2 pages and object bodies; records every URL."""

    def __init__(self, pages: dict[str, list[bytes]], bodies: dict[str, bytes]):
        self.pages = pages
        self.bodies = bodies
        self.urls: list[str] = []

    def urlopen(self, url: str, timeout: float | None = None) -> io.BytesIO:
        self.urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        if parsed.path in {"", "/"}:
            query = urllib.parse.parse_qs(parsed.query)
            pages = self.pages.get(query["prefix"][0], [listing([])])
            index = int(query.get("continuation-token", ["0"])[0])
            return io.BytesIO(pages[index])
        return io.BytesIO(self.bodies[urllib.parse.unquote(parsed.path[1:])])


def http_error(url: str, code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(body))  # type: ignore[arg-type]
