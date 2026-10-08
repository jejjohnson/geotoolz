"""Lightweight CMR REST adapter — no `earthaccess` dependency.

Direct calls to NASA's Common Metadata Repository search API. Useful
when:

- You don't want the full `earthaccess` dependency (no token broker,
  no DAAC presets) but still need to enumerate granules.
- You need fine-grained control over the CMR query parameters that
  `earthaccess` doesn't surface (provider, version, etc.).

Most users should prefer `EarthAccessSource`. This adapter trades
features (DAAC auto-discovery, S3 credentials) for footprint
(stdlib `urllib` + JSON).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger

from geocatalog._src.sources._base import (
    AuthStatus,
    Bounds,
    Source,
    SourceRow,
    wants_no_rows,
)
from geocatalog._src.sources._umm import granule_to_source_row
from geocatalog._src.utils.retry import retry_transient_io
from geocatalog._src.utils.timeutil import to_utc_ts


# CMR public search root. Granule and collection endpoints branch
# off this path. The adapter uses `urllib` so no extras are needed.
_CMR_ROOT = "https://cmr.earthdata.nasa.gov/search"

# CMR caps a single request at 2000 results; for `limit=None` we
# paginate via `search-after` until exhausted.
_CMR_PAGE_SIZE = 2000

# Query parameters the adapter sets itself; `filters` may not override them.
_RESERVED_PARAMS = frozenset({"bounding_box", "short_name", "temporal", "page_size"})


def _cmr_time(value: Any, *, upper: bool = False) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` in UTC — the form CMR's ``temporal`` takes.

    Sub-second precision is rounded *outward* (the lower bound down, the
    upper bound up): CMR's range is inclusive, so flooring the upper
    bound would drop granules inside the requested window.
    """
    ts = to_utc_ts(value)
    ts = ts.ceil("s") if upper else ts.floor("s")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


class CMRSource(Source):
    """Direct CMR REST adapter.

    Construct without arguments. Anonymous queries cover most public
    collections; restricted ones need an EDL bearer token via the
    ``token`` argument.

    Args:
        token: Optional EDL bearer token for protected collections.
        endpoint: CMR root URL — override for non-prod environments.
        retries: Retries per page on transient failures (network
            errors, HTTP 408 / 429 / 5xx). ``0`` disables retry.
    """

    name = "cmr"

    def __init__(
        self,
        *,
        token: str | None = None,
        endpoint: str = _CMR_ROOT,
        retries: int = 3,
    ) -> None:
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.retries = retries

    def query(
        self,
        bounds: Bounds,
        interval: pd.Interval | None = None,
        *,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[SourceRow]:
        """Yield `SourceRow`s for CMR granules.

        Args:
            bounds: ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.
            interval: Optional time window → ``temporal`` parameter.
            collection: CMR ``short_name``.
            filters: Forwarded directly as URL parameters. Useful for
                ``version``, ``provider``, ``platform``,
                ``cloud_cover[min]`` / ``cloud_cover[max]``, etc.
            limit: Cap on rows. ``None`` paginates all results; ``0``
                yields nothing without a request; negative raises.

        Yields:
            `SourceRow` per matching granule. Streamed via pagination
            so a large collection doesn't materialise in one chunk.
        """
        # `limit=0` is a dry run: nothing, and no request. Without this
        # guard the page-size math below would clamp to `max(..., 1)`.
        if wants_no_rows(limit):
            return

        query_id = uuid.uuid4().hex
        fetched_at = datetime.now(tz=UTC)

        clashes = sorted(set(filters or ()) & _RESERVED_PARAMS)
        if clashes:
            raise ValueError(
                f"CMRSource.query: filters {clashes} would override the "
                "parameters the adapter builds from bounds / interval / "
                "collection; pass those arguments instead."
            )
        params: dict[str, Any] = {
            "bounding_box": ",".join(str(x) for x in bounds),
        }
        if collection is not None:
            params["short_name"] = collection
        if interval is not None:
            # CMR wants RFC 3339 instants; aware inputs in other zones are
            # converted, naive ones are UTC, sub-second noise is dropped.
            params["temporal"] = (
                f"{_cmr_time(interval.left)},{_cmr_time(interval.right, upper=True)}"
            )
        for k, v in (filters or {}).items():
            # A list is sent as a repeated key (`provider=A&provider=B`, the
            # form CMR documents for provider / version); spell the key
            # `platform[]` yourself where CMR wants bracket syntax.
            params[k] = list(v) if isinstance(v, list | tuple | set) else v

        # Stream pages via the `search-after` header until done or
        # the user's limit is reached.
        emitted = 0
        search_after: str | None = None
        while True:
            page_size = (
                _CMR_PAGE_SIZE
                if limit is None
                else min(_CMR_PAGE_SIZE, max(limit - emitted, 1))
            )
            params["page_size"] = page_size
            url = f"{self.endpoint}/granules.umm_json?" + urllib.parse.urlencode(
                params, doseq=True
            )
            logger.debug("CMR GET: {!r}", url)
            data, next_search_after = retry_transient_io(
                _fetch_page,
                url,
                token=self.token,
                search_after=search_after,
                retries=self.retries,
            )
            items = data.get("items", [])
            for item in items:
                row = _cmr_item_to_source_row(
                    item,
                    source_name=self.name,
                    query_id=query_id,
                    fetched_at=fetched_at,
                )
                if row is not None:
                    yield row
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return
            if not next_search_after or not items:
                break
            search_after = next_search_after

    def auth_status(self) -> AuthStatus:
        """Probe the CMR root.

        Anonymous queries against public collections always work,
        so "authenticated" here means "we can reach the endpoint".
        Token presence is reported via `detail`.
        """
        try:
            req = urllib.request.Request(
                f"{self.endpoint}/granules.umm_json?page_size=1"
            )
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status = resp.status
        except Exception as exc:
            return AuthStatus(
                source=self.name,
                authenticated=False,
                detail=f"could not reach {self.endpoint}: {exc}",
            )
        ok = status == 200
        if ok:
            detail = f"reachable at {self.endpoint}" + (
                " (token set)" if self.token else " (anonymous)"
            )
        else:
            detail = f"{self.endpoint} returned status {status}"
        return AuthStatus(
            source=self.name,
            authenticated=ok,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# HTTP + UMM mapping helpers
# ---------------------------------------------------------------------------


def _fetch_page(
    url: str, *, token: str | None, search_after: str | None
) -> tuple[dict[str, Any], str | None]:
    """GET one CMR UMM-JSON page; return (body, next-search-after)."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if search_after:
        req.add_header("CMR-Search-After", search_after)
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        # CMR uses both Mixed-Case and lower-case header names
        # depending on the proxy in front; httplib normalises but
        # we accept either.
        next_after = resp.headers.get("CMR-Search-After") or resp.headers.get(
            "cmr-search-after"
        )
    return body, next_after


def _cmr_item_to_source_row(
    item: Mapping[str, Any],
    *,
    source_name: str,
    query_id: str,
    fetched_at: datetime,
) -> SourceRow | None:
    """Map a CMR UMM-JSON ``items[...]`` entry to a `SourceRow`.

    Thin wrapper over the mapper shared with the earthaccess adapter
    (`geocatalog._src.sources._umm.granule_to_source_row`).
    """
    return granule_to_source_row(
        item,
        source_name=source_name,
        query_id=query_id,
        fetched_at=fetched_at,
        source_version="cmr/umm_json",
    )
