"""NASA `earthaccess` adapter — CMR-backed granule discovery.

Wraps the upstream `earthaccess` library so a single
``EarthAccessSource(...).query(bounds, interval)`` call returns
normalized `SourceRow` instances regardless of DAAC / collection.

The mapping is driven by CMR's UMM (Unified Metadata Model)
granule schema. Field paths consulted:

* ``umm.GranuleUR`` — stable identifier.
* ``umm.TemporalExtent.RangeDateTime.{Beginning,Ending}DateTime`` —
  observation interval. Single-`SingleDateTime` variant also handled.
* ``umm.SpatialExtent.HorizontalSpatialDomain.Geometry`` —
  footprint. Supports the three common shapes: ``GPolygons``,
  ``BoundingRectangles``, ``Points``.
* ``granule.data_links()`` — asset URLs.
* ``umm`` — a bounded subset (see ``_umm.umm_essentials``) stored
  under ``SourceRow.properties["umm"]``. Full UMM dicts can be
  several KB per granule; downstream code wanting the raw record
  should re-query via earthaccess.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from geocatalog._src._extras import missing_extra
from geocatalog._src.sources._base import (
    AuthStatus,
    Bounds,
    Source,
    SourceRow,
    wants_no_rows,
)
from geocatalog._src.sources._umm import granule_to_source_row
from geocatalog._src.utils.retry import retry_transient_io


if TYPE_CHECKING:
    pass


try:
    import earthaccess
except ImportError:
    earthaccess = None  # type: ignore[assignment]


class EarthAccessSource(Source):
    """NASA CMR / earthaccess data discovery.

    Construct without arguments; authentication is handled by the
    underlying `earthaccess` library (`earthaccess.login()` or a
    netrc / token in the standard locations). Call ``auth_status``
    to check whether credentials are usable.

    Args:
        daac: Optional DAAC short-name filter (e.g. ``"LPDAAC"``).
            Defaults to None (search all DAACs).
        cloud_hosted: When True, restrict to Earthdata-Cloud
            granules. Useful if you want direct S3 reads
            downstream and don't care about on-prem holdings.
        retries: Retries of the search call on transient failures
            (network errors, HTTP 408 / 429 / 5xx). ``0`` disables retry.
    """

    name = "earthaccess"

    def __init__(
        self,
        *,
        daac: str | None = None,
        cloud_hosted: bool | None = None,
        retries: int = 3,
    ) -> None:
        if earthaccess is None:
            raise missing_extra(
                "`EarthAccessSource`", "earthaccess", packages="earthaccess>=0.10"
            )
        self.daac = daac
        self.cloud_hosted = cloud_hosted
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
        """Yield `SourceRow`s for CMR granules matching the query.

        Args:
            bounds: ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.
            interval: Optional time window. earthaccess expects
                ``temporal=(date_from, date_to)``.
            collection: CMR ``short_name``. Required by CMR for
                non-trivial searches; if omitted the caller almost
                certainly gets an empty result.
            filters: Forwarded as additional kwargs to
                `earthaccess.search_data` (e.g. ``cloud_cover``,
                ``platform``, ``provider``, ``version``). Unknown
                keys silently passed through — `earthaccess` itself
                validates.
            limit: Cap on the number of granules. ``None`` → all;
                ``0`` yields nothing without a request; negative raises.

        Yields:
            `SourceRow` per matching granule.
        """
        # earthaccess reads `count=0` as "all"; `limit=0` means nothing
        # here (#237).
        if wants_no_rows(limit):
            return
        query_id = uuid.uuid4().hex
        fetched_at = datetime.now(tz=UTC)
        version = _earthaccess_version()

        kwargs: dict[str, Any] = {"bounding_box": tuple(bounds)}
        if collection is not None:
            kwargs["short_name"] = collection
        if interval is not None:
            kwargs["temporal"] = (
                pd.Timestamp(interval.left).isoformat(),
                pd.Timestamp(interval.right).isoformat(),
            )
        if self.daac is not None:
            kwargs["daac"] = self.daac
        if self.cloud_hosted is not None:
            kwargs["cloud_hosted"] = self.cloud_hosted
        if filters:
            # `earthaccess.search_data` accepts an open set of kwargs;
            # we forward everything the user passed.
            kwargs.update(filters)

        # `count=-1` means "all" in earthaccess; map our None likewise.
        count = limit if limit is not None else -1

        logger.debug("earthaccess.search_data: {!r} count={!r}", kwargs, count)
        # earthaccess returns the full result list (it has no streaming
        # search API); retry transient HTTP failures of that one call.
        granules = retry_transient_io(
            earthaccess.search_data, count=count, retries=self.retries, **kwargs
        )
        for granule in granules:
            row = _granule_to_source_row(
                granule,
                source_name=self.name,
                query_id=query_id,
                fetched_at=fetched_at,
                source_version=version,
            )
            if row is not None:
                yield row

    def auth_status(self) -> AuthStatus:
        """Check whether `earthaccess` is logged in.

        Builds a requests session via
        ``earthaccess.get_requests_https_session()`` and inspects
        the ``earthaccess.__auth__`` singleton's ``authenticated``
        flag — a cheap, non-network probe of the cached login
        state. The adapter does *not* call ``earthaccess.login()``
        automatically; surface the bare credential state and let
        the caller decide whether to prompt for credentials.
        """
        try:
            session = earthaccess.get_requests_https_session()
        except Exception as exc:
            return AuthStatus(
                source=self.name,
                authenticated=False,
                detail=f"earthaccess session unavailable: {exc}",
            )
        # The cheapest signal: was the session built from an actual
        # auth object? `earthaccess.__auth__` is the singleton.
        auth = getattr(earthaccess, "__auth__", None)
        authenticated = bool(getattr(auth, "authenticated", False))
        del session
        return AuthStatus(
            source=self.name,
            authenticated=authenticated,
            detail=(
                "earthaccess logged in"
                if authenticated
                else "earthaccess not authenticated — call earthaccess.login() "
                "or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD"
            ),
        )


# ---------------------------------------------------------------------------
# Helpers — granule → SourceRow mapping
# ---------------------------------------------------------------------------


def _earthaccess_version() -> str:
    if earthaccess is None:
        return "earthaccess/?"
    return f"earthaccess/{getattr(earthaccess, '__version__', '?')}"


def _granule_to_source_row(
    granule: Any,
    *,
    source_name: str,
    query_id: str,
    fetched_at: datetime,
    source_version: str,
) -> SourceRow | None:
    """Map an `earthaccess.results.DataGranule` to a `SourceRow`.

    A ``DataGranule`` is the CMR UMM-JSON item dict, so this delegates
    to the mapper `CMRSource` uses
    (`geocatalog._src.sources._umm.granule_to_source_row`): the same
    document yields the same row from either adapter. Assets come from
    the UMM ``RelatedUrls`` (HTTPS and S3 direct-access links under
    distinct keys) rather than ``data_links()``, which returns only one
    access type.
    """
    item = {"umm": granule.get("umm"), "meta": granule.get("meta")}
    return granule_to_source_row(
        item,
        source_name=source_name,
        query_id=query_id,
        fetched_at=fetched_at,
        source_version=source_version,
    )
