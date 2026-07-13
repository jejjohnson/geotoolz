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
* ``umm`` — a bounded subset (see ``_umm_essentials``) stored
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

from geocatalog._src.sources._base import AuthStatus, Bounds, Source, SourceRow
from geocatalog._src.sources._extras import _missing_extra
from geocatalog._src.sources._umm import (
    asset_key_from_url as _asset_key_from_url,
    extract_cloud_cover as _extract_cloud_cover,
    granule_geometry as _granule_geometry,
    granule_interval as _granule_interval,
    umm_essentials as _umm_essentials,
)


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
    """

    name = "earthaccess"

    def __init__(
        self,
        *,
        daac: str | None = None,
        cloud_hosted: bool | None = None,
    ) -> None:
        if earthaccess is None:
            raise _missing_extra(
                "EarthAccessSource", "earthaccess", "earthaccess>=0.10"
            )
        self.daac = daac
        self.cloud_hosted = cloud_hosted

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
            limit: Cap on the number of granules. ``None`` → all.

        Yields:
            `SourceRow` per matching granule.
        """
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
        granules = earthaccess.search_data(count=count, **kwargs)
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

    Returns ``None`` when the granule lacks a usable geometry —
    the caller (a streaming iterator) silently skips it rather
    than raising, because CMR occasionally returns granules with
    only collection-level footprints we can't faithfully bind to
    an item-level row.
    """
    umm = granule.get("umm") if hasattr(granule, "get") else {}
    if not isinstance(umm, Mapping):
        umm = {}

    granule_ur = umm.get("GranuleUR")
    if not granule_ur:
        # Fall back to the concept-id if no UR is set. CMR
        # guarantees one of them; we prefer the human-readable UR.
        meta = granule.get("meta") if hasattr(granule, "get") else {}
        granule_ur = meta.get("concept-id", "<no-id>")

    geometry = _granule_geometry(umm)
    if geometry is None or geometry.is_empty:
        logger.debug("earthaccess: skipping granule {!r} (no geometry)", granule_ur)
        return None

    interval = _granule_interval(umm)
    if interval is None:
        logger.debug(
            "earthaccess: skipping granule {!r} (no temporal extent)", granule_ur
        )
        return None

    # Asset map: earthaccess's `data_links()` already filters down
    # to GETDATA-type URLs. We use the trailing path segment as the
    # key (e.g. ".tif" / ".nc") so the dict-like layout matches
    # STAC's asset map shape; falling back to the URL itself when
    # nothing better is available.
    assets: dict[str, str] = {}
    try:
        links = list(granule.data_links())
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        # Malformed granule metadata shouldn't kill the whole query, but
        # an asset-less row is surprising downstream — say why it happened.
        logger.warning(
            "earthaccess: could not extract data links for granule {!r} "
            "({}); row will carry no assets",
            granule_ur,
            exc,
        )
        links = []
    for link in links:
        key = _asset_key_from_url(link)
        # Avoid clobbering when two links happen to share a key
        # (rare — but a numeric suffix keeps both).
        if key in assets:
            assets[f"{key}__{len(assets)}"] = link
        else:
            assets[key] = link

    collection_short_name = ""
    coll_ref = umm.get("CollectionReference", {})
    if isinstance(coll_ref, Mapping):
        collection_short_name = str(coll_ref.get("ShortName", "")) or ""

    properties: dict[str, Any] = {"umm": _umm_essentials(umm)}
    # Lift commonly-used fields to the top level for cheap access.
    cloud_cover = _extract_cloud_cover(umm)
    if cloud_cover is not None:
        properties["eo:cloud_cover"] = cloud_cover

    return SourceRow(
        id=str(granule_ur),
        source=source_name,
        collection=collection_short_name,
        geometry=geometry,
        interval=interval,
        assets=assets,
        properties=properties,
        provenance={
            "query_id": query_id,
            "fetched_at": fetched_at.isoformat(),
            "source_version": source_version,
        },
    )
