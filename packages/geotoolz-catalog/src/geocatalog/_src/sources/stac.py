"""STAC API adapter — generic, with named-provider factory helpers.

A single ``STACSource(endpoint=...)`` covers any STAC-compliant
catalog: Microsoft Planetary Computer, Earth Search, USGS Landsat
Look, NASA HLS, in-house deployments. Two class-method factories —
``STACSource.planetary_computer()`` and ``STACSource.earth_search()``
— are conveniences for the most common endpoints and arrange any
provider-specific auth (Planetary Computer's SAS-token signing).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import shapely.geometry
from loguru import logger

from geocatalog._src._timeutil import (
    to_rfc3339 as _to_iso,
    to_utc_ts as _to_utc_timestamp,
)
from geocatalog._src.sources._base import AuthStatus, Bounds, Source, SourceRow
from geocatalog._src.sources._extras import _missing_extra


if TYPE_CHECKING:
    import pystac
    import pystac_client as _pystac_client_t


try:
    import pystac_client
except ImportError:
    pystac_client = None  # type: ignore[assignment]

try:
    import planetary_computer
except ImportError:
    planetary_computer = None  # type: ignore[assignment]


# Public well-known STAC endpoints. Kept here so the factory methods
# stay readable; users can always pass a custom URL.
_PC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
_EARTH_SEARCH_ENDPOINT = "https://earth-search.aws.element84.com/v1"


class STACSource(Source):
    """STAC API data discovery.

    Args:
        endpoint: Root STAC API URL (the catalog landing page).
        sign_assets: If ``True``, sign asset URLs on access (needed
            for Planetary Computer's blob-storage tokens). Defaults
            to whatever the factory method sets.
        name: Stable adapter identifier. Defaults to ``"stac"``;
            factories override with ``"stac.pc"``, ``"stac.es"``.

    Examples:
        >>> src = STACSource.planetary_computer()
        >>> rows = list(src.query(
        ...     bounds=(-10, 35, 5, 45),
        ...     interval=pd.Interval(
        ...         pd.Timestamp("2024-06-01", tz="UTC"),
        ...         pd.Timestamp("2024-06-30", tz="UTC"),
        ...         closed="both",
        ...     ),
        ...     collection="sentinel-2-l2a",
        ...     filters={"eo:cloud_cover": {"lt": 20}},
        ...     limit=10,
        ... ))
        >>> rows[0].assets["B04"]  # signed red-band URL
        'https://sentinel2l2a01.blob.core.windows.net/...?...&sig=...'
    """

    def __init__(
        self,
        endpoint: str,
        *,
        sign_assets: bool = False,
        name: str = "stac",
    ) -> None:
        if pystac_client is None:
            raise _missing_extra(
                "STACSource", "stac", "pystac-client>=0.7 planetary-computer>=1.0"
            )
        if sign_assets and planetary_computer is None:
            raise _missing_extra(
                "STACSource(sign_assets=True)",
                "stac",
                "pystac-client>=0.7 planetary-computer>=1.0",
            )
        self.endpoint = endpoint
        self.sign_assets = sign_assets
        self.name = name
        # Cache the client across calls — pystac-client's Client.open
        # does a network round-trip to fetch the root catalog, and
        # there's no reason to redo it for repeated queries.
        self._client: _pystac_client_t.Client | None = None

    @classmethod
    def planetary_computer(cls) -> STACSource:
        """Microsoft Planetary Computer — signs blob URLs automatically."""
        return cls(_PC_ENDPOINT, sign_assets=True, name="stac.pc")

    @classmethod
    def earth_search(cls) -> STACSource:
        """Element 84 Earth Search — public AWS-hosted Sentinel/Landsat."""
        return cls(_EARTH_SEARCH_ENDPOINT, sign_assets=False, name="stac.es")

    @property
    def client(self) -> _pystac_client_t.Client:
        """Lazily-opened, cached `pystac_client.Client`."""
        if self._client is None:
            assert pystac_client is not None  # guarded in __init__
            self._client = pystac_client.Client.open(self.endpoint)
        return self._client

    def query(
        self,
        bounds: Bounds,
        interval: pd.Interval | None = None,
        *,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[SourceRow]:
        """Yield `SourceRow`s for STAC items matching the query.

        Args:
            bounds: ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.
            interval: Optional time window. STAC's ``datetime`` param
                takes ``"start/end"`` ISO 8601.
            collection: STAC collection id (e.g. ``"sentinel-2-l2a"``,
                ``"landsat-c2-l2"``). Many STAC servers will refuse
                searches without one; pystac-client itself does not
                require it.
            filters: Forwarded to pystac-client's ``query`` keyword
                (the ``eo:cloud_cover`` / property-filter syntax).
                For CQL-2 use a ``"filter"`` key inside this mapping
                — it falls through to pystac-client's ``filter``
                argument.
            limit: Cap on the number of items. ``None`` = paginate
                all results.

        Yields:
            `SourceRow` per matching STAC item; never opens or
            downloads asset data. Asset URLs are signed in place if
            ``self.sign_assets`` is True.
        """
        # Build query_id once per `query()` call so every row from a
        # single search shares provenance metadata (useful when the
        # rows are persisted side-by-side and a user wants to know
        # "which call produced these?").
        query_id = uuid.uuid4().hex
        fetched_at = datetime.now(tz=UTC)
        source_version = _pystac_client_version()

        search_kwargs: dict[str, Any] = {"bbox": list(bounds)}
        if collection is not None:
            search_kwargs["collections"] = [collection]
        if interval is not None:
            search_kwargs["datetime"] = _interval_to_stac_datetime(interval)
        if limit is not None:
            search_kwargs["max_items"] = limit
        # `filters` is a soft passthrough: a "filter" key goes to
        # pystac-client's `filter=` (CQL-2), everything else goes to
        # `query=` (legacy property-filter syntax). Both round-trip
        # cleanly through the persisted catalog's `queries.parquet`
        # row.
        if filters is not None:
            filters = dict(filters)
            cql2 = filters.pop("filter", None)
            if cql2 is not None:
                search_kwargs["filter"] = cql2
            if filters:
                search_kwargs["query"] = filters

        logger.debug("STAC search on {}: {!r}", self.endpoint, search_kwargs)
        search = self.client.search(**search_kwargs)
        for item in search.items():
            if self.sign_assets:
                # planetary_computer.sign returns a new Item with
                # signed asset hrefs; the rest of the item is
                # untouched.
                item = planetary_computer.sign(item)
            yield _item_to_source_row(
                item,
                source_name=self.name,
                query_id=query_id,
                fetched_at=fetched_at,
                source_version=source_version,
            )

    def auth_status(self) -> AuthStatus:
        """Open the client and probe the root catalog.

        Most public STAC catalogs (PC, Earth Search) don't actually
        gate the search endpoint — auth is on individual assets
        (PC blob SAS, USGS M2M token). So "authenticated" here
        really means "the endpoint is reachable and serves STAC".
        """
        try:
            _ = self.client.get_self_href()
        except Exception as exc:
            return AuthStatus(
                source=self.name,
                authenticated=False,
                detail=f"could not reach {self.endpoint}: {exc}",
            )
        return AuthStatus(
            source=self.name,
            authenticated=True,
            detail=f"reachable at {self.endpoint}",
        )


def _pystac_client_version() -> str:
    """Stable provenance string. Includes pystac-client + planetary-computer."""
    parts = []
    if pystac_client is not None:
        parts.append(f"pystac-client/{getattr(pystac_client, '__version__', '?')}")
    if planetary_computer is not None:
        parts.append(
            f"planetary-computer/{getattr(planetary_computer, '__version__', '?')}"
        )
    return ";".join(parts)


def _interval_to_stac_datetime(interval: pd.Interval) -> str:
    """`pd.Interval` → STAC `datetime` string (``"start/end"`` ISO 8601).

    STAC expects timezone-aware ISO 8601 with ``Z`` for UTC; the
    UTC coercion / ``Z`` serialization live in
    `geocatalog._src._timeutil` (imported here as ``_to_iso`` /
    ``_to_utc_timestamp``).
    """
    return f"{_to_iso(interval.left)}/{_to_iso(interval.right)}"


def _item_to_source_row(
    item: pystac.Item,
    *,
    source_name: str,
    query_id: str,
    fetched_at: datetime,
    source_version: str,
) -> SourceRow:
    """Map a `pystac.Item` to a `SourceRow`.

    Geometry is decoded with `shapely.geometry.shape` (handles
    Polygon, MultiPolygon, etc.). The temporal window prefers
    ``start_datetime`` / ``end_datetime`` if the item is a range,
    else a zero-width interval at ``item.datetime``.
    """
    if item.geometry is None:
        raise ValueError(
            f"STAC item {item.id!r} has no geometry; cannot build SourceRow."
        )
    geom = shapely.geometry.shape(item.geometry)

    # Interval: STAC items either have a single `datetime` or a
    # `start_datetime`/`end_datetime` pair (the "datetime range"
    # convention). Prefer the range when present. Every endpoint is
    # normalized to UTC because:
    #   - STAC items can carry any tz (or none) per the spec;
    #   - pandas refuses to compare tz-aware against tz-naive
    #     Timestamps, so a mixed-tz dataset would explode in
    #     downstream IntervalIndex / merge_asof operations;
    #   - the catalog's stored time-axis contract is UTC.
    props = dict(item.properties or {})
    start_dt = props.get("start_datetime")
    end_dt = props.get("end_datetime")
    if start_dt is not None and end_dt is not None:
        left = _to_utc_timestamp(start_dt)
        right = _to_utc_timestamp(end_dt)
    elif item.datetime is not None:
        left = right = _to_utc_timestamp(item.datetime)
    else:
        raise ValueError(
            f"STAC item {item.id!r} has neither `datetime` nor "
            f"`start_datetime`/`end_datetime`; cannot build interval."
        )
    interval = pd.Interval(left, right, closed="both")

    # Assets: STAC native form is `{key: Asset}`; we want `{key: href}`.
    assets = {key: asset.href for key, asset in (item.assets or {}).items()}

    return SourceRow(
        id=item.id,
        source=source_name,
        collection=item.collection_id or "",
        geometry=geom,
        interval=interval,
        assets=assets,
        properties=props,
        provenance={
            "query_id": query_id,
            "fetched_at": fetched_at.isoformat(),
            "source_version": source_version,
        },
    )
