"""`Source` Protocol and the normalized `SourceRow` it produces.

A `Source` is anything that can answer "what's in this bbox + interval +
collection?" against a remote catalog (NASA earthaccess, a STAC
endpoint, Google Earth Engine, CMR). It is distinct from a
`GeoCatalog`, which represents "what I already know about" — a
`Source` is *external*, queried each time; a `GeoCatalog` is *local*,
persisted in GeoParquet.

The two come together via ``CatalogBundle.ingest(source, bounds=...)``
(see `geocatalog._src.bundle`), which materializes a remote query into
local catalog rows and records a ``QueryRecord`` for provenance. See
``docs/design/query-matchup.md`` §4 for the full picture.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    import pandas as pd
    import shapely.geometry.base


# A bounding box in CRS units: ``(xmin, ymin, xmax, ymax)``. By
# convention adapters accept lon/lat in EPSG:4326 for their public
# entry point and reproject internally if needed; that lines up with
# every upstream API (CMR, STAC, GEE) which all accept WGS84 bboxes.
Bounds = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class AuthStatus:
    """Reported by `Source.auth_status` so callers can fail fast.

    Adapters never raise on construction; they raise on
    ``query`` / ``ingest`` if credentials are missing. ``auth_status``
    is the introspection hook for tooling that wants to surface "you
    need to log in" before kicking off a long workflow.
    """

    source: str
    authenticated: bool
    detail: str | None = None


@dataclasses.dataclass(frozen=True)
class SourceRow:
    """Normalized output of every ``Source.query`` adapter.

    Decoupled from the existing in-catalog ``CatalogRow`` because the
    set of fields is broader (STAC-style asset map, per-row provenance,
    sensor-specific properties). Ingestion maps `SourceRow` to a
    ``CatalogRow`` by promoting the primary asset to ``filepath`` and
    folding the rest into ``extras``. See design §4.3.

    Attributes:
        id: Stable identifier — granule UR, STAC item id, EE asset
            path. Round-trips into ``items.parquet`` as the primary key.
        source: Stable adapter name (``"earthaccess"``, ``"stac.pc"``,
            ``"gee"``, ``"cmr"``). Lets matchups filter by origin.
        collection: Upstream collection identifier — e.g.
            ``MOD09GA``, ``sentinel-2-l2a``,
            ``COPERNICUS/S2_SR_HARMONIZED``.
        geometry: Footprint polygon (or multipolygon) in EPSG:4326.
            The catalog will reproject into its target CRS at
            ingestion time.
        interval: Observation time window. ``closed='both'``; for
            instantaneous granules ``time_start == time_end``.
        assets: Mapping of asset key → URI. For STAC this is the
            native asset map; for earthaccess it's the granule's
            data-URL set; for GEE it's a single ``{"asset": <path>}``.
        properties: Sensor-specific metadata — cloud cover, sun zenith,
            orbit number, processing level, etc. Carried opaquely.
        provenance: ``{"query_id": ..., "fetched_at": ...,
            "source_version": ...}``. Stamped by `Source.query`
            so re-runs are traceable.
    """

    id: str
    source: str
    collection: str
    geometry: shapely.geometry.base.BaseGeometry
    interval: pd.Interval
    assets: Mapping[str, str] = dataclasses.field(default_factory=dict)
    properties: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """A remote catalog that can be queried by bounds + interval.

    Implementations live in `geocatalog._src.sources.<name>` and are
    extras-gated: ``EarthAccessSource`` requires
    ``pip install 'geocatalog[earthaccess]'``, etc. Importing this
    Protocol never pulls in any optional dependency.

    Two methods. ``query`` returns an iterator (so adapters can
    stream paginated results); ``auth_status`` is the introspection
    hook callers use to fail fast.
    """

    #: Stable adapter name. Used as the value of ``SourceRow.source``
    #: and as a filter key in matchups (``primary={"source": "earthaccess"}``).
    name: str

    def query(
        self,
        bounds: Bounds,
        interval: pd.Interval | None = None,
        *,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[SourceRow]:
        """Yield matching rows from the upstream catalog.

        Args:
            bounds: ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.
            interval: Optional time window. ``None`` means no temporal
                filter (the adapter may impose its own default if the
                upstream API requires one).
            collection: Upstream collection / short-name. Adapters
                that target a single collection may ignore this; most
                require it.
            filters: Free-form key/value pairs forwarded to the
                upstream API (e.g. STAC's CQL-2 filters, earthaccess's
                ``cloud_cover``). Adapter-specific; unknown keys are
                ignored or rejected per the adapter's policy.
            limit: Cap on the number of rows returned. ``None`` means
                no cap (paginate all results).

        Yields:
            `SourceRow` instances; never opens or downloads data.
        """
        ...

    def auth_status(self) -> AuthStatus:
        """Report whether the adapter can talk to its upstream.

        Never raises. Tools can call this before ``query`` to surface
        login problems early; ``query`` itself raises on auth failures.
        """
        ...
