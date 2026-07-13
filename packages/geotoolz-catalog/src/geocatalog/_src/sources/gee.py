"""Google Earth Engine adapter — asset enumeration in v1.

Phase 1 scope (per ``docs/design/query-matchup.md`` §8): enumerate
EE ``ImageCollection`` assets that intersect the query bbox + interval,
returning footprints and asset paths. Materializing pixels via
``ee.Image.getDownloadURL`` is deferred to the staging layer (§4.7);
running arbitrary ``ee.Image`` recipes is out of scope until a
later phase.

Scaffolding only.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from geocatalog._src.sources._base import AuthStatus, Bounds, Source, SourceRow
from geocatalog._src.sources._extras import _missing_extra


if TYPE_CHECKING:
    import pandas as pd


try:
    import ee
except ImportError:
    ee = None  # type: ignore[assignment]


class GEESource(Source):
    """Google Earth Engine asset discovery.

    Construct without arguments; authentication is handled by the
    underlying `ee` client (`ee.Authenticate()` + `ee.Initialize()`
    or a service-account credentials file). Call ``auth_status`` to
    check.

    Args:
        project: GCP project the EE API requests are billed to. Some
            collections refuse anonymous reads; pin a project here.
    """

    name = "gee"

    def __init__(self, *, project: str | None = None) -> None:
        if ee is None:
            raise _missing_extra("GEESource", "gee", "earthengine-api>=0.1.380")
        self.project = project

    def query(
        self,
        bounds: Bounds,
        interval: pd.Interval | None = None,
        *,
        collection: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[SourceRow]:
        """Enumerate EE assets intersecting ``bounds`` + ``interval``.

        Scaffolding — not yet implemented (Phase 3, design §8).

        Raises:
            NotImplementedError: Always, until the staging layer can
                materialize ``ee.Image`` assets.
        """
        raise NotImplementedError(
            "GEESource.query is scaffolding — Phase 3 PR (see design §8). "
            "Phase 1 ships earthaccess + STAC + CMR; GEE follows once "
            "the staging layer can materialize ee.Image assets."
        )

    def auth_status(self) -> AuthStatus:
        """Report whether the `ee` client can reach Earth Engine.

        Scaffolding — not yet implemented (Phase 3, design §8).

        Raises:
            NotImplementedError: Always, until the Phase 3 PR lands.
        """
        raise NotImplementedError("GEESource.auth_status is scaffolding — Phase 3 PR.")
