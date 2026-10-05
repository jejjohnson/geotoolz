"""Google Earth Engine adapter — scaffolding, not implemented.

The planned scope (``docs/catalog/design/query-matchup.md`` §8):
enumerate EE ``ImageCollection`` assets that intersect the query
bbox + interval, returning footprints and asset paths. The staging
layer has no ``ee.Image`` download path yet, and running arbitrary
``ee.Image`` recipes is out of scope.

Scaffolding only: `GEESource.query` and `GEESource.auth_status`
raise `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from geocatalog._src._extras import missing_extra
from geocatalog._src.sources._base import (
    AuthStatus,
    Bounds,
    Source,
    SourceRow,
    wants_no_rows,
)


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
            raise missing_extra(
                "`GEESource`", "gee", packages="earthengine-api>=0.1.380"
            )
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

        Scaffolding — not yet implemented (design §8). The shared
        ``limit`` contract already holds: ``limit=0`` yields nothing
        and a negative limit raises `ValueError`.

        Raises:
            NotImplementedError: For any query that could return rows,
                until the staging layer can materialize ``ee.Image``
                assets.
        """
        if wants_no_rows(limit):
            return iter(())
        raise NotImplementedError(
            "GEESource.query is not implemented yet; use STACSource, "
            "EarthAccessSource or CMRSource. GEE follows once the "
            "staging layer can materialize ee.Image assets."
        )

    def auth_status(self) -> AuthStatus:
        """Report whether the `ee` client can reach Earth Engine.

        Scaffolding — not yet implemented (design §8).

        Raises:
            NotImplementedError: Always, until the adapter is implemented.
        """
        raise NotImplementedError("GEESource.auth_status is not implemented yet.")
