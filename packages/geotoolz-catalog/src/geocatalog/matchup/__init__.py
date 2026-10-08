"""`geocatalog.matchup` — join rows from two sources in space and time.

`matchup` pairs each primary row with the secondary rows that satisfy a
spatial and a temporal strategy, and yields `MatchupRow` records
(persisted to ``matchups.parquet`` by `geocatalog.storage.CatalogBundle`).

- Spatial strategies (`SpatialStrategy`): `Intersects`, `IouAtLeast`,
  `CentroidWithin`, `Contains`.
- Temporal strategies (`TemporalStrategy`): `NearestInTime`,
  `WithinWindow`, `Synchronous`.

Either side can be a catalog, a `CatalogBundle` or an iterable of
`geocatalog.sources.SourceRow`.

```python
from geocatalog.matchup import Intersects, MatchupRow, NearestInTime, matchup

pairs: list[MatchupRow] = list(
    matchup(sentinel2, landsat, spatial=Intersects(), temporal=NearestInTime("1D"))
)
```
"""

from __future__ import annotations

from geocatalog._src.matchup import (
    CentroidWithin,
    Contains,
    Intersects,
    IouAtLeast,
    MatchupRow,
    NearestInTime,
    SpatialStrategy,
    Synchronous,
    TemporalStrategy,
    WithinWindow,
    matchup,
)


__all__ = [
    "CentroidWithin",
    "Contains",
    "Intersects",
    "IouAtLeast",
    "MatchupRow",
    "NearestInTime",
    "SpatialStrategy",
    "Synchronous",
    "TemporalStrategy",
    "WithinWindow",
    "matchup",
]
