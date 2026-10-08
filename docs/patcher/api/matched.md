# Matched multi-source patching — `geopatcher.matched`

Co-located patching across N sources: a `MatchedField` wraps one
primary `Field` plus named secondaries and per-secondary coregistration
callables, and is split and merged per source by the matched patchers. See
ADR-003 in [Design decisions](../decisions.md) and the
[query → matchup → patch design](../design/query-matchup.md).

```python
from geopatcher.matched import MatchedField, MatchedSpatialPatcher
```

## Field

::: geopatcher.matched.MatchedField

## Carriers

::: geopatcher.matched.MatchedPatch
::: geopatcher.matched.MatchedTemporalPatch
::: geopatcher.matched.MatchedSpatioTemporalPatch

## Patchers

::: geopatcher.matched.MatchedSpatialPatcher
::: geopatcher.matched.MatchedTemporalPatcher
::: geopatcher.matched.MatchedSpatioTemporalPatcher
